#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Integration-test failure triage → Serge auto-fix dispatcher.

Run daily (e.g. from a GitHub Actions cron). This script:

  1. Pulls the last N daily ``run_models_gpu`` reports from the CI dataset
     ``hf-internal-testing/transformers_daily_ci``.
  2. Keeps only **integration tests** (a pytest class whose name ends with
     ``IntegrationTest``/``IntegrationTests``) that failed on >= ``--min-days``
     of the last ``--window`` days (drops flakes).
  3. Classifies each surviving failure into a coarse mode (OOM, output_mismatch,
     cuda_runtime, load_error, import_or_config, other) and joins the latest
     day's CI ``git bisect`` attribution to cluster failures by ``bad_commit``.
  4. Picks the **most common failure group** — the largest bad-commit cluster,
     falling back to the largest failure-mode group when CI pinned nothing.
  5. Renders a Markdown report and hands the focused failure group to **Serge**
     (``POST /tasks``), which opens a PR proposing a fix. Serge runs no tests;
     verification stays in CI.

This is a self-contained port of the ``integration-failure-triage`` Space's
report pipeline (fetch + filter + classify + cluster). The HTML renderer, the
bucket-persist layer, the HTTP server, the 90-day history sweep, and the local
``git bisect`` helper are intentionally left out — none are needed to compute
the daily report and dispatch the top failure group.

Usage:

    # Dry-run: compute the report, print it + the Serge payload, POST nothing.
    python utils/integration_failure_triage.py --dry-run

    # Real run (from CI): mint an OIDC token, then dispatch to Serge.
    python utils/integration_failure_triage.py \\
        --repo huggingface/transformers \\
        --serge-url "$SERGE_URL" --base-ref main

Environment:
    HF_TOKEN           optional. The CI dataset is public, so anonymous access
                       works; only set this if the dataset is ever gated.
    SERGE_OIDC_TOKEN   GitHub Actions OIDC JWT (aud=serge) used as the bearer
                       token for ``POST /tasks``. Required unless --dry-run.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


# ─────────────────────────────────────────────────────────────────────────────
# Fetch — last N daily run_models_gpu reports from the CI dataset.
# ─────────────────────────────────────────────────────────────────────────────

CI_DATASET = "hf-internal-testing/transformers_daily_ci"
JOB_DIR = "ci_results_run_models_gpu"
MODEL_RESULTS = "model_results.json"
NEW_FAILURES = "new_failures_with_bad_commit_grouped_by_authors.json"


def list_recent_dates(api: HfApi, n: int = 7) -> list[str]:
    """Top-level dirs under the dataset look like YYYY-MM-DD. Return the n most recent."""
    files = api.list_repo_files(repo_id=CI_DATASET, repo_type="dataset")
    dates = set()
    for f in files:
        head = f.split("/", 1)[0]
        try:
            datetime.date.fromisoformat(head)
        except ValueError:
            continue
        dates.add(head)
    return sorted(dates, reverse=True)[:n]


def fetch_day(date: str, cache_dir: str | None = None) -> dict[str, dict | None]:
    """Download both JSONs for a given day; missing files return None instead of raising."""
    out: dict[str, dict | None] = {}
    for label, fname in (("model_results", MODEL_RESULTS), ("new_failures", NEW_FAILURES)):
        try:
            path = hf_hub_download(
                repo_id=CI_DATASET,
                repo_type="dataset",
                filename=f"{date}/{JOB_DIR}/{fname}",
                cache_dir=cache_dir,
            )
            with open(path) as f:
                out[label] = json.load(f)
        except EntryNotFoundError:
            out[label] = None
    return out


def fetch_last_n(n: int = 7, cache_dir: str | None = None) -> dict[str, dict[str, dict | None]]:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    dates = list_recent_dates(api, n)
    return {d: fetch_day(d, cache_dir=cache_dir) for d in dates}


# ─────────────────────────────────────────────────────────────────────────────
# Filter — integration tests only, intersected across the window.
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_SUFFIXES = ("IntegrationTest", "IntegrationTests")


def is_integration_test(test_path: str) -> bool:
    """`tests/models/foo/test_modeling_foo.py::FooIntegrationTest::test_x` → True."""
    if "::" not in test_path:
        return False
    cls = test_path.split("::")[1]
    return cls.endswith(INTEGRATION_SUFFIXES)


def model_name_from_key(key: str) -> str:
    """`models_align` → `align`. (CI keys model_results entries this way.)"""
    return key.removeprefix("models_")


def iter_failures(model_results: dict) -> Iterable[dict]:
    """Yield one record per (model, gpu, test) integration-test failure from a
    single day's ``model_results.json``."""
    for key, entry in model_results.items():
        if not isinstance(entry, dict):
            continue
        model = model_name_from_key(key)
        failures = entry.get("failures") or {}
        for gpu, items in failures.items():
            gpu = gpu.replace("-gpu", "")
            for it in items or []:
                test = it.get("line", "")
                if not is_integration_test(test):
                    continue
                yield {
                    "model": model,
                    "gpu": gpu,
                    "test": test,
                    "trace": (it.get("trace") or "").strip(),
                }


def per_day_integration_failures(
    daily: dict[str, dict[str, dict | None]],
) -> dict[str, list[dict]]:
    """`daily` is the output of `fetch_last_n`."""
    out: dict[str, list[dict]] = defaultdict(list)
    for date, payload in daily.items():
        mr = payload.get("model_results") if payload else None
        if not mr:
            continue
        out[date] = list(iter_failures(mr))
    return out


def intersect_across_days(per_day: dict[str, list[dict]], min_days: int = 5) -> list[dict]:
    """Keep `(model, gpu, test)` triples seen on >= min_days days, enriched with
    `days_seen`, `first_seen`, `latest_seen`, `latest_trace`."""
    seen: dict[tuple[str, str, str], dict] = {}
    for date in sorted(per_day):  # ascending so latest_trace ends up newest
        for rec in per_day[date]:
            key = (rec["model"], rec["gpu"], rec["test"])
            existing = seen.get(key)
            if existing is None:
                seen[key] = {
                    **rec,
                    "days_seen": 1,
                    "first_seen": date,
                    "latest_seen": date,
                    "latest_trace": rec["trace"],
                }
            else:
                existing["days_seen"] += 1
                existing["latest_seen"] = date
                existing["latest_trace"] = rec["trace"]
    return [r for r in seen.values() if r["days_seen"] >= min_days]


# ─────────────────────────────────────────────────────────────────────────────
# Classify — coarse failure mode from the raw trace.
# ─────────────────────────────────────────────────────────────────────────────

_OOM_PAT = re.compile(r"OutOfMemoryError|CUDA out of memory|MallocFailure|HIP out of memory", re.IGNORECASE)
_LOAD_PAT = re.compile(
    r"from_pretrained|safetensors\.|HFValidationError|Repository Not Found|gated|"
    r"Cannot read|UnboundLocalError.*loading|FileNotFoundError|access requested|"
    r"401 Client Error|403 Client Error",
    re.IGNORECASE,
)
_CUDA_RUNTIME_PAT = re.compile(
    r"CUDA error|CUBLAS_STATUS|CUDNN_STATUS|cudnn[_ ]frontend|nvrtc|"
    r"triton\.compiler|RuntimeError: Triton|c10::Error|NCCL.*error",
    re.IGNORECASE,
)
_OUTPUT_MISMATCH_PAT = re.compile(
    r"Tensor-likes are not close|"
    r"assertEqual|assertSequenceEqual|self\.assertListEqual|"
    r"assertAlmostEqual|assertGreater|expected_text|"
    r"AssertionError",  # generic fallback — most assertion failures are output mismatches
    re.IGNORECASE | re.DOTALL,
)
_IMPORT_CFG_PAT = re.compile(
    r"^.*ImportError|ModuleNotFoundError|"
    r"AttributeError:.*(config|object has no attribute)|"
    r"TypeError:.*(__init__|got an unexpected keyword argument)|"
    r"ValueError:.*Unrecognized configuration",
    re.IGNORECASE | re.MULTILINE,
)


def classify(trace: str) -> str:
    if not trace:
        return "other"
    for tag, pat in (
        ("OOM", _OOM_PAT),
        ("load_error", _LOAD_PAT),
        ("cuda_runtime", _CUDA_RUNTIME_PAT),
        ("import_or_config", _IMPORT_CFG_PAT),
        ("output_mismatch", _OUTPUT_MISMATCH_PAT),
    ):
        if pat.search(trace):
            return tag
    return "other"


def short_excerpt(trace: str, max_chars: int = 240) -> str:
    """Last non-empty line of the trace (the actual exception line), trimmed."""
    if not trace:
        return ""
    for line in reversed(trace.splitlines()):
        line = line.strip()
        if line:
            return (line[: max_chars - 1] + "…") if len(line) > max_chars else line
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Cluster — join CI bisect attribution and group by bad_commit.
# ─────────────────────────────────────────────────────────────────────────────

_GOOD_STATUS = "git bisect found the bad commit."


def _index_attribution(new_failures: dict) -> dict[tuple[str, str, str], dict]:
    """Flatten `{author -> {model -> {gpu -> [records]}}}` to
    `{(model, gpu, test) -> record}`. Adds `author` to each record."""
    out: dict[tuple[str, str, str], dict] = {}
    if not new_failures:
        return out
    for author, by_model in (new_failures or {}).items():
        if not isinstance(by_model, dict):
            continue
        for model, by_gpu in by_model.items():
            if not isinstance(by_gpu, dict):
                continue
            for gpu_label, items in by_gpu.items():
                gpu = gpu_label.replace("-gpu", "")
                for rec in items or []:
                    test = rec.get("test", "")
                    enriched = {**rec, "author": author if author != "null" else None}
                    out[(model, gpu, test)] = enriched
    return out


def cluster_failures(filtered: list[dict], new_failures_latest: dict | None) -> dict:
    """Produce the triage report data structure.

    Returns a dict with keys:
      `clusters`  {bad_commit: {meta..., failures: [...]}}, sorted by size desc
      `flaky`     [failure, ...] (CI marked status="flaky:...")
      `unpinned`  [failure, ...] (no trustworthy CI attribution found)
      `totals`    {total, clusters, in_clusters, flaky, unpinned}
    """
    attr = _index_attribution(new_failures_latest or {})

    clusters: dict[str, dict] = {}
    flaky: list[dict] = []
    unpinned: list[dict] = []

    for f in filtered:
        key = (f["model"], f["gpu"], f["test"])
        rec = attr.get(key)
        f = {**f, "failure_mode": classify(f.get("latest_trace") or f.get("trace") or "")}
        if rec is None:
            unpinned.append(f)
            continue
        status = rec.get("status") or ""
        if status.startswith("flaky"):
            flaky.append({**f, "status": status, "author": rec.get("author")})
            continue
        if status != _GOOD_STATUS:
            unpinned.append({**f, "status": status, "author": rec.get("author")})
            continue
        bc = rec.get("bad_commit")
        if not bc:
            unpinned.append({**f, "author": rec.get("author")})
            continue
        c = clusters.setdefault(
            bc,
            {
                "bad_commit": bc,
                "pr_number": rec.get("pr_number"),
                "author": rec.get("author"),
                "merged_by": rec.get("merged_by"),
                "parent": rec.get("parent"),
                "job_link": rec.get("job_link"),
                "failure_excerpt": (rec.get("failure_at_bad_commit") or "").strip(),
                "failures": [],
            },
        )
        c["failures"].append(f)

    clusters_sorted = dict(
        sorted(clusters.items(), key=lambda kv: (-len(kv[1]["failures"]), kv[1].get("author") or ""))
    )

    return {
        "clusters": clusters_sorted,
        "flaky": flaky,
        "unpinned": unpinned,
        "totals": {
            "total": len(filtered),
            "in_clusters": sum(len(c["failures"]) for c in clusters_sorted.values()),
            "clusters": len(clusters_sorted),
            "flaky": len(flaky),
            "unpinned": len(unpinned),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Target selection — the "most common failure group".
# ─────────────────────────────────────────────────────────────────────────────


def pick_target(report: dict) -> dict | None:
    """Choose the most common failure group to hand to Serge.

    Primary: the largest bad-commit cluster (CI bisect blamed one commit for the
    most tests). Fallback (no cluster pinned): the largest failure-mode group
    among the non-clustered failures (unpinned + flaky).

    Returns a normalized descriptor::

        {
          "kind": "cluster" | "failure_mode",
          "label": "...",            # human summary
          "failures": [...],         # the member failures
          "cluster": {...} | None,   # cluster meta when kind == "cluster"
        }

    or None when there are no failures at all.
    """
    clusters = report.get("clusters") or {}
    if clusters:
        # `clusters` is already sorted by size desc — take the first.
        bc, c = next(iter(clusters.items()))
        pr = c.get("pr_number")
        return {
            "kind": "cluster",
            "label": (
                f"{len(c['failures'])} integration tests regressed by commit "
                f"{bc[:12]}" + (f" (PR #{pr})" if pr else "")
            ),
            "failures": c["failures"],
            "cluster": c,
        }

    # Fallback: largest failure-mode group across non-clustered failures.
    pool = list(report.get("unpinned") or []) + list(report.get("flaky") or [])
    if not pool:
        return None
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for f in pool:
        by_mode[f.get("failure_mode") or "other"].append(f)
    mode, items = max(by_mode.items(), key=lambda kv: len(kv[1]))
    return {
        "kind": "failure_mode",
        "label": f"{len(items)} unattributed integration tests sharing failure mode `{mode}`",
        "failures": items,
        "cluster": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown rendering.
# ─────────────────────────────────────────────────────────────────────────────

_GH = "https://github.com/huggingface/transformers"


def _failure_lines(failures: list[dict], window_len: int, limit: int = 60) -> list[str]:
    lines: list[str] = []
    ordered = sorted(failures, key=lambda f: (f.get("failure_mode") or "", f["model"], f["gpu"], f["test"]))
    for f in ordered[:limit]:
        mode = f.get("failure_mode", "other")
        lines.append(f"- `{f['test']}` [{f['gpu']}-gpu] ({mode}, seen {f['days_seen']}/{window_len})")
        excerpt = short_excerpt(f.get("latest_trace") or f.get("trace") or "")
        if excerpt:
            lines.append(f"  - {excerpt}")
    if len(failures) > limit:
        lines.append(f"- … and {len(failures) - limit} more")
    return lines


def render_report(report: dict, target: dict | None, window: list[str]) -> str:
    """Full Markdown triage summary (for the action log / artifact)."""
    t = report["totals"]
    win = f"{window[0]} → {window[-1]}" if window else "?"
    out = [
        "# transformers · integration-test failure triage",
        "",
        f"Window `{win}` ({len(window)} daily runs) · generated "
        f"{datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## TL;DR",
        f"- **{t['total']}** persistent integration-test failures (>= window threshold)",
        f"- **{t['in_clusters']}** attributed to **{t['clusters']}** distinct bad commits (CI bisect)",
        f"- **{t['flaky']}** tagged flaky by CI",
        f"- **{t['unpinned']}** unpinned (CI bisect did not converge)",
        "",
    ]
    if target:
        out.append("## Most common failure group (dispatched to Serge)")
        out.append(f"**{target['label']}**")
        out.append("")
        out.extend(_failure_lines(target["failures"], len(window)))
        out.append("")
    if report["clusters"]:
        out.append("## Pinned clusters (CI bisect)")
        for bc, c in report["clusters"].items():
            pr = c.get("pr_number")
            pr_str = f"PR #{pr}" if pr else "no PR"
            out.append(f"- `{bc[:12]}` · {pr_str} · {c.get('author') or '?'} · {len(c['failures'])} failures")
        out.append("")
    return "\n".join(out)


def render_serge_context(target: dict, window: list[str]) -> str:
    """The focused, untrusted failure report Serge receives as `context`."""
    win = f"{window[0]} → {window[-1]}" if window else "?"
    out = [
        f"transformers integration-test failures — daily CI window {win}.",
        "",
        f"Most common failure group: {target['label']}.",
        "",
    ]
    c = target.get("cluster")
    if c:
        bc = c["bad_commit"]
        pr = c.get("pr_number")
        out.append("Attribution (from CI `git bisect`):")
        out.append(f"- bad commit: {bc} ({_GH}/commit/{bc})")
        if pr:
            out.append(f"- introduced by PR #{pr} ({_GH}/pull/{pr})")
        if c.get("author"):
            out.append(f"- author: {c['author']}  (merged by {c.get('merged_by') or '?'})")
        out.append("")
        modes = Counter(f.get("failure_mode", "other") for f in c["failures"])
        out.append("Failure-mode mix: " + ", ".join(f"{m} ({n})" for m, n in modes.most_common()))
        out.append("")

    out.append("Failing tests:")
    out.extend(_failure_lines(target["failures"], len(window), limit=200))
    out.append("")

    if c and c.get("failure_excerpt"):
        out.append("CI trace captured at the bad commit (truncated):")
        out.append("```")
        out.append(c["failure_excerpt"][:4000])
        out.append("```")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Serge dispatch — POST /tasks (GitHub Actions OIDC bearer).
# ─────────────────────────────────────────────────────────────────────────────

_INSTRUCTION = (
    "Fix the failing transformers integration tests described in the report below. "
    "The report identifies the single most common failure group from the latest daily "
    "CI run. Investigate the listed tests, determine the root cause of the shared "
    "failure, and propose a minimal patch that makes them pass without touching "
    "unrelated code. If the correct expected values genuinely changed, update them; "
    "if the regression is in library code, fix the library code. Do not run the test "
    "suite — CI will verify your PR."
)


def build_task_payload(repo: str, base_ref: str, context: str, title: str | None) -> dict:
    output: dict = {"mode": "new_pr", "branch_prefix": "serge/fix"}
    if title:
        output["title"] = title
    return {
        "repo": repo,
        "base_ref": base_ref,
        "instruction": _INSTRUCTION,
        "context": context,
        "output": output,
    }


def dispatch_to_serge(serge_url: str, token: str, payload: dict, timeout: int = 60) -> dict:
    """POST the task to Serge. Returns the parsed 202 response body."""
    url = serge_url.rstrip("/") + "/tasks"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise SystemExit(f"error: Serge POST /tasks failed: {e.code} {e.reason}\n{detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"error: could not reach Serge at {url}: {e.reason}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=int, default=7, help="number of recent daily CI reports to read")
    p.add_argument("--min-days", type=int, default=5, help="keep failures seen on >= this many days")
    p.add_argument("--cache-dir", default=os.environ.get("ITF_CACHE_DIR"), help="hf_hub_download cache dir")
    p.add_argument("--repo", default="huggingface/transformers", help="target repo for the Serge PR")
    p.add_argument("--base-ref", default="main", help="branch the fix PR starts from")
    p.add_argument(
        "--serge-url", default=os.environ.get("SERGE_URL"), help="Serge base URL (e.g. https://serge.example.com)"
    )
    p.add_argument("--report-out", help="write the full Markdown report to this path")
    p.add_argument("--dry-run", action="store_true", help="compute + print everything but POST nothing to Serge")
    args = p.parse_args(argv)

    print(f"[1/4] Fetching last {args.window} daily CI reports…", flush=True)
    daily = fetch_last_n(args.window, cache_dir=args.cache_dir)
    if not daily:
        print("error: no daily CI reports found", file=sys.stderr)
        return 2
    window = sorted(daily.keys())
    print(f"      window {window[0]} → {window[-1]}", flush=True)

    print(f"[2/4] Filter to IntegrationTest + >= {args.min_days}/{args.window} days…", flush=True)
    per_day = per_day_integration_failures(daily)
    kept = intersect_across_days(per_day, min_days=args.min_days)
    print(f"      {len(kept)} persistent integration-test failures", flush=True)

    print("[3/4] Cluster with CI bisect attribution + pick the most common group…", flush=True)
    nf_latest = daily[max(daily)].get("new_failures")
    report = cluster_failures(kept, nf_latest)
    target = pick_target(report)

    report_md = render_report(report, target, window)
    if args.report_out:
        with open(args.report_out, "w") as f:
            f.write(report_md)
        print(f"      wrote report to {args.report_out}", flush=True)
    print("\n" + report_md + "\n", flush=True)

    if target is None:
        print("[4/4] No integration-test failures to fix — nothing to dispatch. ✅", flush=True)
        return 0

    context = render_serge_context(target, window)
    title = f"Fix {target['label']}"[:120]
    payload = build_task_payload(args.repo, args.base_ref, context, title)

    print(f"[4/4] Most common group: {target['label']}", flush=True)

    if args.dry_run:
        print("\n--- DRY RUN: Serge POST /tasks payload ---", flush=True)
        print(json.dumps(payload, indent=2), flush=True)
        print("\n--- context (untrusted, fed to Serge) ---\n" + context, flush=True)
        return 0

    if not args.serge_url:
        print("error: --serge-url (or SERGE_URL) is required unless --dry-run", file=sys.stderr)
        return 2
    token = os.environ.get("SERGE_OIDC_TOKEN")
    if not token:
        print("error: SERGE_OIDC_TOKEN env var is required unless --dry-run", file=sys.stderr)
        return 2

    print(f"      dispatching to Serge at {args.serge_url} …", flush=True)
    resp = dispatch_to_serge(args.serge_url, token, payload)
    print(f"      ✅ Serge accepted task: {json.dumps(resp)}", flush=True)
    job_url = resp.get("url")
    if job_url:
        print(f"      follow: {args.serge_url.rstrip('/')}{job_url}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
