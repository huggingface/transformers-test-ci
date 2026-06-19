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

import os
import sys
import unittest


REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(REPO_PATH, "utils"))

import integration_failure_triage as itf  # noqa: E402


def _failure(model, gpu, test, trace, mode="output_mismatch", days=6):
    return {
        "model": model,
        "gpu": gpu,
        "test": test,
        "trace": trace,
        "latest_trace": trace,
        "days_seen": days,
        "failure_mode": mode,
    }


class FailureSignatureTest(unittest.TestCase):
    def test_known_symptoms(self):
        self.assertEqual(
            itf.failure_signature("AssertionError: Tensor-likes are not close!"),
            "tensor values differ",
        )
        self.assertEqual(
            itf.failure_signature("AssertionError: Tensor-likes are not equal!"),
            "tensor values differ",
        )
        self.assertEqual(
            itf.failure_signature("AssertionError: Lists differ: [1] != [2]"),
            "list output differs",
        )

    def test_fallback_to_exception_type(self):
        self.assertEqual(itf.failure_signature("ValueError: bad thing"), "ValueError")

    def test_empty_trace(self):
        self.assertEqual(itf.failure_signature(""), "unknown")


class PickTargetsGroupingTest(unittest.TestCase):
    def _report(self, unpinned, clusters=None, flaky=None):
        return {
            "clusters": clusters or {},
            "flaky": flaky or [],
            "unpinned": unpinned,
            "totals": {"total": len(unpinned)},
        }

    def test_groups_by_model_not_one_bucket(self):
        # The old behavior lumped all `output_mismatch` failures (across many
        # unrelated models) into a single unfixable bucket. Each model must now
        # be its own coherent group.
        unpinned = [
            _failure("dac", "single", "tests/models/dac/t.py::DacIntegrationTest::a", "Tensor-likes are not close!"),
            _failure("dac", "multi", "tests/models/dac/t.py::DacIntegrationTest::b", "Tensor-likes are not close!"),
            _failure("whisper", "single", "tests/models/whisper/t.py::WhisperIntegrationTest::c", "Lists differ: [1] != [2]"),
        ]
        targets = itf.pick_targets(self._report(unpinned))

        self.assertEqual(len(targets), 2)
        self.assertTrue(all(t["kind"] == "model_failures" for t in targets))
        models = {t["model"] for t in targets}
        self.assertEqual(models, {"dac", "whisper"})
        # No cross-model group leaks more than one model's failures.
        for t in targets:
            self.assertEqual({f["model"] for f in t["failures"]}, {t["model"]})

    def test_largest_group_first(self):
        unpinned = [
            _failure("solo", "single", "tests/models/solo/t.py::SoloIntegrationTest::a", "Tensor-likes are not close!"),
            _failure("big", "single", "tests/models/big/t.py::BigIntegrationTest::a", "Tensor-likes are not close!"),
            _failure("big", "multi", "tests/models/big/t.py::BigIntegrationTest::b", "Tensor-likes are not close!"),
        ]
        targets = itf.pick_targets(self._report(unpinned))
        self.assertEqual(targets[0]["model"], "big")
        self.assertEqual(len(targets[0]["failures"]), 2)

    def test_distinct_failure_modes_stay_separate(self):
        unpinned = [
            _failure("m", "single", "tests/models/m/t.py::MIntegrationTest::a", "Tensor-likes are not close!"),
            _failure("m", "single", "tests/models/m/t.py::MIntegrationTest::b", "CUDA out of memory", mode="OOM"),
        ]
        targets = itf.pick_targets(self._report(unpinned))
        self.assertEqual(len(targets), 2)
        self.assertEqual({t["failure_mode"] for t in targets}, {"output_mismatch", "OOM"})

    def test_clusters_rank_before_model_groups(self):
        unpinned = [
            _failure("m", "single", "tests/models/m/t.py::MIntegrationTest::a", "Tensor-likes are not close!"),
        ]
        clusters = {
            "deadbeef" * 5: {
                "bad_commit": "deadbeef" * 5,
                "pr_number": 123,
                "author": "octocat",
                "failures": [
                    _failure("x", "single", "tests/models/x/t.py::XIntegrationTest::a", "Tensor-likes are not close!"),
                ],
            }
        }
        targets = itf.pick_targets(self._report(unpinned, clusters=clusters))
        self.assertEqual(targets[0]["kind"], "cluster")
        self.assertEqual(targets[1]["kind"], "model_failures")

    def test_label_mentions_model_mode_and_signature(self):
        unpinned = [
            _failure("dac", "single", "tests/models/dac/t.py::DacIntegrationTest::a", "Tensor-likes are not close!"),
        ]
        label = itf.pick_targets(self._report(unpinned))[0]["label"]
        self.assertIn("`dac`", label)
        self.assertIn("output_mismatch", label)
        self.assertIn("tensor values differ", label)

    def test_fingerprints_differ_per_group(self):
        unpinned = [
            _failure("a", "single", "tests/models/a/t.py::AIntegrationTest::a", "Tensor-likes are not close!"),
            _failure("b", "single", "tests/models/b/t.py::BIntegrationTest::a", "Tensor-likes are not close!"),
        ]
        targets = itf.pick_targets(self._report(unpinned))
        fps = {itf.target_fingerprint(t) for t in targets}
        self.assertEqual(len(fps), 2)


if __name__ == "__main__":
    unittest.main()
