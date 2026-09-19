from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tree_reinforcement_learning.trace_audit import audit_trace, sha256, SCHEMA, V4_CHECKPOINT, PROMPT, BOOLS, VECTORS


def fixture(root):
    """SYNTHETIC TEST fixture; never used as real data or training evidence."""
    m = {"schema": SCHEMA, "source": "real_robot", "base_checkpoint": V4_CHECKPOINT, "prompt": PROMPT,
         "base_checkpoint_sha256": "a"*64, "observation_contract": "synthetic_test_observation",
         "command_contract": "synthetic_test_command", "frame_id": "test_frame", "clock_id": "test_clock",
         "automatic_labeler_version": "synthetic_test_only", "label_validation_passed": True, "evidence_files": {}}
    for role in ("execution_log", "calibration", "label_validation"):
        path = root / f"{role}.txt"
        path.write_text("SYNTHETIC TEST EVIDENCE, NOT REAL ROBOT DATA", encoding="utf-8")
        m["evidence_files"][role] = {"path": path.name, "sha256": sha256(path)}
    m["calibration_sha256"] = m["evidence_files"]["calibration"]["sha256"]
    n = 24
    ids = np.repeat(np.arange(12), 2)
    t = ids.astype(float) + np.tile([0., .1], 12)
    z = {"manifest_json": json.dumps(m), "observations": np.zeros((n, 5)), "next_observations": np.zeros((n, 5)),
         "episode_ids": np.array([f"ep{x}" for x in ids]), "scene_ids": np.array([f"scene{x}" for x in ids]),
         "splits": np.repeat(["train", "validation", "holdout"], 8), "cases": np.tile(np.repeat([4, 5, 6, 7], 2), 3),
         "phases": np.array(["grasp"]*n), "action_origins": np.array(["v4"]*n, dtype="U32"), "rewards": np.zeros(n),
         "observation_time_s": t, "command_time_s": t+.02, "execution_time_s": t+.03,
         "next_observation_time_s": t+.1, "feedback_time_s": t.copy(), "next_feedback_time_s": t+.1}
    z.update({k: np.zeros((n, d)) for k, d in VECTORS.items()})
    z.update({k: np.zeros(n, dtype=bool) for k in BOOLS})
    z["executed"][:] = True; z["label_verified"][:] = True
    z["terminated"][1::2] = True
    return z


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.z = fixture(self.root)
    def tearDown(self):
        self.temp.cleanup()
    def run_audit(self):
        path = self.root / "SYNTHETIC_TEST_ONLY.npz"
        np.savez(path, **self.z)
        return audit_trace(path)
    def test_valid_fixture_passes_but_never_hardware(self):
        report = self.run_audit()
        self.assertEqual(report["reasons"], [])
        self.assertTrue(report["trainable"])
        self.assertFalse(report["hardware_authorized"])
        self.assertEqual(report["checks"]["episodes"], 12)
    def test_legacy_provenance_unknown_not_automatically_simulation(self):
        del self.z["manifest_json"]
        report = self.run_audit()
        self.assertFalse(report["trainable"])
        self.assertEqual(report["source"], "unknown")
    def test_explicit_proxy_is_not_real(self):
        m = json.loads(self.z["manifest_json"]); m["source"] = "proxy_sim"
        self.z["manifest_json"] = json.dumps(m)
        report = self.run_audit()
        self.assertFalse(report["trainable"])
        self.assertTrue(report["checks"]["simulation_only"])
    def test_missing_fields_nan_and_wrong_shape(self):
        for key, value in (("observations", np.zeros((24,))), ("rewards", np.full(24, np.nan)), ("executed", np.ones(24))):
            original = self.z[key]; self.z[key] = value
            self.assertFalse(self.run_audit()["trainable"])
            self.z[key] = original
        del self.z["ee_feedback"]
        self.assertFalse(self.run_audit()["trainable"])
    def test_shadow_has_no_counterfactual_outcome(self):
        self.z["action_origins"][0] = "shadow_residual"
        self.assertFalse(self.run_audit()["trainable"])
        self.z["action_origins"][0] = "executed_residual"
        self.z["executed"][0] = False
        self.assertFalse(self.run_audit()["trainable"])
    def test_v4_action_must_equal_executed_command(self):
        self.z["executed_commands"][0, 0] = .01
        self.assertFalse(self.run_audit()["trainable"])
    def test_unsynced_feedback_and_nonmonotonic_time(self):
        self.z["feedback_time_s"][0] += 2
        self.assertFalse(self.run_audit()["trainable"])
        self.z["feedback_time_s"][0] -= 2
        self.z["observation_time_s"][1] = -1
        self.assertFalse(self.run_audit()["trainable"])
    def test_no_episode_or_scene_split_leakage(self):
        self.z["scene_ids"][8:10] = self.z["scene_ids"][0]
        self.assertFalse(self.run_audit()["trainable"])
    def test_episode_boundary_and_state_continuity(self):
        self.z["terminated"][0] = True
        self.assertFalse(self.run_audit()["trainable"])
        self.z["terminated"][0] = False
        self.z["next_observations"][0, 0] = 10
        self.assertFalse(self.run_audit()["trainable"])
    def test_unverified_label_and_evidence_digest(self):
        self.z["label_verified"][0] = False
        self.assertFalse(self.run_audit()["trainable"])
        self.z["label_verified"][0] = True
        (self.root / "calibration.txt").write_text("changed")
        self.assertFalse(self.run_audit()["trainable"])
    def test_evidence_path_cannot_escape_dataset_directory(self):
        m = json.loads(self.z["manifest_json"])
        m["evidence_files"]["calibration"]["path"] = "../outside"
        self.z["manifest_json"] = json.dumps(m)
        self.assertFalse(self.run_audit()["trainable"])
    def test_verified_failure_is_valid_transition_not_success_only_filter(self):
        self.z["safety_violation"][1] = True
        self.z["rewards"][1] = -10
        self.assertTrue(self.run_audit()["trainable"])
    def test_malformed_and_missing_file_fail_closed(self):
        self.assertFalse(audit_trace(self.root / "missing.npz")["trainable"])
        self.z["manifest_json"] = "[]"
        self.assertFalse(self.run_audit()["trainable"])
