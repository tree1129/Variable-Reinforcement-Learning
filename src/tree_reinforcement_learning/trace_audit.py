"""Versioned, non-pickle trace checks; never authorizes robot motion.

The schema checks recorded evidence, not the truth of a camera-based label or
physical calibration. A valid file is eligible for offline data processing only.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np

SCHEMA = "v4_executed_trace_v1"
V4_CHECKPOINT = "/ssd/tree_openpi/checkpoints/pi05_tree_match_shape_v4/match_shape_grasp_state_v4/5999"
PROMPT = "match each shape block to its corresponding slot"
VECTORS = {"raw_v4_actions": 20, "base_commands": 20, "executed_commands": 20,
           "ee_feedback": 20, "next_ee_feedback": 20}
TIMES = ("observation_time_s", "command_time_s", "execution_time_s", "next_observation_time_s",
         "feedback_time_s", "next_feedback_time_s")
BOOLS = ("executed", "terminated", "truncated", "grasp_verified", "insertion_verified", "label_verified", "safety_violation")
STRINGS = ("episode_ids", "scene_ids", "splits", "phases", "action_origins")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_trace(path: Path, *, max_latency_s: float = .1, max_step_s: float = 1.) -> dict:
    path = Path(path)
    report = {"dataset": str(path), "trainable": False, "hardware_authorized": False,
              "source": "unknown", "reasons": [], "checks": {},
              "eligibility_scope": "offline_trace_processing_only_not_a_training_or_deployment_certificate",
              "limitations": ["Recorded declarations/hashes do not independently verify physical calibration or label accuracy.",
                              "Passing this schema does not connect V4 to the simulation residual trainer."]}
    reasons, checks = report["reasons"], report["checks"]
    if not np.isfinite([max_latency_s, max_step_s]).all() or not 0 < max_latency_s <= max_step_s:
        reasons.append("invalid_audit_timing_limits")
        return report
    try:
        with np.load(path, allow_pickle=False) as loaded:
            checks["keys"] = sorted(loaded.files)
            if "manifest_json" not in loaded.files:
                reasons.append("missing_manifest:unknown_provenance_not_real_v4_evidence")
                return report
            z = {k: loaded[k] for k in loaded.files}
        checks["dataset_sha256"] = sha256(path)
        manifest_value = z["manifest_json"]
        if manifest_value.shape != () or manifest_value.dtype.kind != "U":
            raise ValueError("manifest_json_must_be_unicode_scalar")
        m = json.loads(str(manifest_value))
        if not isinstance(m, dict):
            raise ValueError("manifest_must_be_object")
        report["source"] = m.get("source", "unknown")
        if m.get("schema") != SCHEMA:
            reasons.append("unsupported_schema")
        if report["source"] != "real_robot":
            reasons.append("not_real_robot_trace")
        checks["simulation_only"] = report["source"] in ("proxy_sim", "simulation")
        if m.get("base_checkpoint") != V4_CHECKPOINT or m.get("prompt") != PROMPT:
            reasons.append("v4_identity_or_prompt_mismatch")
        for field in ("base_checkpoint_sha256", "calibration_sha256"):
            if not isinstance(m.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", m[field]):
                reasons.append(f"invalid_manifest:{field}")
        # These identifiers must point to documented, independently reviewed contracts.
        for field in ("observation_contract", "command_contract", "frame_id", "clock_id", "automatic_labeler_version"):
            if not isinstance(m.get(field), str) or not m[field].strip():
                reasons.append(f"missing_manifest:{field}")
        if m.get("label_validation_passed") is not True:
            reasons.append("labeler_validation_missing")
        evidence = m.get("evidence_files", {})
        if not isinstance(evidence, dict):
            raise ValueError("invalid_evidence_manifest")
        for role in ("execution_log", "calibration", "label_validation"):
            entry = evidence.get(role, {})
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                reasons.append(f"missing_evidence:{role}")
                continue
            rel = Path(entry["path"])
            target = (path.parent / rel).resolve()
            if rel.is_absolute() or ".." in rel.parts or path.parent.resolve() not in target.parents:
                reasons.append(f"invalid_evidence_path:{role}")
                continue
            if not target.is_file() or sha256(target) != entry.get("sha256"):
                reasons.append(f"evidence_digest_mismatch:{role}")
            if role == "calibration" and entry.get("sha256") != m.get("calibration_sha256"):
                reasons.append("calibration_identity_mismatch")
        required = set(VECTORS) | set(TIMES) | set(BOOLS) | set(STRINGS) | {"observations", "next_observations", "cases", "rewards"}
        missing = sorted(required - z.keys())
        if missing:
            reasons.extend(f"missing:{k}" for k in missing)
            return report
        obs = z["observations"]
        if obs.ndim != 2 or not obs.shape[0] or not obs.shape[1]:
            raise ValueError("empty_or_invalid_observations")
        n = obs.shape[0]
        checks["transitions"] = n
        checks["observation_dim"] = obs.shape[1]
        numeric_shapes = {**{k: (n, d) for k, d in VECTORS.items()}, "observations": obs.shape,
                          "next_observations": obs.shape, **{k: (n,) for k in (*TIMES, "rewards")}}
        for key, shape in numeric_shapes.items():
            a = z[key]
            if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
                reasons.append(f"invalid_numeric_shape_or_finite:{key}")
        for key in BOOLS:
            if z[key].shape != (n,) or z[key].dtype.kind != "b":
                reasons.append(f"invalid_boolean:{key}")
        for key in STRINGS:
            if z[key].shape != (n,) or z[key].dtype.kind != "U" or np.any(z[key] == ""):
                reasons.append(f"invalid_string_array:{key}")
        if z["cases"].shape != (n,) or z["cases"].dtype.kind not in "iu" or not np.isin(z["cases"], [4, 5, 6, 7]).all():
            reasons.append("invalid_cases")
        # Shape/type failures must not fall through to broadcasting operations.
        if any(s.startswith("invalid_") for s in reasons):
            return report
        if not z["executed"].all() or not np.isin(z["action_origins"], ["v4", "executed_residual"]).all():
            reasons.append("unexecuted_or_shadow_actions_cannot_have_observed_outcomes")
        base_only = z["action_origins"] == "v4"
        if not np.allclose(z["base_commands"][base_only], z["executed_commands"][base_only], rtol=0., atol=1e-6):
            reasons.append("v4_only_command_mismatch")
        if not z["label_verified"].all():
            reasons.append("unverified_outcome_labels")
        if not np.isin(z["phases"], ["grasp", "align"]).all():
            reasons.append("invalid_phase")
        if not np.isin(z["splits"], ["train", "validation", "holdout"]).all():
            reasons.append("invalid_split")
        if np.any(z["terminated"] & z["truncated"]):
            reasons.append("terminated_and_truncated")
        if np.any(z["safety_violation"] & ~z["terminated"]):
            reasons.append("safety_fault_not_terminal")
        t, c, e, nt = (z[k] for k in TIMES[:4])
        if np.any((c < t) | (e < c) | (nt <= e) | (e-t > max_latency_s) | (nt-t > max_step_s)):
            reasons.append("invalid_transition_timing")
        if np.any(np.abs(z["feedback_time_s"] - t) > max_latency_s) or np.any(np.abs(z["next_feedback_time_s"] - nt) > max_latency_s):
            reasons.append("feedback_out_of_sync")
        for episode in np.unique(z["episode_ids"]):
            idx = np.flatnonzero(z["episode_ids"] == episode)
            if np.any(np.diff(idx) != 1):
                reasons.append(f"noncontiguous_episode:{episode}")
            for key in ("splits", "scene_ids", "cases"):
                if len(np.unique(z[key][idx])) != 1:
                    reasons.append(f"episode_mixed_{key}:{episode}")
            done = z["terminated"][idx] | z["truncated"][idx]
            if not done[-1] or done[:-1].any():
                reasons.append(f"invalid_episode_boundary:{episode}")
            if len(idx) > 1:
                if np.any(np.diff(t[idx]) <= 0) or not np.allclose(nt[idx[:-1]], t[idx[1:]], rtol=0., atol=1e-6):
                    reasons.append(f"episode_time_discontinuity:{episode}")
                for a, b in (("next_observations", "observations"), ("next_ee_feedback", "ee_feedback")):
                    if not np.allclose(z[a][idx[:-1]], z[b][idx[1:]], rtol=0., atol=1e-6):
                        reasons.append(f"episode_state_discontinuity:{episode}:{a}")
        for scene in np.unique(z["scene_ids"]):
            if len(np.unique(z["splits"][z["scene_ids"] == scene])) != 1:
                reasons.append(f"scene_split_leakage:{scene}")
        for split in ("train", "validation", "holdout"):
            if set(z["cases"][z["splits"] == split].tolist()) != {4, 5, 6, 7}:
                reasons.append(f"missing_case_coverage:{split}")
        checks["episodes"] = len(np.unique(z["episode_ids"]))
        checks["split_counts"] = {s: int(np.sum(z["splits"] == s)) for s in ("train", "validation", "holdout")}
        checks["safety_failure_transitions"] = int(z["safety_violation"].sum())
        checks["grasp_positive_labels"] = int(z["grasp_verified"].sum())
        checks["insertion_positive_labels"] = int(z["insertion_verified"].sum())
        # Verified failures remain useful transitions; success-only selection is not required.
        report["trainable"] = not reasons
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        reasons.append(f"load_or_schema_error:{exc}")
    return report
