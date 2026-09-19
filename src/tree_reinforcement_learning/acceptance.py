"""Fail-closed, paired evaluation gates. Passing is NOT hardware authorization."""
from __future__ import annotations

import math
from collections import defaultdict


def _validate(suite):
    if not isinstance(suite, dict) or not isinstance(suite.get("records"), list) or not suite["records"]:
        raise ValueError("missing_records")
    records = suite["records"]
    keys, groups = set(), defaultdict(list)
    for r in records:
        if not isinstance(r, dict):
            raise ValueError("invalid_record")
        case, seed = r.get("case"), r.get("seed")
        if type(case) is not int or case not in (4, 5, 6, 7) or type(seed) is not int:
            raise ValueError("invalid_case_or_seed")
        if (case, seed) in keys:
            raise ValueError("duplicate_case_seed")
        keys.add((case, seed))
        if type(r.get("success")) is not bool or type(r.get("safety_violation")) is not bool:
            raise ValueError("invalid_boolean")
        score = r.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 10:
            raise ValueError("invalid_score")
        if r["success"] and r["safety_violation"]:
            raise ValueError("success_with_safety_violation")
        groups[case].append(r)
    if set(groups) != {4, 5, 6, 7}:
        raise ValueError("missing_case")
    metrics = {c: {"episodes": len(rs), "successes": sum(r["success"] for r in rs),
                   "mean_score": sum(r["score"] for r in rs) / len(rs),
                   "safety_failures": sum(r["safety_violation"] for r in rs)} for c, rs in groups.items()}
    aggregate = {"episodes": len(records), "successes": sum(r["success"] for r in records),
                 "mean_score": sum(r["score"] for r in records) / len(records),
                 "safety_failures": sum(r["safety_violation"] for r in records)}
    for k, v in aggregate.items():
        if k in suite:
            value = suite[k]
            if type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(value, v, abs_tol=1e-9):
                raise ValueError(f"inconsistent_summary:{k}")
    return keys, metrics


def compare_evaluations(reference: dict, candidate: dict) -> dict:
    """Every suite AND every case must independently preserve success and score.

    Seeds/episode counts must match. Zero observed safety failures is required
    for both reference and candidate. No statistical confidence is implied.
    """
    reasons, comparisons = [], {}
    if not reference or set(reference) != set(candidate):
        return {"accepted": False, "reasons": ["evaluation_suite_mismatch"], "hardware_authorized": False}
    for name in sorted(reference):
        try:
            ref_keys, ref = _validate(reference[name])
            cand_keys, cand = _validate(candidate[name])
            if ref_keys != cand_keys:
                raise ValueError("paired_seeds_or_counts_mismatch")
            comparisons[name] = {"reference": ref, "candidate": cand}
            for case in sorted(ref):
                prefix = f"{name}:case{case}"
                if ref[case]["safety_failures"] or cand[case]["safety_failures"]:
                    reasons.append(f"{prefix}:safety_failure")
                for metric in ("successes", "mean_score"):
                    if cand[case][metric] < ref[case][metric]:
                        reasons.append(f"{prefix}:regressed:{metric}")
        except (ValueError, TypeError, KeyError) as exc:
            reasons.append(f"{name}:invalid:{exc}")
    return {"accepted": not reasons, "reasons": reasons, "comparisons": comparisons,
            "hardware_authorized": False, "evidence_scope": "paired_observed_episodes_only"}


def assert_disjoint_evaluations(selection: dict, holdout: dict) -> None:
    """Holdout is evaluated only after selection; caller must avoid reusing it."""
    def seeds(suites):
        return set().union(*(_validate(s)[0] for s in suites.values()))
    if seeds(selection) & seeds(holdout):
        raise ValueError("selection_and_holdout_overlap")
