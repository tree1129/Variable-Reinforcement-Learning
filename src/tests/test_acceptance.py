from __future__ import annotations
import copy
import unittest
from tree_reinforcement_learning.acceptance import compare_evaluations, assert_disjoint_evaluations


def suites():
    return {name: {"records": [{"case": c, "seed": offset+c*100+i, "success": i == 0,
                               "score": 10. if i == 0 else 0., "safety_violation": False}
                              for c in range(4, 8) for i in range(2)]}
            for name, offset in (("fixed", 0), ("randomized", 10000))}


class AcceptanceTests(unittest.TestCase):
    def test_equal_is_nonregression_not_hardware_authorization(self):
        result = compare_evaluations(suites(), suites())
        self.assertTrue(result["accepted"])
        self.assertFalse(result["hardware_authorized"])

    def test_no_cross_suite_tradeoff(self):
        old = suites(); new = copy.deepcopy(old)
        new["randomized"]["records"][1].update(success=True, score=10.)
        new["fixed"]["records"][0].update(success=False, score=0.)
        self.assertFalse(compare_evaluations(old, new)["accepted"])

    def test_no_case_tradeoff_with_equal_aggregate(self):
        old = suites(); new = copy.deepcopy(old)
        new["fixed"]["records"][0].update(success=False, score=0.)
        new["fixed"]["records"][3].update(success=True, score=10.)
        self.assertFalse(compare_evaluations(old, new)["accepted"])

    def test_score_cannot_drop_when_success_increases(self):
        old = suites(); new = copy.deepcopy(old)
        new["fixed"]["records"][1].update(success=True, score=1.)
        new["fixed"]["records"][0].update(score=8.)
        self.assertFalse(compare_evaluations(old, new)["accepted"])

    def test_missing_duplicate_and_unpaired_seeds(self):
        for mutate in (lambda rs: rs.pop(), lambda rs: rs.append(rs[0]), lambda rs: rs[0].update(seed=99)):
            old = suites(); new = copy.deepcopy(old)
            mutate(new["fixed"]["records"])
            self.assertFalse(compare_evaluations(old, new)["accepted"])

    def test_nan_summary_or_metric_and_bad_boolean(self):
        for field, value in (("score", float("nan")), ("score", float("inf")), ("success", 1)):
            new = suites(); new["fixed"]["records"][0][field] = value
            self.assertFalse(compare_evaluations(suites(), new)["accepted"])
        new = suites(); new["fixed"]["successes"] = 8
        self.assertFalse(compare_evaluations(suites(), new)["accepted"])

    def test_unsafe_baseline_or_candidate_rejected(self):
        old = suites(); new = suites()
        old["fixed"]["records"][1]["safety_violation"] = True
        self.assertFalse(compare_evaluations(old, new)["accepted"])
        self.assertFalse(compare_evaluations(new, old)["accepted"])

    def test_holdout_must_be_disjoint(self):
        with self.assertRaises(ValueError):
            assert_disjoint_evaluations(suites(), suites())
        holdout = suites()
        for s in holdout.values():
            for r in s["records"]:
                r["seed"] += 1000000
        assert_disjoint_evaluations(suites(), holdout)

    def test_missing_suite_and_missing_case(self):
        self.assertFalse(compare_evaluations(suites(), {})["accepted"])
        candidate = suites()
        candidate["fixed"]["records"] = candidate["fixed"]["records"][:-2]
        self.assertFalse(compare_evaluations(suites(), candidate)["accepted"])
