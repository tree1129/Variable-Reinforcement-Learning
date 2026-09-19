from __future__ import annotations
import unittest
import numpy as np
from tree_reinforcement_learning.v4_residual_contract import (
    CONTRACT, ContractBlocked, DecodedResidualLimits, compose_decoded_v4_shadow, rows6_to_matrix)


class V4ContractTests(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros(20)
        self.base[3:9] = [1, 0, 0, 0, 1, 0]
        self.base[13:19] = [1, 0, 0, 0, 1, 0]
        self.base[[9, 19]] = .5
        self.kw = dict(contract_id=CONTRACT, arm_index=1, frame_id="TEST_ONLY_base_frame",
                       stage="grasp", phase="grasp", grasp_verified=False,
                       limits=DecodedResidualLimits(.005, np.deg2rad(5), .02, (0., 1.)))
    def test_zero_residual_preserves_both_arms_exactly(self):
        result = compose_decoded_v4_shadow(self.base, np.zeros(7), **self.kw)
        np.testing.assert_array_equal(result["candidate_command"], self.base)
        self.assertFalse(result["hardware_authorized"])
    def test_so3_composition_norm_bounds_and_other_arm_unchanged(self):
        result = compose_decoded_v4_shadow(self.base, np.ones(7), **self.kw)
        command = result["candidate_command"]
        np.testing.assert_array_equal(command[:10], self.base[:10])
        self.assertLessEqual(np.linalg.norm(command[10:13]-self.base[10:13]), .005+1e-12)
        rotation = rows6_to_matrix(command[13:19])
        np.testing.assert_allclose(rotation@rotation.T, np.eye(3), atol=1e-12)
        angle = np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))
        self.assertLessEqual(angle, np.deg2rad(5)+1e-12)
        self.assertAlmostEqual(command[19], .52)
    def test_rotation_residual_is_base_frame_left_composition(self):
        self.kw["arm_index"] = 0
        self.base[3:9] = [1, 0, 0, 0, 0, -1]  # Rx(90 deg)
        residual = np.zeros(7); residual[4] = 1
        result = compose_decoded_v4_shadow(self.base, residual, **self.kw)
        angle = np.deg2rad(5)
        ry = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        expected = ry @ rows6_to_matrix(self.base[3:9])
        np.testing.assert_allclose(rows6_to_matrix(result["candidate_command"][3:9]), expected, atol=1e-12)
        np.testing.assert_array_equal(result["candidate_command"][10:], self.base[10:])

    def test_gate_does_not_add_both_stages(self):
        self.kw.update(phase="transport")
        result = compose_decoded_v4_shadow(self.base, np.ones(7), **self.kw)
        self.assertFalse(result["enabled"])
        np.testing.assert_array_equal(result["candidate_command"], self.base)
    def test_alignment_without_verified_grasp_blocks(self):
        self.kw.update(stage="align", phase="align")
        with self.assertRaises(ContractBlocked):
            compose_decoded_v4_shadow(self.base, np.zeros(7), **self.kw)
        self.kw["grasp_verified"] = True
        self.assertTrue(compose_decoded_v4_shadow(self.base, np.zeros(7), **self.kw)["enabled"])
    def test_bad_frame_rotation_nan_or_chunk_blocks(self):
        for command in (np.zeros(32), np.zeros((32, 20)), np.zeros(20), np.full(20, np.nan)):
            with self.assertRaises(ContractBlocked): compose_decoded_v4_shadow(command, np.zeros(7), **self.kw)
        self.kw["frame_id"] = ""
        with self.assertRaises(ContractBlocked): compose_decoded_v4_shadow(self.base, np.zeros(7), **self.kw)
    def test_fault_stops_instead_of_falling_back_to_base_action(self):
        self.kw["phase"] = "fault"
        with self.assertRaises(ContractBlocked): compose_decoded_v4_shadow(self.base, np.zeros(7), **self.kw)
