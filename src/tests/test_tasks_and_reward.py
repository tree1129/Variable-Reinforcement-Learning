from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
src = PROJECT_ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from tree_reinforcement_learning.env import OBSERVATION_DIM, encode_observation
from tree_reinforcement_learning.reward import RewardConfig, benchmark_score, compute_reward
from tree_reinforcement_learning.tasks import CASES, FACE_ORDER, SHAPE_ORDER, build_prompt, get_task


def _observation(*, inserted=(False, False, False), retracted=False, forbidden_contact=False,
                 box_displacement_m=0.0, contact_force_n=0.0):
    """Create a minimal metric observation accepted by the independent API."""
    return {
        "object_positions": np.zeros((4, 3), dtype=np.float32),
        "object_orientations": np.zeros((4, 3), dtype=np.float32),
        "box_pose": np.zeros(6, dtype=np.float32),
        "hole_poses": np.zeros((3, 6), dtype=np.float32),
        "ee_pose": np.zeros(6, dtype=np.float32),
        "gripper": 1.0,
        "inserted": np.asarray(inserted, dtype=bool),
        "retracted": retracted,
        "grasped": False,
        "grasp_success": False,
        "forbidden_contact": forbidden_contact,
        "box_displacement_m": box_displacement_m,
        "contact_force_n": contact_force_n,
    }


class CaseDefinitionTests(unittest.TestCase):
    def test_case_mapping_and_excluded_objects(self):
        expected = {
            4: ({"front": "triangle", "right": "square", "left": "sphere"}, "trapezoid"),
            5: ({"front": "square", "right": "trapezoid", "left": "triangle"}, "sphere"),
            6: ({"front": "trapezoid", "right": "sphere", "left": "square"}, "triangle"),
            7: ({"front": "sphere", "right": "triangle", "left": "trapezoid"}, "square"),
        }
        for case_id, (faces, excluded) in expected.items():
            task = get_task(case_id)
            self.assertEqual(dict((face, shape) for shape, face in task.hole_faces.items()), faces)
            self.assertEqual(task.excluded, excluded)
            self.assertEqual(set(task.targets), set(SHAPE_ORDER) - {excluded})
            self.assertEqual(set(task.hole_faces.values()), set(FACE_ORDER))

    def test_prompt_contains_safety_and_retraction_instruction(self):
        for case_id in CASES:
            prompt = build_prompt(case_id)
            self.assertIn("Do not insert", prompt)
            self.assertIn("retract", prompt.lower())
            self.assertIn("Keep the coin box fixed", prompt)

    def test_observation_encoder_is_case_conditioned(self):
        observation = _observation()
        encoded = encode_observation(observation, 4)
        self.assertEqual(encoded.shape, (OBSERVATION_DIM,))
        # Case 4: front=triangle, right=square, left=sphere, excluded=trapezoid.
        self.assertEqual(encoded[64 + 0], 1.0)  # front / triangle
        self.assertEqual(encoded[64 + 5], 1.0)  # right / square
        self.assertEqual(encoded[64 + 11], 1.0)  # left / sphere
        self.assertEqual(encoded[64 + 2 + 12], 1.0)  # excluded trapezoid


class RewardAndScoreTests(unittest.TestCase):
    def test_approach_progress_rewards_reaching_next_target(self):
        previous = _observation()
        current = _observation()
        previous["ee_pose"][:3] = (-0.30, -0.30, 0.35)
        current["ee_pose"][:3] = (-0.18, 0.10, 0.35)
        previous["object_positions"][0] = (-0.18, 0.10, 0.03)
        current["object_positions"][0] = (-0.18, 0.10, 0.03)
        result = compute_reward(previous, current, np.zeros(7, dtype=np.float32), 4)
        self.assertGreater(result.reward, 0.0)

    def test_three_insertions_and_retraction_score_ten(self):
        previous = _observation()
        current = _observation(inserted=(True, True, True), retracted=True)
        result = compute_reward(previous, current, np.zeros(7, dtype=np.float32), 4)
        self.assertTrue(result.terminated)
        self.assertTrue(result.success)
        self.assertEqual(benchmark_score(current, 4), 10.0)

    def test_partial_insertion_score_is_three_points_each(self):
        self.assertEqual(benchmark_score(_observation(inserted=(True, False, False)), 4), 3.0)
        self.assertEqual(benchmark_score(_observation(inserted=(True, True, False)), 5), 6.0)
        self.assertEqual(benchmark_score(_observation(inserted=(True, True, True)), 6), 9.0)
        self.assertEqual(benchmark_score(_observation(inserted=(True, True, True), retracted=True), 7), 10.0)

    def test_forbidden_contact_is_safety_termination(self):
        previous = _observation()
        current = _observation(forbidden_contact=True)
        result = compute_reward(previous, current, np.zeros(7, dtype=np.float32), 4)
        self.assertTrue(result.terminated)
        self.assertFalse(result.success)
        self.assertTrue(result.safety_violation)
        self.assertIn("forbidden_contact", result.events)

    def test_box_motion_and_force_are_safety_violations(self):
        config = RewardConfig(max_box_displacement_m=0.005, max_contact_force_n=25.0)
        for current in (
            _observation(box_displacement_m=0.006),
            _observation(contact_force_n=25.1),
        ):
            result = compute_reward(_observation(), current, np.zeros(7, dtype=np.float32), 4, config)
            self.assertTrue(result.terminated)
            self.assertTrue(result.safety_violation)


if __name__ == "__main__":
    unittest.main()
