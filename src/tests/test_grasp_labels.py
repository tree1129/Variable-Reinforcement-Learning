"""Synthetic rigid-track unit tests, NOT independent real labeler validation."""
import copy
from dataclasses import asdict
import math
import unittest
import numpy as np
from tree_reinforcement_learning.grasp_labels import GraspCandidateConfig, GraspCandidateLabeler, label_track_records


def config():
    # Test-fixture thresholds only; never hardware/safety calibration.
    return GraspCandidateConfig(min_clearance_m=.015, min_lift_m=.015,
        min_follow_translation_m=.02, hold_s=.4, max_gap_s=.15,
        max_pair_skew_s=.02, max_relative_drift_m=.005,
        max_support_clearance_m=.002, min_confidence=.9)


def sample(i, supported=False, **overrides):
    t = round(i*.1, 7)
    ee = [i*.01, 0., .1]
    obj = [ee[0], 0., .08]
    s = {'episode_id': 'test-episode', 'scene_id': 'test-scene', 'case': 4,
         'object_id': 'block-A', 'target_object_id': 'block-A', 'track_id': 'track-A',
         'arm_id': 'right', 'frame_id': 'fixture-base', 'clock_id': 'fixture-clock',
         'calibration_id': 'synthetic-only-not-real-calibration', 'calibration_verified': True,
         'visible': True, 'support_contact_observed': supported, 'safety_violation': False,
         'confidence': .99, 't_s': t, 'ee_stamp_s': t, 'object_stamp_s': t,
         'object_position_m': obj, 'ee_position_m': ee,
         'ee_rotation_world_from_tool': np.eye(3).tolist(),
         'object_bottom_clearance_m': 0. if supported else .02}
    s.update(overrides)
    return s


def stable_sequence():
    return [sample(0, supported=True)] + [sample(i) for i in range(1, 8)]


class GraspLabelTest(unittest.TestCase):
    def setUp(self): self.labeler = GraspCandidateLabeler(config())

    def feed_stable(self):
        return [self.labeler.update(s) for s in stable_sequence()]

    def test_lift_follow_hold_is_only_an_unverified_candidate(self):
        rows = self.feed_stable()
        self.assertTrue(rows[-1]['candidate_grasp_stable'])
        for r in rows:
            self.assertFalse(r['label_verified'])
            self.assertFalse(r['trainable'])
            self.assertFalse(r['hardware_authorized'])
        self.assertEqual(rows[0]['status'], 'supported')
        self.assertFalse(rows[1]['candidate_grasp_stable'])

    def test_ee_lift_or_closed_jaws_alone_not_success(self):
        for i in range(8):
            s = sample(i, supported=True, object_position_m=[0, 0, 0], gripper_closed=True)
            r = self.labeler.update(s)
            self.assertFalse(r['candidate_grasp_stable'])

    def test_stationary_suspended_object_cannot_prove_follow(self):
        self.labeler.update(sample(0, supported=True))
        for i in range(1, 8):
            r = self.labeler.update(sample(i, ee_position_m=[0, 0, .1], object_position_m=[0, 0, .08]))
        self.assertFalse(r['candidate_grasp_stable'])

    def test_tool_rotation_uses_rigid_local_offset_not_world_offset(self):
        self.labeler.update(sample(0, supported=True))
        for i in range(1, 9):
            theta = i*.1
            R = np.array([[math.cos(theta), -math.sin(theta), 0],
                          [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
            ee = np.array([i*.01, 0, .1])
            r = self.labeler.update(sample(i, ee_position_m=ee.tolist(),
                object_position_m=(ee+R@np.array([.02, 0, -.02])).tolist(),
                ee_rotation_world_from_tool=R.tolist()))
        self.assertTrue(r['candidate_grasp_stable'])

    def test_drop_visibility_and_fault_are_distinct(self):
        self.feed_stable()
        self.assertEqual(self.labeler.update(sample(8, supported=True))['status'], 'candidate_drop')
        self.setUp(); self.feed_stable()
        self.assertEqual(self.labeler.update(sample(8, visible=False))['status'], 'unknown')
        self.setUp(); self.feed_stable()
        self.assertEqual(self.labeler.update(sample(8, safety_violation=True))['status'], 'fault')
        self.assertEqual(self.labeler.update(sample(9))['status'], 'fault')

    def test_wrong_object_identity_and_changed_frames(self):
        for key, value in [('object_id', 'wrong'), ('track_id', 'B'), ('frame_id', 'other'),
                           ('clock_id', 'other'), ('arm_id', 'left'), ('calibration_id', 'other')]:
            self.setUp(); self.feed_stable()
            r = self.labeler.update(sample(8, **{key: value}))
            self.assertEqual(r['status'], 'unknown')

    def test_missing_baseline_never_stable(self):
        for i in range(8):
            self.assertFalse(self.labeler.update(sample(i))['candidate_grasp_stable'])

    def test_gaps_duplicate_and_stale_timestamps_break_hold(self):
        for times in ((2., 2., 2.), (.7, .7, .7), (.8, .5, .8)):
            self.setUp(); self.feed_stable()
            r = self.labeler.update(sample(8, t_s=times[0], object_stamp_s=times[1], ee_stamp_s=times[2]))
            self.assertEqual(r['status'], 'unknown')
            # Returning to clean tracking without a new supported baseline cannot recover a positive.
            self.assertFalse(self.labeler.update(sample(9))['candidate_grasp_stable'])

    def test_slip_does_not_remain_stable(self):
        self.feed_stable()
        r = self.labeler.update(sample(8, object_position_m=[.2, 0, .08]))
        self.assertEqual(r['status'], 'unknown')
        self.assertIn('rigidly', r['reason'])

    def test_reset_episode_has_no_carried_hold(self):
        self.feed_stable()
        r = self.labeler.update(sample(8, episode_id='next-episode'))
        self.assertFalse(r['candidate_grasp_stable'])
        self.assertIn('baseline', r['reason'])

    def test_nan_shape_confidence_boolean_and_calibration_rejected(self):
        for key, value in [('object_position_m', [float('nan'), 0, 0]),
                           ('ee_position_m', ['0', '0', '0']), ('ee_rotation_world_from_tool', [[1]*3]*3),
                           ('confidence', .2), ('calibration_verified', False), ('visible', 1),
                           ('safety_violation', None), ('object_bottom_clearance_m', -1)]:
            self.setUp(); self.labeler.update(sample(0, supported=True))
            self.assertEqual(self.labeler.update(sample(1, **{key: value}))['status'], 'unknown')

    def test_support_and_large_clearance_contradiction(self):
        r = self.labeler.update(sample(0, supported=True, object_bottom_clearance_m=.1))
        self.assertEqual(r['status'], 'unknown')

    def test_invalid_config_is_rejected(self):
        for key, value in [('hold_s', 0), ('max_gap_s', float('nan')), ('min_confidence', 2),
                           ('max_pair_skew_s', .5), ('max_support_clearance_m', .1)]:
            d = asdict(config()); d[key] = value
            with self.assertRaises(ValueError): GraspCandidateConfig(**d)

    def test_batch_preserves_unknown_and_failure_records(self):
        samples = stable_sequence()+[sample(8, visible=False), sample(9, safety_violation=True)]
        out = label_track_records(samples, config())
        self.assertEqual(len(out), len(samples))
        self.assertEqual(out[-1]['status'], 'fault')
        self.assertEqual(out[-2]['status'], 'unknown')
