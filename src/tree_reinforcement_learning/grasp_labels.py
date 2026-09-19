"""Conservative *candidate* grasp labels from externally calibrated object tracks.

This is not a tracker, calibration tool, validated success detector, reward
function, or controller. Every result remains label_verified=False. Closed jaws,
EE lift alone, unknown visibility, and a dispatcher ACK cannot prove a grasp.
Thresholds must be supplied by the caller; no production safety defaults exist.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
import numpy as np


@dataclass(frozen=True)
class GraspCandidateConfig:
    min_clearance_m: float
    min_lift_m: float
    min_follow_translation_m: float
    hold_s: float
    max_gap_s: float
    max_pair_skew_s: float
    max_relative_drift_m: float
    max_support_clearance_m: float
    min_confidence: float

    def __post_init__(self):
        for f in fields(self):
            x = getattr(self, f.name)
            if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
                raise ValueError('invalid_candidate_threshold:' + f.name)
        if self.min_confidence > 1 or self.max_support_clearance_m >= self.min_clearance_m:
            raise ValueError('invalid_confidence_or_clearance_order')
        if self.max_pair_skew_s > self.max_gap_s or self.hold_s < self.max_gap_s:
            raise ValueError('invalid_candidate_time_limits')


def _vector(value, shape, name):
    # Numeric strings/bools must not be silently accepted as geometric evidence.
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError('invalid_' + name)
    return a.astype(float)


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('invalid_' + name)
    return float(value)


class GraspCandidateLabeler:
    def __init__(self, config: GraspCandidateConfig):
        self.config = config
        self.identity = None
        self.episode = None
        self.last_t = None
        self.fault_latched = False
        self._clear_evidence()

    def _clear_evidence(self):
        self.baseline_clearance = None
        self.lift_t = None
        self.lift_ee = None
        self.local_offset = None
        self.frames = 0
        self.stable = False

    def _result(self, status, reason, **metrics):
        return {'schema': 'grasp_candidate_label_v1', 'status': status, 'reason': reason,
                'candidate_grasp_stable': status == 'candidate_stable',
                'candidate_drop': status == 'candidate_drop',
                'label_verified': False, 'trainable': False, 'hardware_authorized': False,
                'required_before_verified': 'independent_heldout_labeler_and_calibration_validation',
                'metrics': metrics}

    def update(self, sample):
        try:
            if not isinstance(sample, dict):
                raise ValueError('sample_not_object')
            names = ('episode_id', 'scene_id', 'object_id', 'target_object_id', 'track_id',
                     'arm_id', 'frame_id', 'clock_id', 'calibration_id')
            for name in names:
                if not isinstance(sample.get(name), str) or not sample[name].strip():
                    raise ValueError('missing_identity:' + name)
            if sample['arm_id'] not in ('left', 'right'):
                raise ValueError('invalid_arm_id')
            if type(sample.get('case')) is not int or sample['case'] not in (4, 5, 6, 7):
                raise ValueError('invalid_case')
            for name in ('visible', 'support_contact_observed', 'safety_violation', 'calibration_verified'):
                if type(sample.get(name)) is not bool:
                    raise ValueError('missing_boolean:' + name)
            # These are unverified declarations, necessary but not sufficient.
            identity = tuple(sample[n] for n in names) + (sample['case'],)
            episode = sample['episode_id']
            if self.episode is None or episode != self.episode:
                self._clear_evidence()
                self.last_t = None
                self.fault_latched = False
                self.episode = episode
                self.identity = identity
            elif identity != self.identity:
                raise ValueError('identity_or_frame_changed_within_episode')
            if sample['safety_violation']:
                self.fault_latched = True
            if self.fault_latched:
                self._clear_evidence()
                return self._result('fault', 'episode_safety_fault_latched')
            if sample['object_id'] != sample['target_object_id']:
                raise ValueError('wrong_object_identity')
            if not sample['visible'] or not sample['calibration_verified']:
                raise ValueError('visibility_or_calibration_unverified')
            t = _number(sample.get('t_s'), 't_s')
            ot = _number(sample.get('object_stamp_s'), 'object_stamp_s')
            et = _number(sample.get('ee_stamp_s'), 'ee_stamp_s')
            if min(t, ot, et) < 0:
                raise ValueError('negative_time')
            c = self.config
            if max(t, ot, et) - min(t, ot, et) > c.max_pair_skew_s:
                raise ValueError('unsynchronized_object_and_ee')
            if self.last_t is not None and (t <= self.last_t or t-self.last_t > c.max_gap_s):
                # Do not use this frame to restart a hold across a missing interval.
                self.last_t = t
                raise ValueError('nonmonotonic_or_gapped_track')
            self.last_t = t
            confidence = _number(sample.get('confidence'), 'confidence')
            if not c.min_confidence <= confidence <= 1:
                raise ValueError('low_or_invalid_confidence')
            obj = _vector(sample.get('object_position_m'), (3,), 'object_position')
            ee = _vector(sample.get('ee_position_m'), (3,), 'ee_position')
            R = _vector(sample.get('ee_rotation_world_from_tool'), (3, 3), 'ee_rotation')
            if not np.allclose(R.T @ R, np.eye(3), atol=1e-5, rtol=0) or not np.isclose(np.linalg.det(R), 1., atol=1e-5):
                raise ValueError('ee_rotation_not_SO3')
            clearance = _number(sample.get('object_bottom_clearance_m'), 'object_bottom_clearance')
            if clearance < -c.max_support_clearance_m:
                raise ValueError('object_below_support_plane')
            local = R.T @ (obj-ee)
            supported = sample['support_contact_observed'] and abs(clearance) <= c.max_support_clearance_m
            if sample['support_contact_observed'] and not supported:
                raise ValueError('support_contact_clearance_contradiction')
            had_lift = self.lift_t is not None
            if supported:
                self._clear_evidence()
                self.baseline_clearance = clearance
                return self._result('candidate_drop' if had_lift else 'supported',
                                    'returned_to_support_after_lift' if had_lift else 'support_baseline_observed')
            if self.baseline_clearance is None:
                raise ValueError('missing_supported_baseline')
            lift = clearance-self.baseline_clearance
            if clearance < c.min_clearance_m or lift < c.min_lift_m:
                if had_lift:
                    self._clear_evidence()
                    return self._result('unknown', 'clearance_lost_without_confirmed_support_contact')
                return self._result('pending', 'insufficient_object_lift')
            if self.lift_t is None:
                self.lift_t, self.lift_ee, self.local_offset = t, ee, local
                self.frames = 1
                return self._result('pending', 'lift_observed_need_rigid_follow_and_hold')
            drift = float(np.linalg.norm(local-self.local_offset))
            if drift > c.max_relative_drift_m:
                raise ValueError('object_not_rigidly_following_tool')
            self.frames += 1
            travel = float(np.linalg.norm(ee-self.lift_ee))
            held = t-self.lift_t
            self.stable = self.stable or (held >= c.hold_s and travel >= c.min_follow_translation_m and self.frames >= 3)
            return self._result('candidate_stable' if self.stable else 'pending',
                                'rigid_follow_and_hold_observed' if self.stable else 'need_more_follow_and_hold',
                                hold_s=held, ee_translation_since_lift_m=travel,
                                local_object_drift_m=drift, object_lift_m=lift, frames=self.frames)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self._clear_evidence()
            return self._result('unknown', str(exc))


def label_track_records(records, config):
    """Keeps failures/unknowns; never filters the stream to successful frames."""
    labeler = GraspCandidateLabeler(config)
    return [{'input_index': i, **labeler.update(sample)} for i, sample in enumerate(records)]
