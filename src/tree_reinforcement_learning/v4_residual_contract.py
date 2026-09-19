"""Pure SHADOW arithmetic for decoded 20-D V4 absolute commands.

This is NOT a V4 client, phase estimator, safety shield, or robot controller.
Apply only AFTER denormalization and TreeAbsoluteActions, to ONE command, with
explicitly verified arm ordering/frame/gripper units. Never add to 32-D latent
or geometric delta actions. The training proxy uses a different contract.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

CONTRACT = "absolute_position_m_rotation6d_rows_gripper_v1"


class ContractBlocked(ValueError):
    """No usable action on a fault or an unverified contract; do not fallback."""


@dataclass(frozen=True)
class DecodedResidualLimits:
    # Caller-supplied test limits; no values here are certified hardware limits.
    position_m: float
    rotation_rad: float
    gripper_delta: float
    gripper_bounds: tuple[float, float]

    def validate(self):
        values = np.asarray([self.position_m, self.rotation_rad, self.gripper_delta, *self.gripper_bounds])
        if values.shape != (5,) or not np.isfinite(values).all() or np.any(values[:3] < 0) or values[3] >= values[4]:
            raise ContractBlocked("invalid_explicit_limits")


def rows6_to_matrix(value):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (6,) or not np.isfinite(a).all():
        raise ContractBlocked("invalid_rotation6d")
    r1, r2 = a[:3], a[3:]
    # Decoded commands must already be orthonormal. Do not silently repair bad
    # telemetry/raw network outputs and pretend the physical contract is valid.
    if not np.allclose([np.linalg.norm(r1), np.linalg.norm(r2), np.dot(r1, r2)], [1, 1, 0], rtol=0, atol=1e-4):
        raise ContractBlocked("rotation6d_not_decoded_orthonormal_rows")
    return np.stack((r1, r2, np.cross(r1, r2)))


def _unit_ball(value):
    return value / max(1., float(np.linalg.norm(value)))


def _rotation_vector_to_matrix(value):
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = value / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k @ k)


def compose_decoded_v4_shadow(base_command, residual, *, contract_id, arm_index, frame_id,
                              stage, phase, grasp_verified, limits):
    """Return a non-executable candidate; fault/unknown -> exception, not V4 motion.

    Residual position/rotation are in the explicitly named command BASE frame.
    Rotations use R_delta @ R_base (SO(3) composition), not 6-D addition.
    """
    if contract_id != CONTRACT or not isinstance(frame_id, str) or not frame_id.strip():
        raise ContractBlocked("unverified_contract_or_frame")
    if type(arm_index) is not int or arm_index not in (0, 1):
        raise ContractBlocked("explicit_arm_index_required")
    if stage not in ("grasp", "align") or phase not in ("grasp", "align", "transport", "idle"):
        raise ContractBlocked("fault_unknown_or_unsupported_phase_stop_required")
    if type(grasp_verified) is not bool:
        raise ContractBlocked("grasp_confirmation_must_be_explicit_boolean")
    limits.validate()
    base = np.asarray(base_command, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    if base.shape != (20,) or r.shape != (7,) or not np.isfinite(base).all() or not np.isfinite(r).all():
        raise ContractBlocked("requires_one_decoded_command20_and_finite_residual7")
    rotations = [rows6_to_matrix(base[i+3:i+9]) for i in (0, 10)]
    low, high = limits.gripper_bounds
    if np.any(base[[9, 19]] < low) or np.any(base[[9, 19]] > high):
        raise ContractBlocked("gripper_units_or_bounds_mismatch")
    if stage == "align" and phase == "align" and not grasp_verified:
        raise ContractBlocked("alignment_requires_verified_grasp")
    enabled = stage == phase and not (stage == "grasp" and grasp_verified)
    candidate = base.copy()
    if enabled:
        offset = 10*arm_index
        dp = limits.position_m * _unit_ball(np.clip(r[:3], -1., 1.))
        dr = limits.rotation_rad * _unit_ball(np.clip(r[3:6], -1., 1.))
        candidate[offset:offset+3] += dp
        if np.any(dr):
            rotation = _rotation_vector_to_matrix(dr) @ rotations[arm_index]
            candidate[offset+3:offset+9] = rotation[:2].reshape(6)
        candidate[offset+9] = np.clip(base[offset+9] + limits.gripper_delta*np.clip(r[6], -1., 1.), low, high)
    return {"candidate_command": candidate, "enabled": enabled, "hardware_authorized": False,
            "mode": "arithmetic_shadow_only", "contract_id": contract_id, "frame_id": frame_id,
            "stage": stage, "phase": phase, "arm_index": arm_index}
