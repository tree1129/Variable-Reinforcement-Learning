#!/usr/bin/env python3
"""Headless MuJoCo adapter validation for the independent Case 4--7 project.

This is a backend/API smoke test, not a claim that the XML is a validated model
of the physical robot.  It deliberately runs without a renderer so it is safe
on an SSH-only A100 host.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
src = PROJECT_ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from tree_reinforcement_learning.env import OBSERVATION_DIM, encode_observation  # noqa: E402
from tree_reinforcement_learning.mujoco_backend import MujocoBackend  # noqa: E402
from tree_reinforcement_learning.tasks import SHAPE_ORDER, get_task  # noqa: E402

REQUIRED_SHAPES = {
    "object_positions": (4, 3),
    "object_orientations": (4, 3),
    "box_pose": (6,),
    "hole_poses": (3, 6),
    "ee_pose": (6,),
    "inserted": (3,),
}


def check_observation(observation: dict, case_id: int) -> None:
    for key, shape in REQUIRED_SHAPES.items():
        value = np.asarray(observation[key])
        if value.shape != shape:
            raise AssertionError(f"Case {case_id}: {key} has {value.shape}, expected {shape}")
        if not np.all(np.isfinite(value.astype(np.float64))):
            raise AssertionError(f"Case {case_id}: {key} contains NaN/Inf")
    encoded = encode_observation(observation, case_id)
    if encoded.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(encoded)):
        raise AssertionError(f"Case {case_id}: invalid encoded observation {encoded.shape}")
    if bool(np.any(observation["inserted"])):
        raise AssertionError(f"Case {case_id}: reset unexpectedly reports inserted objects")
    if bool(observation.get("forbidden_contact", False)):
        raise AssertionError(f"Case {case_id}: reset reports forbidden contact")
    if float(observation.get("box_displacement_m", 0.0)) != 0.0:
        raise AssertionError(f"Case {case_id}: reset box displacement is non-zero")


def validate_state_machine() -> None:
    """Exercise grasp/release transitions with a synthetic, permissive radius.

    The permissive insertion radius isolates the adapter state machine from the
    particular reachability of the external XML.  Physical reachability is
    checked separately by actual training/evaluation and is not inferred here.
    """
    backend = MujocoBackend(insertion_radius_m=10.0)
    try:
        backend.reset(4)
        target = "triangle"
        meta = backend.object_meta[target]
        adr = meta["qpos_adr"]
        backend.data.qpos[adr : adr + 3] = backend._ee_position() - np.asarray(
            (0.0, 0.0, backend.interaction_height_m), dtype=np.float64
        )
        close = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        backend.apply_action(close)
        backend.apply_action(close)
        if not backend.get_observation()["grasped"]:
            raise AssertionError("target grasp transition did not engage")
        open_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        backend.apply_action(open_action)
        backend.apply_action(open_action)
        observation = backend.get_observation()
        target_index = get_task(4).targets.index(target)
        if not bool(observation["inserted"][target_index]):
            raise AssertionError("target release transition did not mark insertion")

        backend.reset(4)
        excluded = get_task(4).excluded
        meta = backend.object_meta[excluded]
        adr = meta["qpos_adr"]
        backend.data.qpos[adr : adr + 3] = backend._ee_position() - np.asarray(
            (0.0, 0.0, backend.interaction_height_m), dtype=np.float64
        )
        backend.apply_action(close)
        backend.apply_action(close)
        backend.apply_action(open_action)
        backend.apply_action(open_action)
        if bool(np.any(backend.get_observation()["inserted"])):
            raise AssertionError("excluded object was incorrectly marked inserted")
    finally:
        backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=None, help="Optional MuJoCo XML path")
    args = parser.parse_args()

    backend = MujocoBackend(xml_path=args.xml) if args.xml else MujocoBackend()
    try:
        for case_id in range(4, 8):
            backend.reset(case_id, seed=case_id)
            check_observation(backend.get_observation(), case_id)
            for _ in range(4):
                backend.apply_action(np.zeros(7, dtype=np.float32))
                check_observation(backend.get_observation(), case_id)
            task = get_task(case_id)
            if set(task.targets) | {task.excluded} != set(SHAPE_ORDER):
                raise AssertionError(f"Case {case_id}: task does not cover all four shapes")
        backend.close()
        validate_state_machine()
    finally:
        backend.close()
    print("OK: MuJoCo backend reset/step/observation/state-machine checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
