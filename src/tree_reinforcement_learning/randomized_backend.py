"""Factories for bounded randomized MuJoCo robustness training."""
from __future__ import annotations

from .mujoco_backend import MujocoBackend


def make_backend() -> MujocoBackend:
    """Create the bounded-jitter backend used only for simulation training."""
    return MujocoBackend(
        randomize_scene=True,
        scene_jitter_xy_m=(0.025, 0.020),
        box_jitter_xy_m=(0.015, 0.010),
    )
