from __future__ import annotations

from typing import Any

import numpy as np

from .tasks import FACE_ORDER, SHAPE_ORDER, get_task


class ToyBackend:
    """Non-physical kinematic backend for API smoke tests only."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.home = np.array([0.0, 0.0, 0.35], dtype=np.float32)
        self.reset(4, seed)

    def reset(self, case_id: int, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.case_id = int(case_id)
        self.task = get_task(case_id)
        self.object_positions = np.array([
            [-0.18, 0.10, 0.03], [0.18, 0.10, 0.03],
            [-0.18, -0.10, 0.03], [0.18, -0.10, 0.03],
        ], dtype=np.float32) + self.rng.normal(0.0, 0.01, (4, 3)).astype(np.float32)
        self.object_orientations = np.zeros((4, 3), dtype=np.float32)
        self.hole_poses = np.zeros((3, 6), dtype=np.float32)
        for i, _face in enumerate(FACE_ORDER):
            self.hole_poses[i, :3] = [0.0, (i - 1) * 0.06, 0.30]
        self.box_pose = np.zeros(6, dtype=np.float32)
        self.ee_pose = np.r_[self.home, np.zeros(3, dtype=np.float32)]
        self.inserted = np.zeros(3, dtype=bool)
        self.gripper = 1.0
        self.grasped_index: int | None = None
        self.grasp_success = False
        self.retracted = False
        self.box_displacement_m = 0.0
        self.contact_force_n = 0.0

    def apply_action(self, action: np.ndarray):
        self.grasp_success = False
        self.retracted = False
        self.ee_pose[:3] += np.asarray(action[:3], dtype=np.float32) * 0.01
        self.ee_pose[3:] += np.asarray(action[3:6], dtype=np.float32) * 0.087
        self.ee_pose[:3] = np.clip(self.ee_pose[:3], [-0.35, -0.35, 0.01], [0.35, 0.35, 0.50])
        self.gripper = float(action[6])
        if self.grasped_index is None and self.gripper < -0.5:
            distances = np.linalg.norm(self.object_positions - self.ee_pose[:3], axis=1)
            index = int(np.argmin(distances))
            if distances[index] < 0.06:
                self.grasped_index = index
                self.grasp_success = True
        if self.grasped_index is not None:
            self.object_positions[self.grasped_index] = self.ee_pose[:3]
            if self.gripper > 0.5:
                shape = SHAPE_ORDER[self.grasped_index]
                if shape in self.task.targets:
                    target_i = self.task.targets.index(shape)
                    face_i = FACE_ORDER.index(self.task.hole_faces[shape])
                    if np.linalg.norm(self.ee_pose[:3] - self.hole_poses[face_i, :3]) < 0.06:
                        self.inserted[target_i] = True
                self.grasped_index = None
        if np.all(self.inserted) and np.linalg.norm(self.ee_pose[:3] - self.home) < 0.06:
            self.retracted = True

    def get_observation(self) -> dict[str, Any]:
        return {
            "object_positions": self.object_positions.copy(), "object_orientations": self.object_orientations.copy(),
            "box_pose": self.box_pose.copy(), "hole_poses": self.hole_poses.copy(), "ee_pose": self.ee_pose.copy(),
            "gripper": self.gripper, "inserted": self.inserted.copy(), "grasp_success": self.grasp_success,
            "grasped": self.grasped_index is not None, "retracted": self.retracted, "forbidden_contact": False,
            "box_displacement_m": self.box_displacement_m, "contact_force_n": self.contact_force_n,
        }

    def close(self):
        return None
