from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .reward import RewardConfig, benchmark_score, compute_reward
from .tasks import CASES, FACE_ORDER, SHAPE_ORDER, get_task

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # allows task/reward unit tests without RL dependencies
    gym = None
    spaces = None

ACTION_DIM = 7
# 24 object pose + 6 box + 18 hole pose + 6 ee + 1 gripper + 3 inserted
# + 3 next-target mask + 3 safety + 16 task condition + 2 active-arm + 2 grippers
# + explicit 3-D subgoal delta and grasp phase = 88.
OBSERVATION_DIM = 88
SHAPE_INDEX = {name: i for i, name in enumerate(SHAPE_ORDER)}
FACE_INDEX = {name: i for i, name in enumerate(FACE_ORDER)}


class Backend(Protocol):
    def reset(self, case_id: int, seed: int | None = None) -> None: ...
    def get_observation(self) -> dict[str, Any]: ...
    def apply_action(self, action: np.ndarray) -> None: ...
    def close(self) -> None: ...


def _array(observation: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(observation[key], dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"{key} has shape {value.shape}; expected {shape}")
    return value


def encode_observation(observation: dict[str, Any], case_id: int) -> np.ndarray:
    """Encode metric backend data and the Case 4–7 condition as one vector."""
    object_positions = _array(observation, "object_positions", (4, 3)) / 0.5
    object_orientations = _array(observation, "object_orientations", (4, 3)) / np.pi
    box_pose = _array(observation, "box_pose", (6,)).copy()
    box_pose[:3] /= 0.5
    box_pose[3:] /= np.pi
    hole_poses = _array(observation, "hole_poses", (3, 6)).copy()
    hole_poses[:, :3] /= 0.5
    hole_poses[:, 3:] /= np.pi
    ee_pose = _array(observation, "ee_pose", (6,)).copy()
    ee_pose[:3] /= 0.5
    ee_pose[3:] /= np.pi
    inserted = _array(observation, "inserted", (3,)).astype(np.float32)

    # Task condition: one-hot shape at each destination face (12) + excluded shape (4).
    task = get_task(case_id)
    condition = np.zeros(16, dtype=np.float32)
    for shape, face in task.hole_faces.items():
        condition[FACE_INDEX[face] * 4 + SHAPE_INDEX[shape]] = 1.0
    condition[12 + SHAPE_INDEX[task.excluded]] = 1.0

    next_target = np.zeros(3, dtype=np.float32)
    for i, value in enumerate(inserted):
        if not value:
            next_target[i] = 1.0
            break
    safety = np.array(
        [
            np.clip(float(observation.get("box_displacement_m", 0.0)) / 0.05, 0.0, 10.0),
            np.clip(float(observation.get("contact_force_n", 0.0)) / 50.0, 0.0, 10.0),
            float(bool(observation.get("grasped", False))),
        ], dtype=np.float32,
    )
    active_arm = str(observation.get("active_arm", "right"))
    arm_context = np.asarray((active_arm == "left", active_arm == "right"), dtype=np.float32)
    gripper_context = np.asarray((
        float(np.clip(observation.get("gripper_left", observation.get("gripper", 0.0)), -1.0, 1.0)),
        float(np.clip(observation.get("gripper_right", observation.get("gripper", 0.0)), -1.0, 1.0)),
    ), dtype=np.float32)
    next_index = next((i for i, done in enumerate(inserted) if not done), None)
    if next_index is None:
        subgoal = np.asarray((0.0, 0.0, 0.35), dtype=np.float32)
    else:
        shape = task.targets[next_index]
        shape_index = SHAPE_INDEX[shape]
        if bool(observation.get("grasped", False)):
            subgoal = hole_poses[FACE_INDEX[task.hole_faces[shape]], :3] + np.asarray((0.0, 0.0, 0.080), dtype=np.float32)
        else:
            subgoal = object_positions[shape_index] + np.asarray((0.0, 0.0, 0.080), dtype=np.float32)
    subgoal_context = np.r_[np.clip((subgoal - ee_pose[:3]) / 0.5, -2.0, 2.0), float(bool(observation.get("grasped", False)))].astype(np.float32)
    result = np.concatenate([
        object_positions.reshape(-1), object_orientations.reshape(-1), box_pose,
        hole_poses.reshape(-1), ee_pose,
        np.array([float(np.clip(observation.get("gripper", 0.0), -1.0, 1.0))], dtype=np.float32),
        inserted, next_target, safety, condition, arm_context, gripper_context, subgoal_context,
    ]).astype(np.float32)
    if result.shape != (OBSERVATION_DIM,):
        raise AssertionError(f"encoder produced {result.shape}; expected {(OBSERVATION_DIM,)}")
    return result


if gym is not None:

    class Case47Env(gym.Env):
        """Gymnasium environment with an injected simulator/robot adapter."""

        metadata = {"render_modes": []}

        def __init__(self, backend: Backend, *, max_episode_steps: int = 600, reward_config: RewardConfig = RewardConfig(), seed: int | None = None):
            super().__init__()
            self.backend = backend
            self.max_episode_steps = int(max_episode_steps)
            self.reward_config = reward_config
            self.rng = np.random.default_rng(seed)
            self.observation_space = spaces.Box(-10.0, 10.0, (OBSERVATION_DIM,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
            self.case_id = 4
            self.steps = 0
            self.previous: dict[str, Any] | None = None

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            super().reset(seed=seed)
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            requested = (options or {}).get("case_id")
            self.case_id = int(requested) if requested is not None else int(self.rng.choice(tuple(CASES)))
            get_task(self.case_id)
            self.backend.reset(self.case_id, seed=seed)
            self.steps = 0
            self.previous = self.backend.get_observation()
            return encode_observation(self.previous, self.case_id), {"case_id": self.case_id}

        def step(self, action: np.ndarray):
            if self.previous is None:
                raise RuntimeError("reset() must be called before step()")
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (ACTION_DIM,):
                raise ValueError(f"action has shape {action.shape}; expected {(ACTION_DIM,)}")
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self.backend.apply_action(action)
            current = self.backend.get_observation()
            result = compute_reward(self.previous, current, action, self.case_id, self.reward_config)
            self.previous = current
            self.steps += 1
            truncated = self.steps >= self.max_episode_steps and not result.terminated
            info = {"case_id": self.case_id, "events": result.events, "success": result.success, "safety_violation": result.safety_violation, "score": benchmark_score(current, self.case_id)}
            return encode_observation(current, self.case_id), result.reward, result.terminated, truncated, info

        def close(self):
            self.backend.close()

else:

    class Case47Env:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            raise ImportError("Case47Env requires gymnasium; run pip install -e '.[rl]'")
