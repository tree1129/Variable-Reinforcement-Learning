from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .tasks import TaskSpec, get_task


@dataclass(frozen=True)
class RewardConfig:
    distance_progress: float = 1.0
    approach_progress: float = 1.25
    orientation_progress: float = 0.25
    grasp_success: float = 1.0
    insertion_success: float = 5.0
    all_inserted: float = 4.0
    retract_success: float = 1.0
    action_cost: float = 0.002
    time_penalty: float = 0.001
    forbidden_contact: float = 8.0
    box_moved: float = 8.0
    excessive_force: float = 2.0
    max_box_displacement_m: float = 0.005
    max_contact_force_n: float = 25.0
    interaction_height_m: float = 0.080


@dataclass(frozen=True)
class RewardResult:
    reward: float
    terminated: bool
    success: bool
    safety_violation: bool
    events: tuple[str, ...]


def _required_array(obs: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(obs[key], dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"{key} has shape {value.shape}; expected {shape}")
    return value


def _flag(obs: Mapping[str, Any], key: str) -> bool:
    return bool(obs.get(key, False))


def _scalar(obs: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    return float(np.asarray(obs.get(key, default)).reshape(-1)[0])


def _target_errors(obs: Mapping[str, Any], task: TaskSpec) -> tuple[float, float]:
    positions = _required_array(obs, "object_positions", (4, 3))
    orientations = _required_array(obs, "object_orientations", (4, 3))
    holes = _required_array(obs, "hole_poses", (3, 6))
    inserted = _required_array(obs, "inserted", (3,)).astype(bool)
    shape_index = {shape: i for i, shape in enumerate(("triangle", "square", "trapezoid", "sphere"))}
    face_index = {face: i for i, face in enumerate(("front", "right", "left"))}
    position_error = orientation_error = 0.0
    active = 0
    for target_index, shape in enumerate(task.targets):
        if inserted[target_index]:
            continue
        object_index = shape_index[shape]
        hole_index = face_index[task.hole_faces[shape]]
        position_error += float(np.linalg.norm(positions[object_index] - holes[hole_index, :3]))
        orientation_error += float(np.linalg.norm(orientations[object_index] - holes[hole_index, 3:]))
        active += 1
    return (position_error / active, orientation_error / active) if active else (0.0, 0.0)


def _progress(previous: float, current: float) -> float:
    return float(np.clip(previous - current, -0.05, 0.05))


def _active_control_error(obs: Mapping[str, Any], task: TaskSpec, config: RewardConfig) -> float:
    """Distance to the currently actionable subgoal.

    The original reward only measured object-to-hole distance.  Before grasping,
    that potential is constant, so SAC had no signal telling it how to reach a
    target object; this was the main failure mode in the first 300k-step round.
    Use the ordered target list as a short-horizon curriculum: approach the next
    uninserted target, then transport it to its matching hole.
    """
    positions = _required_array(obs, "object_positions", (4, 3))
    holes = _required_array(obs, "hole_poses", (3, 6))
    ee_pose = _required_array(obs, "ee_pose", (6,))
    inserted = _required_array(obs, "inserted", (3,)).astype(bool)
    shape_index = {shape: i for i, shape in enumerate(("triangle", "square", "trapezoid", "sphere"))}
    face_index = {face: i for i, face in enumerate(("front", "right", "left"))}
    next_index = next((i for i, done in enumerate(inserted) if not done), None)
    if next_index is None:
        return float(np.linalg.norm(ee_pose[:3] - np.asarray([0.0, 0.0, 0.35], dtype=np.float32)))
    shape = task.targets[next_index]
    safe_offset = np.asarray((0.0, 0.0, config.interaction_height_m), dtype=np.float32)
    if bool(obs.get("grasped", False)):
        return float(np.linalg.norm(ee_pose[:3] - (holes[face_index[task.hole_faces[shape]], :3] + safe_offset)))
    return float(np.linalg.norm(ee_pose[:3] - (positions[shape_index[shape]] + safe_offset)))


def compute_reward(previous: Mapping[str, Any], current: Mapping[str, Any], action: np.ndarray, case_id: int | TaskSpec, config: RewardConfig = RewardConfig()) -> RewardResult:
    """Compute shaped reward and hard termination for one backend transition."""
    task = case_id if isinstance(case_id, TaskSpec) else get_task(case_id)
    previous_position, previous_orientation = _target_errors(previous, task)
    current_position, current_orientation = _target_errors(current, task)
    previous_control = _active_control_error(previous, task, config)
    current_control = _active_control_error(current, task, config)
    reward = config.distance_progress * _progress(previous_position, current_position)
    reward += config.approach_progress * _progress(previous_control, current_control)
    reward += config.orientation_progress * _progress(previous_orientation, current_orientation)
    reward -= config.action_cost * float(np.linalg.norm(np.asarray(action, dtype=np.float32)))
    reward -= config.time_penalty
    events: list[str] = []

    previous_inserted = _required_array(previous, "inserted", (3,)).astype(bool)
    current_inserted = _required_array(current, "inserted", (3,)).astype(bool)
    new_insertions = int(np.count_nonzero(current_inserted & ~previous_inserted))
    if new_insertions:
        reward += config.insertion_success * new_insertions
        events.append(f"inserted:{new_insertions}")
    if _flag(current, "grasp_success") and not _flag(previous, "grasp_success"):
        reward += config.grasp_success
        events.append("grasp_success")
    if bool(np.all(current_inserted)) and not bool(np.all(previous_inserted)):
        reward += config.all_inserted
        events.append("all_inserted")
    if _flag(current, "retracted") and not _flag(previous, "retracted"):
        reward += config.retract_success
        events.append("retracted")

    forbidden = _flag(current, "forbidden_contact")
    box_moved = _scalar(current, "box_displacement_m") > config.max_box_displacement_m
    excessive_force = _scalar(current, "contact_force_n") > config.max_contact_force_n
    safety_violation = forbidden or box_moved or excessive_force
    if forbidden:
        reward -= config.forbidden_contact
        events.append("forbidden_contact")
    if box_moved:
        reward -= config.box_moved
        events.append("box_moved")
    if excessive_force:
        reward -= config.excessive_force
        events.append("excessive_force")

    success = bool(np.all(current_inserted)) and _flag(current, "retracted") and not safety_violation
    terminated = success or safety_violation
    if success:
        events.append("task_success")
    return RewardResult(float(reward), terminated, success, safety_violation, tuple(events))


def benchmark_score(observation: Mapping[str, Any], case_id: int) -> float:
    """Supplied 10-point rubric: 3+3+3 for insertion and 1 for retraction."""
    inserted = _required_array(observation, "inserted", (3,)).astype(bool)
    score = 3.0 * float(np.count_nonzero(inserted))
    if bool(np.all(inserted)) and _flag(observation, "retracted"):
        score += 1.0
    return min(10.0, score)
