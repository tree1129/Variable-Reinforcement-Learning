"""Headless MuJoCo adapter for the Case 4--7 SAC environment.

This adapter is intentionally kept inside the independent RL project.  It does
not modify the existing robot-simulation project; it only loads its XML model
read-only and drives one arm through a bounded Cartesian IK controller.

The default XML path is the A100 path used by the existing MuJoCo workcell:
``/root/thu_robot_sim/project/artifacts/scene_twin/maps/v0010_operable_workcell/operable_workcell.xml``.
Override it with ``TREE_MUJOCO_XML`` when the simulator is installed elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .tasks import FACE_ORDER, SHAPE_ORDER, get_task

try:  # Optional dependency; toy/unit-test workflows do not need MuJoCo.
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised only without extra
    mujoco = None  # type: ignore[assignment]
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None


DEFAULT_XML = Path(
    "/root/thu_robot_sim/project/artifacts/scene_twin/maps/"
    "v0010_operable_workcell/operable_workcell.xml"
)
OBJECT_JOINTS = {
    "sphere": "sphere_free",
    "square": "cube_free",
    "triangle": "triangular_prism_free",
    "trapezoid": "trapezoidal_prism_free",
}
OBJECT_BODIES = {
    "sphere": "sphere",
    "square": "cube",
    "triangle": "triangular_prism",
    "trapezoid": "trapezoidal_prism",
}
# Calibrated bilateral tabletop workspace.  Positive y is exclusively served
# by the left arm and negative y by the right arm.  The physical sorter stays
# on the shared centre line, close enough for both arms to reach its front
# face; the side faces are only approached by their matching-side arm.
PLACEMENT_XY = {
    "left_near": (0.80, 0.12),
    "left_middle": (0.90, 0.16),
    "left_far": (1.00, 0.12),
    "right_near": (0.80, -0.12),
    "right_middle": (0.90, -0.16),
    "right_far": (1.00, -0.12),
}
# All cases use the same collision-checked central sorter pose.  Case labels
# are retained in TaskSpec for the instruction mapping, not as a request to
# move the physical box during an episode.
BOX_XY = {"center": (1.08, 0.0), "middle_left": (1.08, 0.0), "middle_right": (1.08, 0.0)}


class MujocoBackend:
    """Headless MuJoCo backend implementing the project's Backend protocol.

    The model supplied with the workcell has position actuators but no
    Cartesian controller.  We therefore solve a bounded damped-least-squares
    IK step for the selected arm, and use a deterministic grasp/insert state
    machine around the simulator state.  This makes the adapter suitable for
    simulator training while preserving the environment's normalized 7-D
    action contract.
    """

    def __init__(
        self,
        xml_path: str | os.PathLike[str] | None = None,
        *,
        arm: str = "right",
        position_step_m: float = 0.010,
        rotation_step_rad: float = 0.0872664626,
        grasp_radius_m: float = 0.055,
        insertion_radius_m: float = 0.060,
        ik_iterations: int = 4,
        orientation_weight: float = 0.05,
        interaction_height_m: float = 0.080,
        grasp_confirm_steps: int = 2,
        release_confirm_steps: int = 2,
        randomize_scene: bool = False,
        scene_jitter_xy_m: tuple[float, float] = (0.025, 0.020),
        box_jitter_xy_m: tuple[float, float] = (0.015, 0.010),
    ) -> None:
        if mujoco is None:  # pragma: no cover
            raise ImportError("MujocoBackend requires `pip install -e '.[mujoco]'`") from _MUJOCO_IMPORT_ERROR
        self.xml_path = Path(xml_path or os.environ.get("TREE_MUJOCO_XML", DEFAULT_XML)).expanduser()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        self.arm = arm
        self.position_step_m = float(position_step_m)
        self.rotation_step_rad = float(rotation_step_rad)
        self.grasp_radius_m = float(grasp_radius_m)
        self.insertion_radius_m = float(insertion_radius_m)
        self.ik_iterations = int(ik_iterations)
        # The supplied workcell has a narrow orientation workspace. Prioritize
        # Cartesian position so the policy can actually reach blocks/holes;
        # keep a small orientation term for the 7-D action contract.
        self.orientation_weight = float(np.clip(orientation_weight, 0.0, 1.0))
        # Grasp/release uses a collision-safe proxy point above the tabletop.
        # This avoids asking the gripper geometry to penetrate the tabletop
        # merely because an object's center is near the table surface.
        self.interaction_height_m = float(interaction_height_m)
        if self.interaction_height_m <= 0.0:
            raise ValueError("interaction_height_m must be positive")
        self.grasp_confirm_steps = int(grasp_confirm_steps)
        self.release_confirm_steps = int(release_confirm_steps)
        if self.grasp_confirm_steps < 1 or self.release_confirm_steps < 1:
            raise ValueError("grasp/release confirmation steps must be positive")
        self.randomize_scene = bool(randomize_scene)
        self.scene_jitter_xy_m = np.asarray(scene_jitter_xy_m, dtype=np.float64)
        self.box_jitter_xy_m = np.asarray(box_jitter_xy_m, dtype=np.float64)
        if self.scene_jitter_xy_m.shape != (2,) or np.any(self.scene_jitter_xy_m < 0.0):
            raise ValueError("scene_jitter_xy_m must contain two non-negative values")
        if self.box_jitter_xy_m.shape != (2,) or np.any(self.box_jitter_xy_m < 0.0):
            raise ValueError("box_jitter_xy_m must contain two non-negative values")
        self.scene_jitter = np.zeros(2, dtype=np.float64)
        self.box_jitter = np.zeros(2, dtype=np.float64)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        # Keep metadata for both physical manipulators.  A 7-D policy action
        # drives the arm assigned to the next object: left-side sources use the
        # left arm and right-side sources use the right arm.  This preserves the
        # simple SAC action contract while preventing a right arm from crossing
        # the table to service a left-side object.
        self.arm_states = {name: self._build_arm_state(name) for name in ("left", "right")}
        self.all_arm_body_ids = set().union(*(state["body_ids"] for state in self.arm_states.values()))
        self.grippers = {"left": 1.0, "right": 1.0}
        self.active_arm = arm
        self._select_arm(arm)
        self.object_meta: dict[str, dict[str, int]] = {}
        for shape, joint_name in OBJECT_JOINTS.items():
            jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODIES[shape])
            self.object_meta[shape] = {
                "joint_id": jid,
                "qpos_adr": int(self.model.jnt_qposadr[jid]),
                "dof_adr": int(self.model.jnt_dofadr[jid]),
                "body_id": body_id,
                "geom_id": self._first_geom_for_body(body_id),
            }
        self.object_geom_to_shape = {
            gid: shape
            for shape, meta in self.object_meta.items()
            for gid, owner in enumerate(self.model.geom_bodyid)
            if int(owner) == meta["body_id"]
        }
        self.box_geom_ids = self._named_box_geoms()
        # The tabletop is static world geometry, so it is not covered by the
        # box list.  Object/table contact is normal at reset, but any selected
        # arm contact with this surface is a safety violation.
        self.table_geom_ids = self._named_table_geoms()
        self.collision = False
        self.collision_type = ""
        self.collision_body_pair = ""
        self.box_geom_initial = {
            gid: (self.model.geom_pos[gid].copy(), self.model.geom_quat[gid].copy()) for gid in self.box_geom_ids
        }
        self.box_pivot_initial = np.asarray((1.27, 0.0), dtype=np.float64)
        self.object_initial: dict[str, np.ndarray] = {}
        self.home_qpos = np.asarray((0.0, 1.10, -1.30, 0.10, 0.0, 0.0), dtype=np.float64)
        self._reset_data()

    def _id(self, obj_type: Any, name: str) -> int:
        ident = int(mujoco.mj_name2id(self.model, obj_type, name))
        if ident < 0:
            raise RuntimeError(f"Missing MuJoCo {obj_type}: {name}")
        return ident

    def _build_arm_state(self, arm: str) -> dict[str, Any]:
        joint_names = [f"{arm}_arm_joint{i}" for i in range(1, 7)]
        joint_ids = [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        return {
            "joint_names": joint_names,
            "joint_ids": joint_ids,
            "actuator_ids": [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_position") for name in joint_names],
            "qpos_adrs": [int(self.model.jnt_qposadr[joint_id]) for joint_id in joint_ids],
            "joint_low": np.asarray([self.model.jnt_range[joint_id, 0] for joint_id in joint_ids], dtype=np.float64),
            "joint_high": np.asarray([self.model.jnt_range[joint_id, 1] for joint_id in joint_ids], dtype=np.float64),
            "gripper_left": self._id(mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_arm_gripper_left_joint"),
            "gripper_right": self._id(mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_arm_gripper_right_joint"),
            "gripper_left_act": self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_arm_gripper_left_joint_position"),
            "gripper_right_act": self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_arm_gripper_right_joint_position"),
            "ee_body": self._id(mujoco.mjtObj.mjOBJ_BODY, f"{arm}_arm_gripper_base_link"),
            "tip_bodies": [
                self._id(mujoco.mjtObj.mjOBJ_BODY, f"{arm}_arm_gripper_left_tip_link"),
                self._id(mujoco.mjtObj.mjOBJ_BODY, f"{arm}_arm_gripper_right_tip_link"),
            ],
            "body_ids": self._descendant_bodies(self._id(mujoco.mjtObj.mjOBJ_BODY, f"{arm}_arm_base_link")),
        }

    def _select_arm(self, arm: str) -> None:
        """Make one side the controlled/action-observed arm without moving it."""
        if arm not in self.arm_states:
            raise ValueError(f"unknown arm {arm!r}")
        state = self.arm_states[arm]
        self.arm = arm
        self.active_arm = arm
        self.joint_names = state["joint_names"]
        self.joint_ids = state["joint_ids"]
        self.actuator_ids = state["actuator_ids"]
        self.qpos_adrs = state["qpos_adrs"]
        self.joint_low = state["joint_low"]
        self.joint_high = state["joint_high"]
        self.gripper_left = state["gripper_left"]
        self.gripper_right = state["gripper_right"]
        self.gripper_left_act = state["gripper_left_act"]
        self.gripper_right_act = state["gripper_right_act"]
        self.ee_body = state["ee_body"]
        self.tip_bodies = state["tip_bodies"]
        self.arm_body_ids = state["body_ids"]
        self.gripper = self.grippers[arm]

    def _arm_for_shape(self, shape: str) -> str:
        """Return the physically reachable manipulator for an object's side."""
        y = float(self.data.xpos[self.object_meta[shape]["body_id"]][1])
        return "left" if y >= 0.0 else "right"

    def _next_target(self) -> str | None:
        for index, shape in enumerate(self.task.targets):
            if not self.inserted[index]:
                return shape
        return None

    def _select_next_target_arm(self) -> None:
        shape = self._next_target()
        if shape is not None:
            self._select_arm(self._arm_for_shape(shape))

    def _first_geom_for_body(self, body_id: int) -> int:
        for gid, owner in enumerate(self.model.geom_bodyid):
            if int(owner) == body_id:
                return gid
        raise RuntimeError(f"Body {body_id} has no geom")

    def _named_box_geoms(self) -> list[int]:
        result = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith(("housing_", "drawer_", "roof_")):
                result.append(gid)
        if not result:
            raise RuntimeError("No shape-sorter box geoms found in MuJoCo model")
        return result

    def _named_table_geoms(self) -> list[int]:
        """Return the explicit tabletop collision geometry from the scene."""
        result = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith(("table", "desk", "work_surface")):
                result.append(gid)
        if not result:
            raise RuntimeError("No tabletop collision geom found in MuJoCo model")
        return result

    def _descendant_bodies(self, root: int) -> set[int]:
        descendants = {root}
        for body_id in range(self.model.nbody):
            parent = int(self.model.body_parentid[body_id])
            visited: set[int] = set()
            while parent >= 0 and parent not in visited:
                if parent == root:
                    descendants.add(body_id)
                    break
                visited.add(parent)
                next_parent = int(self.model.body_parentid[parent])
                if next_parent == parent:
                    break
                parent = next_parent
        return descendants

    def _reset_data(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.object_initial = {
            shape: self.data.qpos[meta["qpos_adr"] : meta["qpos_adr"] + 7].copy()
            for shape, meta in self.object_meta.items()
        }
        self.case_id = 4
        self.task = get_task(4)
        self.inserted = np.zeros(3, dtype=bool)
        self.grasped_shape: str | None = None
        self.grasped_arm: str | None = None
        self.grasp_success = False
        self.grasp_candidate_shape: str | None = None
        self.grasp_close_steps = 0
        self.release_open_steps = 0
        self.retracted = False
        self.forbidden_contact = False
        self.box_displacement_m = 0.0
        self.contact_force_n = 0.0
        for arm_name in ("left", "right"):
            self._select_arm(arm_name)
            self._set_arm_qpos(self.home_qpos)
            self._set_gripper(1.0)
        self._select_arm("right")
        mujoco.mj_forward(self.model, self.data)

    def _set_arm_qpos(self, q: np.ndarray) -> None:
        q = np.clip(np.asarray(q, dtype=np.float64), self.joint_low, self.joint_high)
        for adr, value, aid in zip(self.qpos_adrs, q, self.actuator_ids):
            self.data.qpos[adr] = value
            if aid >= 0:
                self.data.ctrl[aid] = value

    def _set_gripper(self, value: float) -> None:
        # Normalized action convention: -1 = close, +1 = open.
        openness = float(np.clip((value + 1.0) * 0.5, 0.0, 1.0))
        left = -0.065 + 0.763 * openness
        right = 0.065 - 0.763 * openness
        left_adr = int(self.model.jnt_qposadr[self.gripper_left])
        right_adr = int(self.model.jnt_qposadr[self.gripper_right])
        self.data.qpos[left_adr] = left
        self.data.qpos[right_adr] = right
        self.data.ctrl[self.gripper_left_act] = left
        self.data.ctrl[self.gripper_right_act] = right
        self.gripper = float(np.clip(value, -1.0, 1.0))
        self.grippers[self.arm] = self.gripper

    def _set_box_pose(self, x: float, y: float) -> None:
        delta = np.asarray((float(x), float(y)), dtype=np.float64) - self.box_pivot_initial
        for gid, (initial_pos, initial_quat) in self.box_geom_initial.items():
            self.model.geom_pos[gid][0] = initial_pos[0] + delta[0]
            self.model.geom_pos[gid][1] = initial_pos[1] + delta[1]
            self.model.geom_pos[gid][2] = initial_pos[2]
            self.model.geom_quat[gid][:] = initial_quat
        self.box_xy = np.asarray((x, y), dtype=np.float64)
        # The box pose is part of the episode initialization.  Since this
        # adapter never applies an action to the box, its displacement from
        # the episode reference remains zero for the safety signal.
        self.box_displacement_m = 0.0

    def _reset_object_pose(self, shape: str, xy: tuple[float, float]) -> None:
        meta = self.object_meta[shape]
        adr, dadr = meta["qpos_adr"], meta["dof_adr"]
        initial = self.object_initial[shape]
        self.data.qpos[adr : adr + 7] = initial
        self.data.qpos[adr : adr + 2] = xy
        self.data.qvel[dadr : dadr + 6] = 0.0

    def reset(self, case_id: int, seed: int | None = None) -> None:
        # Keep the historical deterministic layout by default.  Optional bounded
        # jitter is for robustness training/evaluation only; it never crosses
        # the left/right arm partition and is intentionally smaller than the
        # tabletop/workcell clearance margin.
        rng = np.random.default_rng(seed)
        self.scene_jitter = rng.uniform(-self.scene_jitter_xy_m, self.scene_jitter_xy_m) if self.randomize_scene else np.zeros(2, dtype=np.float64)
        self.box_jitter = rng.uniform(-self.box_jitter_xy_m, self.box_jitter_xy_m) if self.randomize_scene else np.zeros(2, dtype=np.float64)
        self.case_id = int(case_id)
        self.task = get_task(self.case_id)
        mujoco.mj_resetData(self.model, self.data)
        self.inserted = np.zeros(3, dtype=bool)
        self.grasped_shape = None
        self.grasped_arm: str | None = None
        self.grasp_success = False
        self.grasp_candidate_shape = None
        self.grasp_close_steps = 0
        self.release_open_steps = 0
        self.retracted = False
        self.forbidden_contact = False
        self.contact_force_n = 0.0
        for arm_name in ("left", "right"):
            self._select_arm(arm_name)
            self._set_arm_qpos(self.home_qpos)
            self._set_gripper(1.0)
        box_x, box_y = np.asarray(BOX_XY[self.task.box_position], dtype=np.float64) + self.box_jitter
        self._set_box_pose(float(box_x), float(box_y))
        for shape in SHAPE_ORDER:
            x, y = np.asarray(PLACEMENT_XY[self.task.placements[shape]], dtype=np.float64) + self.scene_jitter
            # Preserve the arm partition under jitter: positive y remains left,
            # negative y remains right.  The chosen jitter is bounded below the
            # closest source-to-centre margin, so no object changes side.
            self._reset_object_pose(shape, (float(x), float(y)))
        mujoco.mj_forward(self.model, self.data)
        self._select_next_target_arm()

    def _ee_position(self) -> np.ndarray:
        return np.mean(self.data.xpos[self.tip_bodies], axis=0).copy()

    def _ee_rotvec(self) -> np.ndarray:
        # The base orientation is returned as a compact axis-angle vector.
        rot = self.data.xmat[self.ee_body].reshape(3, 3)
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, rot.reshape(-1))
        out = np.zeros(3, dtype=np.float64)
        mujoco.mju_quat2Vel(out, quat, 1.0)
        return out

    def _ee_pose(self) -> np.ndarray:
        return np.r_[self._ee_position(), self._ee_rotvec()].astype(np.float32)

    def _hole_poses(self) -> np.ndarray:
        x, y = self.box_xy
        return np.asarray(
            [
                (x - 0.075, y, 0.010, 0.0, 0.0, 0.0),  # front
                (x, y - 0.075, 0.010, 0.0, 0.0, 0.0),  # right
                (x, y + 0.075, 0.010, 0.0, 0.0, 0.0),  # left
            ],
            dtype=np.float32,
        )

    def _position_jacobian(self) -> np.ndarray:
        jac = np.zeros((3, self.model.nv), dtype=np.float64)
        for body_id in self.tip_bodies:
            one = np.zeros((3, self.model.nv), dtype=np.float64)
            angular = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, self.data, one, angular, body_id)
            jac += one
        return jac / len(self.tip_bodies)

    def _orientation_jacobian(self) -> np.ndarray:
        jac = np.zeros((3, self.model.nv), dtype=np.float64)
        angular = np.zeros((3, self.model.nv), dtype=np.float64)
        linear = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, self.data, linear, angular, self.ee_body)
        return angular

    def _ik_step(self, target_position: np.ndarray, target_rotvec: np.ndarray) -> None:
        for _ in range(self.ik_iterations):
            mujoco.mj_forward(self.model, self.data)
            pos_error = target_position - self._ee_position()
            rot_error = target_rotvec - self._ee_rotvec()
            error = np.r_[
                pos_error,
                self.orientation_weight * np.clip(rot_error, -0.30, 0.30),
            ]
            if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 2e-3:
                break
            full_jac = np.vstack((self._position_jacobian(), self._orientation_jacobian()))
            joint_columns = []
            for jid in self.joint_ids:
                joint_columns.append(int(self.model.jnt_dofadr[jid]))
            jac = full_jac[:, joint_columns]
            damping = 0.025
            dq = jac.T @ np.linalg.solve(jac @ jac.T + (damping**2) * np.eye(6), error)
            self._set_arm_qpos(np.asarray([self.data.qpos[adr] for adr in self.qpos_adrs]) + np.clip(dq, -0.08, 0.08))
        mujoco.mj_forward(self.model, self.data)

    def _interaction_pose(self, position: np.ndarray) -> np.ndarray:
        return np.asarray(position, dtype=np.float64) + np.asarray((0.0, 0.0, self.interaction_height_m))

    def _set_object_to_ee(self, shape: str) -> None:
        meta = self.object_meta[shape]
        adr, dadr = meta["qpos_adr"], meta["dof_adr"]
        qpos = self.data.qpos[adr : adr + 7]
        # The policy approaches from above. Keep the object's center below the
        # fingertip midpoint so grasping does not visibly teleport it upward.
        qpos[:3] = self._ee_position() - np.asarray(
            (0.0, 0.0, self.interaction_height_m), dtype=np.float64
        )
        qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dadr : dadr + 6] = 0.0

    def _geom_name(self, geom_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom_{geom_id}"

    def _contact_is_dangerous(self, g1: int, g2: int) -> tuple[bool, str, str]:
        """Classify one MuJoCo contact for the task safety monitor.

        Permanent arm self-contact is ignored.  Contacts with the box, the
        excluded object, or an ungrasped/uninserted object are violations.  A
        currently grasped target is explicitly allowed because the adapter
        uses kinematic attachment and therefore necessarily keeps the gripper
        in contact with it during transport.
        """
        b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
        if b1 in self.all_arm_body_ids and b2 in self.all_arm_body_ids:
            return False, "arm_self_contact", f"{self._geom_name(g1)}<->{self._geom_name(g2)}"

        arm_geom = (b1 in self.arm_body_ids, b2 in self.arm_body_ids)
        box_geom = (g1 in self.box_geom_ids, g2 in self.box_geom_ids)
        table_geom = (g1 in self.table_geom_ids, g2 in self.table_geom_ids)
        shape1 = self.object_geom_to_shape.get(g1)
        shape2 = self.object_geom_to_shape.get(g2)
        if any(arm_geom) and any(table_geom):
            return True, "arm_table", f"{self._geom_name(g1)}<->{self._geom_name(g2)}"
        if any(arm_geom) and any(box_geom):
            return True, "arm_box", f"{self._geom_name(g1)}<->{self._geom_name(g2)}"
        if any(arm_geom) and (shape1 is not None or shape2 is not None):
            shape = shape1 if shape1 is not None else shape2
            # The excluded shape is never legal to touch, even if a faulty
            # policy closes the gripper on it and marks it as grasped.
            if shape == self.task.excluded:
                return True, "arm_excluded_object", f"{shape}:{self._geom_name(g1)}<->{self._geom_name(g2)}"
            if shape == self.grasped_shape or shape in self.task.targets and self.inserted[self.task.targets.index(shape)]:
                return False, "arm_allowed_object", f"{shape}:{self._geom_name(g1)}<->{self._geom_name(g2)}"
            return True, "arm_object", f"{shape}:{self._geom_name(g1)}<->{self._geom_name(g2)}"
        if shape1 is not None and shape2 is not None:
            inserted = set(shape for i, shape in enumerate(self.task.targets) if self.inserted[i])
            if shape1 in inserted or shape2 in inserted:
                return False, "inserted_object_contact", f"{shape1}<->{shape2}"
            return True, "object_object", f"{shape1}<->{shape2}"
        return False, "ignored", f"{self._geom_name(g1)}<->{self._geom_name(g2)}"

    def _update_contacts(self) -> None:
        self.forbidden_contact = False
        self.collision = False
        self.collision_type = ""
        self.collision_body_pair = ""
        maximum_force = 0.0
        for i in range(int(self.data.ncon)):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            dangerous, collision_type, pair = self._contact_is_dangerous(g1, g2)
            if not dangerous:
                continue
            self.forbidden_contact = True
            self.collision = True
            if not self.collision_type:
                self.collision_type = collision_type
                self.collision_body_pair = pair
            # Read force for this contact only.  Global qfrc_constraint also
            # contains unrelated arm self-contact constraints and caused false
            # excessive-force terminations in the previous implementation.
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            maximum_force = max(maximum_force, float(abs(force[0])))
        self.contact_force_n = maximum_force

    def apply_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"action has shape {action.shape}; expected (7,)")
        action = np.clip(action, -1.0, 1.0)
        current = self._ee_pose().astype(np.float64)
        target_position = current[:3] + action[:3].astype(np.float64) * self.position_step_m
        target_rotvec = current[3:] + action[3:6].astype(np.float64) * self.rotation_step_rad
        self._ik_step(target_position, target_rotvec)
        self._set_gripper(float(action[6]))
        self.grasp_success = False
        self.retracted = False

        if self.grasped_shape is None and action[6] < -0.5:
            # Only the next required object is eligible.  This prevents a
            # side arm from closing on an excluded or opposite-side block.
            shape = self._next_target()
            if shape is not None and self._arm_for_shape(shape) == self.active_arm:
                distance = float(np.linalg.norm(
                    self._interaction_pose(self.data.xpos[self.object_meta[shape]["body_id"]]) - self._ee_position()
                ))
                if distance <= self.grasp_radius_m:
                    if self.grasp_candidate_shape == shape:
                        self.grasp_close_steps += 1
                    else:
                        self.grasp_candidate_shape = shape
                        self.grasp_close_steps = 1
                    if self.grasp_close_steps >= self.grasp_confirm_steps:
                        self.grasped_shape = shape
                        self.grasped_arm = self.active_arm
                        self.grasp_success = True
                        self.grasp_candidate_shape = None
                        self.grasp_close_steps = 0
                else:
                    self.grasp_candidate_shape = None
                    self.grasp_close_steps = 0
            else:
                self.grasp_candidate_shape = None
                self.grasp_close_steps = 0
        elif self.grasped_shape is None:
            self.grasp_candidate_shape = None
            self.grasp_close_steps = 0

        if self.grasped_shape is not None:
            self._set_object_to_ee(self.grasped_shape)
            shape = self.grasped_shape
            target_index = self.task.targets.index(shape)
            face_index = FACE_ORDER.index(self.task.hole_faces[shape])
            hole = self._hole_poses()[face_index, :3]
            at_hole = np.linalg.norm(self._ee_position() - self._interaction_pose(hole)) <= self.insertion_radius_m
            if action[6] > 0.5 and at_hole:
                self.release_open_steps += 1
            else:
                self.release_open_steps = 0
            if self.release_open_steps >= self.release_confirm_steps:
                self.inserted[target_index] = True
                self.data.qpos[self.object_meta[shape]["qpos_adr"] : self.object_meta[shape]["qpos_adr"] + 3] = hole
                self.grasped_shape = None
                self.grasped_arm = None
                self.release_open_steps = 0
                self._select_next_target_arm()
            else:
                # Keep the gripper visibly clamped while carrying. A release
                # only takes effect after two open commands at the right hole.
                self._set_gripper(-1.0)

        if np.all(self.inserted):
            home_position = self._home_position()
            self.retracted = bool(np.linalg.norm(self._ee_position() - home_position) <= 0.075)
        mujoco.mj_forward(self.model, self.data)
        self._update_contacts()

    def _home_position(self) -> np.ndarray:
        current = np.asarray([self.data.qpos[adr] for adr in self.qpos_adrs], dtype=np.float64)
        self._set_arm_qpos(self.home_qpos)
        mujoco.mj_forward(self.model, self.data)
        result = self._ee_position()
        self._set_arm_qpos(current)
        mujoco.mj_forward(self.model, self.data)
        return result

    def get_observation(self) -> dict[str, Any]:
        positions = np.zeros((4, 3), dtype=np.float32)
        orientations = np.zeros((4, 3), dtype=np.float32)
        for index, shape in enumerate(SHAPE_ORDER):
            meta = self.object_meta[shape]
            positions[index] = self.data.xpos[meta["body_id"]]
            # Objects are reset/placed with identity orientation in this
            # adapter; the field is still exposed for the common protocol.
        box_pose = np.asarray((self.box_xy[0], self.box_xy[1], 0.0, 0.0, 0.0, 0.0), dtype=np.float32)
        return {
            "object_positions": positions,
            "object_orientations": orientations,
            "box_pose": box_pose,
            "hole_poses": self._hole_poses(),
            "ee_pose": self._ee_pose(),
            "gripper": self.gripper,
            "active_arm": self.active_arm,
            "gripper_left": self.grippers["left"],
            "gripper_right": self.grippers["right"],
            "inserted": self.inserted.copy(),
            "retracted": self.retracted,
            "grasped": self.grasped_shape is not None,
            "grasp_success": self.grasp_success,
            "forbidden_contact": self.forbidden_contact,
            "collision": self.collision,
            "collision_type": self.collision_type,
            "collision_body_pair": self.collision_body_pair,
            "box_displacement_m": self.box_displacement_m,
            "contact_force_n": self.contact_force_n,
        }

    def close(self) -> None:
        # No renderer/context is created, so there is no headless resource to
        # release.  Keeping this method makes the adapter protocol complete.
        return None


def make_backend() -> MujocoBackend:
    """Factory used by ``--backend-factory`` on the A100 host."""
    return MujocoBackend()
