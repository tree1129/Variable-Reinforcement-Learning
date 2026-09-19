"""Single-first-block grasp/lift proxy objective. NOT a physical grasp test."""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .residual import compose_proxy_action
from .tasks import SHAPE_ORDER, get_task


class SingleGraspProxyEnv(gym.Wrapper):
    def __init__(self, env, base, *, hold_frames=5, lift_m=.015, gate_distance_m=.10):
        super().__init__(env)
        # Explicit allowlist: never route this trainer through a hardware adapter.
        if type(env.backend).__module__ not in ("tree_reinforcement_learning.toy_backend", "tree_reinforcement_learning.mujoco_backend"):
            raise ValueError("only_bundled_nonphysical_proxy_backends_allowed")
        if env.observation_space.shape != (88,) or env.action_space.shape != (7,):
            raise ValueError("proxy_contract_mismatch")
        self.base = base
        self.hold_frames, self.lift_m, self.gate_distance_m = hold_frames, lift_m, gate_distance_m
        self.observation_space = spaces.Box(-10., 10., (96,), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., (7,), dtype=np.float32)
        self.finished = True

    def _augment(self, obs):
        self.base_action = self.base(obs)
        # Validates base finiteness/bounds even when the gate is closed.
        compose_proxy_action(self.base_action, np.zeros(7), 0)
        raw = self.env.previous
        distance = np.linalg.norm(raw["ee_pose"][:3] - raw["object_positions"][self.target_index])
        self.gate = int(not raw["grasped"] and distance <= self.gate_distance_m)
        return np.r_[obs, self.base_action, self.gate].astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.target_shape = get_task(self.env.case_id).targets[0]
        self.target_index = SHAPE_ORDER.index(self.target_shape)
        self.start_z = float(self.env.previous["object_positions"][self.target_index, 2])
        self.hold_count, self.finished = 0, False
        return self._augment(obs), info

    def step(self, residual):
        if self.finished:
            raise RuntimeError("reset_required_after_terminal")
        action, effective = compose_proxy_action(self.base_action, residual, self.gate)
        base_before = self.base_action.copy()
        gate_before = self.gate
        obs, reward, term, trunc, info = self.env.step(action)
        raw, backend = self.env.previous, self.env.backend
        if hasattr(backend, "grasped_shape"):
            correct = backend.grasped_shape == self.target_shape
        else:
            correct = backend.grasped_index == self.target_index
        lifted = bool(correct and raw["object_positions"][self.target_index, 2] >= self.start_z + self.lift_m)
        self.hold_count = self.hold_count + 1 if lifted else 0
        success = self.hold_count >= self.hold_frames and not info["safety_violation"]
        # Proxy grasp success is a separate objective from three-block insertion.
        term = bool(term or success)
        trunc = bool(trunc and not term)
        self.finished = term or trunc
        info.update(task_score=info["score"], score=float(10 if success else 6 if lifted else 3 if correct else 0),
                    success=bool(success), proxy_grasp_label=True, base_action=base_before,
                    applied_action=action, effective_residual=effective, residual_gate=gate_before)
        return self._augment(obs), float(reward + (10. if success else 0.)), term, trunc, info
