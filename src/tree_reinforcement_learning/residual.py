"""Frozen 7-D proxy base + bounded deterministic residual learning.

This action contract is NOT OpenPI/V4's 20-D pose contract. There is no robot
publisher, V4 loader, or hardware enable flag in this module.
"""
from __future__ import annotations
import copy
import hashlib
import numpy as np
import torch
from torch import nn

PROXY_SCALES = np.array([.1, .1, .1, .1, .1, .1, .1], dtype=np.float32)


def compose_proxy_action(base, residual, gate, scales=PROXY_SCALES):
    base, residual, scales = (np.asarray(x, dtype=np.float32) for x in (base, residual, scales))
    if any(x.shape != (7,) or not np.isfinite(x).all() for x in (base, residual, scales)):
        raise ValueError("proxy_action_requires_finite_7d_vectors")
    if np.any(np.abs(base) > 1) or np.any(scales <= 0) or np.any(scales > 1) or gate not in (0, 1):
        raise ValueError("invalid_proxy_bounds_or_gate")
    action = np.clip(base + scales * np.clip(residual, -1, 1) * gate, -1, 1)
    effective = (action - base) / scales
    return action.astype(np.float32), effective.astype(np.float32)


def module_digest(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenSACBase:
    """Use only the deterministic actor of an existing simulation checkpoint."""
    def __init__(self, checkpoint):
        from stable_baselines3 import SAC
        self.model = SAC.load(str(checkpoint), device="cpu")
        if self.model.observation_space.shape != (88,) or self.model.action_space.shape != (7,):
            raise ValueError("expected_proxy_observation88_action7_NOT_V4")
        if not np.all(self.model.action_space.low == -1) or not np.all(self.model.action_space.high == 1):
            raise ValueError("expected_normalized_proxy_actions")
        self.model.policy.requires_grad_(False)
        self.model.policy.set_training_mode(False)
        self.initial_digest = module_digest(self.model.policy)

    def __call__(self, obs):
        with torch.no_grad():
            action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    def assert_frozen(self):
        if any(p.requires_grad for p in self.model.policy.parameters()) or module_digest(self.model.policy) != self.initial_digest:
            raise RuntimeError("frozen_base_changed")


class ScriptedToyBase:
    """Synthetic fixture controller only, not a learned policy or V4 substitute."""
    initial_digest = "scripted_toy_fixture_v1"

    def __call__(self, obs):
        target = int(np.argmax(obs[64:68]))  # first target is the front-face shape
        position = obs[target*3:target*3+3] * .5
        ee = obs[48:51] * .5
        if obs[87] > .5:
            return np.array([0, 0, .7, 0, 0, 0, -1], dtype=np.float32)
        delta = position - ee
        grip = -1. if np.linalg.norm(delta) < .025 else 1.
        return np.r_[np.clip(delta/.01, -1., 1.), [0., 0., 0., grip]].astype(np.float32)

    def assert_frozen(self):
        pass


def project_tensor(state, residual, scales):
    base, gate = state[..., -8:-1], state[..., -1:]
    clipped = torch.clamp(base + scales * residual.clamp(-1, 1) * gate, -1, 1)
    return (clipped - base) / scales


class ResidualActor(nn.Module):
    def __init__(self, obs_dim, scales=PROXY_SCALES):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 7))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("scales", torch.as_tensor(scales.copy()))

    def forward(self, state):
        return project_tensor(state, torch.tanh(self.net(state)), self.scales)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        def net():
            return nn.Sequential(nn.Linear(obs_dim+7, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.q1, self.q2 = net(), net()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        return self.q1(x), self.q2(x)


class Replay:
    def __init__(self, capacity, obs_dim, seed=0):
        if capacity <= 0 or obs_dim <= 0:
            raise ValueError("invalid_replay_dimensions")
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.arrays = {k: np.zeros((capacity, n), np.float32) for k, n in
                       (("state", obs_dim), ("action", 7), ("reward", 1), ("next_state", obs_dim), ("terminal", 1))}

    def add(self, state, action, reward, next_state, terminal):
        values = dict(state=state, action=action, reward=[reward], next_state=next_state, terminal=[float(terminal)])
        converted = {}
        if not isinstance(terminal, (bool, np.bool_)):
            raise ValueError("terminal_must_be_boolean_not_truncation_or_arbitrary_number")
        for k, v in values.items():
            a = np.asarray(v, np.float32)
            if a.shape != self.arrays[k].shape[1:] or not np.isfinite(a).all():
                raise ValueError(f"invalid_replay_transition:{k}")
            converted[k] = a
        if np.any(np.abs(converted["action"]) > 1.00001):
            raise ValueError("residual_replay_action_outside_normalized_contract")
        # Validate the entire transition before touching a live ring-buffer slot.
        for k, a in converted.items():
            self.arrays[k][self.pos] = a
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError("insufficient_replay")
        idx = self.rng.integers(self.size, size=batch_size)
        return {k: torch.as_tensor(v[idx]) for k, v in self.arrays.items()}


class ResidualTD3:
    """CPU reference trainer: replay prefill in caller, then REAL critic-only warmup.

    Actor/target actor stay unchanged for critic_warmup critic updates. Targets
    bootstrap through time-limit truncations, but not actual terminal outcomes.
    """
    def __init__(self, obs_dim, *, critic_warmup=100, actor_lr=1e-4, critic_lr=3e-4, anchor_weight=.1):
        if critic_warmup < 1 or actor_lr <= 0 or critic_lr <= 0 or anchor_weight < 0:
            raise ValueError("invalid_training_configuration")
        self.actor = ResidualActor(obs_dim)
        self.critic = TwinCritic(obs_dim)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_warmup = critic_warmup
        self.anchor_weight = anchor_weight
        self.critic_updates, self.actor_updates = 0, 0
        self.gamma, self.tau, self.policy_delay = .99, .005, 2

    def predict(self, state):
        a = np.asarray(state, dtype=np.float32)
        if a.ndim != 1 or not np.isfinite(a).all():
            raise ValueError("invalid_residual_state")
        with torch.no_grad():
            return self.actor(torch.as_tensor(a).unsqueeze(0))[0].numpy()

    @staticmethod
    def _step(loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite_training_loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
        optimizer.step()

    def update(self, batch):
        state, action, reward, next_state, terminal = (batch[k] for k in ("state", "action", "reward", "next_state", "terminal"))
        with torch.no_grad():
            noise = (.1 * torch.randn_like(action)).clamp(-.2, .2)
            next_action = project_tensor(next_state, self.target_actor(next_state) + noise, self.actor.scales)
            tq1, tq2 = self.target_critic(next_state, next_action)
            target = reward + self.gamma * (1.-terminal) * torch.minimum(tq1, tq2)
        q1, q2 = self.critic(state, action)
        critic_loss = ((q1-target)**2).mean() + ((q2-target)**2).mean()
        self._step(critic_loss, self.critic_optimizer, self.critic.parameters())
        self.critic_updates += 1
        actor_loss = None
        active_state = state[state[:, -1] == 1]
        if (self.critic_updates > self.critic_warmup
                and (self.critic_updates-self.critic_warmup) % self.policy_delay == 0
                and len(active_state) > 0):
            # Do not count Adam momentum-only steps on closed-gate batches as
            # grasp learning, or dilute actor gradients with transport states.
            self.critic.requires_grad_(False)
            try:
                residual = self.actor(active_state)
                aq1, _ = self.critic(active_state, residual)
                loss = -aq1.mean() + self.anchor_weight * residual.square().mean()
                self._step(loss, self.actor_optimizer, self.actor.parameters())
                actor_loss = float(loss.detach())
                self.actor_updates += 1
            finally:
                self.critic.requires_grad_(True)
            with torch.no_grad():
                for target_p, p in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_p.lerp_(p, self.tau)
        with torch.no_grad():
            for target_p, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_p.lerp_(p, self.tau)
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": actor_loss,
                "critic_updates": self.critic_updates, "actor_updates": self.actor_updates}

    def snapshot(self):
        return copy.deepcopy({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                              "target_actor": self.target_actor.state_dict(), "target_critic": self.target_critic.state_dict(),
                              "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
                              "critic_updates": self.critic_updates, "actor_updates": self.actor_updates})

    def restore(self, data):
        for name in ("actor", "critic", "target_actor", "target_critic", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(data[name])
        self.critic_updates, self.actor_updates = data["critic_updates"], data["actor_updates"]
