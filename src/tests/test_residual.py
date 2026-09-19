from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
HAS_RL = all(importlib.util.find_spec(k) for k in ("torch", "gymnasium", "stable_baselines3"))
if HAS_RL:
    import torch
    from tree_reinforcement_learning.residual import (ResidualActor, ResidualTD3, Replay, module_digest,
        compose_proxy_action, FrozenSACBase, ScriptedToyBase)
    from tree_reinforcement_learning.env import Case47Env
    from tree_reinforcement_learning.grasp_proxy import SingleGraspProxyEnv
    from tree_reinforcement_learning.toy_backend import ToyBackend


@unittest.skipUnless(HAS_RL, "requires RL environment; remote CI must run with no skips")
class ResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
    def state(self):
        state = np.zeros(96, np.float32); state[-1] = 1
        return state
    def batch(self):
        replay = Replay(64, 96)
        for _ in range(64):
            replay.add(self.state(), np.zeros(7), 1., self.state(), False)
        return replay.sample(32)
    def test_zero_init_exact_and_disabled_phase_zero(self):
        actor = ResidualActor(96)
        state = self.state(); state[-8:-1] = .5
        self.assertTrue(torch.equal(actor(torch.tensor(state)), torch.zeros(7)))
        with torch.no_grad():
            actor.net[-1].bias.fill_(1)
        state[-1] = 0
        self.assertTrue(torch.equal(actor(torch.tensor(state)), torch.zeros(7)))
    def test_action_bounds_effective_residual_and_nan_rejected(self):
        base = np.full(7, .98, np.float32)
        action, residual = compose_proxy_action(base, np.ones(7), 1)
        self.assertTrue(np.all(action <= 1))
        np.testing.assert_allclose(action, base+.1*residual, atol=1e-7)
        np.testing.assert_array_equal(compose_proxy_action(base, np.ones(7), 0)[0], base)
        with self.assertRaises(ValueError):
            compose_proxy_action(base, np.full(7, np.nan), 1)
        with self.assertRaises(ValueError):
            compose_proxy_action(np.zeros(20), np.zeros(7), 1)
    def test_critic_only_warmup_then_actor_changes(self):
        agent = ResidualTD3(96, critic_warmup=4)
        a0, c0, target0 = module_digest(agent.actor), module_digest(agent.critic), module_digest(agent.target_actor)
        for _ in range(4):
            result = agent.update(self.batch())
            self.assertIsNone(result["actor_loss"])
        self.assertEqual(a0, module_digest(agent.actor))
        self.assertEqual(target0, module_digest(agent.target_actor))
        self.assertNotEqual(c0, module_digest(agent.critic))
        agent.update(self.batch()); agent.update(self.batch())
        self.assertEqual(agent.actor_updates, 1)
        self.assertNotEqual(a0, module_digest(agent.actor))
    def test_actor_skips_batches_without_enabled_grasp_phase(self):
        agent = ResidualTD3(96, critic_warmup=1)
        for _ in range(3): agent.update(self.batch())
        before = module_digest(agent.actor)
        updates_before = agent.actor_updates
        batch = self.batch(); batch["state"][:, -1] = 0
        batch["next_state"][:, -1] = 0
        for _ in range(4):
            result = agent.update(batch)
            self.assertIsNone(result["actor_loss"])
        self.assertEqual(agent.actor_updates, updates_before)
        self.assertEqual(module_digest(agent.actor), before)

    def test_rollback_restores_actor_critic_targets_optimizers(self):
        agent = ResidualTD3(96, critic_warmup=1)
        saved = agent.snapshot()
        actor0, critic0 = module_digest(agent.actor), module_digest(agent.critic)
        for _ in range(6): agent.update(self.batch())
        agent.restore(saved)
        self.assertEqual(actor0, module_digest(agent.actor))
        self.assertEqual(critic0, module_digest(agent.critic))
        self.assertEqual(agent.actor_updates, 0)
        self.assertEqual(len(agent.actor_optimizer.state), 0)
    def test_frozen_sac_base_has_no_trainable_parameters_and_same_actions(self):
        from stable_baselines3 import SAC
        env = Case47Env(ToyBackend())
        try:
            model = SAC("MlpPolicy", env, device="cpu", policy_kwargs={"net_arch": [16, 16]})
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "base.zip"; model.save(path)
                base = FrozenSACBase(path)
                obs, _ = env.reset(seed=1, options={"case_id": 4})
                expected, _ = model.predict(obs, deterministic=True)
                np.testing.assert_array_equal(base(obs), expected)
                base.assert_frozen()
                self.assertFalse(any(p.requires_grad for p in base.model.policy.parameters()))
        finally: env.close()
    def test_actual_executed_proxy_transitions_and_single_grasp_stop(self):
        base = ScriptedToyBase()
        env = SingleGraspProxyEnv(Case47Env(ToyBackend(), max_episode_steps=120), base)
        try:
            for case in range(4, 8):
                obs, _ = env.reset(seed=case, options={"case_id": case})
                for _ in range(120):
                    obs, _, term, trunc, info = env.step(np.zeros(7))
                    np.testing.assert_array_equal(info["base_action"], info["applied_action"])
                    if term or trunc: break
                self.assertTrue(info["success"], f"case{case}")
                self.assertEqual(info["score"], 10.)
                with self.assertRaises(RuntimeError): env.step(np.zeros(7))
        finally: env.close()
    def test_replay_refuses_nan_and_insufficient_batch(self):
        replay = Replay(4, 96)
        with self.assertRaises(ValueError): replay.sample(1)
        with self.assertRaises(ValueError): replay.add(self.state(), np.zeros(7), float("nan"), self.state(), False)
    def test_failed_replay_write_does_not_partially_overwrite_full_buffer(self):
        replay = Replay(1, 96)
        replay.add(self.state(), np.zeros(7), 1., self.state(), False)
        before = {k: a.copy() for k, a in replay.arrays.items()}
        with self.assertRaises(ValueError):
            replay.add(self.state()+1, np.zeros(7), float("nan"), self.state(), False)
        for k, value in before.items():
            np.testing.assert_array_equal(replay.arrays[k], value)

    def test_time_limit_is_truncation_not_terminal(self):
        env = SingleGraspProxyEnv(Case47Env(ToyBackend(), max_episode_steps=1), ScriptedToyBase())
        try:
            env.reset(seed=1, options={"case_id": 4})
            _, _, term, trunc, _ = env.step(np.zeros(7))
            self.assertFalse(term)
            self.assertTrue(trunc)
        finally: env.close()
