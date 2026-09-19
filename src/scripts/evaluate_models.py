#!/usr/bin/env python3
"""Evaluate saved SAC seeds, without changing training code or checkpoints."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stochastic-episodes', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/evaluation_7models.json')
    args = parser.parse_args()
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.utils import set_random_seed
    from tree_reinforcement_learning.env import Case47Env
    from tree_reinforcement_learning.mujoco_backend import MujocoBackend
    from tree_reinforcement_learning.cli import load_config, DEFAULT_CONFIG
    from tree_reinforcement_learning.reward import RewardConfig

    torch.set_num_threads(1)
    config = load_config(DEFAULT_CONFIG)
    env = Case47Env(MujocoBackend(), max_episode_steps=config['max_episode_steps'],
                    reward_config=RewardConfig(**config['reward']))
    results = {'limitations': ['Fixed deterministic scene reset; different seeds do not randomize geometry.',
                               'Proxy kinematic grasp/insertion, not validated physical insertion.',
                               'Deterministic: one rollout per Case. Stochastic: policy-action sampling only.'],
               'models': {}}
    try:
        for gpu in (0, 1, 2, 3, 4, 5, 7):
            folder = ROOT / 'outputs' / ('case4_7_sac_a100' if gpu == 0 else f'case4_7_sac_7gpu/gpu_{gpu}')
            path = folder / 'sac_case4_7.zip'
            model = SAC.load(path, device='cpu')
            records = []
            for mode, count in [('deterministic', 1), ('stochastic', args.stochastic_episodes)]:
                for case in (4, 5, 6, 7):
                    for episode in range(count):
                        seed = 910000 + case * 100 + episode
                        set_random_seed(seed)
                        obs, _ = env.reset(seed=seed, options={'case_id': case})
                        total = 0.0
                        events = Counter()
                        for step in range(1, config['max_episode_steps'] + 1):
                            action, _ = model.predict(obs, deterministic=mode == 'deterministic')
                            obs, reward, terminated, truncated, info = env.step(action)
                            total += reward
                            events.update(info['events'])
                            if terminated or truncated:
                                break
                        raw = env.backend.get_observation()
                        records.append(dict(mode=mode, case=case, seed=seed, steps=step,
                            reward=float(total), score=info['score'], success=bool(info['success']),
                            safety_violation=bool(info['safety_violation']), timeout=bool(truncated),
                            inserted=np.asarray(raw['inserted']).tolist(), retracted=bool(raw['retracted']),
                            events=dict(events)))
            results['models'][str(gpu)] = dict(checkpoint=str(path), num_timesteps=model.num_timesteps, episodes=records)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temp = args.output.with_suffix('.tmp')
            temp.write_text(json.dumps(results, indent=2), encoding='utf-8')
            temp.replace(args.output)
            for mode in ('deterministic', 'stochastic'):
                selected = [r for r in records if r['mode'] == mode]
                print(json.dumps(dict(gpu=gpu, mode=mode, n=len(selected),
                    mean_score=float(np.mean([r['score'] for r in selected])),
                    successes=sum(r['success'] for r in selected),
                    mean_reward=float(np.mean([r['reward'] for r in selected])),
                    safety_failures=sum(r['safety_violation'] for r in selected),
                    grasp_events=sum(r['events'].get('grasp_success', 0) for r in selected))), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
