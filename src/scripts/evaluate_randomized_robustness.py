#!/usr/bin/env python3
"""Read-only robustness evaluation for the best Case 4-7 SAC policy.

The evaluator intentionally uses a bounded, seed-deterministic XY scene jitter. It
never commands the real robot; it only exercises the MuJoCo backend.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes-per-case", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=2026091500)
    p.add_argument("--stochastic", action="store_true")
    args = p.parse_args()

    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.utils import set_random_seed
    from tree_reinforcement_learning.cli import DEFAULT_CONFIG, load_config
    from tree_reinforcement_learning.env import Case47Env
    from tree_reinforcement_learning.mujoco_backend import MujocoBackend
    from tree_reinforcement_learning.reward import RewardConfig

    torch.set_num_threads(1)
    config = load_config(DEFAULT_CONFIG)
    backend = MujocoBackend(randomize_scene=True)
    env = Case47Env(backend, max_episode_steps=int(config["max_episode_steps"]),
                    reward_config=RewardConfig(**config["reward"]))
    model = SAC.load(str(args.checkpoint), device="cpu")
    records = []
    try:
        for case in (4, 5, 6, 7):
            for episode in range(args.episodes_per_case):
                seed = args.seed_base + case * 100 + episode
                set_random_seed(seed)
                obs, _ = env.reset(seed=seed, options={"case_id": case})
                total = 0.0
                events = Counter()
                prev_positions = np.asarray(backend.get_observation()["object_positions"], dtype=float).copy()
                max_object_jump = 0.0
                max_contact_force = 0.0
                active_arm_trace = []
                for step in range(1, int(config["max_episode_steps"]) + 1):
                    action, _ = model.predict(obs, deterministic=not args.stochastic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += float(reward)
                    events.update(info["events"])
                    raw = backend.get_observation()
                    positions = np.asarray(raw["object_positions"], dtype=float)
                    max_object_jump = max(max_object_jump, float(np.max(np.linalg.norm(positions - prev_positions, axis=1))))
                    prev_positions = positions
                    max_contact_force = max(max_contact_force, float(raw.get("contact_force_n", 0.0)))
                    active_arm_trace.append(str(raw.get("active_arm", "")))
                    if terminated or truncated:
                        break
                raw = backend.get_observation()
                records.append({
                    "mode": "stochastic" if args.stochastic else "deterministic",
                    "case": case, "episode": episode, "seed": seed, "steps": step,
                    "reward": total, "score": float(info["score"]),
                    "success": bool(info["success"]),
                    "safety_violation": bool(info["safety_violation"]),
                    "timeout": bool(truncated),
                    "inserted": np.asarray(raw["inserted"]).astype(bool).tolist(),
                    "retracted": bool(raw["retracted"]),
                    "grasped": bool(raw["grasped"]),
                    "collision": bool(raw.get("collision", False)),
                    "collision_type": str(raw.get("collision_type", "")),
                    "max_object_jump_m": max_object_jump,
                    "max_contact_force_n": max_contact_force,
                    "active_arm_trace": active_arm_trace,
                    "events": dict(events),
                })
                print(json.dumps({"case": case, "episode": episode, "seed": seed,
                                  "success": bool(info["success"]), "score": float(info["score"]),
                                  "safety_violation": bool(info["safety_violation"]),
                                  "max_object_jump_m": max_object_jump}, ensure_ascii=False), flush=True)
    finally:
        env.close()

    by_case = {}
    for case in (4, 5, 6, 7):
        subset = [r for r in records if r["case"] == case]
        by_case[str(case)] = {
            "episodes": len(subset),
            "successes": sum(r["success"] for r in subset),
            "mean_score": float(np.mean([r["score"] for r in subset])),
            "safety_failures": sum(r["safety_violation"] for r in subset),
            "collisions": sum(r["collision"] for r in subset),
            "max_object_jump_m": float(max(r["max_object_jump_m"] for r in subset)),
            "mean_steps": float(np.mean([r["steps"] for r in subset])),
        }
    summary = {
        "checkpoint": str(args.checkpoint),
        "randomize_scene": True,
        "scene_jitter_xy_m": [0.025, 0.020],
        "box_jitter_xy_m": [0.015, 0.010],
        "mode": "stochastic" if args.stochastic else "deterministic",
        "limitations": [
            "MuJoCo proxy backend only; not a real-robot success rate.",
            "Bounded XY jitter preserves left/right partition and does not replace camera/hand-eye validation.",
        ],
        "overall": {
            "episodes": len(records), "successes": sum(r["success"] for r in records),
            "mean_score": float(np.mean([r["score"] for r in records])),
            "safety_failures": sum(r["safety_violation"] for r in records),
            "collisions": sum(r["collision"] for r in records),
            "max_object_jump_m": float(max(r["max_object_jump_m"] for r in records)),
        },
        "by_case": by_case,
        "episodes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps({"output": str(args.output), "overall": summary["overall"], "by_case": by_case}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
