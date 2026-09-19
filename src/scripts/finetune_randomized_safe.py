#!/usr/bin/env python3
"""Legacy independent SAC continuation, NOT V4 residual learning.

Requires a trained critic and its replay buffer. Actor-only BC checkpoints must
use the separate frozen-base residual entry instead. Paired selection scores
are not a hardware deployment gate or an independent holdout.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tree_reinforcement_learning.acceptance import compare_evaluations


def run_eval(model, config, *, randomized: bool, seed_base: int, episodes_per_case: int = 2):
    import torch
    from tree_reinforcement_learning.env import Case47Env
    from tree_reinforcement_learning.mujoco_backend import MujocoBackend
    from tree_reinforcement_learning.reward import RewardConfig
    torch.set_num_threads(1)
    env = Case47Env(MujocoBackend(randomize_scene=randomized), max_episode_steps=int(config["max_episode_steps"]),
                    reward_config=RewardConfig(**config["reward"]))
    rec = []
    try:
        for case in range(4, 8):
            for ep in range(episodes_per_case):
                seed = seed_base + case * 100 + ep
                obs, _ = env.reset(seed=seed, options={"case_id": case})
                for step in range(1, int(config["max_episode_steps"]) + 1):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = env.step(action)
                    if term or trunc:
                        break
                rec.append({"case": case, "episode": ep, "seed": seed, "success": bool(info["success"]),
                            "score": float(info["score"]), "safety_violation": bool(info["safety_violation"]), "steps": step})
    finally:
        env.close()
    return {"episodes": len(rec), "successes": sum(x["success"] for x in rec),
            "mean_score": float(np.mean([x["score"] for x in rec])),
            "safety_failures": sum(x["safety_violation"] for x in rec), "records": rec}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--replay-buffer", type=Path, required=True, help="Trusted replay matching the checkpoint; pickle format")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps-per-stage", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026091512)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.steps_per_stage <= 0:
        p.error("steps-per-stage must be positive")
    import torch
    from stable_baselines3 import SAC
    from tree_reinforcement_learning.cli import DEFAULT_CONFIG, load_config
    from tree_reinforcement_learning.env import Case47Env
    from tree_reinforcement_learning.mujoco_backend import MujocoBackend
    from tree_reinforcement_learning.reward import RewardConfig
    torch.set_num_threads(1)
    config = load_config(DEFAULT_CONFIG)
    checkpoint = args.checkpoint if args.checkpoint.is_file() else args.checkpoint.with_suffix(".zip")
    baseline_model = SAC.load(str(checkpoint), device="cpu")
    if baseline_model._n_updates <= 0:
        p.error("critic has no recorded RL updates; refusing actor-only BC continuation")
    baseline_model.load_replay_buffer(str(args.replay_buffer))
    if baseline_model.replay_buffer.size() < baseline_model.batch_size:
        p.error("replay buffer smaller than batch_size")
    args.output.mkdir(parents=True, exist_ok=False)
    def evaluate(model):
        return {"deterministic": run_eval(model, config, randomized=False, seed_base=2026092600),
                "randomized": run_eval(model, config, randomized=True, seed_base=2026092700)}
    reference = evaluate(baseline_model)
    baseline_gate = compare_evaluations(reference, reference)
    best_path = args.output / "sac_case4_7_best.zip"
    replay_path = args.output / "best_replay.pkl"
    history = [{"stage": "baseline", **reference, "gate": baseline_gate, "accepted": baseline_gate["accepted"]}]
    current, current_replay = checkpoint, args.replay_buffer
    if baseline_gate["accepted"]:
        shutil.copy2(checkpoint, best_path)
    for name, scale in [("jitter25", .25), ("jitter50", .5), ("jitter75", .75), ("jitter100", 1.)]:
        if not history[-1]["accepted"]:
            break  # Do not increase difficulty after a rejected stage.
        env = Case47Env(MujocoBackend(randomize_scene=True, scene_jitter_xy_m=(.025*scale, .020*scale),
                                     box_jitter_xy_m=(.015*scale, .010*scale)),
                        max_episode_steps=int(config["max_episode_steps"]), reward_config=RewardConfig(**config["reward"]), seed=args.seed)
        try:
            model = SAC.load(str(current), env=env, device=args.device)
            model.load_replay_buffer(str(current_replay))
            model.set_random_seed(args.seed + int(scale*100))
            model.learn(total_timesteps=args.steps_per_stage, reset_num_timesteps=False, progress_bar=False)
            model.save(str(args.output / f"candidate_{name}.zip"))
        finally:
            env.close()
        candidate = evaluate(model)
        gate = compare_evaluations(reference, candidate)
        history.append({"stage": name, "scale": scale, **candidate, "gate": gate, "accepted": gate["accepted"]})
        if gate["accepted"]:
            model.save(str(best_path))
            model.save_replay_buffer(str(replay_path))
            reference, current, current_replay = candidate, best_path, replay_path
        print(json.dumps({"stage": name, "gate": gate}), flush=True)
    result = {"checkpoint_start": str(checkpoint), "best_checkpoint": str(best_path) if best_path.exists() else None,
              "stages": history, "hardware_authorized": False, "limitations": [
                  "Independent SAC, not V4 residual learning.", "Proxy MuJoCo, not contact/real-robot validation.",
                  "Paired selection only; no independent holdout or statistical guarantee."]}
    (args.output / "finetune_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
