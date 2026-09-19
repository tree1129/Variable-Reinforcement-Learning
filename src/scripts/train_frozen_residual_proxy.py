#!/usr/bin/env python3
"""Train grasp-only residuals around a FROZEN proxy base, never V4/hardware.

No --hardware, ROS, V4 service, or arbitrary backend option exists. The toy base
is a synthetic software fixture. MuJoCo uses the existing nonphysical grasp proxy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("toy", "mujoco"), default="toy")
    base_group = p.add_mutually_exclusive_group(required=True)
    base_group.add_argument("--base-checkpoint", type=Path)
    base_group.add_argument("--scripted-toy-base", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--prefill", type=int, default=512)
    p.add_argument("--critic-warmup", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--episodes-per-case", type=int, default=3)
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=47)
    args = p.parse_args()
    if args.scripted_toy_base and args.backend != "toy":
        p.error("scripted fixture only supports --backend toy")
    for name in ("steps", "prefill", "critic_warmup", "batch_size", "eval_every", "episodes_per_case", "max_episode_steps"):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    if args.prefill < args.batch_size or args.steps < args.prefill + args.critic_warmup + 2:
        p.error("need prefill >= batch-size and enough steps for critic warmup then actor updates")
    import torch
    from tree_reinforcement_learning.acceptance import compare_evaluations, assert_disjoint_evaluations
    from tree_reinforcement_learning.env import Case47Env
    from tree_reinforcement_learning.grasp_proxy import SingleGraspProxyEnv
    from tree_reinforcement_learning.residual import FrozenSACBase, ScriptedToyBase, Replay, ResidualTD3, module_digest
    from tree_reinforcement_learning.trace_audit import sha256
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    base = ScriptedToyBase() if args.scripted_toy_base else FrozenSACBase(args.base_checkpoint)
    checkpoint_digest = sha256(args.base_checkpoint) if args.base_checkpoint else None
    args.output.mkdir(parents=True, exist_ok=False)
    learner = ResidualTD3(96, critic_warmup=args.critic_warmup)
    initial_actor_digest = module_digest(learner.actor)
    replay = Replay(max(args.steps, args.batch_size), 96, args.seed)

    def make_env(randomized):
        if args.backend == "toy":
            from tree_reinforcement_learning.toy_backend import ToyBackend
            backend = ToyBackend()
        else:
            from tree_reinforcement_learning.mujoco_backend import MujocoBackend
            backend = MujocoBackend(randomize_scene=randomized)
        return SingleGraspProxyEnv(Case47Env(backend, max_episode_steps=args.max_episode_steps), base)

    def evaluate(predict, seed_offset):
        suites = {}
        for name, randomized, offset in (("fixed", False, 0), ("randomized", True, 100000)):
            env = make_env(randomized)
            records = []
            try:
                for case in range(4, 8):
                    for episode in range(args.episodes_per_case):
                        seed = seed_offset + offset + case * 1000 + episode
                        obs, _ = env.reset(seed=seed, options={"case_id": case})
                        for step in range(args.max_episode_steps):
                            obs, _, term, trunc, info = env.step(predict(obs))
                            if term or trunc:
                                break
                        records.append({"case": case, "seed": seed, "success": info["success"], "score": info["score"],
                                        "safety_violation": bool(info["safety_violation"]), "steps": step+1})
            finally:
                env.close()
            suites[name] = {"records": records}
        return suites

    def save(name, state):
        torch.save({"format": "frozen_proxy_grasp_residual_v1", "state": state,
                    "base_checkpoint": str(args.base_checkpoint) if args.base_checkpoint else "scripted_toy_fixture",
                    "base_checkpoint_sha256": checkpoint_digest, "base_policy_digest": base.initial_digest,
                    "action_contract": "proxy_normalized7_augmented_obs96", "hardware_authorized": False,
                    "v4_connected": False, "stage": "single_first_block_grasp"}, args.output / name)

    report = {"status": "running", "backend": args.backend, "base_checkpoint": str(args.base_checkpoint),
              "base_checkpoint_sha256": checkpoint_digest, "base_policy_digest": base.initial_digest,
              "hardware_authorized": False, "v4_connected": False, "stage": "single_first_block_grasp",
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "history": [], "limitations": ["Nonphysical proxy grasp/hold, not real shallow-grasp or slip validation.",
                  "7-D simulation controller, NOT the V4 20-D action interface.",
                  "Finite paired episodes, not a statistical safety guarantee.",
                  "Toy fixed/randomized suites differ only by seeds; toy has no physical randomization.",
                  "Holdout used once at end; do not repeatedly tune on these same holdout seeds.",
                  "Saved checkpoints are diagnostic artifacts, not hardware or exact-resume packages."]}
    env = None
    logs = []
    try:
        selection = evaluate(learner.predict, 2000000)
        baseline_gate = compare_evaluations(selection, selection)
        report["baseline_selection"] = selection
        report["baseline_gate"] = baseline_gate
        if not baseline_gate["accepted"]:
            report["status"] = "blocked_unsafe_or_invalid_proxy_baseline"
            return 2
        best = learner.snapshot()
        save("zero_residual.pt", best)
        save("selected_residual.pt", best)
        env = make_env(True)
        episode_index = 0
        obs, _ = env.reset(seed=args.seed, options={"case_id": 4})
        stop_reason = "step_budget"
        for step in range(1, args.steps+1):
            # Small bounded exploration, never a full random hardware action.
            proposed = np.clip(learner.predict(obs) + rng.normal(0, .1, 7), -1, 1)
            next_obs, reward, term, trunc, info = env.step(proposed)
            replay.add(obs, info["effective_residual"], reward, next_obs, term)
            # Record only actions actually applied to THIS proxy transition.
            logs.append({"state": obs.copy(), "next_state": next_obs.copy(), "base": info["base_action"],
                         "applied": info["applied_action"], "residual": info["effective_residual"],
                         "reward": reward, "terminal": term, "truncated": trunc,
                         "case": env.env.case_id, "episode": episode_index})
            obs = next_obs
            if term or trunc:
                episode_index += 1
                obs, _ = env.reset(seed=args.seed+episode_index, options={"case_id": 4+episode_index % 4})
            metrics = None
            if step > args.prefill:
                metrics = learner.update(replay.sample(args.batch_size))
            if step % args.eval_every == 0 or step == args.steps:
                base.assert_frozen()
                if learner.actor_updates == 0 and module_digest(learner.actor) != initial_actor_digest:
                    raise RuntimeError("actor_changed_during_prefill_or_critic_only_warmup")
                candidate = evaluate(learner.predict, 2000000)
                gate = compare_evaluations(selection, candidate)
                record = {"step": step, "evaluation": candidate, "gate": gate, "updates": metrics,
                          "actor_updates": learner.actor_updates, "critic_updates": learner.critic_updates}
                report["history"].append(record)
                save(f"candidate_{step}.pt", learner.snapshot())
                print(json.dumps({"step": step, "accepted": gate["accepted"], "reasons": gate["reasons"],
                                  "actor_updates": learner.actor_updates, "critic_updates": learner.critic_updates}), flush=True)
                if gate["accepted"]:
                    best, selection = learner.snapshot(), candidate
                    save("selected_residual.pt", best)
                else:
                    learner.restore(best)
                    stop_reason = "candidate_rejected_restored_selected_and_stopped"
                    break
        base.assert_frozen()
        if args.base_checkpoint and sha256(args.base_checkpoint) != checkpoint_digest:
            raise RuntimeError("base_checkpoint_file_changed")
        # Locked holdout: comparison is not fed back into further actor updates.
        holdout_base = evaluate(lambda obs: np.zeros(7, np.float32), 4000000)
        holdout_selected = evaluate(learner.predict, 4000000)
        assert_disjoint_evaluations(selection, holdout_selected)
        holdout_gate = compare_evaluations(holdout_base, holdout_selected)
        report.update(status="completed_proxy_only", stop_reason=stop_reason, steps_collected=len(logs),
                      base_unchanged=True, selected_actor_updates=learner.actor_updates,
                      selected_critic_updates=learner.critic_updates,
                      selected_actor_changed=module_digest(learner.actor) != initial_actor_digest,
                      selected_selection=selection, holdout_base=holdout_base, holdout_selected=holdout_selected,
                      holdout_gate=holdout_gate, selected_checkpoint=str(args.output / "selected_residual.pt"))
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if env is not None:
            env.close()
        if logs:
            # Deliberately NOT v4_executed_trace_v1 and cannot pass real-data audit.
            np.savez_compressed(args.output / "executed_proxy_transitions.npz",
                                **{key: np.asarray([r[key] for r in logs]) for key in logs[0]},
                                manifest_json=json.dumps({"schema": "proxy_residual_diagnostics_v1", "source": "proxy_sim", "v4_connected": False}))
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(args.output / "report.json"),
                          "hardware_authorized": False, "v4_connected": False}), flush=True)

if __name__ == "__main__":
    raise SystemExit(main())
