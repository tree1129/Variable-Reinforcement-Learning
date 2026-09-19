#!/usr/bin/env python3
"""Collision-checked demonstration warm start for the bilateral Case 4--7 SAC policy.

The demonstrations are executed through the same 84-D observation, 7-D action,
MuJoCo safety monitor, and arm-routing logic used at deployment.  We optimize
only the SAC actor to reproduce those safe transitions, then save a normal SB3
SAC checkpoint for subsequent SAC fine tuning.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tree_reinforcement_learning.cli import DEFAULT_CONFIG, load_config
from tree_reinforcement_learning.env import Case47Env
from tree_reinforcement_learning.mujoco_backend import MujocoBackend
from tree_reinforcement_learning.reward import RewardConfig
from tree_reinforcement_learning.tasks import FACE_ORDER, SHAPE_ORDER


def make_env(config, seed: int) -> Case47Env:
    return Case47Env(MujocoBackend(), max_episode_steps=config["max_episode_steps"],
                     reward_config=RewardConfig(**config["reward"]), seed=seed)


def collect_episode(env: Case47Env, case: int, seed: int):
    obs, _ = env.reset(seed=seed, options={"case_id": case})
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    backend = env.backend
    def issue(position: np.ndarray, gripper: float, limit: int = 260) -> None:
        nonlocal obs
        for _ in range(limit):
            delta = np.asarray(position, dtype=np.float64) - backend._ee_position()
            action = np.zeros(7, dtype=np.float32)
            action[:3] = np.clip(delta / backend.position_step_m, -1.0, 1.0)
            action[6] = gripper
            observations.append(obs.copy()); actions.append(action.copy())
            obs, _, terminated, truncated, info = env.step(action)
            if terminated:
                if info["success"]:
                    return
                raise RuntimeError(f"unsafe expert transition case={case}: {info['events']}")
            if np.linalg.norm(delta) <= 0.030:
                return
        raise RuntimeError(f"expert cannot reach {position.tolist()} in case {case}")

    for target_index, shape in enumerate(backend.task.targets):
        object_index = SHAPE_ORDER.index(shape)
        obj = backend.get_observation()["object_positions"][object_index].copy()
        grasp = backend._interaction_pose(obj)
        # Above-table approach, then close the selected physical gripper.
        issue(grasp + np.asarray((0.0, 0.0, 0.10)), 1.0)
        issue(grasp, 1.0)
        for _ in range(3):
            observations.append(obs.copy()); actions.append(np.asarray([0,0,0,0,0,0,-1], dtype=np.float32))
            obs, _, terminated, _, info = env.step(actions[-1])
            if terminated: raise RuntimeError(f"unsafe expert close: {info['events']}")
            if backend.grasped_shape == shape: break
        if backend.grasped_shape != shape:
            raise RuntimeError(f"expert failed grasp: expected {shape}, got {backend.grasped_shape}")
        face = FACE_ORDER.index(backend.task.hole_faces[shape])
        hole = backend._hole_poses()[face, :3]
        issue(backend._interaction_pose(hole), -1.0)
        for _ in range(3):
            observations.append(obs.copy()); actions.append(np.asarray([0,0,0,0,0,0,1], dtype=np.float32))
            obs, _, terminated, _, info = env.step(actions[-1])
            if terminated: raise RuntimeError(f"unsafe expert release: {info['events']}")
            if backend.inserted[target_index]: break
        if not backend.inserted[target_index]:
            raise RuntimeError(f"expert failed insertion of {shape}")
    home = backend._home_position()
    issue(home, 1.0)
    return np.asarray(observations, np.float32), np.asarray(actions, np.float32)


def evaluate(model: SAC, config: dict, episodes_per_case: int = 2) -> dict:
    records=[]
    for case in range(4,8):
        for episode in range(episodes_per_case):
            env=make_env(config, 880000+case*10+episode)
            obs,_=env.reset(seed=880000+case*10+episode, options={"case_id":case})
            for step in range(900):
                action,_=model.predict(obs, deterministic=True)
                obs,_,terminated,truncated,info=env.step(action)
                if terminated or truncated: break
            records.append({"case":case,"score":float(info["score"]),"success":bool(info["success"]),
                            "safety":bool(info["safety_violation"]),"steps":step+1,
                            "inserted":env.backend.inserted.tolist(),"retracted":bool(env.backend.retracted)})
            env.close()
    return {"successes":sum(r["success"] for r in records), "mean_score":float(np.mean([r["score"] for r in records])), "records":records}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--copies", type=int, default=96)
    p.add_argument("--seed", type=int, default=20260970)
    p.add_argument("--device", default="cuda")
    args=p.parse_args(); np.random.seed(args.seed); torch.manual_seed(args.seed)
    config=load_config(DEFAULT_CONFIG); args.output.mkdir(parents=True,exist_ok=True)
    # Each case is repeated to balance minibatches while retaining only verified trajectories.
    dataset=[]
    env=make_env(config,args.seed)
    for case in range(4,8):
        ob,ac=collect_episode(env,case,args.seed+case)
        print(f"demo case={case}: transitions={len(ob)}", flush=True)
        dataset.append((ob,ac))
    env.close()
    obs=np.concatenate([x[0] for x in dataset]); act=np.concatenate([x[1] for x in dataset])
    # Oversample close/release actions: they are safety-critical but sparse.
    critical=np.abs(act[:,6]) > .5
    obs=np.concatenate([np.tile(obs,(args.copies,1)), np.tile(obs[critical],(32,1))])
    act=np.concatenate([np.tile(act,(args.copies,1)), np.tile(act[critical],(32,1))])
    rng=np.random.default_rng(args.seed); order=rng.permutation(len(obs)); obs,act=obs[order],act[order]
    env=make_env(config,args.seed)
    model=SAC("MlpPolicy",env,learning_rate=3e-4,buffer_size=100000,batch_size=512,
              learning_starts=0,gamma=.99,tau=.005,ent_coef="auto",seed=args.seed,device=args.device,verbose=0)
    actor=model.policy.actor; actor.train(); batch=512
    weights=torch.tensor([1,1,1,.15,.15,.15,12.0],dtype=torch.float32,device=model.device)
    best=(-1.0, None)
    for epoch in range(1,args.epochs+1):
        order=rng.permutation(len(obs)); losses=[]
        for start in range(0,len(obs),batch):
            ix=order[start:start+batch]
            o=torch.as_tensor(obs[ix],device=model.device); a=torch.as_tensor(act[ix],device=model.device)
            predicted=actor(o,deterministic=True)
            loss=(((predicted-a)**2)*weights).mean()
            actor.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0); actor.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % 100 == 0 or epoch == args.epochs:
            actor.eval(); result=evaluate(model,config); actor.train()
            print(json.dumps({"epoch":epoch,"loss":float(np.mean(losses)),**{k:v for k,v in result.items() if k!='records'}},ensure_ascii=False),flush=True)
            quality=(result["successes"],result["mean_score"])
            if quality > best[0:2]:
                best=(quality[0],quality[1],result)
                model.save(str(args.output/"sac_case4_7_best"))
    model.save(str(args.output/"sac_case4_7"))
    actor.eval(); final=evaluate(model,config,episodes_per_case=3)
    (args.output/"bc_metrics.json").write_text(json.dumps({"dataset_transitions":int(len(obs)),"best":best[2],"final":final},indent=2))
    print(json.dumps({"final":final,"output":str(args.output)},ensure_ascii=False),flush=True)
    env.close()
if __name__ == "__main__": main()
