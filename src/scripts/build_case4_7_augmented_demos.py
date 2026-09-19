#!/usr/bin/env python3
"""Build audited, successful-only Case 4-7 demonstrations with scene jitter.

Simulation-only: no ROS imports, publishers, services, actions, or robot control.
The generated demonstrations add explicit lift/transport/descend waypoints so
objects remain clamped and the arms clear the tabletop and sorter housing.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tree_reinforcement_learning.cli import DEFAULT_CONFIG, load_config
from tree_reinforcement_learning.env import Case47Env
from tree_reinforcement_learning.mujoco_backend import MujocoBackend
from tree_reinforcement_learning.reward import RewardConfig
from tree_reinforcement_learning.tasks import FACE_ORDER, SHAPE_ORDER


def collect(env: Case47Env, case: int, seed: int):
    obs, _ = env.reset(seed=seed, options={"case_id": case})
    b = env.backend
    observations, actions, phases = [], [], []
    events = Counter()

    def step_action(action, phase):
        nonlocal obs
        observations.append(obs.copy()); actions.append(np.asarray(action, np.float32)); phases.append(phase)
        obs, _, terminated, truncated, info = env.step(actions[-1])
        events.update(info["events"])
        if terminated and not info["success"]:
            raise RuntimeError(f"unsafe:{tuple(info['events'])}")
        if truncated:
            raise RuntimeError("timeout")
        return terminated, info

    def move(position, grip, phase, tolerance=.025, limit=300):
        for _ in range(limit):
            delta = np.asarray(position, float) - b._ee_position()
            action = np.zeros(7, np.float32)
            action[:3] = np.clip(delta / b.position_step_m, -1, 1)
            action[6] = grip
            term, info = step_action(action, phase)
            if term or np.linalg.norm(delta) <= tolerance:
                return
        raise RuntimeError(f"unreachable:{phase}")

    for target_index, shape in enumerate(b.task.targets):
        obj = b.get_observation()["object_positions"][SHAPE_ORDER.index(shape)].copy()
        grasp = b._interaction_pose(obj)
        # High approach and vertical descent provide tabletop clearance.
        move(grasp + np.array((0., 0., .12)), 1., "approach_high")
        move(grasp, 1., "approach_descend")
        for _ in range(4):
            term, info = step_action([0,0,0,0,0,0,-1], "grasp_confirm")
            if b.grasped_shape == shape: break
        if b.grasped_shape != shape:
            raise RuntimeError(f"grasp_failed:{shape}")
        # Keep the gripper closed while lifting and crossing toward the box.
        move(grasp + np.array((0., 0., .14)), -1., "lift_clamped")
        face = FACE_ORDER.index(b.task.hole_faces[shape])
        hole = b._interaction_pose(b._hole_poses()[face, :3])
        move(hole + np.array((0., 0., .14)), -1., "transport_high_clamped")
        move(hole, -1., "place_descend_clamped")
        for _ in range(4):
            term, info = step_action([0,0,0,0,0,0,1], "release_confirm")
            if b.inserted[target_index]: break
        if not b.inserted[target_index]:
            raise RuntimeError(f"insert_failed:{shape}")
    move(b._home_position(), 1., "retract")
    raw = b.get_observation()
    if not (np.all(raw["inserted"]) and raw["retracted"] and not raw["forbidden_contact"]):
        raise RuntimeError("final_gate_failed")
    return np.asarray(observations,np.float32), np.asarray(actions,np.float32), np.asarray(phases), dict(events)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--episodes-per-case-scale",type=int,default=12)
    p.add_argument("--seed",type=int,default=2026091500)
    args=p.parse_args(); cfg=load_config(DEFAULT_CONFIG); args.output.mkdir(parents=True,exist_ok=True)
    scales=(0.0,0.25,0.5,0.75,1.0)
    all_o=[]; all_a=[]; all_case=[]; all_seed=[]; all_scale=[]; all_phase=[]; episodes=[]
    started=time.time()
    for scale_index,scale in enumerate(scales):
        b=MujocoBackend(randomize_scene=scale>0,
            scene_jitter_xy_m=(.025*scale,.020*scale),box_jitter_xy_m=(.015*scale,.010*scale))
        env=Case47Env(b,max_episode_steps=cfg["max_episode_steps"],reward_config=RewardConfig(**cfg["reward"]))
        for case in range(4,8):
            for ep in range(args.episodes_per_case_scale):
                seed=args.seed+scale_index*10000+case*100+ep
                try:
                    o,a,phase,events=collect(env,case,seed)
                    ok=True; error=""
                    all_o.append(o);all_a.append(a);all_case.append(np.full(len(o),case,np.int8));all_seed.append(np.full(len(o),seed,np.int64));all_scale.append(np.full(len(o),scale,np.float32));all_phase.append(phase)
                except Exception as exc:
                    ok=False;error=str(exc);o=np.empty((0,88));events={}
                episodes.append({"case":case,"seed":seed,"scale":scale,"success":ok,"transitions":len(o),"events":events,"error":error})
                print(json.dumps(episodes[-1],ensure_ascii=False),flush=True)
        env.close()
    if not all_o: raise SystemExit("no successful demonstrations")
    obs=np.concatenate(all_o); actions=np.concatenate(all_a); cases=np.concatenate(all_case); seeds=np.concatenate(all_seed); scales_arr=np.concatenate(all_scale); phases=np.concatenate(all_phase)
    np.savez_compressed(args.output/"case4_7_augmented_success_demos.npz",observations=obs,actions=actions,cases=cases,seeds=seeds,jitter_scales=scales_arr,phases=phases)
    manifest={"created_unix":time.time(),"simulation_only":True,"hardware_motion_enabled":False,
      "episodes_requested":len(episodes),"episodes_successful":sum(x["success"] for x in episodes),"transitions":len(obs),
      "observation_dim":int(obs.shape[1]),"action_dim":int(actions.shape[1]),"scales":list(scales),
      "phase_counts":dict(Counter(phases.tolist())),"case_counts":{str(c):int(np.sum(cases==c)) for c in range(4,8)},
      "episodes":episodes,"safety_gate":"SUCCESSFUL_NO_FORBIDDEN_CONTACT_ONLY"}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ("episodes_requested","episodes_successful","transitions","phase_counts","case_counts")},ensure_ascii=False),flush=True)
if __name__=="__main__": main()
