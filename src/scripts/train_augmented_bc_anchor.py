#!/usr/bin/env python3
"""Conservative actor fine-tuning on audited augmented demonstrations.

The original successful policy acts as a teacher/anchor. New randomized expert
actions are blended in according to jitter scale, while gripper labels always
come from successful demonstrations. Candidate checkpoints are accepted only
when deterministic and randomized safety gates pass. Simulation-only.
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def evaluate(model, cfg, randomized, episodes_per_case, seed_base):
    from tree_reinforcement_learning.env import Case47Env
    from tree_reinforcement_learning.mujoco_backend import MujocoBackend
    from tree_reinforcement_learning.reward import RewardConfig
    records=[]
    env=Case47Env(MujocoBackend(randomize_scene=randomized),max_episode_steps=cfg["max_episode_steps"],reward_config=RewardConfig(**cfg["reward"]))
    try:
        for case in range(4,8):
            for ep in range(episodes_per_case):
                seed=seed_base+case*100+ep
                obs,_=env.reset(seed=seed,options={"case_id":case})
                max_jump=0.; prev=np.asarray(env.backend.get_observation()["object_positions"],float)
                for step in range(1,cfg["max_episode_steps"]+1):
                    action,_=model.predict(obs,deterministic=True)
                    obs,_,term,trunc,info=env.step(action)
                    cur=np.asarray(env.backend.get_observation()["object_positions"],float)
                    max_jump=max(max_jump,float(np.max(np.linalg.norm(cur-prev,axis=1))));prev=cur
                    if term or trunc: break
                records.append({"case":case,"seed":seed,"success":bool(info["success"]),"score":float(info["score"]),"safety":bool(info["safety_violation"]),"steps":step,"max_object_jump_m":max_jump,"events":list(info["events"])})
    finally: env.close()
    return {"episodes":len(records),"successes":sum(x["success"] for x in records),"mean_score":float(np.mean([x["score"] for x in records])),"safety_failures":sum(x["safety"] for x in records),"max_object_jump_m":max(x["max_object_jump_m"] for x in records),"records":records}


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--dataset",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--epochs",type=int,default=80);p.add_argument("--eval-every",type=int,default=10);p.add_argument("--batch-size",type=int,default=1024);p.add_argument("--learning-rate",type=float,default=3e-5);p.add_argument("--device",default="cuda");p.add_argument("--seed",type=int,default=2026091513);args=p.parse_args()
    import torch
    from stable_baselines3 import SAC
    from tree_reinforcement_learning.cli import DEFAULT_CONFIG,load_config
    torch.set_num_threads(1);np.random.seed(args.seed);torch.manual_seed(args.seed);cfg=load_config(DEFAULT_CONFIG);args.output.mkdir(parents=True,exist_ok=True)
    z=np.load(args.dataset);obs=z["observations"].astype(np.float32);expert=z["actions"].astype(np.float32);cases=z["cases"];scales=z["jitter_scales"].astype(np.float32);phases=z["phases"].astype(str)
    model=SAC.load(str(args.checkpoint),device=args.device);actor=model.policy.actor
    # Teacher outputs are frozen before any optimization.
    teacher=[];actor.eval()
    with torch.no_grad():
        for s in range(0,len(obs),4096): teacher.append(actor(torch.as_tensor(obs[s:s+4096],device=model.device),deterministic=True).cpu().numpy())
    teacher=np.concatenate(teacher).astype(np.float32)
    alpha=np.clip(.20+.80*scales,0,1)[:,None]
    targets=teacher*(1-alpha)+expert*alpha
    targets[:,6]=expert[:,6]  # never average open and close commands
    # Balance cases and oversample sparse gripper confirmation transitions.
    rng=np.random.default_rng(args.seed);indices=[];max_case=max(int(np.sum(cases==c)) for c in range(4,8))
    for c in range(4,8):
        ix=np.flatnonzero(cases==c);indices.append(rng.choice(ix,max_case,replace=len(ix)<max_case))
    indices=np.concatenate(indices)
    critical=np.flatnonzero(np.isin(phases,["grasp_confirm","release_confirm"]))
    indices=np.concatenate([indices,np.tile(critical,4)]);rng.shuffle(indices)
    initial={k:v.detach().clone() for k,v in actor.named_parameters()}
    optimizer=torch.optim.Adam(actor.parameters(),lr=args.learning_rate)
    weights=torch.tensor([1,1,1,.05,.05,.05,10.],dtype=torch.float32,device=model.device)
    baseline_det=evaluate(model,cfg,False,5,2026093000);baseline_rnd=evaluate(model,cfg,True,8,2026094000)
    report={"simulation_only":True,"dataset":str(args.dataset),"transitions_raw":len(obs),"samples_balanced":len(indices),"baseline":{"deterministic":baseline_det,"randomized":baseline_rnd},"epochs":[]}
    best_quality=(baseline_rnd["successes"],baseline_det["successes"],baseline_rnd["mean_score"]+baseline_det["mean_score"])
    model.save(str(args.output/"sac_case4_7_best"))
    print(json.dumps({"stage":"baseline","det":baseline_det["successes"],"rnd":baseline_rnd["successes"],"rnd_safety":baseline_rnd["safety_failures"],"samples":len(indices)}),flush=True)
    for epoch in range(1,args.epochs+1):
        rng.shuffle(indices);actor.train();losses=[]
        for s in range(0,len(indices),args.batch_size):
            ix=indices[s:s+args.batch_size];o=torch.as_tensor(obs[ix],device=model.device);y=torch.as_tensor(targets[ix],device=model.device)
            pred=actor(o,deterministic=True);bc=(((pred-y)**2)*weights).mean()
            # Small parameter trust region prevents catastrophic policy drift.
            anchor=sum((v-initial[k]).pow(2).mean() for k,v in actor.named_parameters())
            loss=bc+1e-3*anchor
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),.5);optimizer.step();losses.append(float(loss.detach().cpu()))
        if epoch%args.eval_every==0 or epoch==args.epochs:
            actor.eval();det=evaluate(model,cfg,False,5,2026093000);rnd=evaluate(model,cfg,True,8,2026094000)
            quality=(rnd["successes"],det["successes"],rnd["mean_score"]+det["mean_score"])
            accepted=(det["successes"]==det["episodes"] and det["safety_failures"]==0 and rnd["safety_failures"]==0 and quality>=best_quality)
            if accepted: best_quality=quality;model.save(str(args.output/"sac_case4_7_best"))
            row={"epoch":epoch,"loss":float(np.mean(losses)),"deterministic":det,"randomized":rnd,"accepted":accepted};report["epochs"].append(row)
            (args.output/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            print(json.dumps({"epoch":epoch,"loss":row["loss"],"det":det["successes"],"rnd":rnd["successes"],"rnd_safety":rnd["safety_failures"],"accepted":accepted}),flush=True)
    model.save(str(args.output/"sac_case4_7_final_candidate"));report["best_quality"]=best_quality;report["completed_unix"]=time.time();(args.output/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"completed":True,"best_quality":best_quality,"best_checkpoint":str(args.output/"sac_case4_7_best.zip")}),flush=True)
if __name__=="__main__":main()
