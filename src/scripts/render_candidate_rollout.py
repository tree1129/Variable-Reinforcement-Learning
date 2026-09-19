#!/usr/bin/env python3
"""Offline GPU4/Case6 diagnostic replay: no hardware interfaces."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys,json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch,mujoco
from PIL import Image,ImageDraw
from stable_baselines3 import SAC
from tree_reinforcement_learning.env import Case47Env
from tree_reinforcement_learning.mujoco_backend import MujocoBackend
from tree_reinforcement_learning.cli import load_config,DEFAULT_CONFIG
from tree_reinforcement_learning.reward import RewardConfig

torch.set_num_threads(1)
config=load_config(DEFAULT_CONFIG)
backend=MujocoBackend()
env=Case47Env(backend,max_episode_steps=config['max_episode_steps'],reward_config=RewardConfig(**config['reward']))
model=SAC.load(ROOT/'outputs/case4_7_sac_7gpu/gpu_4/sac_case4_7.zip',device='cpu')
out=ROOT/'outputs/real_comparison'
renderer=mujoco.Renderer(backend.model,height=640,width=960)
try:
    obs,_=env.reset(seed=910600,options={'case_id':6})
    record=[]
    for step in range(1,601):
        action,_=model.predict(obs,deterministic=True)
        obs,reward,terminated,truncated,info=env.step(action)
        if info['events']:
            record.append(dict(step=step,events=list(info['events']),score=info['score']))
        if 'grasp_success' in info['events'] or terminated or truncated:
            renderer.update_scene(backend.data,camera='pilot_overview')
            image=Image.new('RGB',(960,700),'white')
            image.paste(Image.fromarray(renderer.render()),(0,40))
            draw=ImageDraw.Draw(image)
            draw.text((15,12),f'GPU4 POLICY | CASE6 | STEP {step} | OFFLINE DETERMINISTIC REPLAY',fill='black')
            draw.text((15,675),f'proxy grasp={backend.grasped_shape} | inserted={backend.inserted.tolist()} | score={info["score"]}/10',fill='black')
            image.save(out/'gpu4_case6_first_event.png')
            (out/'gpu4_case6_first_event.json').write_text(json.dumps(record,indent=2))
            print(json.dumps(record))
            break
finally:
    renderer.close()
    env.close()
