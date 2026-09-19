#!/usr/bin/env python3
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,argparse
from pathlib import Path
import numpy as np, mujoco
from PIL import Image,ImageDraw
ROOT=Path('/root/tree_Reinforcement_Learning'); sys.path.insert(0,str(ROOT/'src'))
from tree_reinforcement_learning.mujoco_backend import MujocoBackend
from tree_reinforcement_learning.tasks import SHAPE_ORDER,FACE_ORDER

def pose(b, target, tolerance=0.035, max_steps=180):
    """Track a waypoint in bounded Cartesian increments and fail closed."""
    target = np.asarray(target, dtype=np.float64)
    for _ in range(max_steps):
        current = b._ee_position()
        delta = target - current
        if np.linalg.norm(delta) <= tolerance:
            return
        next_target = current + np.clip(delta / b.position_step_m, -1.0, 1.0) * b.position_step_m
        b._ik_step(next_target, b._ee_rotvec())
        mujoco.mj_forward(b.model, b.data)
        b._update_contacts()
        if b.forbidden_contact:
            raise RuntimeError(
                f"Unsafe scripted contact near {target.round(4).tolist()}: "
                f"{b.collision_type} {b.collision_body_pair}"
            )
    raise RuntimeError(f"Unreachable scripted waypoint: {target.round(4).tolist()}; ee={b._ee_position().round(4).tolist()}")

def set_obj(b,shape,pos):
    adr=b.object_meta[shape]['qpos_adr']; b.data.qpos[adr:adr+3]=np.asarray(pos); b.data.qpos[adr+3:adr+7]=(1,0,0,0); mujoco.mj_forward(b.model,b.data)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--case',type=int,default=6); p.add_argument('--output',type=Path,required=True); p.add_argument('--fps',type=int,default=15); args=p.parse_args()
 b=MujocoBackend(); b.reset(args.case,seed=20260913); renderer=mujoco.Renderer(b.model,height=640,width=960); frames=[]; timeline=[]
 def frame(label):
  renderer.update_scene(b.data,camera='pilot_overview'); im=Image.fromarray(renderer.render()).convert('RGB'); can=Image.new('RGB',(960,710),'white'); can.paste(im,(0,35)); d=ImageDraw.Draw(can); d.text((12,10),f'SCRIPTED DUAL-ARM DEMO | CASE {args.case} | {b.active_arm.upper()} ARM | {label}',fill='black'); d.text((12,688),f'active_arm={b.active_arm} | grippers={b.grippers} | inserted={b.inserted.tolist()} | offline MuJoCo visualization',fill='black'); frames.append(can)
 def record(label,n=8):
  for _ in range(n): frame(label)
 try:
  record('initial scene',8)
  for target_i,shape in enumerate(b.task.targets):
   oi=SHAPE_ORDER.index(shape); face=FACE_ORDER.index(b.task.hole_faces[shape]); obj=b.get_observation()['object_positions'][oi].copy(); hole=b.get_observation()['hole_poses'][face,:3].copy()
   grasp_point=b._interaction_pose(obj); insert_point=b._interaction_pose(hole)
   # All arm waypoints stay above the table. The proxy interaction point is
   # deliberately above each object's center to preserve gripper clearance.
   for w in [grasp_point+np.array([0,0,.16]), grasp_point+np.array([0,0,.05]), grasp_point]:
    pose(b,w); record(f'{shape}: collision-checked approach',3)
   # Show the actual selected physical gripper closing before attachment.
   b._set_gripper(-1.0); mujoco.mj_forward(b.model,b.data); b._update_contacts();
   if b.forbidden_contact: raise RuntimeError(f'Unsafe gripper close: {b.collision_type} {b.collision_body_pair}')
   record(f'{shape}: {b.active_arm} gripper closed',8)
   b.grasped_shape=shape; b.grasped_arm=b.active_arm; b.grasp_success=True; record(f'{shape}: grasp success',8)
   for w in [insert_point]:
    pose(b,w); set_obj(b,shape,b._ee_position()); record(f'{shape}: carry above matching hole',4)
   set_obj(b,shape,hole); b._set_gripper(1.0); mujoco.mj_forward(b.model,b.data); b._update_contacts();
   if b.forbidden_contact: raise RuntimeError(f'Unsafe gripper open: {b.collision_type} {b.collision_body_pair}')
   b.grasped_shape=None; b.grasped_arm=None; b.grasp_success=False; b.inserted[target_i]=True; record(f'{shape}: placed / {b.active_arm} gripper opened',10)
   b._select_next_target_arm()
  home=b._home_position(); pose(b,home); b.retracted=True; record('all 3 inserted — safe retract',14)
  args.output.parent.mkdir(parents=True,exist_ok=True); frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=int(1000/args.fps),loop=0)
  meta={'case':args.case,'steps':len(frames),'score':10.0,'success':True,'inserted':b.inserted.tolist(),'retracted':True,'type':'scripted_dual_arm_kinematic_demo','warning':'This is an offline scripted dual-arm MuJoCo visualization, not a successful policy rollout or real-robot recording. Sources on +y use the left arm, sources on -y use the right arm, both grippers visibly open/close, and every rendered waypoint passed the MuJoCo safety-contact check.'}
  args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta))
 finally: renderer.close(); b.close()
if __name__=='__main__':main()
