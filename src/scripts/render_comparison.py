#!/usr/bin/env python3
"""Render training-backend scenes; no robot connection or command publishing."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import json
import mujoco
from PIL import Image, ImageDraw
from tree_reinforcement_learning.mujoco_backend import MujocoBackend
from tree_reinforcement_learning.tasks import SHAPE_ORDER

out = ROOT / 'outputs/real_comparison'
out.mkdir(parents=True, exist_ok=True)
backend = MujocoBackend()
renderer = mujoco.Renderer(backend.model, height=640, width=960)
records = {}
canvas = Image.new('RGB', (1920, 1440), 'white')
try:
    for idx, case in enumerate((4, 5, 6, 7)):
        backend.reset(case)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.96, 0, 0]
        camera.distance = 1.55
        camera.azimuth = 0
        camera.elevation = -89.9
        renderer.update_scene(backend.data, camera=camera)
        image = Image.fromarray(renderer.render())
        panel = Image.new('RGB', (960, 720), 'white')
        panel.paste(image, (0, 40))
        draw = ImageDraw.Draw(panel)
        draw.text((15, 12), f'CASE {case} | TRAINING BACKEND RESET | TOP VIEW | NOT REAL ROBOT', fill='black')
        task = backend.task
        draw.text((15, 686), f'front={task.targets[0]}   right={task.targets[1]}   left={task.targets[2]}   excluded={task.excluded}', fill='black')
        panel.save(out / f'case{case}_top.png')
        canvas.paste(panel, (idx % 2 * 960, idx // 2 * 720))
        obs = backend.get_observation()
        records[str(case)] = {'objects_world_m': {s: obs['object_positions'][i].tolist() for i,s in enumerate(SHAPE_ORDER)},
                             'box_reference_world_m': obs['box_pose'].tolist(),
                             'proxy_holes_world_m': obs['hole_poses'].tolist(),
                             'warning': 'Simulation world coordinates only. No robot-frame calibration. Proxy holes are NOT verified physical openings.'}
        renderer.update_scene(backend.data, camera='pilot_overview')
        Image.fromarray(renderer.render()).save(out / f'case{case}_overview.png')
    canvas.save(out / 'cases4_7_layout.png')
    (out / 'scene_coordinates.json').write_text(json.dumps(records, indent=2))
finally:
    renderer.close()
    backend.close()
print(out)
