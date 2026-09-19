#!/usr/bin/env python3
"""Read-only robot observation -> SAC action shadow bridge.

Safety invariant: no ROS publisher is created and no command topic is touched.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np

try:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
except ImportError as e: raise SystemExit("Run inside the SDK ROS container: %s" % e)

class Bridge:
    def __init__(self, model_path: str | None):
        self.model = None
        if model_path:
            from stable_baselines3 import SAC
            self.model = SAC.load(model_path, device="cpu")
        self.pose = None; self.joints = None
        self.node = rclpy.create_node("sac_sdk_shadow_bridge", start_parameter_services=False, enable_rosout=False)
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.node.create_subscription(PoseStamped, "/right_arm/end_pose", self.pose_cb, qos)
        self.node.create_subscription(JointState, "/right_arm/joint_states", self.joint_cb, qos)
    def pose_cb(self, msg): self.pose = msg
    def joint_cb(self, msg): self.joints = msg
    def sample(self, seconds):
        out=[]; end=time.monotonic()+seconds
        while time.monotonic()<end:
            rclpy.spin_once(self.node, timeout_sec=.05)
            if self.pose is not None:
                # Deliberately do not fabricate object/hole observations.
                out.append({"t":time.time(), "frame_id":self.pose.header.frame_id,
                            "position":[self.pose.pose.position.x,self.pose.pose.position.y,self.pose.pose.position.z],
                            "model_action":None, "execution":"blocked_missing_real_scene_observation"})
        return out
    def close(self): self.node.destroy_node(); rclpy.shutdown()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model"); p.add_argument("--seconds",type=float,default=10); p.add_argument("--output",default="sac_shadow.json"); a=p.parse_args()
    rclpy.init(args=[]); b=Bridge(a.model)
    rows=b.sample(a.seconds)
    report={"schema":"sac-sdk-shadow/v1","hardware_motion_enabled":False,"publishers_created":0,
            "robot_observation_topics":["/right_arm/end_pose","/right_arm/joint_states"],
            "model_loaded":b.model is not None,"samples":rows,
            "blocked_reasons":["no_real_object_hole_calibration","feedback_frame_not_validated_for_sdk_target","proxy_node_owns_pose_command_topic","simulation_to_real_validation_incomplete"]}
    Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    b.close(); print(json.dumps({k:report[k] for k in report if k!="samples"},ensure_ascii=False))
if __name__=="__main__": main()
