#!/usr/bin/env python3
"""Read-only preflight probe for the Case 4--7 SAC policy.

This process deliberately creates *only subscriptions*.  It has no publisher,
service client, action client, controller-switch call, or hardware-enable path.
It is safe to run alongside the existing X2Robot stack and writes a bounded
JSON audit describing whether the robot's sensor streams are fresh enough for
subsequent calibration work.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState


TOPICS = {
    "left_pose": "/left_arm/end_pose",
    "right_pose": "/right_arm/end_pose",
    "left_joints": "/left_arm/joint_states",
    "right_joints": "/right_arm/joint_states",
    "left_gripper": "/left_gripper/joint_states",
    "right_gripper": "/right_gripper/joint_states",
    "left_wrench": "/left_arm/wrench_ext_world",
    "right_wrench": "/right_arm/wrench_ext_world",
    "color_info": "/camera_head_front/color/camera_info",
    "depth_info": "/camera_head_front/depth/camera_info",
    "color_image": "/camera_head_front/color/image_raw/compressed",
    "depth_image": "/camera_head_front/depth/image_raw/compressedDepth",
}


def _stamp_seconds(msg: Any) -> float:
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


class ReadinessProbe(Node):
    def __init__(self) -> None:
        super().__init__("case4_7_sac_readiness_probe", enable_rosout=False)
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.last: dict[str, dict[str, Any]] = {}
        self.counts: dict[str, int] = {name: 0 for name in TOPICS}

        self.create_subscription(PoseStamped, TOPICS["left_pose"], lambda m: self._pose("left_pose", m), qos)
        self.create_subscription(PoseStamped, TOPICS["right_pose"], lambda m: self._pose("right_pose", m), qos)
        self.create_subscription(JointState, TOPICS["left_joints"], lambda m: self._joint("left_joints", m), qos)
        self.create_subscription(JointState, TOPICS["right_joints"], lambda m: self._joint("right_joints", m), qos)
        self.create_subscription(JointState, TOPICS["left_gripper"], lambda m: self._joint("left_gripper", m), qos)
        self.create_subscription(JointState, TOPICS["right_gripper"], lambda m: self._joint("right_gripper", m), qos)
        self.create_subscription(WrenchStamped, TOPICS["left_wrench"], lambda m: self._wrench("left_wrench", m), qos)
        self.create_subscription(WrenchStamped, TOPICS["right_wrench"], lambda m: self._wrench("right_wrench", m), qos)
        self.create_subscription(CameraInfo, TOPICS["color_info"], lambda m: self._camera_info("color_info", m), qos)
        self.create_subscription(CameraInfo, TOPICS["depth_info"], lambda m: self._camera_info("depth_info", m), qos)
        self.create_subscription(CompressedImage, TOPICS["color_image"], lambda m: self._image("color_image", m), qos)
        self.create_subscription(CompressedImage, TOPICS["depth_image"], lambda m: self._image("depth_image", m), qos)

    def _record(self, name: str, payload: dict[str, Any]) -> None:
        self.counts[name] += 1
        payload["received_at_monotonic"] = time.monotonic()
        self.last[name] = payload

    def _pose(self, name: str, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        self._record(name, {
            "frame_id": msg.header.frame_id,
            "stamp_s": _stamp_seconds(msg),
            "position_m": [p.x, p.y, p.z],
            "orientation_xyzw": [q.x, q.y, q.z, q.w],
            "finite": _finite(values),
        })

    def _joint(self, name: str, msg: JointState) -> None:
        values = [float(v) for v in msg.position]
        self._record(name, {
            "frame_id": msg.header.frame_id,
            "stamp_s": _stamp_seconds(msg),
            "names": list(msg.name),
            "position": values,
            "finite": _finite(values),
        })

    def _wrench(self, name: str, msg: WrenchStamped) -> None:
        f, t = msg.wrench.force, msg.wrench.torque
        values = [f.x, f.y, f.z, t.x, t.y, t.z]
        self._record(name, {
            "frame_id": msg.header.frame_id,
            "stamp_s": _stamp_seconds(msg),
            "force_n": [f.x, f.y, f.z],
            "torque_nm": [t.x, t.y, t.z],
            "finite": _finite(values),
        })

    def _camera_info(self, name: str, msg: CameraInfo) -> None:
        self._record(name, {
            "frame_id": msg.header.frame_id,
            "stamp_s": _stamp_seconds(msg),
            "resolution": [int(msg.width), int(msg.height)],
            "intrinsics_k": [float(v) for v in msg.k],
            "distortion_model": msg.distortion_model,
        })

    def _image(self, name: str, msg: CompressedImage) -> None:
        self._record(name, {
            "frame_id": msg.header.frame_id,
            "stamp_s": _stamp_seconds(msg),
            "format": msg.format,
            "payload_bytes": len(msg.data),
        })

    def report(self, duration_s: float) -> dict[str, Any]:
        now = time.monotonic()
        streams: dict[str, Any] = {}
        for name in TOPICS:
            latest = self.last.get(name)
            age = None if latest is None else now - latest["received_at_monotonic"]
            streams[name] = {
                "topic": TOPICS[name],
                "messages_received": self.counts[name],
                "latest_age_s": age,
                "fresh": bool(latest is not None and age is not None and age <= 1.0),
                "sample": None if latest is None else {k: v for k, v in latest.items() if k != "received_at_monotonic"},
            }
        robot_state_ready = all(streams[name]["fresh"] for name in (
            "left_pose", "right_pose", "left_joints", "right_joints", "left_gripper", "right_gripper",
        ))
        camera_ready = all(streams[name]["fresh"] for name in ("color_info", "depth_info", "color_image", "depth_image"))
        return {
            "schema": "case4_7_sac_real_robot_readiness/v1",
            "created_unix_s": time.time(),
            "duration_s": duration_s,
            "hardware_motion_enabled": False,
            "publishers_created": 0,
            "services_called": 0,
            "actions_sent": 0,
            "controllers_changed": 0,
            "model_loaded": False,
            "model_action_generated": False,
            "policy_observation_dimension": 88,
            "robot_state_ready": robot_state_ready,
            "camera_stream_ready": camera_ready,
            "real_scene_observation_ready": False,
            "deployment_state": "shadow_preflight_only",
            "streams": streams,
            "required_before_motion": [
                "validated camera-to-base and fixture calibration",
                "object and hole 6D-pose estimator matching the 88D policy observation contract",
                "robot-specific normalized-action to controller conversion",
                "workspace, joint, velocity, acceleration, force/torque and desk-collision safety wrapper",
                "watchdog, E-stop operator and explicit supervised first-motion approval",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Case 4--7 SAC real-robot readiness probe")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--output", default="/tmp/case4_7_sac_readiness.json")
    args = parser.parse_args()
    if not 1.0 <= args.seconds <= 120.0:
        raise SystemExit("--seconds must be between 1 and 120")

    rclpy.init(args=[])
    node = ReadinessProbe()
    try:
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        result = node.report(args.seconds)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "output": str(output),
            "hardware_motion_enabled": result["hardware_motion_enabled"],
            "publishers_created": result["publishers_created"],
            "robot_state_ready": result["robot_state_ready"],
            "camera_stream_ready": result["camera_stream_ready"],
            "deployment_state": result["deployment_state"],
        }, ensure_ascii=False))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
