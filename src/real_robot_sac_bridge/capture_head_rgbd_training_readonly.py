#!/usr/bin/env python3
"""Capture reviewable head RGB-D/domain-adaptation data without robot control.

Safety invariant: this node uses ROS subscriptions only.  It creates no
publisher, service/action client, controller switch, or policy executor.
The resulting manifest deliberately marks all frames UNREVIEWED, so they cannot
be consumed by a supervised object/hole-pose training job until a reviewer adds
semantic labels and validates calibration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState


TOPICS = {
    "color": "/camera_head_front/color/image_raw",
    "depth": "/camera_head_front/depth/image_raw",
    "color_info": "/camera_head_front/color/camera_info",
    "depth_info": "/camera_head_front/depth/camera_info",
    "left_pose": "/left_arm/end_pose",
    "right_pose": "/right_arm/end_pose",
    "left_gripper": "/left_gripper/joint_states",
    "right_gripper": "/right_gripper/joint_states",
}


def timestamp_s(message: Any) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def image_array(message: Image) -> np.ndarray:
    if message.encoding == "rgb8":
        return np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)[:, : message.width * 3].reshape(message.height, message.width, 3).copy()
    if message.encoding == "bgr8":
        raw = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)[:, : message.width * 3].reshape(message.height, message.width, 3)
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    if message.encoding == "16UC1":
        raw = np.frombuffer(message.data, np.uint16).reshape(message.height, message.step // 2)[:, : message.width]
        return raw.copy()
    raise ValueError(f"Unsupported image encoding: {message.encoding}")


class Capture(Node):
    def __init__(self) -> None:
        super().__init__("case4_7_head_rgbd_training_capture", enable_rosout=False)
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        # Keep a short history so RGB and depth are matched by their source
        # timestamps, not merely whichever callbacks happened to arrive last.
        self.colors: list[tuple[float, str, np.ndarray]] = []
        self.depths: list[tuple[float, str, np.ndarray]] = []
        self.last_emitted_depth_stamp: float | None = None
        self.color_info: CameraInfo | None = None
        self.depth_info: CameraInfo | None = None
        self.left_pose: PoseStamped | None = None
        self.right_pose: PoseStamped | None = None
        self.left_gripper: JointState | None = None
        self.right_gripper: JointState | None = None
        self.create_subscription(Image, TOPICS["color"], self.color_cb, qos)
        self.create_subscription(Image, TOPICS["depth"], self.depth_cb, qos)
        self.create_subscription(CameraInfo, TOPICS["color_info"], self.color_info_cb, qos)
        self.create_subscription(CameraInfo, TOPICS["depth_info"], self.depth_info_cb, qos)
        self.create_subscription(PoseStamped, TOPICS["left_pose"], lambda m: setattr(self, "left_pose", m), qos)
        self.create_subscription(PoseStamped, TOPICS["right_pose"], lambda m: setattr(self, "right_pose", m), qos)
        self.create_subscription(JointState, TOPICS["left_gripper"], lambda m: setattr(self, "left_gripper", m), qos)
        self.create_subscription(JointState, TOPICS["right_gripper"], lambda m: setattr(self, "right_gripper", m), qos)

    def color_cb(self, message: Image) -> None:
        self.colors.append((timestamp_s(message), message.header.frame_id, image_array(message)))
        self.colors = self.colors[-90:]

    def depth_cb(self, message: Image) -> None:
        self.depths.append((timestamp_s(message), message.header.frame_id, image_array(message)))
        self.depths = self.depths[-30:]

    def color_info_cb(self, message: CameraInfo) -> None:
        self.color_info = message

    def depth_info_cb(self, message: CameraInfo) -> None:
        self.depth_info = message

    @staticmethod
    def pose(msg: PoseStamped | None) -> dict[str, Any] | None:
        if msg is None:
            return None
        p, q = msg.pose.position, msg.pose.orientation
        return {"frame_id": msg.header.frame_id, "stamp_s": timestamp_s(msg), "position_m": [p.x, p.y, p.z], "orientation_xyzw": [q.x, q.y, q.z, q.w]}

    @staticmethod
    def gripper(msg: JointState | None) -> dict[str, Any] | None:
        if msg is None:
            return None
        return {"frame_id": msg.header.frame_id, "stamp_s": timestamp_s(msg), "names": list(msg.name), "position": [float(v) for v in msg.position]}

    def take_sample(self, out: Path, index: int, max_pair_delta_s: float) -> dict[str, Any] | None:
        if not self.colors or not self.depths:
            return None
        # Use the newest depth frame that has not previously been emitted and
        # choose its closest color frame. Depth runs at 10 Hz and color at 30
        # Hz on the robot, so this preserves measurement time rather than
        # dropping a valid pair due to callback order.
        candidates = [d for d in self.depths if self.last_emitted_depth_stamp is None or d[0] > self.last_emitted_depth_stamp]
        if not candidates:
            return None
        depth_stamp, depth_frame, depth = candidates[-1]
        color_stamp, color_frame, color = min(self.colors, key=lambda c: abs(c[0] - depth_stamp))
        pair_delta_s = abs(color_stamp - depth_stamp)
        if pair_delta_s > max_pair_delta_s:
            return None
        if self.color_info is None or self.depth_info is None:
            return None
        stem = f"frame_{index:05d}"
        color_path = out / "color" / f"{stem}.jpg"
        depth_path = out / "depth" / f"{stem}.npy"
        if not cv2.imwrite(str(color_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Failed to write {color_path}")
        np.save(depth_path, depth)
        self.last_emitted_depth_stamp = depth_stamp
        return {
            "frame_id": index,
            "capture_time_unix_s": time.time(),
            "annotation_status": "UNREVIEWED_NOT_TRAINABLE",
            "case_id": None,
            "color": {"path": str(color_path.relative_to(out)), "stamp_s": color_stamp, "frame_id": color_frame, "shape": list(color.shape), "encoding": "rgb8"},
            "depth": {"path": str(depth_path.relative_to(out)), "stamp_s": depth_stamp, "frame_id": depth_frame, "shape": list(depth.shape), "encoding": "16UC1", "scale_to_m": 0.001},
            "rgb_depth_delta_ms": pair_delta_s * 1000.0,
            "robot_state": {"left_pose": self.pose(self.left_pose), "right_pose": self.pose(self.right_pose), "left_gripper": self.gripper(self.left_gripper), "right_gripper": self.gripper(self.right_gripper)},
            "objects": [],
            "holes": [],
            "calibration_status": "UNVERIFIED_CAMERA_TO_BASE",
        }


def intrinsics(message: CameraInfo | None) -> dict[str, Any] | None:
    if message is None:
        return None
    return {"frame_id": message.header.frame_id, "width": int(message.width), "height": int(message.height), "k": [float(v) for v in message.k], "d": [float(v) for v in message.d], "distortion_model": message.distortion_model}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only head RGB-D capture for later reviewed training labels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--sample-hz", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--max-rgb-depth-delta-ms", type=float, default=35.0)
    args = parser.parse_args()
    if not (1.0 <= args.seconds <= 300.0 and 0.1 <= args.sample_hz <= 10.0 and 1 <= args.max_samples <= 1000):
        raise SystemExit("invalid capture bounds")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "color").mkdir()
    (args.output / "depth").mkdir()

    rclpy.init(args=[])
    node = Capture()
    records: list[dict[str, Any]] = []
    rejected_pairs = 0
    try:
        deadline = time.monotonic() + args.seconds
        next_capture = time.monotonic()
        while time.monotonic() < deadline and len(records) < args.max_samples:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if now < next_capture:
                continue
            next_capture = now + 1.0 / args.sample_hz
            sample = node.take_sample(args.output, len(records), args.max_rgb_depth_delta_ms / 1000.0)
            if sample is None:
                rejected_pairs += 1
            else:
                records.append(sample)
        manifest = {
            "schema": "case4_7_real_head_rgbd_training_capture/v1",
            "created_unix_s": time.time(),
            "hardware_motion_enabled": False,
            "publishers_created": 0,
            "services_called": 0,
            "actions_sent": 0,
            "controllers_changed": 0,
            "supervised_training_eligible": False,
            "reason_not_trainable": ["semantic objects/holes are UNREVIEWED", "camera_to_base calibration is UNVERIFIED"],
            "capture": {"seconds_requested": args.seconds, "sample_hz": args.sample_hz, "max_rgb_depth_delta_ms": args.max_rgb_depth_delta_ms, "captured": len(records), "rejected_or_unready": rejected_pairs},
            "topics": TOPICS,
            "color_intrinsics": intrinsics(node.color_info),
            "depth_intrinsics": intrinsics(node.depth_info),
            "frames": records,
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / "ANNOTATION_REQUIRED.md").write_text(
            "# Annotation required before training\n\n"
            "Every frame is intentionally UNREVIEWED_NOT_TRAINABLE. Add reviewed object and hole instance labels, Case 4-7 ID, and validate camera-to-base calibration before using it for supervised training or robot motion.\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(args.output), "captured": len(records), "rejected_or_unready": rejected_pairs, "hardware_motion_enabled": False, "supervised_training_eligible": False}, ensure_ascii=False))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
