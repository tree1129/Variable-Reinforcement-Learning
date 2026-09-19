#!/usr/bin/env python3
"""Read-only TF connectivity preflight for Case 4--7 head-camera calibration.

It subscribes to /tf, /tf_static and camera-info topics only.  It does not
create a ROS publisher, service/action client, controller request, or robot
motion command.  A TF connection is evidence of graph connectivity only; it is
never treated as a validated camera-to-base calibration.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage

COLOR_INFO_TOPIC = "/camera_head_front/color/camera_info"
DEPTH_INFO_TOPIC = "/camera_head_front/depth/camera_info"
TF_TOPICS = ("/tf", "/tf_static")


def stamp_s(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class TfPreflight(Node):
    def __init__(self) -> None:
        super().__init__("case4_7_tf_preflight_readonly", enable_rosout=False)
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        static_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.transforms: dict[str, dict[str, Any]] = {}
        self.color_info: CameraInfo | None = None
        self.depth_info: CameraInfo | None = None
        self.create_subscription(CameraInfo, COLOR_INFO_TOPIC, self._on_color_info, sensor_qos)
        self.create_subscription(CameraInfo, DEPTH_INFO_TOPIC, self._on_depth_info, sensor_qos)
        self.create_subscription(TFMessage, "/tf", lambda msg: self._on_tf(msg, is_static=False), sensor_qos)
        self.create_subscription(TFMessage, "/tf_static", lambda msg: self._on_tf(msg, is_static=True), static_qos)

    def _on_color_info(self, msg: CameraInfo) -> None:
        self.color_info = msg

    def _on_depth_info(self, msg: CameraInfo) -> None:
        self.depth_info = msg

    def _on_tf(self, msg: TFMessage, is_static: bool) -> None:
        for transform in msg.transforms:
            parent = transform.header.frame_id.lstrip("/")
            child = transform.child_frame_id.lstrip("/")
            if not parent or not child:
                continue
            self.transforms[child] = {
                "parent": parent,
                "child": child,
                "stamp_s": stamp_s(transform.header.stamp),
                "is_static": is_static,
            }

    @staticmethod
    def info(msg: CameraInfo | None) -> dict[str, Any] | None:
        if msg is None:
            return None
        return {
            "frame_id": msg.header.frame_id.lstrip("/"),
            "stamp_s": stamp_s(msg.header.stamp),
            "width": int(msg.width),
            "height": int(msg.height),
            "distortion_model": msg.distortion_model,
        }

    def graph(self) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {}
        for entry in self.transforms.values():
            parent, child = entry["parent"], entry["child"]
            graph.setdefault(parent, set()).add(child)
            graph.setdefault(child, set()).add(parent)
        return graph


def has_path(graph: dict[str, set[str]], source: str | None, target: str | None) -> bool | None:
    if not source or not target:
        return None
    if source not in graph or target not in graph:
        return False
    pending = deque([source])
    seen = {source}
    while pending:
        current = pending.popleft()
        if current == target:
            return True
        for nxt in graph.get(current, ()):
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only head-camera TF connectivity preflight")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--base-candidate", action="append", default=["base_link", "left_arm_neutral", "right_arm_neutral"])
    args = parser.parse_args()
    if not 1.0 <= args.seconds <= 60.0:
        raise SystemExit("--seconds must be within [1, 60]")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite: {args.output}")

    rclpy.init(args=[])
    node = TfPreflight()
    try:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        color = node.info(node.color_info)
        depth = node.info(node.depth_info)
        graph = node.graph()
        color_frame = color["frame_id"] if color else None
        depth_frame = depth["frame_id"] if depth else None
        candidate_results = [
            {
                "frame_id": candidate.lstrip("/"),
                "present_in_observed_tf": candidate.lstrip("/") in graph,
                "connected_to_head_color": has_path(graph, candidate.lstrip("/"), color_frame),
                "connected_to_head_depth": has_path(graph, candidate.lstrip("/"), depth_frame),
            }
            for candidate in args.base_candidate
        ]
        output = {
            "schema": "case4_7_tf_calibration_preflight/v1",
            "created_unix_s": time.time(),
            "read_only_safety": {
                "hardware_motion_enabled": False,
                "publishers_created": 0,
                "services_created": 0,
                "services_called": 0,
                "actions_sent": 0,
                "controllers_changed": 0,
                "subscriptions_only": [COLOR_INFO_TOPIC, DEPTH_INFO_TOPIC, *TF_TOPICS],
            },
            "camera": {
                "color": color,
                "depth": depth,
                "color_depth_tf_connected": has_path(graph, color_frame, depth_frame),
            },
            "tf_observation": {
                "frames_observed": len(graph),
                "edges_observed": len(node.transforms),
                "base_candidates": candidate_results,
            },
            "calibration": {
                "camera_to_base_transform_observed": any(x["connected_to_head_color"] for x in candidate_results),
                "camera_to_base_calibration_validated": False,
                "status": "BLOCKED_PENDING_VALIDATED_CAMERA_TO_BASE_CALIBRATION",
                "required_next": [
                    "identify the robot control/base frame accepted by the physical controller",
                    "perform a measured camera-to-base calibration with a fixed target",
                    "validate reprojection and 3D residuals on held-out poses",
                    "record transform ID, uncertainty, and the calibration fixture evidence",
                ],
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "frames_observed": len(graph), "color_depth_tf_connected": output["camera"]["color_depth_tf_connected"], "camera_to_base_transform_observed": output["calibration"]["camera_to_base_transform_observed"], "hardware_motion_enabled": False}, ensure_ascii=False))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
