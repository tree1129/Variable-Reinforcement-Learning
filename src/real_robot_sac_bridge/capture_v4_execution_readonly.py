#!/usr/bin/env python3
"""Bounded ROS observer. No robot commands, services, actions, policy or reset.

JSONL is diagnostic evidence, NOT a v4_executed_trace_v1 dataset. Camera payloads
are intentionally excluded. Source, receipt and ROS clocks are kept separate;
DDS source timestamps are not a clock calibration. Works via python - --stdout.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

# Fixed allowlist: subscription to a command topic is not command execution.
TOPICS = {
    '/x2robot/action_chunk/arm1': 'trajectory_msgs/msg/JointTrajectory',
    '/x2robot/action_chunk/arm2': 'trajectory_msgs/msg/JointTrajectory',
    '/x2robot_action_dispatcher/chunk_ack': 'std_msgs/msg/Int64MultiArray',
    '/x2robot_action_dispatcher/executed_steps': 'std_msgs/msg/Int64',
    '/x2robot_action_dispatcher/reset_epoch_ack': 'std_msgs/msg/Int64',
    '/x2robot_action_dispatcher/capabilities': 'std_msgs/msg/String',
    '/left_arm_cartesian_controller/pose_cmd': 'geometry_msgs/msg/PoseStamped',
    '/right_arm_cartesian_controller/pose_cmd': 'geometry_msgs/msg/PoseStamped',
    '/left_gripper_controller/commands': 'std_msgs/msg/Float64MultiArray',
    '/right_gripper_controller/commands': 'std_msgs/msg/Float64MultiArray',
    '/left_arm/end_pose': 'geometry_msgs/msg/PoseStamped',
    '/right_arm/end_pose': 'geometry_msgs/msg/PoseStamped',
    '/left_gripper/joint_states': 'sensor_msgs/msg/JointState',
    '/right_gripper/joint_states': 'sensor_msgs/msg/JointState',
    '/application/left_arm_wrist_wrench': 'std_msgs/msg/Float32',
    '/application/right_arm_wrist_wrench': 'std_msgs/msg/Float32',
    '/application/error_codes': 'protocol/msg/ErrorCodes',
    '/application/safety_ctrl_locally_status': 'std_msgs/msg/Bool',
    '/x2robot_infer/inference_status': 'protocol/msg/InferenceStatus',
    '/camera_head_front/color/image_raw/compressed': 'sensor_msgs/msg/CompressedImage',
    '/camera_head_front/depth/image_raw': 'sensor_msgs/msg/Image',
    '/camera_head_front/color/camera_info': 'sensor_msgs/msg/CameraInfo',
    '/camera_head_front/depth/camera_info': 'sensor_msgs/msg/CameraInfo',
}
SCHEMA = 'v4_readonly_ros_events_v1'


def project_message(topic, msg, convert):
    """Bounded projections; never serialize images or unrelated user instructions."""
    if topic.endswith('/inference_status'):
        return {'infer_state': int(msg.infer_state), 'projection': 'state_only'}
    if topic.endswith('/image_raw') or topic.endswith('/image_raw/compressed'):
        data = {'header': convert(msg.header), 'payload_recorded': False,
                'payload_bytes': len(msg.data)}
        for field in ('height', 'width', 'encoding', 'step', 'is_bigendian', 'format'):
            if hasattr(msg, field):
                data[field] = getattr(msg, field)
        return data
    return convert(msg)


def info_fields(info):
    """Jazzy deployments return either a mapping or MessageInfo object.

    Missing transport metadata stays null; never substitute receipt timestamps.
    """
    def get(name):
        return info.get(name) if isinstance(info, Mapping) else getattr(info, name, None)
    fields = {}
    for source, dest in [('source_timestamp', 'dds_source_timestamp_ns'),
                         ('received_timestamp', 'dds_received_timestamp_ns')]:
        value = get(source)
        fields[dest] = value if type(value) is int and value >= 0 else None
    gid = get('publisher_gid')
    fields['publisher_gid'] = bytes(gid).hex() if gid is not None else None
    return fields


def capture(seconds, stream, max_events=20000, max_bytes=32 * 1024 * 1024):
    # Import only on real capture, so schema helpers can be tested without ROS.
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rosidl_runtime_py.utilities import get_message
    from rosidl_runtime_py.convert import message_to_ordereddict

    session = str(uuid.uuid4())
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        boot = 'unavailable'
    report = {'schema': SCHEMA, 'kind': 'capture_end', 'session_id': session,
              'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'source': 'real_robot_readonly', 'boot_id': boot,
              'hardware_authorized': False, 'trainable': False,
              'command_publishers_created': 0, 'service_clients_created': 0,
              'action_clients_created': 0, 'policy_loaded': False,
              'requested_seconds': seconds, 'counts': {t: 0 for t in TOPICS},
              'setup_errors': {}, 'callback_errors': {}, 'capture_complete': False,
              'camera_payloads_recorded': False,
              'clock_mapping_validated': False, 'stop_reason': 'duration',
              'qos': 'best_effort_volatile_depth_100; delivery_not_guaranteed',
              'limitations': ['Dispatcher acknowledgements/counters do not prove physical execution.',
                              'No V4 raw inference, checkpoint identity, calibrated object tracks or labels.',
                              'No feedback-frame conversion and no cross-clock timestamp substitution.']}
    digest = hashlib.sha256()
    count = 0
    used = 0
    stopped = False
    rclpy.init(args=[])
    node = rclpy.create_node('v4_execution_observer_' + session[:8],
                            enable_rosout=False, start_parameter_services=False)
    refs = []
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)

    def callback(topic, msg, info):
        nonlocal count, used, stopped
        if stopped:
            return
        try:
            row = {'schema': SCHEMA, 'kind': 'ros_message', 'session_id': session,
                   'seq': count, 'topic': topic, 'message_type': TOPICS[topic],
                   'receipt_monotonic_ns': time.monotonic_ns(),
                   'receipt_wall_ns': time.time_ns(), 'receipt_ros_ns': node.get_clock().now().nanoseconds,
                   **info_fields(info),
                   'data': project_message(topic, msg, message_to_ordereddict)}
            line = (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()
            if count >= max_events or used + len(line) > max_bytes:
                stopped = True
                report['stop_reason'] = 'resource_limit_incomplete'
                return
            stream.write(line.decode())
            digest.update(line)
            count += 1
            used += len(line)
            report['counts'][topic] += 1
        except Exception as exc:
            report['callback_errors'][topic] = type(exc).__name__ + ':' + str(exc)[:250]
            stopped = True
            report['stop_reason'] = 'callback_error_incomplete'

    try:
        for topic, cls in TOPICS.items():
            try:
                # Two-argument callback is required for rclpy MessageInfo delivery.
                def make_callback(name):
                    def on_message(msg, info):
                        callback(name, msg, info)
                    return on_message
                refs.append(node.create_subscription(get_message(cls), topic, make_callback(topic), qos))
            except Exception as exc:
                report['setup_errors'][topic] = str(exc)[:250]
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not stopped:
            rclpy.spin_once(node, timeout_sec=min(.05, max(0., deadline-time.monotonic())))
        report['publishers_at_end'] = {}
        for topic in TOPICS:
            report['publishers_at_end'][topic] = [
                {'node_name': p.node_name, 'node_namespace': p.node_namespace,
                 'topic_type': p.topic_type, 'endpoint_gid': bytes(p.endpoint_gid).hex()}
                for p in node.get_publishers_info_by_topic(topic)]
        report['capture_complete'] = not stopped and not report['setup_errors']
    except (Exception, KeyboardInterrupt) as exc:
        report['stop_reason'] = 'observer_error_incomplete'
        report['error'] = type(exc).__name__ + ':' + str(exc)[:300]
    finally:
        node.destroy_node()
        rclpy.shutdown()
    report.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  events=count, event_bytes=used, event_lines_sha256=digest.hexdigest())
    stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=10)
    dest = p.add_mutually_exclusive_group(required=True)
    dest.add_argument('--stdout', action='store_true')
    dest.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 30:
        p.error('--seconds must be finite and in [1,30]')
    if args.stdout:
        report = capture(args.seconds, sys.stdout)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / 'events.jsonl').open('x', encoding='utf-8') as stream:
            report = capture(args.seconds, stream)
        (args.output_dir / 'capture_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['capture_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
