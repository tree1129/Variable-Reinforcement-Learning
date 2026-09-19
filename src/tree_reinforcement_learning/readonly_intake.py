"""Offline inspection of the deployed ROS dispatch ABI, never RL attribution.

The C++ chunk ACK is [epoch, chunk_id, start, end, load_publish_counter].
Its counter increments after ROS command publication, not controller feedback.
Raw Euler keyframes are interpolated/blended; they are NOT V4's raw 20D output.
Only diagnostic associations are made. No motion interface or NPZ export exists.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'v4_readonly_ros_events_v1'
ACK = '/x2robot_action_dispatcher/chunk_ack'
COUNTER = '/x2robot_action_dispatcher/executed_steps'
CHUNKS = ('/x2robot/action_chunk/arm1', '/x2robot/action_chunk/arm2')
RAW_NAMES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']


def _integer(x, name):
    if type(x) is not int or x < 0:
        raise ValueError('invalid_nonnegative_integer:' + name)
    return x


def _json(line):
    def bad(value):
        raise ValueError('nonfinite_json:' + value)
    result = json.loads(line, parse_constant=bad)
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite_json_number')
        if isinstance(value, dict):
            for v in value.values(): finite(v)
        elif isinstance(value, list):
            for v in value: finite(v)
    finite(result)
    if not isinstance(result, dict):
        raise ValueError('record_not_object')
    return result


def parse_ack(data):
    if not isinstance(data, list) or len(data) not in (2, 4, 5):
        raise ValueError('unsupported_chunk_ack_shape')
    values = [_integer(v, 'ack') for v in data]
    out = {'epoch': values[0], 'chunk_id': values[1],
           'semantics': 'queue_acceptance_not_physical_execution',
           'physical_execution_verified': False}
    if len(values) >= 4:
        if values[3] <= values[2]:
            raise ValueError('invalid_ack_window')
        out.update(start=values[2], end_exclusive=values[3])
    if len(values) == 5:
        out['load_publish_counter'] = values[4]
    return out


def parse_chunk(data):
    if not isinstance(data, dict) or data.get('joint_names') != RAW_NAMES:
        raise ValueError('not_supported_raw_euler_chunk')
    header = data.get('header', {})
    tokens = header.get('frame_id', '').split('|')
    if tokens[0] not in ('arm1', 'arm2'):
        raise ValueError('invalid_arm_metadata')
    meta = {}
    for token in tokens[1:]:
        if '=' not in token:
            raise ValueError('invalid_chunk_metadata')
        k, v = token.split('=', 1)
        if k in meta:
            raise ValueError('duplicate_chunk_metadata:' + k)
        meta[k] = v
    for key in ('x', 'c'):
        if not re.fullmatch(r'[0-9]+', meta.get(key, '')):
            raise ValueError('missing_or_invalid_chunk_identity:' + key)
    stamp = header.get('stamp', {})
    sec, ns = _integer(stamp.get('sec'), 'sec'), _integer(stamp.get('nanosec'), 'nanosec')
    if ns >= 10**9:
        raise ValueError('invalid_nanosec')
    points = data.get('points')
    if not isinstance(points, list) or not points:
        raise ValueError('empty_chunk')
    for p in points:
        pos = p.get('positions') if isinstance(p, dict) else None
        if not isinstance(pos, list) or len(pos) != 7 or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in pos):
            raise ValueError('invalid_euler_keyframe')
    return {'arm': tokens[0], 'epoch': int(meta['x']), 'chunk_id': int(meta['c']),
            'header_stamp_ns': sec * 10**9 + ns, 'keyframes': len(points),
            'metadata': meta, 'semantics': 'client_euler_keyframes_not_raw_v4_20d'}


def inspect_capture(path):
    path = Path(path)
    report = {'schema': 'v4_readonly_intake_report_v1', 'capture': str(path.resolve()),
              'source': 'unknown', 'capture_integrity_passed': False,
              'trainable': False, 'hardware_authorized': False,
              'physical_execution_verified_transitions': 0, 'verified_grasp_labels': 0,
              'integrity_errors': [], 'warnings': [], 'topic_counts': {},
              'chunk_windows': [], 'missing_for_real_v4_training': [
                  'frozen_v4_checkpoint_identity_and_raw_inference_with_observation_ids',
                  'decoded_base_and_actual_command_lineage_including_interpolation_and_blending',
                  'controller_applied_command_id_with_source_stamped_feedback_not_dispatch_counter',
                  'validated_action_arm_frame_gripper_and_cross_clock_contracts',
                  'calibrated_object_identity_support_clearance_and_rigid_tracking',
                  'independent_automatic_labeler_validation_evidence',
                  'episode_case_scene_reset_boundaries_and_disjoint_data_splits'],
              'limitations': ['Capture integrity is not physical calibration or controller verification.',
                              'No raw-V4 observations or rewards are reconstructed from ROS topics.']}
    counts, frames, publishers = Counter(), defaultdict(set), defaultdict(set)
    chunks, acks = defaultdict(list), []
    counters, states = [], set()
    seq, last_time, session = 0, None, None
    end = None
    h = hashlib.sha256()
    total_bytes = 0
    try:
        with path.open('rb') as f:
            for lineno, line in enumerate(f, 1):
                if lineno > 40001 or len(line) > 4 * 1024 * 1024:
                    raise ValueError('input_resource_bound_exceeded')
                row = _json(line)
                if row.get('schema') != SCHEMA:
                    raise ValueError('unsupported_schema')
                if end is not None:
                    raise ValueError('data_after_capture_end')
                if row.get('kind') == 'capture_end':
                    end = row
                    continue
                if row.get('kind') != 'ros_message':
                    raise ValueError('unknown_record_kind')
                if row.get('seq') != seq or type(row.get('seq')) is not int:
                    raise ValueError('sequence_gap_duplicate_or_reorder')
                sid = row.get('session_id')
                if not isinstance(sid, str) or not sid:
                    raise ValueError('missing_session')
                session = session or sid
                if sid != session:
                    raise ValueError('mixed_capture_sessions')
                t = _integer(row.get('receipt_monotonic_ns'), 'receipt_monotonic_ns')
                if last_time is not None and t < last_time:
                    raise ValueError('nonmonotonic_receipt_clock')
                last_time = t
                topic = row.get('topic')
                if not isinstance(topic, str) or not topic.startswith('/'):
                    raise ValueError('invalid_topic')
                if not isinstance(row.get('data'), dict):
                    raise ValueError('invalid_message_data')
                h.update(line); total_bytes += len(line); counts[topic] += 1; seq += 1
                gid = row.get('publisher_gid')
                if isinstance(gid, str) and gid:
                    publishers[topic].add(gid)
                else:
                    report['warnings'].append('missing_publisher_gid:' + topic)
                data = row['data']
                if isinstance(data.get('header'), dict):
                    frames[topic].add(str(data['header'].get('frame_id', '')))
                try:
                    if topic == ACK:
                        ack = parse_ack(data.get('data'))
                        ack['event_seq'] = row['seq']; acks.append(ack)
                    elif topic in CHUNKS:
                        chunk = parse_chunk(data)
                        if chunk['arm'] != topic.rsplit('/', 1)[1]:
                            raise ValueError('topic_arm_metadata_mismatch')
                        chunk['event_seq'] = row['seq']
                        chunks[(chunk['epoch'], chunk['chunk_id'])].append(chunk)
                    elif topic == COUNTER:
                        value = _integer(data.get('data'), 'dispatch_counter')
                        counters.append(value)
                    elif topic == '/x2robot_infer/inference_status':
                        states.add(_integer(data.get('infer_state'), 'infer_state'))
                except (ValueError, TypeError, AttributeError) as exc:
                    report['warnings'].append('unusable_event:' + str(row['seq']) + ':' + str(exc))
        if end is None:
            raise ValueError('missing_capture_end_truncated_stream')
        if end.get('source') != 'real_robot_readonly':
            raise ValueError('not_real_robot_readonly_source')
        report['source'] = end['source']
        if (session is not None and session != end.get('session_id')) or not end.get('session_id'):
            raise ValueError('end_session_mismatch')
        if end.get('events') != seq or end.get('event_bytes') != total_bytes or end.get('event_lines_sha256') != h.hexdigest():
            raise ValueError('event_digest_count_or_length_mismatch')
        declared = end.get('counts')
        if not isinstance(declared, dict) or any(type(v) is not int or v < 0 for v in declared.values()) or {k: v for k, v in declared.items() if v} != dict(counts):
            raise ValueError('topic_counts_mismatch')
        for key in ('command_publishers_created', 'service_clients_created', 'action_clients_created'):
            if type(end.get(key)) is not int or end[key] != 0:
                raise ValueError('not_declared_readonly:' + key)
        for key in ('hardware_authorized', 'trainable', 'policy_loaded'):
            if end.get(key) is not False:
                raise ValueError('invalid_readonly_declaration:' + key)
        if end.get('capture_complete') is not True or end.get('setup_errors') or end.get('callback_errors'):
            raise ValueError('capture_incomplete_or_observer_error')
        report['capture_integrity_passed'] = True
        report['capture_started_at'] = end.get('started_at')
        report['capture_ended_at'] = end.get('ended_at')
        report['publishers_at_end'] = end.get('publishers_at_end', {})
    except (OSError, ValueError, TypeError, AttributeError, OverflowError) as exc:
        report['integrity_errors'].append(str(exc))
    ack_counts = Counter((a['epoch'], a['chunk_id']) for a in acks)
    for ack in acks:
        key = (ack['epoch'], ack['chunk_id'])
        candidates = chunks[key]
        arms = Counter(c['arm'] for c in candidates)
        paired = (arms == Counter({'arm1': 1, 'arm2': 1})
                  and len({c['header_stamp_ns'] for c in candidates}) == 1
                  and len({c['keyframes'] for c in candidates}) == 1
                  and len({json.dumps(c['metadata'], sort_keys=True) for c in candidates}) == 1)
        report['chunk_windows'].append({**ack, 'paired_arm_keyframes_observed': paired,
                                       'duplicate_ack_identity': ack_counts[key] > 1,
                                       'chunk_event_seqs': [c['event_seq'] for c in candidates],
                                       'raw_v4_identity_verified': False,
                                       'executed': False,
                                       'attribution': 'diagnostic_only_not_an_rl_transition'})
    if any(b < a for a, b in zip(counters, counters[1:])):
        report['warnings'].append('dispatcher_counter_regression_reset_or_restart_do_not_join_across_it')
    for topic, gids in publishers.items():
        if len(gids) > 1:
            report['warnings'].append('multiple_publisher_gids_do_not_assume_one_controller:' + topic)
    report.update(topic_counts=dict(counts), feedback_frames={k: sorted(v) for k, v in frames.items()},
                  inference_state_values=sorted(states),
                  dispatch_counter={'samples': len(counters),
                                    'first': counters[0] if counters else None,
                                    'last': counters[-1] if counters else None,
                                    'observed_increases': sum(b > a for a, b in zip(counters, counters[1:])),
                                    'semantics': 'ROS_publish_count_not_physical_execution'})
    report['warnings'] = sorted(set(report['warnings']))
    return report
