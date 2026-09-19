"""Synthetic ABI fixtures, not measured V4 trajectories or label validation."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tree_reinforcement_learning.readonly_intake import SCHEMA, RAW_NAMES, ACK, CHUNKS, COUNTER, inspect_capture, parse_ack, parse_chunk

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('readonly_capture', ROOT / 'real_robot_sac_bridge/capture_v4_execution_readonly.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def chunk(arm='arm1', **kwargs):
    data = {'header': {'frame_id': arm+'|x=10|c=5|r=100|m=10|n=50',
                       'stamp': {'sec': 1, 'nanosec': 12}},
            'joint_names': RAW_NAMES.copy(), 'points': [{'positions': [0.] * 7}]}
    data.update(kwargs)
    return data


def events():
    pairs = [(CHUNKS[0], chunk()), (CHUNKS[1], chunk('arm2')),
             (ACK, {'data': [10, 5, 3, 50, 100]}),
             (COUNTER, {'data': 100}), (COUNTER, {'data': 101}),
             (COUNTER, {'data': 101})]  # Last event is a heartbeat, not another step.
    return [{'schema': SCHEMA, 'kind': 'ros_message', 'session_id': 'synthetic-test-only',
             'seq': i, 'receipt_monotonic_ns': 100+i, 'topic': topic, 'publisher_gid': 'aa',
             'data': data} for i, (topic, data) in enumerate(pairs)]


def serialize(rows):
    lines = [json.dumps(row).encode()+b'\n' for row in rows]
    data = b''.join(lines)
    counts = {'/never_published': 0}
    for row in rows:
        counts[row['topic']] = counts.get(row['topic'], 0)+1
    end = {'schema': SCHEMA, 'kind': 'capture_end', 'session_id': 'synthetic-test-only',
           'source': 'real_robot_readonly', 'synthetic_fixture': True,
           'counts': counts, 'events': len(rows), 'event_bytes': len(data),
           'event_lines_sha256': hashlib.sha256(data).hexdigest(),
           'hardware_authorized': False, 'trainable': False, 'policy_loaded': False,
           'command_publishers_created': 0, 'service_clients_created': 0,
           'action_clients_created': 0, 'capture_complete': True,
           'setup_errors': {}, 'callback_errors': {}}
    return data, end


class ReadonlyIntakeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'events.jsonl'

    def run_report(self, rows=None, mutate_end=None, raw=None):
        data, end = serialize(events() if rows is None else rows)
        if mutate_end:
            mutate_end(end)
        self.path.write_bytes(raw if raw is not None else data+json.dumps(end).encode()+b'\n')
        return inspect_capture(self.path)

    def test_readonly_pair_is_never_an_executed_transition(self):
        r = self.run_report()
        self.assertTrue(r['capture_integrity_passed'], r['integrity_errors'])
        self.assertFalse(r['trainable'])
        self.assertFalse(r['hardware_authorized'])
        self.assertEqual(r['physical_execution_verified_transitions'], 0)
        self.assertTrue(r['chunk_windows'][0]['paired_arm_keyframes_observed'])
        self.assertFalse(r['chunk_windows'][0]['executed'])
        self.assertEqual(r['dispatch_counter']['observed_increases'], 1)

    def test_ack_versions_do_not_claim_execution(self):
        for v in ([10, 5], [10, 5, 3, 50], [10, 5, 3, 50, 100]):
            self.assertFalse(parse_ack(v)['physical_execution_verified'])
        self.assertNotIn('load_publish_counter', parse_ack([10, 5]))

    def test_invalid_ack_rejected(self):
        for v in (None, [], [1], [1, 2, 3], [1, 2, 4, 4], [True, 2], [-1, 2], [1., 2], [1, 2, 3, 4, -1]):
            with self.subTest(v=v), self.assertRaises(ValueError): parse_ack(v)

    def test_no_action_dimension_or_rotation_inference(self):
        for change in ({'joint_names': ['x']*8}, {'points': [{'positions': [0.]*20}]},
                       {'points': [{'positions': ['0']*7}]}, {'points': [{'positions': [float('nan')]*7}]}):
            with self.subTest(change=change), self.assertRaises(ValueError): parse_chunk(chunk(**change))

    def test_chunk_metadata_identity_must_exist_and_be_unique(self):
        for meta in ('arm1', 'arm1|x=1|c=2|c=3', 'arm1|x=-1|c=2', 'right|x=1|c=2'):
            d = chunk(); d['header']['frame_id'] = meta
            with self.assertRaises(ValueError): parse_chunk(d)

    def test_one_arm_or_duplicate_arm_cannot_pair(self):
        for idx in (0, 1):
            rows = events(); rows[idx]['topic'] = '/unrelated'
            self.assertFalse(self.run_report(rows)['chunk_windows'][0]['paired_arm_keyframes_observed'])
        rows = events(); rows[1]['topic'] = CHUNKS[0]; rows[1]['data'] = chunk()
        self.assertFalse(self.run_report(rows)['chunk_windows'][0]['paired_arm_keyframes_observed'])

    def test_mismatched_stamp_or_metadata_cannot_pair(self):
        for key in ('stamp', 'metadata'):
            rows = events()
            if key == 'stamp': rows[1]['data']['header']['stamp']['nanosec'] += 1
            else: rows[1]['data']['header']['frame_id'] += '|t=100'
            self.assertFalse(self.run_report(rows)['chunk_windows'][0]['paired_arm_keyframes_observed'])

    def test_chunk_epoch_must_match_ack(self):
        rows = events(); rows[2]['data']['data'][0] = 11
        self.assertFalse(self.run_report(rows)['chunk_windows'][0]['paired_arm_keyframes_observed'])

    def test_counter_reset_is_reported_not_a_success(self):
        rows = events(); rows[-1]['data']['data'] = 0
        r = self.run_report(rows)
        self.assertTrue(any('regression' in x for x in r['warnings']))
        self.assertFalse(r['trainable'])

    def test_multiple_publishers_are_reported(self):
        rows = events(); rows[-1]['publisher_gid'] = 'bb'
        self.assertTrue(any('multiple_publisher' in x for x in self.run_report(rows)['warnings']))

    def test_hash_end_count_and_completeness(self):
        for field, value in [('event_lines_sha256', '0'*64), ('events', 20), ('counts', {}),
                             ('capture_complete', False), ('session_id', 'other'),
                             ('command_publishers_created', 1), ('trainable', True)]:
            with self.subTest(field=field):
                self.assertFalse(self.run_report(mutate_end=lambda m: m.update({field: value}))['capture_integrity_passed'])

    def test_sequence_clock_and_session_fail_closed(self):
        for key, value in [('seq', 0), ('receipt_monotonic_ns', 1), ('session_id', 'other')]:
            rows = events(); rows[1][key] = value
            self.assertFalse(self.run_report(rows)['capture_integrity_passed'])

    def test_truncation_extra_data_and_nonfinite(self):
        data, end = serialize(events())
        for raw in (data, data+json.dumps(end).encode()+b'\n{}\n', b'{"bad":NaN}\n', b'{"bad":1e999}\n', b'[]\n'):
            self.assertFalse(self.run_report(raw=raw)['capture_integrity_passed'])

    def test_dict_and_object_message_info_supported(self):
        d = {'source_timestamp': 1, 'received_timestamp': 2, 'publisher_gid': [1, 2]}
        for value in (d, SimpleNamespace(**d)):
            self.assertEqual(recorder.info_fields(value), {'dds_source_timestamp_ns': 1,
                             'dds_received_timestamp_ns': 2, 'publisher_gid': '0102'})
        self.assertEqual(recorder.info_fields({}), {'dds_source_timestamp_ns': None,
                         'dds_received_timestamp_ns': None, 'publisher_gid': None})

    def test_image_projection_does_not_serialize_payload(self):
        msg = SimpleNamespace(header=SimpleNamespace(stamp=1), data=b'large image', height=2)
        def convert(v):
            self.assertIs(v, msg.header)
            return {'stamp': 1}
        out = recorder.project_message('/camera_head_front/depth/image_raw', msg, convert)
        self.assertFalse(out['payload_recorded'])
        self.assertNotIn('data', out)

    def test_recorder_has_no_control_or_policy_api(self):
        import ast
        tree = ast.parse((ROOT/'real_robot_sac_bridge/capture_v4_execution_readonly.py').read_text())
        methods = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(methods & {'create_publisher', 'publish', 'create_client', 'call_async', 'send_goal_async', 'predict', 'infer'})
