#!/usr/bin/env python3
"""Label externally supplied calibrated tracks as UNVERIFIED candidates only."""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tree_reinforcement_learning.grasp_labels import GraspCandidateConfig, label_track_records
from tree_reinforcement_learning.readonly_intake import _json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True, help='JSON thresholds; no production defaults')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.resolve() in (a.tracks.resolve(), a.config.resolve()):
        p.error('output cannot overwrite input')
    content = a.tracks.read_bytes()
    config = GraspCandidateConfig(**_json(a.config.read_text()))
    records = [_json(line) for line in content.splitlines()]
    results = label_track_records(records, config)
    report = {'schema': 'grasp_candidate_batch_v1', 'trainable': False,
              'hardware_authorized': False, 'label_verified': False,
              'input_sha256': hashlib.sha256(content).hexdigest(),
              'config': asdict(config), 'records': results}
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({'records': len(results), 'label_verified': False, 'trainable': False}))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
