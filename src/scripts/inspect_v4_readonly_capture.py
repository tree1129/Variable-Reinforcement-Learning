#!/usr/bin/env python3
"""Inspect bounded read-only ROS capture. Exit 2 means NOT trainable (expected)."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tree_reinforcement_learning.readonly_intake import inspect_capture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    if a.capture.resolve() == a.report.resolve():
        p.error('report cannot overwrite input')
    report = inspect_capture(a.capture)
    # Exclusive creation also prevents overwriting source/evidence via hardlinks.
    with a.report.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ('capture_integrity_passed', 'trainable', 'hardware_authorized', 'integrity_errors')}, ensure_ascii=False))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
