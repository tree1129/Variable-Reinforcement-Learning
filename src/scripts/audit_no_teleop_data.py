#!/usr/bin/env python3
"""Validate executed V4 traces; passing never authorizes hardware motion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tree_reinforcement_learning.trace_audit import audit_trace


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if a.dataset.resolve() == a.report.resolve():
        p.error("report must not overwrite the input dataset")
    report = audit_trace(a.dataset)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["trainable"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
