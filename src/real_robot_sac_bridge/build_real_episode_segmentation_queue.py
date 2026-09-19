#!/usr/bin/env python3
"""Build a conservative time-window review queue for a mixed real-robot episode.

This tool never labels success from motion heuristics. Every window remains
UNREVIEWED until an operator marks result/collision/grasp stability. It only
creates metadata for later extraction from an existing ROS bag.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def duration_from_info(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^Duration:\s*([0-9]+(?:\.[0-9]+)?)s", text, re.MULTILINE)
    if not m:
        raise SystemExit(f"Cannot find bag duration in {path}")
    return float(m.group(1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--rosbag-info", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window-seconds", type=float, default=2.0)
    args = p.parse_args()
    if not (0.5 <= args.window_seconds <= 30.0):
        raise SystemExit("--window-seconds must be between 0.5 and 30")
    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    duration = duration_from_info(args.rosbag_info)
    if episode.get("result") != "partial_success":
        raise SystemExit("This queue is intended for an episode reviewed as partial_success")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    windows = []
    i = 0
    start = 0.0
    while start < duration - 1e-6:
        end = min(duration, start + args.window_seconds)
        windows.append({
            "window_id": f"{i:04d}",
            "start_s": round(start, 3),
            "end_s": round(end, 3),
            "case_id": episode.get("case_id"),
            "result": "unreviewed",
            "collision": "no_inherited_from_episode_but_verify",
            "stable_grasp": "unreviewed",
            "arm_assignment": "unreviewed",
            "object_retained": "unreviewed",
            "placement_stable": "unreviewed",
            "train_split": "pending_manual_review",
            "review_notes": "Mark success/failure/abort for this exact time interval; do not infer from the parent episode.",
        })
        i += 1
        start = end
    manifest = {
        "schema": "case4_7_real_episode_segmentation_queue/v1",
        "episode_path": str(args.episode),
        "bag_info_path": str(args.rosbag_info),
        "case_id": episode.get("case_id"),
        "parent_result": episode.get("result"),
        "parent_collision": episode.get("collision"),
        "parent_stable_grasp": episode.get("stable_grasp"),
        "duration_s": duration,
        "window_seconds": args.window_seconds,
        "window_count": len(windows),
        "all_windows_trainable": False,
        "review_required": True,
        "windows": windows,
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    jsonl = output.with_suffix(".jsonl")
    with jsonl.open("w", encoding="utf-8") as f:
        for item in windows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "jsonl": str(jsonl), "duration_s": duration, "windows": len(windows), "trainable": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
