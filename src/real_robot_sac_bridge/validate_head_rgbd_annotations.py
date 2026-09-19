#!/usr/bin/env python3
"""Strictly gate reviewed Case 4--7 head RGB-D annotations before training.

This program is deliberately a data-quality gate, not a trainer and not a
robot-control bridge.  It never imports/executes a policy, creates ROS objects,
or changes files other than its explicitly requested new report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OBJECT_CLASSES = {"triangle", "square", "trapezoid", "sphere"}
FACE_LOCATIONS = {"front", "left", "right"}
CASE_IDS = {4, 5, 6, 7}


def fail(issues: list[str], sample: str, message: str) -> None:
    issues.append(f"{sample}: {message}")


def in_image_bbox(value: Any, width: int, height: int) -> bool:
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(x, (int, float)) for x in value):
        return False
    x1, y1, x2, y2 = value
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def path_is_file(capture: Path, relative: Any) -> bool:
    return isinstance(relative, str) and bool(relative) and (capture / relative).is_file()


def validate_pose(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    p, q = value.get("position_m"), value.get("orientation_xyzw")
    return (
        isinstance(p, list) and len(p) == 3 and all(isinstance(x, (int, float)) for x in p)
        and isinstance(q, list) and len(q) == 4 and all(isinstance(x, (int, float)) for x in q)
        and isinstance(value.get("method"), str) and bool(value["method"])
        and value.get("uncertainty") is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate reviewed real head RGB-D labels before any training job")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="New report path; refuses overwrite")
    parser.add_argument("--fail-on-blocked", action="store_true", help="Exit nonzero when training remains blocked")
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit(f"Refusing to overwrite: {args.report}")
    manifest_path = args.capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "case4_7_real_head_rgbd_training_capture/v1":
        raise SystemExit("Unsupported capture manifest")
    if manifest.get("hardware_motion_enabled") is not False:
        raise SystemExit("Refusing non-read-only data")
    source_frames = {int(f["frame_id"]): f for f in manifest.get("frames", [])}
    if not source_frames:
        raise SystemExit("No source frames")
    if not args.annotations.is_file():
        raise SystemExit(f"Annotation file missing: {args.annotations}")

    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(args.annotations.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"line {lineno}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(row, dict):
            issues.append(f"line {lineno}: expected object")
            continue
        rows.append(row)

    seen: set[int] = set()
    for row in rows:
        source = row.get("source", {})
        frame_id = source.get("capture_frame_id")
        sample = str(row.get("sample_id", f"line_for_{frame_id}"))
        if not isinstance(frame_id, int) or frame_id not in source_frames:
            fail(issues, sample, "unknown source capture frame")
            continue
        if frame_id in seen:
            fail(issues, sample, "duplicate source capture frame")
        seen.add(frame_id)
        original = source_frames[frame_id]
        if source.get("color", {}).get("path") != original["color"]["path"] or source.get("depth", {}).get("path") != original["depth"]["path"]:
            fail(issues, sample, "source color/depth references differ from immutable capture manifest")

        review = row.get("review", {})
        if review.get("status") != "ACCEPTED":
            fail(issues, sample, "review.status must be ACCEPTED")
            continue
        if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            fail(issues, sample, "accepted record lacks reviewer")
        if not isinstance(review.get("reviewed_at_unix_s"), (int, float)):
            fail(issues, sample, "accepted record lacks review timestamp")

        task = row.get("task", {})
        if task.get("case_id") not in CASE_IDS:
            fail(issues, sample, "case_id must be one of 4,5,6,7")
        if not isinstance(task.get("operation_phase"), str) or not task["operation_phase"].strip():
            fail(issues, sample, "operation_phase is required")
        if task.get("scene_is_usable") is not True:
            fail(issues, sample, "scene_is_usable must be true")

        for group, cls_key, valid_values in (("objects", "class", OBJECT_CLASSES), ("holes", "shape_class", OBJECT_CLASSES)):
            instances = row.get(group)
            if not isinstance(instances, list) or not instances:
                fail(issues, sample, f"{group} must contain at least one reviewed instance")
                continue
            for index, instance in enumerate(instances):
                item = f"{sample}.{group}[{index}]"
                if not isinstance(instance, dict):
                    fail(issues, item, "must be an object")
                    continue
                if not isinstance(instance.get("instance_id"), str) or not instance["instance_id"].strip():
                    fail(issues, item, "instance_id is required")
                if instance.get(cls_key) not in valid_values:
                    fail(issues, item, f"{cls_key} invalid")
                if not in_image_bbox(instance.get("bbox_xyxy_px"), 1280, 720):
                    fail(issues, item, "bbox_xyxy_px invalid for 1280x720 RGB image")
                if not path_is_file(args.capture, instance.get("mask_path")):
                    fail(issues, item, "reviewed mask_path missing")
                if instance.get("occluded") is not False:
                    fail(issues, item, "occluded samples must be rejected from initial training set")
                if not validate_pose(instance.get("pose_in_head_camera")):
                    fail(issues, item, "validated camera-frame 6D pose and uncertainty required")
                if group == "holes" and instance.get("face_location") not in FACE_LOCATIONS:
                    fail(issues, item, "hole face_location invalid")

        scene = row.get("scene", {})
        if not in_image_bbox(scene.get("box_bbox_xyxy_px"), 1280, 720):
            fail(issues, sample, "box_bbox_xyxy_px invalid or missing")
        if not path_is_file(args.capture, scene.get("table_mask_path")):
            fail(issues, sample, "reviewed table_mask_path missing")
        if not path_is_file(args.capture, scene.get("collision_or_workspace_mask_path")):
            fail(issues, sample, "reviewed collision_or_workspace_mask_path missing")
        if not validate_pose(scene.get("box_pose_in_head_camera")):
            fail(issues, sample, "validated box camera-frame 6D pose and uncertainty required")

        calibration = row.get("calibration", {})
        if calibration.get("camera_to_base_status") != "VALIDATED":
            fail(issues, sample, "camera_to_base_status must be VALIDATED")
        if not isinstance(calibration.get("transform_id"), str) or not calibration["transform_id"].strip():
            fail(issues, sample, "validated transform_id required")
        if calibration.get("rgb_depth_registration_verified") is not True:
            fail(issues, sample, "rgb_depth_registration_verified must be true")

    missing = sorted(set(source_frames) - seen)
    if missing:
        issues.append(f"dataset: annotations missing {len(missing)} source frames (first: {missing[:5]})")
    duplicate_count = len(rows) - len(seen)
    status = "ELIGIBLE_FOR_PERCEPTION_TRAINING_ONLY" if not issues else "BLOCKED_PENDING_REVIEW_AND_CALIBRATION"
    report = {
        "schema": "case4_7_real_head_rgbd_annotation_gate/v1",
        "capture": str(args.capture),
        "annotations": str(args.annotations),
        "status": status,
        "policy_training_allowed": False,
        "perception_training_allowed": status == "ELIGIBLE_FOR_PERCEPTION_TRAINING_ONLY",
        "robot_motion_allowed": False,
        "counts": {"source_frames": len(source_frames), "annotation_rows": len(rows), "unique_source_frames": len(seen), "duplicate_or_invalid_rows": duplicate_count, "issues": len(issues)},
        "requirements_after_eligibility": [
            "train and validate perception separately on held-out real-camera data",
            "build the SAC 88-D state only from validated perception outputs",
            "validate collision/workspace safety wrapper independently",
            "obtain explicit approval before any guarded physical test",
        ],
        "issues": issues,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "counts": report["counts"], "perception_training_allowed": report["perception_training_allowed"], "policy_training_allowed": False, "robot_motion_allowed": False}, ensure_ascii=False))
    if args.fail_on_blocked and issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
