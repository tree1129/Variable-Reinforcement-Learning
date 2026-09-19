#!/usr/bin/env python3
"""Create a review queue for read-only Case 4--7 head RGB-D captures.

This tool intentionally creates *pending* annotations only.  It never infers
object/hole labels, estimates camera-to-base calibration, or authorizes model
training / robot control.  A reviewer must complete every required field and a
separate calibration validator must pass before any downstream training job may
consume the reviewed data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OBJECT_CLASSES = ("triangle", "square", "trapezoid", "sphere")
HOLE_FACE_LOCATIONS = ("front", "left", "right")
CASE_IDS = (4, 5, 6, 7)


def read_manifest(capture_dir: Path) -> dict[str, Any]:
    path = capture_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing capture manifest: {path}") from exc
    if manifest.get("schema") != "case4_7_real_head_rgbd_training_capture/v1":
        raise SystemExit("Unsupported capture manifest schema")
    if manifest.get("hardware_motion_enabled") is not False:
        raise SystemExit("Refusing non-read-only capture: hardware_motion_enabled must be false")
    if manifest.get("supervised_training_eligible") is not False:
        raise SystemExit("Refusing capture incorrectly marked training-eligible")
    for key in ("publishers_created", "services_called", "actions_sent", "controllers_changed"):
        if manifest.get(key) != 0:
            raise SystemExit(f"Refusing capture: {key} must be 0")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise SystemExit("Capture has no frames")
    return manifest


def queue_record(frame: dict[str, Any]) -> dict[str, Any]:
    if frame.get("annotation_status") != "UNREVIEWED_NOT_TRAINABLE":
        raise SystemExit(f"Frame {frame.get('frame_id')} is not an unreviewed capture frame")
    return {
        "schema": "case4_7_head_rgbd_annotation/v1",
        "sample_id": f"head_rgbd_frame_{int(frame['frame_id']):05d}",
        "source": {
            "capture_frame_id": int(frame["frame_id"]),
            "color": frame["color"],
            "depth": frame["depth"],
            "robot_state": frame.get("robot_state"),
        },
        "review": {
            "status": "PENDING",
            "reviewer": None,
            "reviewed_at_unix_s": None,
            "reject_reason": None,
        },
        "task": {
            "case_id": None,
            "allowed_case_ids": list(CASE_IDS),
            "operation_phase": None,
            "scene_is_usable": None,
        },
        "objects": [
            {
                "instance_id": None,
                "class": None,
                "allowed_classes": list(OBJECT_CLASSES),
                "bbox_xyxy_px": None,
                "mask_path": None,
                "visible_fraction": None,
                "occluded": None,
                "keypoints_xy_px": None,
                "pose_in_head_camera": {
                    "position_m": None,
                    "orientation_xyzw": None,
                    "method": None,
                    "uncertainty": None,
                },
            }
        ],
        "holes": [
            {
                "instance_id": None,
                "shape_class": None,
                "allowed_shape_classes": list(OBJECT_CLASSES),
                "face_location": None,
                "allowed_face_locations": list(HOLE_FACE_LOCATIONS),
                "bbox_xyxy_px": None,
                "mask_path": None,
                "keypoints_xy_px": None,
                "pose_in_head_camera": {
                    "position_m": None,
                    "orientation_xyzw": None,
                    "method": None,
                    "uncertainty": None,
                },
            }
        ],
        "scene": {
            "box_bbox_xyxy_px": None,
            "box_pose_in_head_camera": {
                "position_m": None,
                "orientation_xyzw": None,
                "method": None,
                "uncertainty": None,
            },
            "table_mask_path": None,
            "collision_or_workspace_mask_path": None,
        },
        "calibration": {
            "camera_to_base_status": "UNVERIFIED",
            "transform_id": None,
            "rgb_depth_registration_verified": False,
        },
        "training_gate": {
            "eligible": False,
            "blockers": [
                "review_pending",
                "semantic_labels_missing",
                "camera_to_base_calibration_unverified",
                "rgb_depth_registration_unverified",
                "table_and_collision_masks_missing",
            ],
        },
    }


def write_guide(path: Path) -> None:
    path.write_text(
        "# Case 4--7 head RGB-D annotation guide\n\n"
        "This queue is **not training data**. Each record remains `eligible: false` until an independent validator accepts it.\n\n"
        "## Required reviewer actions\n\n"
        "1. Reject blurred, occluded, ambiguous, out-of-workspace, or incorrectly synchronized scenes.\n"
        "2. Set exactly one Case ID (4, 5, 6, or 7) only when the task arrangement is visible.\n"
        "3. Replace the placeholder object/hole arrays with every visible instance: class, instance ID, box/mask, visibility, and occlusion.\n"
        "4. Label box, table, and collision/workspace masks. Do not use a color-threshold proposal as ground truth without review.\n"
        "5. Add only measured/validated camera-frame poses; do not invent 6D poses.\n"
        "6. Keep camera-to-base status `UNVERIFIED` until an external-hand-eye calibration check supplies a validated transform ID.\n"
        "7. A separate dataset validator must check label completeness, calibration, RGB-depth registration, and a held-out split before any perception or policy training.\n\n"
        "No record in this file authorizes robot motion.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a pending human-annotation queue from a read-only RGB-D capture")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL output path; refuses to overwrite")
    parser.add_argument("--guide", type=Path, help="Optional guide path; defaults next to --output")
    args = parser.parse_args()

    manifest = read_manifest(args.capture)
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing queue: {args.output}")
    guide = args.guide or args.output.with_name("ANNOTATION_GUIDE.md")
    if guide.exists():
        raise SystemExit(f"Refusing to overwrite existing guide: {guide}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = [queue_record(frame) for frame in manifest["frames"]]
    with args.output.open("x", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_guide(guide)
    print(json.dumps({"queue": str(args.output), "guide": str(guide), "records": len(records), "supervised_training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
