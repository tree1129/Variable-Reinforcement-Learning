# SAC -> SDK bridge (shadow only)

This bridge connects A100 SAC inference to read-only robot observations and validates the action conversion. It deliberately does **not** publish to any robot command topic (`hardware_motion_enabled=false`). The model is simulation-only and its observation contract is not calibrated to the real robot.

Run on `robot` (ROS container) with the model copied from A100:

```bash
python3 shadow_bridge.py --model /path/to/sac_case4_7.zip --seconds 10
```

The output JSON records robot pose samples, model actions, finite/range checks, and the reason execution is blocked. To enable hardware, a separate reviewed implementation must establish calibration, controller ownership, workspace/velocity/force limits, watchdog and an explicit hardware gate; this file intentionally has no hardware-enable flag.

## Real-robot readiness probe (read-only)

`readiness_probe.py` is the first deployment-stage check. It subscribes only to
both arms' end poses, arm and gripper joint states, wrist wrenches, and the head
RGB-D streams. It creates no ROS publisher, service/action client, or controller
switch request and does not load or run the policy. Run it in a ROS-enabled
container:

```bash
python3 readiness_probe.py --seconds 12 --output /tmp/case4_7_sac_readiness.json
```

A successful sensor-stream report is **not** authorization to move hardware. It
only establishes that calibration and perception work can begin; all gates named
in `required_before_motion` must remain unresolved until independently validated.

## Head RGB-D training-data capture (read-only)

`capture_head_rgbd_training_readonly.py` stores time-paired head RGB, measured
head depth, camera intrinsics, both-arm pose feedback, and gripper feedback. It
uses subscriptions only and emits all samples as `UNREVIEWED_NOT_TRAINABLE`.
It must not be used for supervised training until each selected frame has
reviewed object/hole annotations, a Case 4--7 identity, and validated
camera-to-base calibration.

## Annotation and calibration gates (no control path)

`build_head_rgbd_annotation_queue.py` turns a read-only capture into a pending
JSONL review queue. It does not infer labels and marks every record
non-trainable. Use a new output path:

```bash
python3 build_head_rgbd_annotation_queue.py \
  --capture deployment_reports/head_rgbd_case4_7_YYYYMMDD_NNN \
  --output deployment_reports/head_rgbd_case4_7_YYYYMMDD_NNN/annotations/review_queue.jsonl
```

`validate_head_rgbd_annotations.py` is the hard data gate. It requires accepted
human review, Case 4--7 identity, reviewed masks and camera-frame poses, a
validated camera-to-base transform ID, and RGB-D registration before it can
allow **perception-only** training. It never authorizes policy training or
robot motion:

```bash
python3 validate_head_rgbd_annotations.py \
  --capture deployment_reports/head_rgbd_case4_7_YYYYMMDD_NNN \
  --annotations deployment_reports/head_rgbd_case4_7_YYYYMMDD_NNN/annotations/review_queue.jsonl \
  --report deployment_reports/head_rgbd_case4_7_YYYYMMDD_NNN/annotations/annotation_gate.json \
  --fail-on-blocked
```

`calibration_tf_preflight_readonly.py` subscribes only to `/tf`, `/tf_static`,
and head-camera `camera_info` topics. It can establish whether a transform graph
connection is visible, but it never treats that as a camera-to-base calibration;
measured hand-eye calibration and held-out residual validation remain required.

## Fast real-robot demonstration capture (manual motion, read-only recorder)

`capture_real_demonstration_readonly.sh` records one operator-controlled Case
4--7 episode as an MCAP ROS bag. It records compressed head RGB-D, camera info,
both arm/gripper feedback, wrist force/torque, TF, and copies of existing
teleoperation command streams as offline action labels. The recorder itself
never publishes a robot command, calls an action/service, switches a controller,
or runs a learned policy.

Run it inside the ROS-enabled debug container while the authorized operator uses
the existing teleoperation system:

```bash
./capture_real_demonstration_readonly.sh \
  --case 4 --episode 001 --seconds 60 --layout random_left_right
```

Use a separate episode for every reset. Keep successes, safe failures, and
collision/abort episodes in separate reviewed groups. A new bag is deliberately
marked `UNREVIEWED_NOT_TRAINABLE`; review the outcome and safety fields and
validate RGB-D registration, camera-to-base calibration, and the real action
contract before conversion to an offline-learning dataset.

## Mixed-episode segmentation gate

For an episode reviewed as `partial_success`, use
`build_real_episode_segmentation_queue.py` to create conservative time windows.
The tool never infers success from motion and leaves every window
`pending_manual_review`. Only windows explicitly reviewed as successful should
enter the pure-success training split; failed windows can be retained for
recovery/stability analysis.

```bash
python3 build_real_episode_segmentation_queue.py \
  --episode /path/to/episode.json \
  --rosbag-info /path/to/rosbag_info.txt \
  --output /path/to/segmentation_queue.json \
  --window-seconds 2
```

## Frozen V4 passive intake (v16; no teleoperation required)

The passive observer `capture_v4_execution_readonly.py` records existing ROS
streams without starting a policy or issuing motion. It preserves chunk/epoch
metadata, dispatch counters, pose/gripper feedback, separate receipt clocks and
camera **metadata only**. It has a bounded 1--30 second duration and no hardware
enable flag. Run inside the ROS container, writing a **new** output directory,
or use `--stdout` to stream the log through SSH without installing a recorder.

The deployed C++ dispatch counter counts command **publications**, not physical
execution. Queue `chunk_ack` is not a controller-applied acknowledgement. Thus
these captures are NOT trainable V4 trajectories and are never auto-converted to
RL transitions. Missing raw V4 identity, calibrated object tracks and validated
outcomes remain blockers. The grasp candidate labeler cannot mark labels verified.

Implementation, measured results and all absolute paths are documented in:
`/Volumes/Extreme Pro/csf_mac/自变量/tree_Reinforcement_Learning/docs/v4_readonly_intake_v16_zh.md`.
