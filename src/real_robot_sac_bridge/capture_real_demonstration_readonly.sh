#!/usr/bin/env bash
# Record one manually operated Case 4--7 real-robot demonstration.
#
# SAFETY: this script only starts `ros2 bag record` subscriptions. It does not
# publish robot commands, call services/actions, switch controllers, or execute
# a learned policy. Robot motion must come from the already-authorized manual
# teleoperation system and remain under the operator's supervision.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  capture_real_demonstration_readonly.sh --case 4|5|6|7|unknown --episode ID [options]

Required:
  --case N             Case ID (4, 5, 6, 7, or unknown for a live unlabelled capture)
  --episode ID         Episode identifier, e.g. 001

Options:
  --seconds N          Maximum recording duration (default: 60, range: 5..300)
  --output-base DIR    Output root (default: /home/xr/workspace/case4_7_real_demos)
  --operator NAME      Non-sensitive operator label (default: manual_teleop)
  --layout LABEL       Scene/layout label (default: unreviewed)
  -h, --help           Show this help

The output remains UNREVIEWED_NOT_TRAINABLE until the episode result, collision
status, action semantics, synchronization, and calibration gates are reviewed.
USAGE
}

case_id=""
episode=""
seconds=60
output_base="/home/xr/workspace/case4_7_real_demos"
operator="manual_teleop"
layout="unreviewed"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) case_id="${2:-}"; shift 2 ;;
    --episode) episode="${2:-}"; shift 2 ;;
    --seconds) seconds="${2:-}"; shift 2 ;;
    --output-base) output_base="${2:-}"; shift 2 ;;
    --operator) operator="${2:-}"; shift 2 ;;
    --layout) layout="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$case_id" =~ ^[4-7]$ || "$case_id" == "unknown" ]] || { echo "--case must be 4, 5, 6, 7, or unknown" >&2; exit 2; }
[[ "$episode" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--episode must use only letters, digits, dot, underscore, or hyphen" >&2; exit 2; }
[[ "$seconds" =~ ^[0-9]+$ ]] && (( seconds >= 5 && seconds <= 300 )) || { echo "--seconds must be an integer from 5 to 300" >&2; exit 2; }
[[ "$operator" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--operator contains unsupported characters" >&2; exit 2; }
[[ "$layout" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--layout contains unsupported characters" >&2; exit 2; }

if ! command -v ros2 >/dev/null 2>&1; then
  setup="$(find /opt/ros -mindepth 2 -maxdepth 2 -name setup.bash 2>/dev/null | head -n 1 || true)"
  if [[ -n "$setup" ]]; then
    set +u
    source "$setup"
    set -u
  fi
fi
command -v ros2 >/dev/null 2>&1 || { echo "ros2 not found; run inside the ROS-enabled debug container" >&2; exit 1; }

stamp="$(date +%Y%m%d_%H%M%S)"
capture_id="case${case_id}_ep${episode}_${stamp}"
out_dir="${output_base%/}/${capture_id}"
bag_dir="$out_dir/bag"
mkdir -p "$out_dir"
[[ ! -e "$bag_dir" ]] || { echo "Refusing to overwrite $bag_dir" >&2; exit 1; }

# Feedback/perception streams plus copies of existing teleoperation commands as
# action labels. Merely subscribing to a command topic does not actuate it.
requested_topics=(
  /camera_head_front/color/image_raw/compressed
  /camera_head_front/depth/image_raw/compressedDepth
  /camera_head_front/color/camera_info
  /camera_head_front/depth/camera_info
  /left_arm/joint_states
  /right_arm/joint_states
  /left_arm/end_pose
  /right_arm/end_pose
  /left_gripper/joint_states
  /right_gripper/joint_states
  /left_arm/wrench_ext_world
  /right_arm/wrench_ext_world
  /left_arm/joint_torque_ext
  /right_arm/joint_torque_ext
  /teleoperation/slave/left_gripper_control
  /teleoperation/slave/right_gripper_control
  /left_arm/pose_command/related
  /right_arm/pose_command/related
  /left_arm_cartesian_controller/pose_cmd
  /right_arm_cartesian_controller/pose_cmd
  /left_arm_joint_controller/commands
  /right_arm_joint_controller/commands
  /left_gripper_controller/commands
  /right_gripper_controller/commands
  /x2robot/action/arm1
  /x2robot/action/arm2
  /tf
  /tf_static
)

mapfile -t live_topics < <(ros2 topic list)
declare -A live=()
for topic in "${live_topics[@]}"; do live["$topic"]=1; done
record_topics=()
missing_topics=()
for topic in "${requested_topics[@]}"; do
  if [[ -n "${live[$topic]+x}" ]]; then
    record_topics+=("$topic")
  else
    missing_topics+=("$topic")
  fi
done
(( ${#record_topics[@]} >= 10 )) || { echo "Too few expected topics are live (${#record_topics[@]}); refusing capture" >&2; exit 1; }

python3 - "$out_dir/episode.json" "$capture_id" "$case_id" "$episode" "$seconds" "$operator" "$layout" "${record_topics[*]}" "${missing_topics[*]}" <<'PY'
import json, sys, time
path, capture_id, case_id, episode, seconds, operator, layout, topics, missing = sys.argv[1:]
payload = {
    "schema": "case4_7_real_robot_demonstration/v1",
    "capture_id": capture_id,
    "created_unix_s": time.time(),
    "case_id": int(case_id) if case_id.isdigit() else None,
    "episode_id": episode,
    "operator_label": operator,
    "layout_label": layout,
    "case_label_status": "UNREVIEWED" if case_id == "unknown" else "OPERATOR_DECLARED_UNREVIEWED",
    "capture_status": "RECORDING",
    "review_status": "UNREVIEWED_NOT_TRAINABLE",
    "result": "unreviewed",
    "collision": "unreviewed",
    "stable_grasp": "unreviewed",
    "hardware_motion_source": "existing_manual_teleoperation_only",
    "hardware_motion_enabled_by_recorder": False,
    "publishers_created_for_robot_commands": 0,
    "services_called": 0,
    "actions_sent": 0,
    "controllers_changed": 0,
    "learned_policy_executed": False,
    "seconds_requested": int(seconds),
    "recorded_topics": topics.split(),
    "missing_optional_topics": missing.split(),
    "training_eligibility": {
        "perception": False,
        "offline_imitation": False,
        "online_policy": False,
        "blocking_gates": [
            "episode outcome and safety review pending",
            "RGB-depth registration review pending",
            "camera-to-base calibration not validated",
            "real action/observation contract not validated"
        ]
    }
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY
printf '%s\n' "${record_topics[@]}" > "$out_dir/recorded_topics.txt"
printf '%s\n' "${missing_topics[@]}" > "$out_dir/missing_optional_topics.txt"

cat > "$out_dir/REVIEW_REQUIRED.md" <<'REVIEW'
# Human review required

Before using this episode, record: success/failure/abort, any table/box/self
collision, whether each intended object was handled by the correct arm, whether
the gripper fully closed and retained the object, and whether placement was
stable. This bag is not authorization for autonomous real-robot motion.
REVIEW

set +e
timeout --signal=INT --kill-after=15 "${seconds}s" \
  ros2 bag record \
    --storage mcap \
    --storage-preset-profile zstd_fast \
    --disable-keyboard-controls \
    --polling-interval 100 \
    --output "$bag_dir" \
    --custom-data "case_id=$case_id" "episode_id=$episode" "operator=$operator" "layout=$layout" "motion_source=manual_teleoperation" \
    --topics "${record_topics[@]}" \
    > "$out_dir/rosbag_record.log" 2>&1
rc=$?
set -e
# timeout commonly returns 124 even after SIGINT produced a valid finalized bag.
if [[ $rc -ne 0 && $rc -ne 124 ]]; then
  status="FAILED"
else
  status="CAPTURED_PENDING_REVIEW"
fi

bytes="$(du -sb "$bag_dir" 2>/dev/null | awk '{print $1}' || echo 0)"
critical_topics=(
  /camera_head_front/color/image_raw/compressed
  /camera_head_front/depth/image_raw/compressedDepth
  /left_arm/joint_states
  /right_arm/joint_states
  /left_arm/end_pose
  /right_arm/end_pose
  /left_gripper/joint_states
  /right_gripper/joint_states
  /left_arm/wrench_ext_world
  /right_arm/wrench_ext_world
  /left_arm_cartesian_controller/pose_cmd
  /right_arm_cartesian_controller/pose_cmd
  /left_gripper_controller/commands
  /right_gripper_controller/commands
)
missing_critical=()
if [[ -f "$bag_dir/metadata.yaml" ]]; then
  ros2 bag info "$bag_dir" > "$out_dir/rosbag_info.txt" 2>&1 || true
  for topic in "${critical_topics[@]}"; do
    if ! grep -F "Topic: $topic |" "$out_dir/rosbag_info.txt" | grep -Eq 'Count: [1-9][0-9]*'; then
      missing_critical+=("$topic")
    fi
  done
else
  status="FAILED_NO_METADATA"
fi
if (( ${#missing_critical[@]} > 0 )); then
  status="FAILED_MISSING_CRITICAL_DATA"
fi
printf '%s\n' "${missing_critical[@]}" > "$out_dir/missing_critical_data.txt"

python3 - "$out_dir/episode.json" "$status" "$rc" "$bytes" "${missing_critical[*]}" <<'PY'
import json, sys, time
path, status, rc, size_bytes, missing_critical = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    payload = json.load(f)
payload.update({
    "capture_status": status,
    "completed_unix_s": time.time(),
    "recorder_exit_code": int(rc),
    "bag_size_bytes": int(size_bytes),
    "critical_stream_gate_passed": not bool(missing_critical.split()),
    "missing_critical_streams": missing_critical.split(),
})
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY

echo "capture_id=$capture_id"
echo "status=$status"
echo "output=$out_dir"
echo "recorded_topics=${#record_topics[@]} missing_optional_topics=${#missing_topics[@]} bag_size_bytes=$bytes"
echo "hardware_motion_enabled_by_recorder=false publishers_for_robot_commands=0 services_called=0 actions_sent=0 controllers_changed=0"
[[ "$status" != FAILED* ]]
