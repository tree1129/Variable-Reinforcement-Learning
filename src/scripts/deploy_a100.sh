#!/usr/bin/env bash
# Deploy this independent project to the SSH host named `a100`.
# This script only touches the remote destination below; it does not touch
# tree_center_brain, Wall-X, OpenPI, or any other local project.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${A100_SSH_HOST:-a100}"
REMOTE_DIR="${A100_REMOTE_DIR:-/root/tree_Reinforcement_Learning}"
REMOTE_PYTHON="${A100_PYTHON:-python3}"
RUN_TRAIN=0
BACKEND="custom"
BACKEND_FACTORY=""
STEPS=""
DEVICE="cuda"

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_a100.sh [options]

Options:
  --train                         Start SAC after deployment (requires --backend-factory for custom backend)
  --backend toy|mujoco|custom     Backend for optional training (default: custom)
  --backend-factory module:func   Factory import path for the real simulator backend
  --steps N                       SAC timesteps (default: config value)
  --device cuda|cpu|auto          SB3 device (default: cuda)
  --host SSH_HOST                 SSH alias (default: a100)
  --remote-dir PATH               Remote install directory (default: /root/tree_Reinforcement_Learning)
  -h, --help                      Show this help

Examples:
  # Copy, create venv, install, and run remote checks only:
  scripts/deploy_a100.sh

  # Deploy and start SAC against a simulator adapter already installed remotely:
  scripts/deploy_a100.sh --train --backend custom \
    --backend-factory my_sim.adapter:make_backend --steps 300000

The default deployment does not start training. ToyBackend is only an API smoke test,
not a physical simulator and not evidence of real robot performance.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) RUN_TRAIN=1; shift ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --backend-factory) BACKEND_FACTORY="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --host) REMOTE_HOST="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$RUN_TRAIN" == 1 && "$BACKEND" == custom && -z "$BACKEND_FACTORY" ]]; then
  echo "--train --backend custom requires --backend-factory module:callable" >&2
  exit 2
fi

SSH_OPTS=(
  -o ConnectTimeout="${A100_CONNECT_TIMEOUT:-10}"
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=3
)

# `tar` over SSH avoids copying macOS metadata files (._*) into the Linux run.
# A temporary archive is removed locally even if a later remote command fails.
ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/tree_rl_a100.XXXXXX.tar.gz")"
cleanup() { rm -f "$ARCHIVE"; }
trap cleanup EXIT

tar -czf "$ARCHIVE" \
  --exclude='._*' \
  --exclude='*/._*' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='outputs/*' \
  -C "$(dirname "$PROJECT_ROOT")" "$(basename "$PROJECT_ROOT")"

echo "[1/4] Uploading $PROJECT_ROOT to $REMOTE_HOST:$REMOTE_DIR"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
scp "${SSH_OPTS[@]}" "$ARCHIVE" "$REMOTE_HOST:/tmp/tree_rl_a100.tar.gz"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "rm -rf '$REMOTE_DIR.new' && mkdir -p '$REMOTE_DIR.new' && tar -xzf /tmp/tree_rl_a100.tar.gz -C '$REMOTE_DIR.new' --strip-components=1 && rm -f /tmp/tree_rl_a100.tar.gz && rm -rf '$REMOTE_DIR.old' && if [ -e '$REMOTE_DIR' ]; then mv '$REMOTE_DIR' '$REMOTE_DIR.old'; fi && mv '$REMOTE_DIR.new' '$REMOTE_DIR' && rm -rf '$REMOTE_DIR.old'"

echo "[2/4] Creating remote virtual environment"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_DIR' && '$REMOTE_PYTHON' -m venv .venv"

echo "[3/4] Installing package and RL dependencies"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_DIR' && .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -e '.[rl,mujoco]'"

echo "[4/4] Running remote validation"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_DIR' && .venv/bin/python -m unittest discover -s tests -v && .venv/bin/python scripts/validate_tasks.py && .venv/bin/python scripts/train.py --dry-run"
if [[ "$BACKEND" == mujoco ]]; then
  echo "Running headless MuJoCo backend validation"
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_DIR' && .venv/bin/python scripts/validate_mujoco_backend.py"
fi

echo "Deployment and validation completed: $REMOTE_HOST:$REMOTE_DIR"

if [[ "$RUN_TRAIN" == 1 ]]; then
  # Build a shell-quoted remote command without relying on Bash 5-only array
  # expansion (the default macOS Bash is often Bash 3).
  quote_arg() { printf '%q' "$1"; }
  TRAIN_CMD="$(quote_arg --backend) $(quote_arg "$BACKEND") $(quote_arg --algorithm) $(quote_arg sac) $(quote_arg --device) $(quote_arg "$DEVICE") $(quote_arg --output) $(quote_arg "$REMOTE_DIR/outputs/case4_7_sac_a100")"
  if [[ "$BACKEND" == mujoco && -z "$BACKEND_FACTORY" ]]; then
    BACKEND_FACTORY="tree_reinforcement_learning.mujoco_backend:make_backend"
  fi
  [[ -n "$BACKEND_FACTORY" ]] && TRAIN_CMD+=" $(quote_arg --backend-factory) $(quote_arg "$BACKEND_FACTORY")"
  [[ -n "$STEPS" ]] && TRAIN_CMD+=" $(quote_arg --steps) $(quote_arg "$STEPS")"
  # Start only after setup/checks pass. Use nohup so the SSH session can close.
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_DIR' && mkdir -p outputs/case4_7_sac_a100 && nohup .venv/bin/python scripts/train.py $TRAIN_CMD > outputs/case4_7_sac_a100/train.log 2>&1 < /dev/null & echo SAC_PID=\$!"
  echo "SAC launch requested; monitor $REMOTE_DIR/outputs/case4_7_sac_a100/train.log on $REMOTE_HOST"
fi
