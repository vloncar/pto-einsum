#!/usr/bin/env bash

set -euo pipefail

# Print usage instructions
usage() {
  cat <<EOF
Usage: $0 [--image IMAGE] [--device DEVICE] HOSTNAME COMMAND...

Run a command in a Docker container on a remote machine.

Options:
  --image IMAGE     Docker image to use. Default: quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12
  --device DEVICE   NPU device ID to use (e.g. 0, 1). Default: auto-detect 0
  -h, --help        Show this help message

Environment Variables:
  REMOTE_DIR        Target directory on the remote host. Default: ~/pto-einsum-remote
EOF
}

# Parse command line options
IMAGE="quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12"
DEVICE_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      if [[ $# -lt 2 ]]; then
        echo "Error: --image requires an argument" >&2
        exit 1
      fi
      IMAGE="$2"
      shift 2
      ;;
    --device)
      if [[ $# -lt 2 ]]; then
        echo "Error: --device requires an argument" >&2
        exit 1
      fi
      DEVICE_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Error: Unknown option $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 2 ]]; then
  echo "Error: Missing HOSTNAME or COMMAND" >&2
  usage >&2
  exit 1
fi

HOSTNAME="$1"
shift
COMMAND=("$@")

# Determine local repository root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${SCRIPT_DIR}"

# Remote directory setup
REMOTE_DIR="${REMOTE_DIR:-~/pto-einsum-remote}"

echo "=== Syncing workspace to ${HOSTNAME}:${REMOTE_DIR} ==="
# Exclude build files, cache dirs, and virtualenvs from sync
rsync -avz --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'build/' \
  "${LOCAL_DIR}/" "${HOSTNAME}:${REMOTE_DIR}/"

# Safely escape command arguments
QUOTED_COMMAND=$(printf '%q ' "${COMMAND[@]}")

echo "=== Executing command inside Docker container on ${HOSTNAME} ==="

# Execute via SSH
# We dynamically check remote UID/GID, docker sudo requirements, and NPU availability.
ssh -t "${HOSTNAME}" "
  REMOTE_UID=\$(id -u)
  REMOTE_GID=\$(id -g)

  # Resolve tilde to remote user's home directory if present
  REAL_REMOTE_DIR="${REMOTE_DIR}"
  if [ \"\${REAL_REMOTE_DIR}\" = \"~\" ]; then
    REAL_REMOTE_DIR=\"\$HOME\"
  elif [[ \"\${REAL_REMOTE_DIR}\" == \~/* ]]; then
    REAL_REMOTE_DIR=\"\$HOME/\${REAL_REMOTE_DIR#\~/}\"
  fi

  # Check if docker requires sudo
  if docker info >/dev/null 2>&1; then
    DOCKER_BIN='docker'
  else
    DOCKER_BIN='sudo docker'
  fi

  # Mount all available NPU devices
  DEVICE_ID='${DEVICE_ID}'
  NPU_FLAGS=''
  echo '[run_ssh] Mounting all available NPU devices...'
  for dev in /dev/davinci[0-9]*; do
    if [ -c \"\$dev\" ]; then
      NPU_FLAGS=\"\$NPU_FLAGS --device=\$dev\"
    fi
  done
  NPU_FLAGS=\"\$NPU_FLAGS --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /usr/local/Ascend/driver:/usr/local/Ascend/driver\"
  if [ -f /usr/local/bin/npu-smi ]; then
    NPU_FLAGS=\"\$NPU_FLAGS -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi\"
  fi

  # Set visibility via ASCEND_RT_VISIBLE_DEVICES
  if [ -n \"\$DEVICE_ID\" ]; then
    echo \"[run_ssh] Setting ASCEND_RT_VISIBLE_DEVICES=\$DEVICE_ID\"
    NPU_FLAGS=\"\$NPU_FLAGS -e ASCEND_RT_VISIBLE_DEVICES=\$DEVICE_ID\"
  else
    echo '[run_ssh] Defaulting ASCEND_RT_VISIBLE_DEVICES=0'
    NPU_FLAGS=\"\$NPU_FLAGS -e ASCEND_RT_VISIBLE_DEVICES=0\"
  fi

  # Run the Docker container
  # We run as user 0:0 (root) inside the container as requested.
  # Before exiting, the container resets the ownership of mounted files back to the host user's UID:GID.
  \$DOCKER_BIN run --rm \
    --ipc=host \
    --privileged \
    -u 0:0 \
    \$NPU_FLAGS \
    -v \"\${REAL_REMOTE_DIR}:/workspace\" \
    -w /workspace \
    \"${IMAGE}\" \
    bash -c \"${QUOTED_COMMAND}; status=\\\$?; chown -R \${REMOTE_UID}:\${REMOTE_GID} /workspace; exit \\\$status\"
"
