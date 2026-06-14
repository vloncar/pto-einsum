#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Run a Python script under `msprof op simulator`.

Usage:
  ./run_sim.sh [options] <script.py> [-- <script args...>]

Options:
  --output <dir>        Final directory to sync results into after the run.
  --soc-version <soc>   Override simulator soc version. Default: Ascend910B4
  --verbose-msprof      Show the full `msprof` simulator log stream.
  --quiet-msprof        Suppress noisy `msprof` INFO/WARN/ERROR lines. Default.
  -h, --help            Show this help.

Environment:
  MSPROF_PRIVATE_ROOT
                        Private root used for the actual `msprof --output`.
                        Defaults to `$XDG_RUNTIME_DIR/msprof` when available,
                        otherwise `$HOME/.cache/msprof`.
  KEEP_MSPROF_STAGING=1
                        Keep the private staging directory after a successful sync.
  MSPROF_LOG_MODE=quiet|verbose
                        Override the default simulator log rendering mode.

Examples:
  ./run_sim.sh test_einsum.py
  ./run_sim.sh \
    --output "$PWD/build/msprof_res/bench_einsum" \
    bench_einsum.py
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

log() {
  echo "[run_sim] $*"
}

ensure_private_dir() {
  local dir="$1"
  umask 077
  mkdir -p "${dir}"
  chmod 700 "${dir}"
}

sync_msprof_output() {
  local src_dir="$1"
  local dst_dir="$2"

  mkdir -p "${dst_dir}"
  cp -a "${src_dir}/." "${dst_dir}/"
}

print_msprof_log() {
  local log_file="$1"
  local mode="$2"
  local status="$3"

  if [[ "${mode}" == "verbose" ]]; then
    cat "${log_file}"
    return
  fi

  grep -Ev '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} \[(INFO|WARN|ERROR|DEBUG)\][[:space:]]' \
    "${log_file}" || true

  if [[ ${status} -ne 0 ]]; then
    local suppressed_count
    suppressed_count=$(wc -l < "${log_file}")
    local filtered_count
    filtered_count=$(
      grep -Evc '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} \[(INFO|WARN|ERROR|DEBUG)\][[:space:]]' \
        "${log_file}" || true
    )
    suppressed_count=$((suppressed_count - filtered_count))
    if [[ ${suppressed_count} -gt 0 ]]; then
      log "msprof failure tail:"
      tail -n 20 "${log_file}"
    fi
    log "full msprof log saved at ${log_file}"
  fi
}

SOC_VERSION="Ascend910B4"
OUTPUT_DIR=""
SCRIPT_PATH=""
SCRIPT_ARGS=()
MSPROF_LOG_MODE="${MSPROF_LOG_MODE:-quiet}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || die "--output requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --soc-version)
      [[ $# -ge 2 ]] || die "--soc-version requires a value"
      SOC_VERSION="$2"
      shift 2
      ;;
    --verbose-msprof)
      MSPROF_LOG_MODE="verbose"
      shift
      ;;
    --quiet-msprof)
      MSPROF_LOG_MODE="quiet"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      SCRIPT_ARGS=("$@")
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      if [[ -z "${SCRIPT_PATH}" ]]; then
        SCRIPT_PATH="$1"
      else
        SCRIPT_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

[[ -n "${SCRIPT_PATH}" ]] || die "missing <script.py>"

if [[ "${SCRIPT_PATH}" != /* ]]; then
  SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_PATH}"
fi
[[ -f "${SCRIPT_PATH}" ]] || die "script not found: ${SCRIPT_PATH}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  die "ASCEND_HOME_PATH is not set; source CANN setenv or export it first"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  SCRIPT_STEM="$(basename -- "${SCRIPT_PATH}" .py)"
  OUTPUT_DIR="${REPO_ROOT}/build/msprof_res/${SCRIPT_STEM}"
else
  SCRIPT_STEM="$(basename -- "${SCRIPT_PATH}" .py)"
fi

SIM_LIB_DIR="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
[[ -d "${SIM_LIB_DIR}" ]] || die "simulator library directory not found: ${SIM_LIB_DIR}"

PRIVATE_ROOT="${MSPROF_PRIVATE_ROOT:-${XDG_RUNTIME_DIR:-${HOME}/.cache}/msprof}"
ensure_private_dir "${PRIVATE_ROOT}"
RUNTIME_OUTPUT_DIR="$(mktemp -d "${PRIVATE_ROOT}/${SCRIPT_STEM}.XXXXXX")"
chmod 700 "${RUNTIME_OUTPUT_DIR}"
MSPROF_STDIO_LOG="${RUNTIME_OUTPUT_DIR}/msprof.stdout.log"

source "${ASCEND_HOME_PATH}/bin/setenv.bash"
export LD_LIBRARY_PATH="${SIM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
ulimit -n 65535

# msprof rejects group/other-writable working directories, so always launch
# from a private directory and use an absolute path for the example script.
cd "${HOME}"

log "staging msprof output in ${RUNTIME_OUTPUT_DIR}"
if [[ "${OUTPUT_DIR}" != "${RUNTIME_OUTPUT_DIR}" ]]; then
  log "final results will be synced to ${OUTPUT_DIR}"
fi

set +e
msprof op simulator \
  --soc-version="${SOC_VERSION}" \
  --output="${RUNTIME_OUTPUT_DIR}" \
  python3 "${SCRIPT_PATH}" "${SCRIPT_ARGS[@]}" \
  > "${MSPROF_STDIO_LOG}" 2>&1
STATUS=$?
set -e

print_msprof_log "${MSPROF_STDIO_LOG}" "${MSPROF_LOG_MODE}" "${STATUS}"

SYNC_STATUS=0
if [[ -d "${RUNTIME_OUTPUT_DIR}" ]]; then
  if sync_msprof_output "${RUNTIME_OUTPUT_DIR}" "${OUTPUT_DIR}"; then
    log "synced msprof results to ${OUTPUT_DIR}"
  else
    SYNC_STATUS=$?
    log "failed to sync msprof results to ${OUTPUT_DIR}"
  fi
fi

if [[ "${KEEP_MSPROF_STAGING:-0}" != "1" && ${SYNC_STATUS} -eq 0 ]]; then
  rm -rf "${RUNTIME_OUTPUT_DIR}"
else
  log "kept staging directory at ${RUNTIME_OUTPUT_DIR}"
fi

if [[ ${STATUS} -ne 0 ]]; then
  exit ${STATUS}
fi

exit ${SYNC_STATUS}
