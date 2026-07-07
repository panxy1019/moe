#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/moe/V16_AttractorMoEROM/test_results_v16"
WAIT_PATTERN="${V16_WAIT_PATTERN:-python .*train_v15_3_hopf_focused.py}"
if [[ "${V16_WAIT_FOR_IDLE:-0}" == "1" ]]; then
  echo "Waiting for existing training tasks matching: ${WAIT_PATTERN}"
  while pgrep -af "${WAIT_PATTERN}" >/dev/null 2>&1; do
    date -Is
    sleep "${V16_WAIT_SLEEP_SECONDS:-300}"
  done
fi

cases=(
  V16_AttractorMoEROM_FullRegimeLoss32
  V16_AttractorMoEROM_AttractorConditionedFramework32
)

mkdir -p "${ROOT}/results"
for case_name in "${cases[@]}"; do
  if pgrep -af "train_v16_attractor_moe_rom.py.*--experiment-tag ${case_name}" >/dev/null 2>&1; then
    echo "${case_name} already running; skip"
    continue
  fi
  log_dir="${ROOT}/results/${case_name}"
  mkdir -p "${log_dir}"
  nohup "${ROOT}/run_v16_one.sh" "${case_name}" >"${log_dir}/nohup.log" 2>&1 &
  echo $! >"${log_dir}/launcher.pid"
  echo "started ${case_name} launcher pid $(cat "${log_dir}/launcher.pid")"
  sleep 10
done
