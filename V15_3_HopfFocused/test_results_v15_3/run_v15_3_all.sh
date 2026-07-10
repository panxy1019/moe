#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/moe/V15_3_HopfFocused/test_results_v15_3"
cases=(
  V15_3_StrongBaseline32_AdaptiveGate_Balanced
  V15_3_HopfAmpEnvelopeLoss32
  V15_3_HopfAmpEnvelopeLoss32_WithWeightedRollout
)
mkdir -p "${ROOT}/results"

for case_name in "${cases[@]}"; do
  name="${case_name}_ru32_rp32"
  out="${ROOT}/results/${case_name}/${name}"
  metrics="${out}/${name}_metrics.json"
  if [[ -f "${metrics}" ]]; then
    echo "${case_name}: metrics already exist, skipping"
    continue
  fi
  if pgrep -af "train_v15_3_hopf_focused.py.*--experiment-tag ${case_name}" >/dev/null; then
    echo "${case_name}: already running, skipping"
    continue
  fi
  mkdir -p "${out}"
  log="${out}/nohup.log"
  echo "${case_name}: starting $(date -Is)" | tee -a "${log}"
  nohup bash "${ROOT}/run_v15_3_one.sh" "${case_name}" >>"${log}" 2>&1 &
  echo "$!" > "${out}/train.pid"
  echo "${case_name}: pid $(cat "${out}/train.pid")"
  sleep 5
done
