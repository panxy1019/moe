#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1"
cases=(V15_1_AdaptiveGate V15_1_FiLMBase V15_1_RegimeAwareROM)
mkdir -p "${ROOT}/results"

for case_name in "${cases[@]}"; do
  name="${case_name}_physics_generalizable_ru16_rp16"
  out="${ROOT}/results/${case_name}/${name}"
  metrics="${out}/${name}_metrics.json"
  if [[ -f "${metrics}" ]]; then
    echo "${case_name}: metrics already exist, skipping"
    continue
  fi
  if pgrep -af "train_v15_1_pressure_base_evolution.py.*--experiment-tag ${case_name}" >/dev/null; then
    echo "${case_name}: already running, skipping"
    continue
  fi
  mkdir -p "${out}"
  log="${out}/nohup.log"
  echo "${case_name}: starting $(date -Is)" | tee -a "${log}"
  nohup bash "${ROOT}/run_v15_1_one.sh" "${case_name}" >>"${log}" 2>&1 &
  echo "$!" > "${out}/train.pid"
  echo "${case_name}: pid $(cat "${out}/train.pid")"
  sleep 5
done
