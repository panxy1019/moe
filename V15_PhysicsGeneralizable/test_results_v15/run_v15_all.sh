#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V15_PhysicsGeneralizable/test_results_v15"
RESULT_ROOT="${ROOT}/results"
AGG_DIR="${RESULT_ROOT}/V15_summary"

mkdir -p "${AGG_DIR}"
echo "===== V15 PhysicsGeneralizable all $(date -Is) =====" | tee "${AGG_DIR}/run_all.log"

cases=(V15_Base V15_LargeROM V15_BalancedTraining)
pids=()
for case_name in "${cases[@]}"; do
  case "${case_name}" in
    V15_LargeROM) ru=32; rp=32 ;;
    *) ru=16; rp=16 ;;
  esac
  name="${case_name}_physics_generalizable_ru${ru}_rp${rp}"
  metrics="${RESULT_ROOT}/${case_name}/${name}/${name}_metrics.json"
  if [[ -f "${metrics}" ]]; then
    echo "skip ${case_name}: metrics already exist" | tee -a "${AGG_DIR}/run_all.log"
    continue
  fi
  if pgrep -af "train_v15_physics_generalizable.py.*--experiment-tag ${case_name}" >/dev/null; then
    echo "skip ${case_name}: already running" | tee -a "${AGG_DIR}/run_all.log"
    continue
  fi
  echo "start ${case_name} $(date -Is)" | tee -a "${AGG_DIR}/run_all.log"
  bash "${ROOT}/run_v15_one.sh" "${case_name}" \
    > "${AGG_DIR}/manual_${case_name}.log" 2>&1 &
  pids+=("$!")
  sleep 15
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more V15 runs failed" | tee -a "${AGG_DIR}/run_all.log"
  exit "${status}"
fi

agg_args=(--output-dir "${AGG_DIR}")
for case_name in "${cases[@]}"; do
  case "${case_name}" in
    V15_LargeROM) ru=32; rp=32 ;;
    *) ru=16; rp=16 ;;
  esac
  name="${case_name}_physics_generalizable_ru${ru}_rp${rp}"
  agg_args+=("${RESULT_ROOT}/${case_name}/${name}/${name}_metrics.json")
done

"${PY}" "${ROOT}/aggregate_v15_physics_generalizable.py" "${agg_args[@]}" \
  2>&1 | tee -a "${AGG_DIR}/run_all.log"

echo "===== V15 PhysicsGeneralizable all done $(date -Is) =====" | tee -a "${AGG_DIR}/run_all.log"
