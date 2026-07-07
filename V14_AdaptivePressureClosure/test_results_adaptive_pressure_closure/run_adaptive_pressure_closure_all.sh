#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V14_AdaptivePressureClosure/test_results_adaptive_pressure_closure"
RESULT_ROOT="${ROOT}/results"
AGG_DIR="${RESULT_ROOT}/v14_adaptive_pressure_closure_summary"
BASELINE_METRICS="${BASELINE_METRICS:-/root/moe/V14_3_PressureInputAblation/test_results_v14_3/results/v14_3_pressure_input_pressure_only_dense_uniform10/v14_3_pressure_input_pressure_only_dense_uniform10_metrics.json}"

mkdir -p "${AGG_DIR}"
echo "===== V14 adaptive pressure closure all $(date -Is) =====" | tee "${AGG_DIR}/run_all.log"

modes=(residual_scaling base_scaling dual_adaptive)
pids=()
for mode in "${modes[@]}"; do
  name="v14_adaptive_pressure_closure_${mode}_dense_uniform10"
  metrics="${RESULT_ROOT}/${name}/${name}_metrics.json"
  if [[ -f "${metrics}" ]]; then
    echo "skip ${mode}: metrics already exist" | tee -a "${AGG_DIR}/run_all.log"
    continue
  fi
  if pgrep -af "train_adaptive_pressure_closure.py.*--closure-mode ${mode}" >/dev/null; then
    echo "skip ${mode}: already running" | tee -a "${AGG_DIR}/run_all.log"
    continue
  fi
  echo "start ${mode} $(date -Is)" | tee -a "${AGG_DIR}/run_all.log"
  bash "${ROOT}/run_adaptive_pressure_closure_one.sh" "${mode}" \
    > "${AGG_DIR}/manual_${mode}.log" 2>&1 &
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
  echo "one or more adaptive closure runs failed" | tee -a "${AGG_DIR}/run_all.log"
  exit "${status}"
fi

agg_args=(--output-dir "${AGG_DIR}")
if [[ -f "${BASELINE_METRICS}" ]]; then
  agg_args+=(--baseline-metrics "${BASELINE_METRICS}")
else
  echo "baseline metrics not found at ${BASELINE_METRICS}; aggregating adaptive modes only" \
    | tee -a "${AGG_DIR}/run_all.log"
fi
for mode in "${modes[@]}"; do
  name="v14_adaptive_pressure_closure_${mode}_dense_uniform10"
  agg_args+=("${RESULT_ROOT}/${name}/${name}_metrics.json")
done

"${PY}" "${ROOT}/aggregate_adaptive_pressure_closure.py" "${agg_args[@]}" \
  2>&1 | tee -a "${AGG_DIR}/run_all.log"

echo "===== V14 adaptive pressure closure all done $(date -Is) =====" | tee -a "${AGG_DIR}/run_all.log"
