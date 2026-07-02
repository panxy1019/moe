#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V14_3_PressureInputAblation/test_results_v14_3"
RESULT_ROOT="${ROOT}/results"
AGG_DIR="${RESULT_ROOT}/v14_3_pressure_input_ablation_summary"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-1}"
MAX_SHARED_UTIL="${MAX_SHARED_UTIL:-35}"
MAX_SHARED_MEM_FRAC="${MAX_SHARED_MEM_FRAC:-0.35}"

gpu_snapshot() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1
}

wait_for_gpu() {
  while true; do
    line="$(gpu_snapshot || true)"
    if [[ -z "${line}" ]]; then
      echo "No nvidia-smi GPU snapshot; continuing with CUDA default."
      return 0
    fi
    IFS=',' read -r gpu util mem total <<< "${line}"
    gpu="${gpu// /}"
    util="${util// /}"
    mem="${mem// /}"
    total="${total// /}"
    busy_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l)"
    mem_frac="$("${PY}" - <<PY
mem=float("${mem}")
total=max(float("${total}"), 1.0)
print(mem/total)
PY
)"
    if [[ "${busy_count}" -eq 0 ]]; then
      export CUDA_VISIBLE_DEVICES="${gpu}"
      echo "Using idle GPU ${gpu}: util=${util}%, mem=${mem}/${total} MiB."
      return 0
    fi
    if [[ "${ALLOW_SHARED_GPU}" == "1" ]]; then
      ok="$("${PY}" - <<PY
util=float("${util}")
mem_frac=float("${mem_frac}")
print("1" if util <= float("${MAX_SHARED_UTIL}") and mem_frac <= float("${MAX_SHARED_MEM_FRAC}") else "0")
PY
)"
      if [[ "${ok}" == "1" ]]; then
        export CUDA_VISIBLE_DEVICES="${gpu}"
        echo "Sharing GPU ${gpu}: util=${util}%, mem=${mem}/${total} MiB, processes=${busy_count}."
        return 0
      fi
    fi
    echo "GPU busy: util=${util}%, mem=${mem}/${total} MiB, processes=${busy_count}; sleeping 10 min."
    sleep 600
  done
}

mkdir -p "${AGG_DIR}"
echo "===== V14_3 pressure input ablation all $(date -Is) =====" | tee "${AGG_DIR}/run_all.log"

for mode in velocity_only pressure_only hybrid; do
  wait_for_gpu | tee -a "${AGG_DIR}/run_all.log"
  bash "${ROOT}/run_v14_3_pressure_input_one.sh" "${mode}" 2>&1 | tee -a "${AGG_DIR}/run_all.log"
done

"${PY}" "${ROOT}/aggregate_pressure_input_ablation.py" \
  --output-dir "${AGG_DIR}" \
  "${RESULT_ROOT}/v14_3_pressure_input_pressure_only_dense_uniform10/v14_3_pressure_input_pressure_only_dense_uniform10_metrics.json" \
  "${RESULT_ROOT}/v14_3_pressure_input_velocity_only_dense_uniform10/v14_3_pressure_input_velocity_only_dense_uniform10_metrics.json" \
  "${RESULT_ROOT}/v14_3_pressure_input_hybrid_dense_uniform10/v14_3_pressure_input_hybrid_dense_uniform10_metrics.json" \
  2>&1 | tee -a "${AGG_DIR}/run_all.log"

echo "===== V14_3 pressure input ablation all done $(date -Is) =====" | tee -a "${AGG_DIR}/run_all.log"
