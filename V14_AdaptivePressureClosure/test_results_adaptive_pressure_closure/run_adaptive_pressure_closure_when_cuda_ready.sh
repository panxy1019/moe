#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V14_AdaptivePressureClosure/test_results_adaptive_pressure_closure"
AGG_DIR="${ROOT}/results/v14_adaptive_pressure_closure_summary"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-600}"

mkdir -p "${AGG_DIR}"
log="${AGG_DIR}/cuda_wait_launcher.log"
echo "===== wait for CUDA $(date -Is) =====" | tee -a "${log}"

while true; do
  if "${PY}" - <<'PY' >/tmp/v14_adaptive_cuda_check.txt 2>&1
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
print(torch.cuda.get_device_name(0))
PY
  then
    echo "CUDA ready $(date -Is): $(cat /tmp/v14_adaptive_cuda_check.txt)" | tee -a "${log}"
    exec bash "${ROOT}/run_adaptive_pressure_closure_all.sh"
  fi
  echo "CUDA unavailable $(date -Is); sleeping ${CHECK_INTERVAL_SECONDS}s" | tee -a "${log}"
  cat /tmp/v14_adaptive_cuda_check.txt | tail -n 4 >> "${log}" || true
  sleep "${CHECK_INTERVAL_SECONDS}"
done
