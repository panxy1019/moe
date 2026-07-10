#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/moe/V16_2_AttractorStabilityMoE/test_results_v16_2"
cd "${ROOT}"

cases=(
  V16_2_SteadyContractivePressureROM32
  V16_2_HopfLogRadiusNormalForm32
  V16_2_RegimeGroupedMoE32
)

mkdir -p "${ROOT}/results/pids" "${ROOT}/results/nohup"

for case_name in "${cases[@]}"; do
  log="${ROOT}/results/nohup/${case_name}.nohup.log"
  pid_file="${ROOT}/results/pids/${case_name}.pid"
  echo "starting ${case_name}; log=${log}"
  (
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export SWANLAB_TRACKING_MODE="${SWANLAB_TRACKING_MODE:-online}"
    export SWANLAB_TRACKING_PROJECT="${SWANLAB_TRACKING_PROJECT:-V16_2_AttractorStabilityMoE}"
    export SWANLAB_TRACKING_GROUP="${SWANLAB_TRACKING_GROUP:-v16-2-attractor-stability-moe}"
    ./run_v16_2_one.sh "${case_name}"
  ) >"${log}" 2>&1 &
  echo $! >"${pid_file}"
  sleep 8
done

echo "launched ${#cases[@]} V16_2 jobs"
for case_name in "${cases[@]}"; do
  echo "${case_name}: pid=$(cat "${ROOT}/results/pids/${case_name}.pid")"
done
