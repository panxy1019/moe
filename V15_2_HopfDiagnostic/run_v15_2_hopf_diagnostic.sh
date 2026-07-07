#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V15_2_HopfDiagnostic"

"${PY}" "${ROOT}/v15_2_hopf_diagnostic.py" \
  --v15-module /root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/train_v15_1_pressure_base_evolution.py \
  --metrics-json /root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/results/V15_1_AdaptiveGate/V15_1_AdaptiveGate_physics_generalizable_ru16_rp16/V15_1_AdaptiveGate_physics_generalizable_ru16_rp16_metrics.json \
  --checkpoint /root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/results/V15_1_AdaptiveGate/V15_1_AdaptiveGate_physics_generalizable_ru16_rp16/V15_1_AdaptiveGate_physics_generalizable_ru16_rp16_Re_24p630436_checkpoint.pt \
  --output-dir "${ROOT}/results" \
  --device cuda
