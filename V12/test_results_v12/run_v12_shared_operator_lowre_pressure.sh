#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V12/test_results_v12"
SCRIPT="${ROOT}/deep_moe_rom_v12.py"
name="v12_r16_shared_operator_lowre_pressure"
out="${ROOT}/results/${name}"

mkdir -p "${out}"
echo "===== ${name} $(date -Is) =====" | tee "${out}/run.log"
"${PY}" "${SCRIPT}" \
  --output-dir "${out}" \
  --experiment-name "${name}" \
  --test-re-indices 10 \
  --r-u 16 \
  --r-p 16 \
  --num-blocks 3 \
  --num-experts 6 \
  --num-shared-experts 2 \
  --top-k 2 \
  --hidden-dim 160 \
  --expert-hidden 256 \
  --batch-size 768 \
  --recon-dim 2048 \
  --rollout-steps 16 \
  --train-rollout-steps 8 \
  --rollout-batch 2 \
  --rollout-every-batches 3 \
  --curriculum-steps 1,2,4,8 \
  --dropout 0.035 \
  --temperature 1.05 \
  --gate-floor 0.08 \
  --shared-scale 0.70 \
  --routed-scale 0.90 \
  --rhs-target residual \
  --pressure-target closure \
  --lambda-pressure 1.20 \
  --lambda-rollout 0.10 \
  --lambda-pressure-rollout 0.45 \
  --lambda-router-balance 0.05 \
  --lambda-router-entropy -0.002 \
  --lambda-router-smooth 0.04 \
  --lambda-expert-diversity 0.006 \
  --lambda-regime-router 0.003 \
  --lambda-alpha-rel 0.05 \
  --lambda-rhs-rel 0.07 \
  --lambda-pressure-rel 1.10 \
  --rollout-relative-mix 0.30 \
  --relative-floor-frac 0.03 \
  --lr 7.0e-4 \
  --weight-decay 1.2e-4 \
  --epochs 170 \
  --min-epochs 90 \
  --patience 35 \
  --eval-every 5 \
  --early-stop-min-delta 8e-4 \
  --device cuda 2>&1 | tee -a "${out}/run.log"
echo "===== ${name} done $(date -Is) =====" | tee -a "${out}/run.log"
