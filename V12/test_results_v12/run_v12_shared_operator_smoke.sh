#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V12/test_results_v12"
SCRIPT="${ROOT}/deep_moe_rom_v12.py"
name="smoke_v12_shared_operator"
out="${ROOT}/smoke"

mkdir -p "${out}"
echo "===== ${name} $(date -Is) =====" | tee "${out}/run.log"
"${PY}" "${SCRIPT}" \
  --output-dir "${out}" \
  --experiment-name "${name}" \
  --test-re-indices 10 \
  --r-u 16 \
  --r-p 16 \
  --num-blocks 3 \
  --num-experts 12 \
  --num-shared-experts 2 \
  --top-k 3 \
  --hidden-dim 160 \
  --expert-hidden 256 \
  --batch-size 768 \
  --recon-dim 512 \
  --rollout-steps 4 \
  --train-rollout-steps 2 \
  --rollout-batch 1 \
  --rollout-every-batches 6 \
  --curriculum-steps 1,2 \
  --dropout 0.035 \
  --temperature 1.0 \
  --gate-floor 0.06 \
  --shared-scale 0.55 \
  --routed-scale 1.0 \
  --rhs-target residual \
  --pressure-target closure \
  --lambda-pressure 0.80 \
  --lambda-rollout 0.02 \
  --lambda-pressure-rollout 0.30 \
  --lambda-router-balance 0.035 \
  --lambda-router-entropy -0.002 \
  --lambda-router-smooth 0.04 \
  --lambda-expert-diversity 0.008 \
  --lambda-regime-router 0.003 \
  --lambda-alpha-rel 0.06 \
  --lambda-rhs-rel 0.08 \
  --lambda-pressure-rel 0.55 \
  --rollout-relative-mix 0.25 \
  --relative-floor-frac 0.06 \
  --lr 7.0e-4 \
  --weight-decay 1.2e-4 \
  --epochs 2 \
  --min-epochs 1 \
  --patience 2 \
  --eval-every 1 \
  --early-stop-min-delta 8e-4 \
  --device cuda 2>&1 | tee -a "${out}/run.log"
echo "===== ${name} done $(date -Is) =====" | tee -a "${out}/run.log"
