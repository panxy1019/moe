#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V11/test_results_v11"
SCRIPT="${ROOT}/deep_moe_rom_v11.py"
name="smoke_v11_operator_space"
out="${ROOT}/smoke"

mkdir -p "${out}"
echo "===== ${name} $(date -Is) =====" | tee "${out}/run.log"
"${PY}" "${SCRIPT}" \
  --output-dir "${out}" \
  --experiment-name "${name}" \
  --test-re-indices 10 \
  --r-u 16 \
  --r-p 16 \
  --num-blocks 2 \
  --num-experts 8 \
  --num-shared-experts 0 \
  --top-k 2 \
  --hidden-dim 144 \
  --expert-hidden 224 \
  --batch-size 768 \
  --recon-dim 512 \
  --rollout-steps 4 \
  --train-rollout-steps 2 \
  --rollout-batch 1 \
  --rollout-every-batches 6 \
  --curriculum-steps 1,2 \
  --dropout 0.035 \
  --temperature 1.05 \
  --gate-floor 0.10 \
  --lambda-pressure 0.65 \
  --lambda-rollout 0.02 \
  --lambda-pressure-rollout 0.30 \
  --lambda-router-balance 0.06 \
  --lambda-router-entropy -0.001 \
  --lambda-router-smooth 0.04 \
  --lambda-expert-diversity 0.005 \
  --lambda-regime-router 0.002 \
  --lambda-alpha-rel 0.06 \
  --lambda-rhs-rel 0.06 \
  --lambda-pressure-rel 0.35 \
  --rollout-relative-mix 0.25 \
  --relative-floor-frac 0.06 \
  --lr 7.5e-4 \
  --weight-decay 1.2e-4 \
  --epochs 2 \
  --min-epochs 1 \
  --patience 2 \
  --eval-every 1 \
  --early-stop-min-delta 8e-4 \
  --device cuda 2>&1 | tee -a "${out}/run.log"
echo "===== ${name} done $(date -Is) =====" | tee -a "${out}/run.log"
