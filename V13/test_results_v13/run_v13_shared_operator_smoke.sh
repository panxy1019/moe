#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V13/test_results_v13"
SCRIPT="${ROOT}/train_v13.py"
name="smoke_v13_hierarchical_moe"
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
  --num-experts 8 \
  --num-shared-experts 2 \
  --top-k 2 \
  --hidden-dim 128 \
  --expert-hidden 512 \
  --expert-blocks 2 \
  --quadratic-rank 2 \
  --quadratic-scale 0.04 \
  --batch-size 512 \
  --recon-dim 512 \
  --rollout-steps 4 \
  --train-rollout-steps 2 \
  --rollout-batch 1 \
  --rollout-every-batches 8 \
  --curriculum-steps 1,2 \
  --dropout 0.04 \
  --temperature 1.05 \
  --gate-floor 0.0 \
  --shared-scale 0.65 \
  --routed-scale 1.0 \
  --rhs-target residual \
  --pressure-target closure \
  --lambda-pressure 0.85 \
  --lambda-rollout 0.01 \
  --lambda-pressure-rollout 0.30 \
  --lambda-router-balance 0.06 \
  --lambda-router-entropy -0.002 \
  --lambda-router-smooth 0.03 \
  --lambda-expert-diversity 0.004 \
  --lambda-regime-router 0.002 \
  --lambda-alpha-rel 0.05 \
  --lambda-rhs-rel 0.07 \
  --lambda-pressure-rel 0.60 \
  --rollout-relative-mix 0.25 \
  --relative-floor-frac 0.05 \
  --lr 6.5e-4 \
  --weight-decay 1.5e-4 \
  --epochs 2 \
  --min-epochs 1 \
  --patience 2 \
  --eval-every 1 \
  --early-stop-min-delta 8e-4 \
  --device cuda 2>&1 | tee -a "${out}/run.log"
echo "===== ${name} done $(date -Is) =====" | tee -a "${out}/run.log"
