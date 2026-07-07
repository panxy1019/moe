#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V14/test_results_v14"
SCRIPT="${ROOT}/train_v14.py"
name="smoke_v14_hprs_moe"
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
  --num-regime-groups 3 \
  --experts-per-group 2 \
  --num-experts 2 \
  --num-shared-experts 1 \
  --top-k 2 \
  --group-top-k 1 \
  --hidden-dim 96 \
  --expert-hidden 192 \
  --expert-blocks 1 \
  --quadratic-rank 2 \
  --quadratic-scale 0.04 \
  --batch-size 512 \
  --recon-dim 512 \
  --rollout-steps 6 \
  --train-rollout-steps 4 \
  --rollout-batch 1 \
  --rollout-every-batches 2 \
  --curriculum-steps 2,4 \
  --dropout 0.04 \
  --temperature 0.95 \
  --gate-floor 0.0 \
  --group-temperature 0.90 \
  --group-gate-floor 0.0 \
  --shared-scale 1.0 \
  --routed-scale 0.85 \
  --rhs-target residual \
  --pressure-target closure \
  --lambda-pressure 0.85 \
  --lambda-rollout 0.05 \
  --lambda-energy 0.02 \
  --lambda-trajectory-consistency 0.04 \
  --lambda-pressure-rollout 0.30 \
  --lambda-router-balance 0.06 \
  --lambda-router-entropy -0.002 \
  --lambda-group-balance 0.02 \
  --lambda-group-entropy 0.0 \
  --lambda-group-supervision 0.02 \
  --lambda-router-smooth 0.03 \
  --lambda-expert-diversity 0.004 \
  --lambda-regime-router 0.002 \
  --lambda-alpha-rel 0.05 \
  --lambda-rhs-rel 0.07 \
  --lambda-pressure-rel 0.60 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.5 \
  --scheduled-sampling-warmup-frac 0.7 \
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
