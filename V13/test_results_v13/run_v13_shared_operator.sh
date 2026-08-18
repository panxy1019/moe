#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V13/test_results_v13"
SCRIPT="${ROOT}/train_v13.py"
name="v13_r16_hier_closure_24x4_top2"
out="${ROOT}/results/${name}"

mkdir -p "${out}"
echo "===== ${name} $(date -Is) =====" | tee "${out}/run.log"
"${PY}" "${SCRIPT}" \
  --output-dir "${out}" \
  --experiment-name "${name}" \
  --test-re-indices 10 59 99 \
  --r-u 16 \
  --r-p 16 \
  --num-blocks 3 \
  --num-experts 24 \
  --num-shared-experts 4 \
  --top-k 2 \
  --hidden-dim 256 \
  --expert-hidden 1024 \
  --expert-blocks 4 \
  --quadratic-rank 4 \
  --quadratic-scale 0.05 \
  --batch-size 512 \
  --recon-dim 2048 \
  --rollout-steps 16 \
  --train-rollout-steps 8 \
  --rollout-batch 1 \
  --rollout-every-batches 4 \
  --curriculum-steps 1,2,4,8 \
  --dropout 0.04 \
  --temperature 1.05 \
  --gate-floor 0.0 \
  --shared-scale 0.65 \
  --routed-scale 1.0 \
  --rhs-target residual \
  --pressure-target closure \
  --lambda-pressure 0.90 \
  --lambda-rollout 0.10 \
  --lambda-pressure-rollout 0.40 \
  --lambda-router-balance 0.08 \
  --lambda-router-entropy -0.003 \
  --lambda-router-smooth 0.035 \
  --lambda-expert-diversity 0.004 \
  --lambda-regime-router 0.003 \
  --lambda-alpha-rel 0.05 \
  --lambda-rhs-rel 0.07 \
  --lambda-pressure-rel 0.75 \
  --rollout-relative-mix 0.28 \
  --relative-floor-frac 0.05 \
  --lr 6.5e-4 \
  --weight-decay 1.5e-4 \
  --epochs 180 \
  --min-epochs 100 \
  --patience 45 \
  --eval-every 5 \
  --early-stop-min-delta 8e-4 \
  --device cuda 2>&1 | tee -a "${out}/run.log"
echo "===== ${name} done $(date -Is) =====" | tee -a "${out}/run.log"
