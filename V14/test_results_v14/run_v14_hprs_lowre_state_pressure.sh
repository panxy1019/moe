#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V14/test_results_v14"
SCRIPT="${ROOT}/train_v14.py"
name="v14_r16_hprs_lowre_state_pressure"
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
  --num-regime-groups 3 \
  --experts-per-group 6 \
  --num-experts 6 \
  --num-shared-experts 1 \
  --top-k 2 \
  --group-top-k 1 \
  --hidden-dim 224 \
  --expert-hidden 768 \
  --expert-blocks 3 \
  --quadratic-rank 4 \
  --quadratic-scale 0.05 \
  --batch-size 256 \
  --recon-dim 2048 \
  --rollout-steps 24 \
  --train-rollout-steps 16 \
  --rollout-batch 2 \
  --rollout-every-batches 1 \
  --curriculum-steps 4,8,12,16 \
  --dropout 0.04 \
  --temperature 0.95 \
  --gate-floor 0.0 \
  --group-temperature 0.90 \
  --group-gate-floor 0.0 \
  --shared-scale 1.05 \
  --routed-scale 0.80 \
  --rhs-target residual \
  --pressure-target state \
  --lambda-coeff 0.75 \
  --lambda-dyn 0.90 \
  --lambda-pressure 1.45 \
  --lambda-rollout 0.42 \
  --lambda-energy 0.06 \
  --lambda-trajectory-consistency 0.20 \
  --lambda-pressure-rollout 0.65 \
  --lambda-router-balance 0.07 \
  --lambda-router-entropy -0.002 \
  --lambda-group-balance 0.05 \
  --lambda-group-entropy 0.0 \
  --lambda-group-supervision 0.05 \
  --lambda-router-smooth 0.04 \
  --lambda-expert-diversity 0.006 \
  --lambda-regime-router 0.004 \
  --lambda-alpha-rel 0.04 \
  --lambda-rhs-rel 0.06 \
  --lambda-pressure-rel 1.20 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.85 \
  --scheduled-sampling-warmup-frac 0.70 \
  --rollout-relative-mix 0.35 \
  --relative-floor-frac 0.03 \
  --lr 5.5e-4 \
  --weight-decay 1.5e-4 \
  --epochs 220 \
  --min-epochs 120 \
  --patience 60 \
  --eval-every 5 \
  --early-stop-min-delta 8e-4 \
  --device cuda 2>&1 | tee -a "${out}/run.log"
echo "===== ${name} done $(date -Is) =====" | tee -a "${out}/run.log"
