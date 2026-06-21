#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V9(2)/test_results_v9_2"
SCRIPT="${ROOT}/deep_moe_rom_v9_2.py"

mkdir -p "${ROOT}/results"

run_exp() {
  local name="$1"
  shift
  local out="${ROOT}/results/${name}"
  mkdir -p "${out}"
  echo "===== ${name} $(date -Is) =====" | tee "${out}/run.log"
  "${PY}" "${SCRIPT}" \
    --output-dir "${out}" \
    --experiment-name "${name}" \
    --test-re-indices 10 59 99 \
    --r-u 16 \
    --r-p 16 \
    --num-blocks 2 \
    --num-experts 8 \
    --top-k 2 \
    --hidden-dim 144 \
    --expert-hidden 224 \
    --batch-size 768 \
    --recon-dim 2048 \
    --rollout-steps 16 \
    --train-rollout-steps 8 \
    --rollout-batch 2 \
    --rollout-every-batches 3 \
    --curriculum-steps 1,2,4,8 \
    --dropout 0.035 \
    --temperature 1.15 \
    --lambda-pressure 0.65 \
    --lambda-pressure-rollout 0.35 \
    --lambda-router-balance 0.08 \
    --lambda-router-entropy -0.0015 \
    --lambda-router-smooth 0.05 \
    --lambda-alpha-rel 0.08 \
    --lambda-rhs-rel 0.08 \
    --lambda-pressure-rel 0.35 \
    --rollout-relative-mix 0.35 \
    --relative-floor-frac 0.05 \
    --lr 7.5e-4 \
    --weight-decay 1.2e-4 \
    --epochs 180 \
    --min-epochs 100 \
    --patience 35 \
    --eval-every 5 \
    --early-stop-min-delta 8e-4 \
    --device cuda \
    "$@" 2>&1 | tee -a "${out}/run.log"
  echo "===== ${name} done $(date -Is) =====" | tee -a "${out}/run.log"
}

run_exp v9_2_r16_rk4_b2_longstable
