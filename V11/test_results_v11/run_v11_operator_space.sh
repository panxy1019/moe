#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V11/test_results_v11"
SCRIPT="${ROOT}/deep_moe_rom_v11.py"

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
    --num-shared-experts 0 \
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
    --temperature 1.05 \
    --gate-floor 0.10 \
    --lambda-pressure 0.65 \
    --lambda-rollout 0.14 \
    --lambda-pressure-rollout 0.35 \
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
    --epochs 200 \
    --min-epochs 100 \
    --patience 40 \
    --eval-every 5 \
    --early-stop-min-delta 8e-4 \
    --device cuda \
    "$@" 2>&1 | tee -a "${out}/run.log"
  echo "===== ${name} done $(date -Is) =====" | tee -a "${out}/run.log"
}

run_exp v11_r16_operator_space_b2
