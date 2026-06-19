#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT=/root/moe/V8/test_results_v8
SCRIPT=${ROOT}/deep_moe_rom_v8.py

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
    --batch-size 768 \
    --recon-dim 1024 \
    --rollout-steps 12 \
    --train-rollout-steps 6 \
    --rollout-batch 1 \
    --rollout-every-batches 4 \
    --curriculum-steps 1,2,4,6 \
    --num-experts 6 \
    --top-k 2 \
    --hidden-dim 128 \
    --expert-hidden 192 \
    --dropout 0.04 \
    --temperature 0.8 \
    --lambda-pressure 0.60 \
    --lambda-pressure-rollout 0.35 \
    --lambda-router-smooth 0.05 \
    --device cuda \
    "$@" 2>&1 | tee -a "${out}/run.log"
  echo "===== ${name} done $(date -Is) =====" | tee -a "${out}/run.log"
}

run_exp v8_r16_rk4_b2_surres \
  --r-u 16 --r-p 16 --num-blocks 2 \
  --epochs 28 --min-epochs 18 --patience 6

run_exp v8_r32_rk4_b3_surres \
  --r-u 32 --r-p 32 --num-blocks 3 \
  --epochs 24 --min-epochs 16 --patience 6
