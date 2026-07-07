#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V14_2_DataAblation/test_results_v14_2"
SCRIPT="${ROOT}/train_v14_2.py"
name="smoke_v14_2_data_ablation"
out="${ROOT}/smoke/${name}"

mkdir -p "${out}"
echo "===== ${name} $(date -Is) =====" | tee "${out}/run.log"
"${PY}" "${SCRIPT}" \
  --output-dir "${out}" \
  --experiment-name "${name}" \
  --test-re-selection uniform \
  --num-uniform-test-re 3 \
  --train-time-stride 5 \
  --train-re-stride 1 \
  --r-u 16 \
  --r-p 16 \
  --num-blocks 1 \
  --num-regime-groups 3 \
  --experts-per-group 2 \
  --num-experts 2 \
  --num-shared-experts 1 \
  --top-k 1 \
  --group-top-k 1 \
  --hidden-dim 64 \
  --expert-hidden 128 \
  --expert-blocks 1 \
  --quadratic-rank 2 \
  --batch-size 128 \
  --recon-dim 256 \
  --rollout-steps 8 \
  --train-rollout-steps 4 \
  --rollout-batch 1 \
  --curriculum-steps 2,4 \
  --epochs 2 \
  --min-epochs 1 \
  --patience 2 \
  --eval-every 1 \
  --device cuda 2>&1 | tee -a "${out}/run.log"
echo "===== ${name} done $(date -Is) =====" | tee -a "${out}/run.log"
