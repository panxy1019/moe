#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: run_adaptive_pressure_closure_one.sh residual_scaling|base_scaling|dual_adaptive}"
case "${mode}" in
  residual_scaling|base_scaling|dual_adaptive) ;;
  *) echo "unknown closure mode: ${mode}" >&2; exit 2 ;;
esac

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V14_AdaptivePressureClosure/test_results_adaptive_pressure_closure"
SCRIPT="${ROOT}/train_adaptive_pressure_closure.py"
name="v14_adaptive_pressure_closure_${mode}_dense_uniform10"
out="${ROOT}/results/${name}"
TRACKING_MODE="${SWANLAB_TRACKING_MODE:-online}"
TRACKING_PROJECT="${SWANLAB_TRACKING_PROJECT:-V14_AdaptivePressureClosure}"
TRACKING_GROUP="${SWANLAB_TRACKING_GROUP:-adaptive-pressure-closure}"
TRACKING_LOG_DIR="${SWANLAB_TRACKING_LOG_DIR:-${ROOT}/results/swanlog}"

if [[ "${TRACKING_MODE}" == "online" && -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY is required when SWANLAB_TRACKING_MODE=online." >&2
  exit 4
fi
# SwanLab itself reads SWANLAB_* environment variables during import. Keep only
# the API key in the environment; pass project/group/mode through CLI flags.
unset SWANLAB_MODE SWANLAB_PROJECT SWANLAB_GROUP SWANLAB_LOG_DIR

"${PY}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA is not available; refusing to run adaptive closure training on CPU.", file=sys.stderr)
    sys.exit(3)
print("CUDA ready:", torch.cuda.get_device_name(0))
PY

mkdir -p "${out}"
echo "===== ${name} $(date -Is) =====" | tee "${out}/run.log"
"${PY}" "${SCRIPT}" \
  --output-dir "${out}" \
  --experiment-name "${name}" \
  --test-re-selection uniform \
  --num-uniform-test-re 10 \
  --train-time-stride 1 \
  --train-re-stride 1 \
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
  --shared-scale 1.0 \
  --routed-scale 0.85 \
  --rhs-target residual \
  --pressure-target closure \
  --pressure-input-mode pressure_only \
  --closure-mode "${mode}" \
  --swanlab-mode "${TRACKING_MODE}" \
  --swanlab-project "${TRACKING_PROJECT}" \
  --swanlab-group "${TRACKING_GROUP}" \
  --swanlab-run-name "v14_apc_${mode}" \
  --swanlab-log-dir "${TRACKING_LOG_DIR}" \
  --swanlab-required \
  --lambda-coeff 0.75 \
  --lambda-dyn 0.90 \
  --lambda-pressure 0.95 \
  --lambda-rollout 0.45 \
  --lambda-energy 0.05 \
  --lambda-trajectory-consistency 0.18 \
  --lambda-pressure-rollout 0.45 \
  --lambda-router-balance 0.06 \
  --lambda-router-entropy -0.002 \
  --lambda-group-balance 0.04 \
  --lambda-group-entropy 0.0 \
  --lambda-group-supervision 0.04 \
  --lambda-router-smooth 0.04 \
  --lambda-expert-diversity 0.006 \
  --lambda-regime-router 0.004 \
  --lambda-alpha-rel 0.04 \
  --lambda-rhs-rel 0.06 \
  --lambda-pressure-rel 0.70 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.85 \
  --scheduled-sampling-warmup-frac 0.70 \
  --rollout-relative-mix 0.35 \
  --relative-floor-frac 0.05 \
  --lr 5.5e-4 \
  --weight-decay 1.5e-4 \
  --epochs 240 \
  --min-epochs 130 \
  --patience 70 \
  --eval-every 5 \
  --early-stop-min-delta 8e-4 \
  --allow-tf32 \
  --device cuda 2>&1 | tee -a "${out}/run.log"
echo "===== ${name} done $(date -Is) =====" | tee -a "${out}/run.log"
