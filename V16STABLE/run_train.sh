#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOE_ROOT="${MOE_ROOT:-/root/moe}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/pt_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-${MOE_ROOT}/ROM_PhysicsGeneralizable/data/Global_POD_AreaWeighted_L2}"
TENSOR_PATH="${TENSOR_PATH:-${MOE_ROOT}/ROM_PhysicsGeneralizable/data/semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz}"
PRESSURE_TENSOR_PATH="${PRESSURE_TENSOR_PATH:-${MOE_ROOT}/ROM_PhysicsGeneralizable/data/pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/V16_1_SteadyPressureAnchor32_ru32_rp32}"
SWANLAB_MODE="${SWANLAB_TRACKING_MODE:-online}"
SWANLAB_PROJECT="${SWANLAB_TRACKING_PROJECT:-V16STABLE}"
SWANLAB_GROUP="${SWANLAB_TRACKING_GROUP:-v16-1-steady-pressure-anchor32}"
SWANLAB_LOG_DIR="${SWANLAB_TRACKING_LOG_DIR:-${SCRIPT_DIR}/results/swanlog}"

for required in \
  "${DATA_ROOT}/pod_snapshot_index.csv" \
  "${DATA_ROOT}/global_velocity_pod_area_weighted_l2.npz" \
  "${DATA_ROOT}/global_pressure_pod_area_weighted_l2.npz" \
  "${TENSOR_PATH}" \
  "${PRESSURE_TENSOR_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required input: ${required}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("CUDA is not available; refusing to run training on CPU.", file=sys.stderr)
    sys.exit(3)
print("CUDA ready:", torch.cuda.get_device_name(0))
PY

mkdir -p "${OUTPUT_DIR}" "${SWANLAB_LOG_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/train_v16_1_steady_pressure_anchor32.py" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name V16_1_SteadyPressureAnchor32_ru32_rp32 \
  --experiment-tag V16_1_SteadyPressureAnchor32 \
  --data-root "${DATA_ROOT}" \
  --tensor-path "${TENSOR_PATH}" \
  --pressure-surrogate-path "${PRESSURE_TENSOR_PATH}" \
  --test-re-selection regime_default \
  --train-time-stride 1 \
  --train-re-stride 1 \
  --r-u 32 \
  --r-p 32 \
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
  --closure-mode adaptive_gate \
  --pressure-base-mode static \
  --attractor-balanced-sampling \
  --hopf-pair 0,1 \
  --hopf-re-min 47.0 \
  --hopf-re-max 59.5 \
  --hopf-focus-re 51.78645 \
  --hopf-focus-width 0.85 \
  --hopf-loss-warmup-epochs 30 \
  --hopf-overshoot-ratio 2.0 \
  --lambda-hopf-log-amp 0.0 \
  --lambda-hopf-energy 0.0 \
  --lambda-hopf-overshoot 0.0 \
  --hopf-sample-weight 1.0 \
  --hopf-focus-sample-weight 1.0 \
  --hopf-rollout-weight 1.0 \
  --hopf-diagnostic-horizons 8,16,24,32 \
  --lambda-steady-rhs 0.02 \
  --lambda-steady-state 0.01 \
  --lambda-attractor-hopf-radius 0.02 \
  --lambda-attractor-hopf-overshoot 0.02 \
  --lambda-attractor-hopf-onset 0.01 \
  --v16-1-hopf-floor-quantile 10 \
  --v16-1-loss-warmup-epochs 0 \
  --lambda-v16-1-hopf-growth 0.0 \
  --lambda-v16-1-hopf-false-growth 0.0 \
  --lambda-v16-1-hopf-floor-rel 0.0 \
  --lambda-v16-1-steady-p-state 0.02 \
  --lambda-v16-1-steady-p-mean 0.02 \
  --lambda-v16-1-steady-p-delta 0.01 \
  --lambda-v16-1-steady-residual-damp 0.01 \
  --lambda-v16-1-steady-p-energy 0.01 \
  --lambda-periodic-energy 0.02 \
  --lambda-periodic-radius 0.01 \
  --lambda-attractor-ce 0.0 \
  --lambda-proto-energy 0.0 \
  --lambda-proto-radius 0.0 \
  --attractor-r-floor 0.0 \
  --attractor-onset-threshold 5e-5 \
  --swanlab-mode "${SWANLAB_MODE}" \
  --swanlab-project "${SWANLAB_PROJECT}" \
  --swanlab-group "${SWANLAB_GROUP}" \
  --swanlab-run-name V16_1_SteadyPressureAnchor32 \
  --swanlab-log-dir "${SWANLAB_LOG_DIR}" \
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
  --seed "${V16STABLE_SEED:-1600}" \
  --epochs "${V16STABLE_EPOCHS:-240}" \
  --min-epochs "${V16STABLE_MIN_EPOCHS:-130}" \
  --patience 70 \
  --eval-every "${V16STABLE_EVAL_EVERY:-5}" \
  --eval-routing-every 20 \
  --early-stop-min-delta 8e-4 \
  --allow-tf32 \
  "$@"
