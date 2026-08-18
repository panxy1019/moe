#!/usr/bin/env bash
set -euo pipefail

case_name="${1:?usage: run_v15_one.sh V15_Base|V15_LargeROM|V15_BalancedTraining}"
case "${case_name}" in
  V15_Base)
    ru=16
    rp=16
    balanced_args=()
    ;;
  V15_LargeROM)
    ru=32
    rp=32
    balanced_args=()
    ;;
  V15_BalancedTraining)
    ru=16
    rp=16
    balanced_args=(--regime-balanced-sampling)
    ;;
  *)
    echo "unknown V15 case: ${case_name}" >&2
    exit 2
    ;;
esac

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V15_PhysicsGeneralizable/test_results_v15"
SCRIPT="${ROOT}/train_v15_physics_generalizable.py"
name="${case_name}_physics_generalizable_ru${ru}_rp${rp}"
out="${ROOT}/results/${case_name}/${name}"
TRACKING_MODE="${SWANLAB_TRACKING_MODE:-online}"
TRACKING_PROJECT="${SWANLAB_TRACKING_PROJECT:-V15_PhysicsGeneralizable}"
TRACKING_GROUP="${SWANLAB_TRACKING_GROUP:-v15-physics-generalizable}"
TRACKING_LOG_DIR="${SWANLAB_TRACKING_LOG_DIR:-${ROOT}/results/swanlog}"

if [[ "${TRACKING_MODE}" == "online" && -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY is required when SWANLAB_TRACKING_MODE=online." >&2
  exit 4
fi
unset SWANLAB_MODE SWANLAB_PROJECT SWANLAB_GROUP SWANLAB_LOG_DIR

"${PY}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA is not available; refusing to run V15 training on CPU.", file=sys.stderr)
    sys.exit(3)
print("CUDA ready:", torch.cuda.get_device_name(0))
PY

mkdir -p "${out}"
echo "===== ${name} $(date -Is) =====" | tee "${out}/run.log"
"${PY}" "${SCRIPT}" \
  --output-dir "${out}" \
  --experiment-name "${name}" \
  --experiment-tag "${case_name}" \
  --test-re-selection regime_default \
  --train-time-stride 1 \
  --train-re-stride 1 \
  --r-u "${ru}" \
  --r-p "${rp}" \
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
  --closure-mode baseline \
  "${balanced_args[@]}" \
  --swanlab-mode "${TRACKING_MODE}" \
  --swanlab-project "${TRACKING_PROJECT}" \
  --swanlab-group "${TRACKING_GROUP}" \
  --swanlab-run-name "${case_name}" \
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
