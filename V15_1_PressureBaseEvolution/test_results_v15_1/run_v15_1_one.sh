#!/usr/bin/env bash
set -euo pipefail

case_name="${1:?usage: run_v15_1_one.sh V15_1_AdaptiveGate|V15_1_FiLMBase|V15_1_RegimeAwareROM}"

case "${case_name}" in
  V15_1_AdaptiveGate)
    closure_mode="adaptive_gate"
    pressure_base_mode="static"
    ;;
  V15_1_FiLMBase)
    closure_mode="baseline"
    pressure_base_mode="film_base"
    ;;
  V15_1_RegimeAwareROM)
    closure_mode="baseline"
    pressure_base_mode="regime_aware_rom"
    ;;
  *)
    echo "unknown V15_1 case: ${case_name}" >&2
    exit 2
    ;;
esac

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1"
SCRIPT="${ROOT}/train_v15_1_pressure_base_evolution.py"
name="${case_name}_physics_generalizable_ru16_rp16"
out="${ROOT}/results/${case_name}/${name}"
TRACKING_MODE="${SWANLAB_TRACKING_MODE:-online}"
TRACKING_PROJECT="${SWANLAB_TRACKING_PROJECT:-${case_name}}"
TRACKING_GROUP="${SWANLAB_TRACKING_GROUP:-v15-1-pressure-base-evolution}"
TRACKING_LOG_DIR="${SWANLAB_TRACKING_LOG_DIR:-${ROOT}/results/swanlog/${case_name}}"

if [[ "${TRACKING_MODE}" == "online" && -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY is required when SWANLAB_TRACKING_MODE=online." >&2
  exit 4
fi

"${PY}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA is not available; refusing to run V15_1 training on CPU.", file=sys.stderr)
    sys.exit(3)
print("CUDA ready:", torch.cuda.get_device_name(0))
PY

mkdir -p "${out}" "${TRACKING_LOG_DIR}"
echo "===== ${name} $(date -Is) =====" | tee "${out}/run.log"
"${PY}" "${SCRIPT}" \
  --output-dir "${out}" \
  --experiment-name "${name}" \
  --experiment-tag "${case_name}" \
  --data-root /root/moe/ROM_PhysicsGeneralizable/data/Global_POD_AreaWeighted_L2 \
  --tensor-path /root/moe/ROM_PhysicsGeneralizable/data/semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz \
  --pressure-surrogate-path /root/moe/ROM_PhysicsGeneralizable/data/pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz \
  --regime-rom-root /root/moe/ROM_PhysicsGeneralizable/Regime_ROM_Library \
  --test-re-selection regime_default \
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
  --closure-mode "${closure_mode}" \
  --pressure-base-mode "${pressure_base_mode}" \
  --film-base-hidden 64 \
  --film-base-scale 0.20 \
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
  --epochs "${V15_1_EPOCHS:-240}" \
  --min-epochs "${V15_1_MIN_EPOCHS:-130}" \
  --patience 70 \
  --eval-every "${V15_1_EVAL_EVERY:-5}" \
  --eval-routing-every 20 \
  --early-stop-min-delta 8e-4 \
  --allow-tf32 \
  2>&1 | tee -a "${out}/run.log"
