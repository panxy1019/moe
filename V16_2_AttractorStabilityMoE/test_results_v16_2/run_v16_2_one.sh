#!/usr/bin/env bash
set -euo pipefail

case_name="${1:?usage: run_v16_2_one.sh V16_2_SteadyContractivePressureROM32|V16_2_HopfLogRadiusNormalForm32|V16_2_RegimeGroupedMoE32}"

ru=32
rp=32
extra_args=()

v16_2_experiment="baseline"

# V16_1_SteadyPressureAnchor32 baseline losses: inherited by all V16_2 cases.
lambda_hopf_radius="0.02"
lambda_hopf_overshoot="0.02"
lambda_hopf_onset="0.01"
lambda_v16_1_hopf_growth="0.0"
lambda_v16_1_hopf_false_growth="0.0"
lambda_v16_1_hopf_floor_rel="0.0"
lambda_v16_1_steady_p_state="0.02"
lambda_v16_1_steady_p_mean="0.02"
lambda_v16_1_steady_p_delta="0.01"
lambda_v16_1_steady_residual_damp="0.01"
lambda_v16_1_steady_p_energy="0.01"
v16_1_warmup="0"

# V16_2 case-specific losses.
lambda_v16_2_fp="0.0"
lambda_v16_2_contract="0.0"
lambda_v16_2_p_contract="0.0"
lambda_v16_2_p_delta="0.0"
lambda_v16_2_logr="0.0"
lambda_v16_2_over="0.0"
lambda_v16_2_phase="0.0"
lambda_v16_2_nf="0.0"
lambda_v16_2_mu="0.0"
lambda_v16_2_group_prior="0.0"
lambda_v16_2_group_balance="0.0"
lambda_v16_2_expert_balance="0.0"
lambda_v16_2_entropy_floor="0.0"
v16_2_expert_dropout="0.0"

# Router/expert defaults: V16_1_SteadyPressureAnchor32.
num_experts="6"
num_regime_groups="3"
experts_per_group="6"
num_shared_experts="1"
group_top_k="1"
top_k="2"
lambda_group_balance="0.04"
lambda_group_supervision="0.04"
group_temperature="0.90"
shared_scale="1.0"
routed_scale="0.85"

case "${case_name}" in
  V16_2_SteadyContractivePressureROM32)
    v16_2_experiment="steady_contractive"
    lambda_v16_2_fp="0.20"
    lambda_v16_2_contract="0.10"
    lambda_v16_2_p_contract="0.20"
    lambda_v16_2_p_delta="0.05"
    ;;
  V16_2_HopfLogRadiusNormalForm32)
    v16_2_experiment="hopf_log_normalform"
    lambda_v16_2_logr="0.25"
    lambda_v16_2_over="0.15"
    lambda_v16_2_phase="0.05"
    lambda_v16_2_nf="0.10"
    lambda_v16_2_mu="0.02"
    ;;
  V16_2_RegimeGroupedMoE32)
    v16_2_experiment="regime_grouped"
    num_experts="16"
    num_regime_groups="4"
    experts_per_group="4"
    num_shared_experts="0"
    group_top_k="4"
    top_k="2"
    lambda_group_balance="0.0"
    lambda_group_supervision="0.0"
    group_temperature="1.0"
    shared_scale="0.0"
    routed_scale="1.0"
    lambda_v16_2_group_prior="0.02"
    lambda_v16_2_group_balance="0.01"
    lambda_v16_2_expert_balance="0.02"
    lambda_v16_2_entropy_floor="0.005"
    v16_2_expert_dropout="0.05"
    ;;
  *)
    echo "unknown V16_2 case: ${case_name}" >&2
    exit 2
    ;;
esac

if [[ -n "${V16_2_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${V16_2_EXTRA_ARGS})
fi

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V16_2_AttractorStabilityMoE/test_results_v16_2"
SCRIPT="${ROOT}/train_v16_2_attractor_stability_moe.py"
name="${case_name}_ru${ru}_rp${rp}"
out="${ROOT}/results/${case_name}/${name}"
TRACKING_MODE="${SWANLAB_TRACKING_MODE:-online}"
TRACKING_PROJECT="${SWANLAB_TRACKING_PROJECT:-V16_2_AttractorStabilityMoE}"
TRACKING_GROUP="${SWANLAB_TRACKING_GROUP:-v16-2-attractor-stability-moe}"
TRACKING_LOG_DIR="${SWANLAB_TRACKING_LOG_DIR:-${ROOT}/results/swanlog/${case_name}}"

"${PY}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA is not available; refusing to run V16_2 training on CPU.", file=sys.stderr)
    sys.exit(3)
print("CUDA ready:", torch.cuda.get_device_name(0))
PY

mkdir -p "${out}" "${TRACKING_LOG_DIR}"
{
  echo "===== ${name} $(date -Is) ====="
  echo "case=${case_name}"
  echo "v16_2_experiment=${v16_2_experiment}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
} | tee "${out}/run.log"

"${PY}" "${SCRIPT}" \
  --output-dir "${out}" \
  --experiment-name "${name}" \
  --experiment-tag "${case_name}" \
  --data-root /root/moe/ROM_PhysicsGeneralizable/data/Global_POD_AreaWeighted_L2 \
  --tensor-path /root/moe/ROM_PhysicsGeneralizable/data/semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz \
  --pressure-surrogate-path /root/moe/ROM_PhysicsGeneralizable/data/pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz \
  --test-re-selection regime_default \
  --train-time-stride 1 \
  --train-re-stride 1 \
  --r-u "${ru}" \
  --r-p "${rp}" \
  --num-blocks 3 \
  --num-regime-groups "${num_regime_groups}" \
  --experts-per-group "${experts_per_group}" \
  --num-experts "${num_experts}" \
  --num-shared-experts "${num_shared_experts}" \
  --top-k "${top_k}" \
  --group-top-k "${group_top_k}" \
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
  --group-temperature "${group_temperature}" \
  --group-gate-floor 0.0 \
  --shared-scale "${shared_scale}" \
  --routed-scale "${routed_scale}" \
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
  --lambda-attractor-hopf-radius "${lambda_hopf_radius}" \
  --lambda-attractor-hopf-overshoot "${lambda_hopf_overshoot}" \
  --lambda-attractor-hopf-onset "${lambda_hopf_onset}" \
  --v16-1-hopf-floor-quantile 10 \
  --v16-1-loss-warmup-epochs "${v16_1_warmup}" \
  --lambda-v16-1-hopf-growth "${lambda_v16_1_hopf_growth}" \
  --lambda-v16-1-hopf-false-growth "${lambda_v16_1_hopf_false_growth}" \
  --lambda-v16-1-hopf-floor-rel "${lambda_v16_1_hopf_floor_rel}" \
  --lambda-v16-1-steady-p-state "${lambda_v16_1_steady_p_state}" \
  --lambda-v16-1-steady-p-mean "${lambda_v16_1_steady_p_mean}" \
  --lambda-v16-1-steady-p-delta "${lambda_v16_1_steady_p_delta}" \
  --lambda-v16-1-steady-residual-damp "${lambda_v16_1_steady_residual_damp}" \
  --lambda-v16-1-steady-p-energy "${lambda_v16_1_steady_p_energy}" \
  --v16-2-experiment "${v16_2_experiment}" \
  --v16-2-warmup-start-epoch 20 \
  --v16-2-warmup-end-epoch 50 \
  --lambda-v16-2-fp "${lambda_v16_2_fp}" \
  --lambda-v16-2-contract "${lambda_v16_2_contract}" \
  --lambda-v16-2-p-contract "${lambda_v16_2_p_contract}" \
  --lambda-v16-2-p-delta "${lambda_v16_2_p_delta}" \
  --lambda-v16-2-logr "${lambda_v16_2_logr}" \
  --lambda-v16-2-over "${lambda_v16_2_over}" \
  --lambda-v16-2-phase "${lambda_v16_2_phase}" \
  --lambda-v16-2-nf "${lambda_v16_2_nf}" \
  --lambda-v16-2-mu "${lambda_v16_2_mu}" \
  --lambda-v16-2-group-prior "${lambda_v16_2_group_prior}" \
  --lambda-v16-2-group-balance "${lambda_v16_2_group_balance}" \
  --lambda-v16-2-expert-balance "${lambda_v16_2_expert_balance}" \
  --lambda-v16-2-entropy-floor "${lambda_v16_2_entropy_floor}" \
  --v16-2-expert-dropout "${v16_2_expert_dropout}" \
  --lambda-periodic-energy 0.02 \
  --lambda-periodic-radius 0.01 \
  --lambda-attractor-ce 0.0 \
  --lambda-proto-energy 0.0 \
  --lambda-proto-radius 0.0 \
  --attractor-r-floor 0.0 \
  --attractor-onset-threshold 5e-5 \
  "${extra_args[@]}" \
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
  --lambda-group-balance "${lambda_group_balance}" \
  --lambda-group-entropy 0.0 \
  --lambda-group-supervision "${lambda_group_supervision}" \
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
  --seed "${V16_2_SEED:-1600}" \
  --epochs "${V16_2_EPOCHS:-240}" \
  --min-epochs "${V16_2_MIN_EPOCHS:-130}" \
  --patience 70 \
  --eval-every "${V16_2_EVAL_EVERY:-5}" \
  --eval-routing-every 20 \
  --early-stop-min-delta 8e-4 \
  --allow-tf32 \
  2>&1 | tee -a "${out}/run.log"
