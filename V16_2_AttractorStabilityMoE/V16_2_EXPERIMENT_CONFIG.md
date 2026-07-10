# V16_2 AttractorStabilityMoE Experiment Config

V16_2 is an isolated fork of `V16_1_TrainStableAttractorMoE`. It does not move, overwrite, or modify V16_1 results, checkpoints, metrics, or reports.

Unified baseline:

- Base case: `V16_1_SteadyPressureAnchor32`
- Dataset: Re=20-200 Physics-Generalizable Attractor Database
- ROM dimension: `ru=32`, `rp=32`
- Backbone: HPRS-MoE + Shared Encoder + Galerkin + RK4
- Pressure: static Pressure Poisson Surrogate + modal AdaptiveGate closure
- Pressure target: `closure`
- Pressure input: `pressure_only` (`[a_t,b_t]`)
- Training split: `regime_default` held-out Re, dense time sampling
- Rollout curriculum: `4,8,12,16`
- Evaluation: 24-step autonomous rollout

## Experiments

### V16_2_SteadyContractivePressureROM32

Purpose: reduce steady/pre-Hopf steady closed-loop pressure drift without changing router or MoE structure.

Additional losses, active only on steady attractor samples and warm-started from epoch 20 to 50:

- `L_fp = ||Phi(x_eq, Re) - x_eq||^2`
- `L_contract = max(0, ||Phi(x_eq+eps, Re)-x_eq||/(||eps||+eps0)-rho_s)^2`
- `L_p_contract = max(0, ||b_pred-b_eq||/(||b_t-b_eq||+eps0)-rho_p)^2`
- `L_p_delta = ||b_pred-b_t||^2` near equilibrium

Weights:

```text
lambda_fp = 0.20
lambda_contract = 0.10
lambda_p_contract = 0.20
lambda_p_delta = 0.05
rho_s = 0.90
rho_p = 0.85
perturb_std_frac = 0.02
```

### V16_2_HopfLogRadiusNormalForm32

Purpose: reduce Hopf near-onset false oscillation and amplitude overshoot without adding steady contraction or grouped MoE changes.

Additional losses, active near `Re in [45,55]` and warm-started from epoch 20 to 50:

- log-radius loss
- bounded overshoot penalty with threshold 3x
- amplitude-masked phase loss
- auxiliary Hopf normal-form one-step prediction
- `mu` sign prior around `Re_c_init=48.0`

Weights:

```text
lambda_logr = 0.25
lambda_over = 0.15
lambda_phase = 0.05
lambda_nf = 0.10
lambda_mu = 0.02
```

### V16_2_RegimeGroupedMoE32

Purpose: test whether explicit grouped routing improves expert usage and reduces top1 collapse.

Only router/MoE grouping changes:

- `num_regime_groups=4`
- group names: `steady`, `hopf`, `periodic`, `shared`
- `experts_per_group=4`
- `num_shared_experts=0`
- total routed experts: `16`
- `group_top_k=4` so the group router remains soft
- in-group `top_k=2`

Soft group prior order: `[steady, hopf, periodic, shared]`.

Default priors:

```text
steady_wake = [0.85, 0.05, 0.00, 0.10]
pre_hopf_steady = [0.65, 0.25, 0.00, 0.10]
hopf_transition = [0.25, 0.60, 0.05, 0.10]
developing_periodic_shedding = [0.00, 0.35, 0.55, 0.10]
mature_periodic_shedding = [0.00, 0.05, 0.85, 0.10]
high_re_2d_periodic_near_modeA = [0.00, 0.05, 0.80, 0.15]
```

Additional router regularization:

```text
lambda_group_prior = 0.02
lambda_group_balance = 0.01
lambda_expert_balance = 0.02
lambda_entropy_floor = 0.005
entropy_floor = 0.30
expert_dropout = 0.05 after epoch 30
```

## Commands

Single case:

```bash
cd /root/moe/V16_2_AttractorStabilityMoE/test_results_v16_2
CUDA_VISIBLE_DEVICES=0 ./run_v16_2_one.sh V16_2_SteadyContractivePressureROM32
CUDA_VISIBLE_DEVICES=0 ./run_v16_2_one.sh V16_2_HopfLogRadiusNormalForm32
CUDA_VISIBLE_DEVICES=0 ./run_v16_2_one.sh V16_2_RegimeGroupedMoE32
```

Parallel launch:

```bash
cd /root/moe/V16_2_AttractorStabilityMoE/test_results_v16_2
CUDA_VISIBLE_DEVICES=0 ./run_v16_2_all.sh
```

Eval-only:

```bash
export V16_2_EXTRA_ARGS="--eval-only-checkpoint /path/to/checkpoint.pt"
./run_v16_2_one.sh V16_2_SteadyContractivePressureROM32
unset V16_2_EXTRA_ARGS
```

Aggregate after all metrics exist:

```bash
python aggregate_v16_2_attractor_stability_moe.py \
  /root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1/results/V16_1_SteadyPressureAnchor32/V16_1_SteadyPressureAnchor32_ru32_rp32/V16_1_SteadyPressureAnchor32_ru32_rp32_metrics.json \
  results/V16_2_SteadyContractivePressureROM32/V16_2_SteadyContractivePressureROM32_ru32_rp32/V16_2_SteadyContractivePressureROM32_ru32_rp32_metrics.json \
  results/V16_2_HopfLogRadiusNormalForm32/V16_2_HopfLogRadiusNormalForm32_ru32_rp32/V16_2_HopfLogRadiusNormalForm32_ru32_rp32_metrics.json \
  results/V16_2_RegimeGroupedMoE32/V16_2_RegimeGroupedMoE32_ru32_rp32/V16_2_RegimeGroupedMoE32_ru32_rp32_metrics.json \
  --output-dir results/aggregate
```
