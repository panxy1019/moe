# V15_1 Pressure Base Evolution

This directory contains the V15_1 pressure/ROM base evolution experiments built on
the V15_Base Physics-Generalizable HPRS-MoE-ROM baseline. The three cases keep
the V15_Base data split, HPRS-MoE experts, routers, Galerkin/RK4 training loop,
loss weights, optimizer, seed, batch size, and epoch schedule fixed. Only the
pressure or ROM base construction changes.

## Cases

- `V15_1_AdaptiveGate`: static V15 pressure Poisson base, plus modal adaptive
  base confidence with `closure_mode=adaptive_gate`.
- `V15_1_FiLMBase`: static velocity ROM, FiLM-calibrated pressure Poisson base,
  standard closure fusion.
- `V15_1_RegimeAwareROM`: regime-specific Steady/Hopf/Periodic velocity ROM and
  pressure Poisson bases mixed by a learned shared Regime Gate.

## Run

On the cluster:

```bash
cd /root/moe/V15_1_PressureBaseEvolution/test_results_v15_1
export SWANLAB_API_KEY=...
export SWANLAB_TRACKING_MODE=online
./run_v15_1_all.sh
```

Each case writes to its own output directory under `test_results_v15_1/results/`
and uses a separate SwanLab project by default.
