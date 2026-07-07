# V14 Adaptive Pressure Closure

This experiment starts from V14_3 and changes only the pressure closure fusion.

Final report: [V14_AdaptivePressureClosure_EXPERIMENT_REPORT.md](V14_AdaptivePressureClosure_EXPERIMENT_REPORT.md).

Short result: adaptive scalar closure improves one-step pressure slightly, but does not improve the mean 24-step autonomous pressure rollout over the V14_3 baseline. Dual Adaptive Closure is the only adaptive case that improves the Low-Re mean rollout pressure, mainly at Re=50, while the baseline remains more stable over mid/high Re.

Fixed components:

- HPRS-MoE shared encoder, group router, local routers, velocity head, pressure head, and physics-aware experts.
- Pressure Head input remains the V14 baseline `[a_t, b_t]`.
- Galerkin RHS, RK4 integration, Poisson pressure surrogate, dense V14 data split, losses, optimizer settings, and training schedule.

Compared closure modes:

- `baseline`: `b_pred = b_base + r`.
- `residual_scaling`: `b_pred = b_base + alpha(x) r`.
- `base_scaling`: `b_pred = alpha(x) b_base + r`.
- `dual_adaptive`: `b_pred = (1+beta(x)) b_base + alpha(x) r`.

The adaptive confidence head is a small `Linear -> SiLU -> Linear` MLP on the existing encoder feature `h`. `alpha` is constrained to `[0,1]` with sigmoid and `beta` is constrained to about `[-0.5,0.5]` with `0.5*tanh`.

Run all adaptive cases on the cluster:

```bash
export SWANLAB_API_KEY=...
bash /root/moe/V14_AdaptivePressureClosure/test_results_adaptive_pressure_closure/run_adaptive_pressure_closure_all.sh
```

SwanLab is enabled by default in the run script with project `V14_AdaptivePressureClosure` and group `adaptive-pressure-closure`. Set `SWANLAB_TRACKING_MODE=disabled` to run without tracking.

The aggregate report and plots are written to:

```text
test_results_adaptive_pressure_closure/results/v14_adaptive_pressure_closure_summary/
```
