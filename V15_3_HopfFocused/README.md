# V15_3 Hopf Focused Experiments

V15_3 focuses on the Hopf transition failure mode found in V15/V15_1. The
dominant oscillation pair is fixed to `(a0, a1)`, and the experiments keep the
Physics-Generalizable Re=20-200 dataset, held-out split, HPRS-MoE backbone,
routers, experts, Galerkin ROM, RK4, pressure Poisson surrogate, optimizer,
batch size, epoch schedule, random seed, and evaluation protocol aligned with
the V15/V15_1 baseline.

## Cases

- `V15_3_StrongBaseline32_AdaptiveGate_Balanced`: ru=32/rp=32, regime-balanced
  sampling, and modal AdaptiveGate pressure closure. No Hopf-specific loss.
- `V15_3_HopfAmpEnvelopeLoss32`: StrongBaseline32 plus Hopf-only log-amplitude,
  log-energy, and overshoot envelope losses for Re in `[46,55]`.
- `V15_3_HopfAmpEnvelopeLoss32_WithWeightedRollout`: adds Hopf sample weighting,
  focus weighting near Re=51.786, Hopf rollout weighting, and an `8,16,24,32`
  rollout curriculum.

All cases write Hopf pair diagnostics into the metrics JSON: `r_true`,
`r_pred`, amplitude/log-amplitude error, overshoot ratio, phase error, and
step-wise rollout growth curves for the focus Reynolds number.

## Run

```bash
cd /root/moe/V15_3_HopfFocused/test_results_v15_3
export SWANLAB_API_KEY=...
export SWANLAB_TRACKING_MODE=online
./run_v15_3_all.sh
```

The three SwanLab runs share the `V15_3_HopfFocused` project and use independent
run names/output directories.
