# V14_3 Pressure Input Ablation

This experiment keeps the V14 HPRS-MoE, Galerkin RHS, RK4 integration, losses,
router/expert counts, training schedule, and dense V14 data split fixed. It only
changes the state input seen by the pressure experts while keeping
`--pressure-target=closure`, so the final pressure remains:

```text
b_pred = b_base + pressure_head
```

## Modes

- `pressure_only`: current V14 baseline. The pressure expert uses the original
  current-state pressure branch input from the code path.
- `velocity_only`: the pressure expert state is `[a_next, 0]`.
- `hybrid`: the pressure expert state is `[a_next, b_base]`.

All modes keep pressure expert state dimension at `r_u+r_p`, so the Linear,
low-rank quadratic, and FFN parameterization is identical across the ablation.

## Run

```bash
cd /root/moe/V14_3_PressureInputAblation/test_results_v14_3
bash run_v14_3_pressure_input_ablation_all.sh
```

The all-run script tests the 10 uniform held-out Reynolds numbers:
`50.0, 78.0906, 105.983, 132.743, 160.785, 187.285, 215.256, 244.354, 274.377, 300.0`.

It writes per-mode metrics plus an aggregate report under:

```text
test_results_v14_3/results/v14_3_pressure_input_ablation_summary/
```
