# V14_3 Pressure Input Ablation Report

## Experiment Design

All runs keep HPRS-MoE, routers, Galerkin RHS, RK4, losses, optimizer settings, training schedule, dense V14 data organization, and `--pressure-target=closure` fixed. Only the pressure expert state input changes.

- PressureOnly: unchanged V14 baseline pressure state from current code path.
- VelocityOnly: pressure state is `[a_next, 0]`, so the head sees next-step velocity only.
- Hybrid: pressure state is `[a_next, b_base]`, so the head sees next-step velocity and the Poisson prior.

The pressure expert state dimension is held at `r_u+r_p` for all modes, so the Linear, low-rank quadratic, and FFN parameterization remains identical.

## Aggregate Metrics

| Mode | one-step pressure | rollout pressure | one-step velocity | rollout velocity | RHS | pressure energy one-step | pressure energy rollout |
|---|---:|---:|---:|---:|---:|---:|---:|
| PressureOnly | 0.121132 | 0.164159 | 0.0230022 | 0.073183 | 0.0925724 | 0.0445221 | 0.0721308 |
| VelocityOnly | 0.126876 | 0.208693 | 0.0259003 | 0.0998976 | 0.0916761 | 0.0449067 | 0.0669514 |
| Hybrid | 0.133363 | 0.29243 | 0.0342301 | 0.18517 | 0.0937745 | 0.059801 | 0.150939 |

## Low-Re Focus

| Mode | Re<=80 one-step pressure | Re<=80 rollout pressure | Re<=80 pressure energy one-step |
|---|---:|---:|---:|
| PressureOnly | 0.513168 | 0.619387 | 0.187532 |
| VelocityOnly | 0.526232 | 0.721061 | 0.177031 |
| Hybrid | 0.531471 | 0.699098 | 0.230862 |

## Per-Re Pressure Metrics

| Re | PressureOnly one-step | VelocityOnly one-step | Hybrid one-step | PressureOnly rollout | VelocityOnly rollout | Hybrid rollout |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.970201 | 0.998312 | 1.00627 | 1.1591 | 1.2857 | 1.17979 |
| 78.0906 | 0.0561346 | 0.0541517 | 0.0566661 | 0.0796697 | 0.156418 | 0.218408 |
| 105.983 | 0.0171477 | 0.0139873 | 0.0374667 | 0.0298593 | 0.0458286 | 0.186274 |
| 132.743 | 0.015303 | 0.0151626 | 0.0302151 | 0.0481885 | 0.0381312 | 0.193097 |
| 160.785 | 0.0200866 | 0.0220659 | 0.035262 | 0.0546321 | 0.0887429 | 0.243618 |
| 187.285 | 0.0178647 | 0.0230841 | 0.0228547 | 0.0537741 | 0.0941559 | 0.240316 |
| 215.256 | 0.0183753 | 0.0257374 | 0.0206524 | 0.0353857 | 0.087956 | 0.160587 |
| 244.354 | 0.0206708 | 0.0242179 | 0.0212571 | 0.0318397 | 0.0649563 | 0.154362 |
| 274.377 | 0.0240637 | 0.0260774 | 0.029654 | 0.0429471 | 0.0502847 | 0.135839 |
| 300 | 0.051472 | 0.065966 | 0.0733236 | 0.106186 | 0.174755 | 0.21201 |

## Direct Answers

- VelocityOnly vs PressureOnly: mean one-step pressure change is -4.74% and rollout pressure change is -27.1% relative improvement.
- Hybrid vs PressureOnly: mean one-step pressure change is -10.1% and rollout pressure change is -78.1% relative improvement.
- Low-Re one-step pressure means: PressureOnly=0.513168, VelocityOnly=0.526232, Hybrid=0.531471.

Interpretation rule: if VelocityOnly clearly beats PressureOnly at low Re, the current pressure-state input is likely a major bottleneck. If Hybrid beats both, the pressure head should use velocity modes plus the Poisson prior. If neither improves, the bottleneck is more likely pressure residual learning, the Poisson base, or rollout coupling.

## Artifacts

- Combined CSV: `results/v14_3_pressure_input_ablation_summary/v14_3_pressure_input_ablation_combined.csv`
- one_step_pressure_vs_re: `results/v14_3_pressure_input_ablation_summary/v14_3_one_step_pressure_vs_re.svg`
- rollout_pressure_vs_re: `results/v14_3_pressure_input_ablation_summary/v14_3_rollout_pressure_vs_re.svg`
- pressure_energy_vs_re: `results/v14_3_pressure_input_ablation_summary/v14_3_pressure_energy_vs_re.svg`

## Source Metrics

- VelocityOnly: `results/v14_3_pressure_input_velocity_only_dense_uniform10/v14_3_pressure_input_velocity_only_dense_uniform10_metrics.json`
- PressureOnly: `results/v14_3_pressure_input_pressure_only_dense_uniform10/v14_3_pressure_input_pressure_only_dense_uniform10_metrics.json`
- Hybrid: `results/v14_3_pressure_input_hybrid_dense_uniform10/v14_3_pressure_input_hybrid_dense_uniform10_metrics.json`
