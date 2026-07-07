# V15_3 WeightedRollout Experiment Report

This addendum reports the completed `V15_3_HopfAmpEnvelopeLoss32_WithWeightedRollout` run and compares it with the two earlier completed V15_3 cases. Results were generated from the A40 cluster outputs completed on 2026-07-07.

## Experiment Setup

- Database: Re=20-200 Physics-Generalizable retained database.
- Held-out Re: 24.630, 32.740, 39.685, 45.143, 47.081, 49.022, 51.786, 70.315, 100.352, 149.059, 189.862.
- Backbone: HPRS-MoE-ROM, ru=32/rp=32, Galerkin + RK4, pressure closure `adaptive_gate`, pressure input `pressure_only`.
- WeightedRollout changes relative to HopfAmpEnvelopeLoss32: Hopf samples weighted 3x, Re=51.786 focus window weighted 4x, Hopf rollout loss weighted 2x, curriculum extended from `4,8,12,16` to `8,16,24,32`.
- Metrics are relative L2 unless noted. `rollout` means autonomous 24-step rollout.

## Aggregate Results

| Metric | Strong | HopfAmp | WeightedRollout | Weighted vs Strong | Weighted vs HopfAmp |
|---|---:|---:|---:|---:|---:|
| `rhs_l2` | 0.51887 | 0.66710 | 0.35679 | 31.238% | 46.517% |
| `pressure_head_l2` | 0.25740 | 0.16284 | 0.10935 | 57.518% | 32.849% |
| `one_step_a_l2` | 0.09107 | 0.11251 | 0.06063 | 33.432% | 46.112% |
| `one_step_b_l2` | 0.26306 | 0.18576 | 0.12289 | 53.283% | 33.844% |
| `rollout_a_l2` | 0.50404 | 0.36648 | 0.26672 | 47.084% | 27.222% |
| `rollout_b_l2` | 0.78247 | 0.41419 | 0.41015 | 47.582% | 0.97666% |
| `one_step_pressure_energy_error` | 0.19475 | 0.15975 | 0.04944 | 74.616% | 69.056% |
| `rollout_pressure_energy_error` | 1.3170 | 0.50811 | 0.44045 | 66.556% | 13.316% |

WeightedRollout gives the best aggregate rollout numbers among the three V15_3 cases. Compared with HopfAmpEnvelopeLoss32, mean velocity rollout improves from 0.3665 to 0.2667, while mean pressure rollout changes slightly from 0.4142 to 0.4101. The gain is concentrated in the Hopf critical region, especially Re=47.081, 49.022, and 51.786; the trade-off is visible on steady-pressure rollout and several mature periodic cases.

## Per-Re Rollout Comparison

| Re | Regime | Strong a | HopfAmp a | Weighted a | Weighted vs HopfAmp | Strong b | HopfAmp b | Weighted b | Weighted vs HopfAmp |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630 | Steady | 0.31008 | 0.29380 | 0.29484 | -0.35395% | 0.95848 | 0.55767 | 1.0212 | -83.127% |
| 32.740 | Steady | 0.20938 | 0.20385 | 0.19887 | 2.4445% | 0.71036 | 0.61098 | 0.86447 | -41.490% |
| 39.685 | Steady | 0.16244 | 0.16119 | 0.15523 | 3.7005% | 0.50557 | 0.35878 | 0.43017 | -19.897% |
| 45.143 | Hopf | 0.12838 | 0.12827 | 0.12188 | 4.9812% | 0.41802 | 0.26016 | 0.25083 | 3.5899% |
| 47.081 | Hopf | 0.80696 | 0.55872 | 0.44077 | 21.111% | 1.0780 | 0.40070 | 0.32213 | 19.608% |
| 49.022 | Hopf | 0.61354 | 0.36265 | 0.19973 | 44.925% | 0.91664 | 0.40469 | 0.35119 | 13.222% |
| 51.786 | Hopf | 3.0745 | 2.1361 | 1.2550 | 41.250% | 3.7244 | 1.7221 | 0.96460 | 43.987% |
| 70.315 | Periodic | 0.13936 | 0.12303 | 0.12929 | -5.0862% | 0.16806 | 0.16318 | 0.14545 | 10.864% |
| 100.352 | Periodic | 0.02288 | 0.02121 | 0.05067 | -138.88% | 0.02566 | 0.02517 | 0.05871 | -133.25% |
| 149.059 | Periodic | 0.02462 | 0.02303 | 0.03594 | -56.017% | 0.02949 | 0.02705 | 0.04034 | -49.138% |
| 189.862 | Periodic | 0.05235 | 0.01943 | 0.05172 | -166.17% | 0.07242 | 0.02565 | 0.06251 | -143.69% |

WeightedRollout directly improves the main Hopf failure point Re=51.786 relative to both earlier cases. It also improves Re=47.081 and Re=49.022, which means the Hopf-weighted long-rollout curriculum is doing what it was designed to do. The cost is that steady-pressure rollout becomes worse at Re=24.630 and Re=32.740, and mature periodic velocity/pressure rollout regresses at Re=100.352, 149.059, and 189.862. So the result is positive but not free: Hopf stability improves by spending capacity away from some non-Hopf attractors.

## Hopf Critical Diagnostics

| Re | Case | a-rollout | b-rollout | r_true_mean | r_pred_mean | amp L2 | log-amp MAE | overshoot mean | overshoot max | phase MAE | freq MAE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 47.081 | StrongBaseline32 | 0.80696 | 1.0780 | 0.00000 | 0.00003 | 20.841 | 2.8423 | 20.127 | 57.326 | 1.8581 | 0.65721 |
| 47.081 | HopfAmpEnvelopeLoss32 | 0.55872 | 0.40070 | 0.00000 | 0.00010 | 71.028 | 4.2563 | 73.211 | 142.89 | 1.6033 | 0.73537 |
| 47.081 | WeightedRollout32 | 0.44077 | 0.32213 | 0.00000 | 0.00004 | 31.445 | 3.4348 | 32.519 | 63.855 | 1.6296 | 0.72489 |
| 49.022 | StrongBaseline32 | 0.61354 | 0.91664 | 0.00000 | 0.00003 | 9.3308 | 2.0559 | 9.3245 | 20.617 | 1.0594 | 0.50435 |
| 49.022 | HopfAmpEnvelopeLoss32 | 0.36265 | 0.40469 | 0.00000 | 0.00009 | 29.843 | 3.3792 | 30.286 | 52.551 | 1.5511 | 0.73233 |
| 49.022 | WeightedRollout32 | 0.19973 | 0.35119 | 0.00000 | 0.00005 | 15.865 | 2.7241 | 16.218 | 29.306 | 1.5038 | 0.71852 |
| 51.786 | StrongBaseline32 | 3.0745 | 3.7244 | 0.00002 | 0.00004 | 1.0894 | 0.61396 | 1.9330 | 4.2434 | 0.48767 | 0.29282 |
| 51.786 | HopfAmpEnvelopeLoss32 | 2.1361 | 1.7221 | 0.00002 | 0.00009 | 2.9713 | 1.3436 | 4.2589 | 12.186 | 1.2694 | 0.73763 |
| 51.786 | WeightedRollout32 | 1.2550 | 0.96460 | 0.00002 | 0.00006 | 1.8619 | 0.99986 | 2.8251 | 7.3371 | 0.91550 | 0.67128 |

At Re=51.786, WeightedRollout gives velocity rollout 1.2550 and pressure rollout 0.9646. This is the best Re=51.786 result among the three V15_3 cases, improving over HopfAmpEnvelopeLoss32's 2.1361/1.7221 velocity/pressure rollout.

## Re=51.786 Horizon Diagnostics for WeightedRollout

| Horizon | velocity rollout L2 | pressure rollout L2 | pressure energy error | r_true_mean | r_pred_mean | amp L2 | overshoot mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.75874 | 1.0516 | 0.17632 | 0.00002 | 0.00006 | 1.9200 | 3.0281 |
| 16 | 1.0788 | 0.97179 | 0.12911 | 0.00002 | 0.00006 | 1.9314 | 2.9485 |
| 24 | 1.2550 | 0.96460 | 0.06304 | 0.00002 | 0.00006 | 1.8619 | 2.8251 |
| 32 | 1.4469 | 0.95885 | 0.06838 | 0.00002 | 0.00006 | 2.0463 | 2.8898 |

The horizon sweep confirms that the difficult case is not just a 24-step reporting artifact: the Re=51.786 trajectory remains amplitude-sensitive across horizons, and the weighted long-rollout objective does not fully suppress the near-Hopf radius mismatch.

## Router and Expert Usage

| Re | Regime | active experts | dead experts | load CV | group load | expert collapse flag |
|---:|---|---:|---:|---:|---|---|
| 24.630 | Steady | 5.0000 | 18 | 2.7115 | [1.000, 0.000, 0.000] | False |
| 32.740 | Steady | 5.0000 | 18 | 2.7115 | [1.000, 0.000, 0.000] | False |
| 39.685 | Steady | 5.0000 | 18 | 2.7115 | [1.000, 0.000, 0.000] | False |
| 45.143 | Hopf | 5.0000 | 18 | 2.7115 | [1.000, 0.000, 0.000] | False |
| 47.081 | Hopf | 5.0000 | 18 | 2.7115 | [1.000, 0.000, 0.000] | False |
| 49.022 | Hopf | 5.0000 | 18 | 2.7115 | [1.000, 0.000, 0.000] | False |
| 51.786 | Hopf | 5.0000 | 18 | 2.7115 | [1.000, 0.000, 0.000] | False |
| 70.315 | Periodic | 4.4241 | 12 | 2.1232 | [0.854, 0.013, 0.133] | False |
| 100.352 | Periodic | 4.5063 | 16 | 2.6144 | [0.019, 0.981, 0.000] | False |
| 149.059 | Periodic | 4.6646 | 14 | 2.4226 | [0.044, 0.924, 0.032] | False |
| 189.862 | Periodic | 4.3987 | 9 | 1.3960 | [0.177, 0.335, 0.487] | False |

The routing remains interpretable: low-Re and Hopf cases mainly use the low-regime group, mid periodic cases move to the middle group, and high-Re cases move toward the high group. However, expert utilization is still narrow; many experts remain below the 1% usage threshold because group-top-k routing selects a small active subset.

## Conclusion

WeightedRollout answers the V15_3 Hopf question positively but with an important caveat. If the priority is the diagnosed Hopf failure, especially Re=51.786, then `V15_3_HopfAmpEnvelopeLoss32_WithWeightedRollout` is the best V15_3 case. It also gives the best aggregate velocity rollout and a slight aggregate pressure-rollout gain. If the priority is preserving mature periodic performance and low-Re pressure stability, then the simpler `HopfAmpEnvelopeLoss32` is safer.

For V16, this supports the shift from treating the database as Hopf transient data toward an attractor-aware formulation. Weighted long-rollout supervision can suppress the near-Hopf failure, but the cross-regime trade-off suggests the model needs explicit Steady/Hopf/Periodic attractor conditioning rather than one global curriculum pressure.

## Artifacts

- Aggregate three-case CSV: `reports/data/v15_3_final_three_case_aggregate.csv`
- Weighted per-Re CSV: `reports/data/v15_3_final_weighted_rollout_per_re.csv`
- Hopf diagnostic CSV: `reports/data/v15_3_final_hopf_weighted_rollout_diagnostics.csv`
- Re=51.786 horizon CSV: `reports/data/v15_3_weighted_rollout_re51_horizons.csv`
- Remote raw summary and metrics remain under `/root/moe/V15_3_HopfFocused/test_results_v15_3/results/...` on the cluster.
