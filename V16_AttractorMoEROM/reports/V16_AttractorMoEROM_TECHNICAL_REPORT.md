# V16_AttractorMoEROM Technical Report

Generated after both V16 runs completed on 2026-07-07. The report compares V16 against the completed V15_3 baselines using the same Re=20-200 Physics-Generalizable retained database and the same 11 held-out Reynolds numbers.

## Database Interpretation

The Re=20-200 retained database is treated here as a Physics-Generalizable Attractor Database rather than a Hopf transient database. Re<47 retained windows are steady fixed-point wakes initialized from simpleFoam and then saved through pimpleFoam. Re>=47 retained windows are accepted after lift-peak amplitude and period stabilize over multiple shedding periods. Therefore the supervised trajectories mostly describe final attractors: steady fixed points, weak near-Hopf periodic attractors, developing periodic attractors, and mature periodic limit cycles.

## Experiments

| Case | Main difference | Curriculum | Runtime |
|---|---|---|---:|
| V15_3 Strong | ru=32/rp=32 + AdaptiveGate + regime-balanced baseline | `[4, 8, 12, 16]` | 11.21 h |
| V15_3 HopfAmp | StrongBaseline32 plus Hopf amplitude envelope loss | `[4, 8, 12, 16]` | 11.23 h |
| V15_3 Weighted | HopfAmp plus Hopf sample/focus/rollout weighting and 8-16-24-32 curriculum | `[8, 16, 24, 32]` | 19.52 h |
| V16 FullRegimeLoss | Attractor-balanced sampling plus Steady/Hopf/Periodic attractor-specific losses, no topology change | `[4, 8, 12, 16]` | 10.41 h |
| V16 AttractorConditioned | FullRegimeLoss plus explicit Attractor Router, latent adapters, and prototype losses | `[4, 8, 12, 16]` | 10.56 h |

## Overall Metrics

| Metric | V15 Strong | V15 HopfAmp | V15 Weighted | V16 FullRegimeLoss | V16 AttractorConditioned | Best |
|---|---:|---:|---:|---:|---:|---|
| `rhs_l2` | 0.51887 | 0.66710 | 0.35679 | 0.30082 | 0.39065 | V16 FullRegimeLoss |
| `pressure_head_l2` | 0.25740 | 0.16284 | 0.10935 | 0.10920 | 0.13523 | V16 FullRegimeLoss |
| `one_step_a_l2` | 0.09107 | 0.11251 | 0.06063 | 0.04069 | 0.05623 | V16 FullRegimeLoss |
| `one_step_b_l2` | 0.26306 | 0.18576 | 0.12289 | 0.11428 | 0.14259 | V16 FullRegimeLoss |
| `rollout_a_l2` | 0.50404 | 0.36648 | 0.26672 | 0.22136 | 0.26201 | V16 FullRegimeLoss |
| `rollout_b_l2` | 0.78247 | 0.41419 | 0.41015 | 0.28881 | 0.35110 | V16 FullRegimeLoss |
| `one_step_pressure_energy_error` | 0.19475 | 0.15975 | 0.04944 | 0.07239 | 0.08638 | V15_3 Weighted |
| `rollout_pressure_energy_error` | 1.3170 | 0.50811 | 0.44045 | 0.33048 | 0.24313 | V16 AttractorConditioned |

V16_FullRegimeLoss32 is the strongest overall run. Relative to the best V15_3 Hopf-focused model, V15_3 WeightedRollout32, it improves mean velocity rollout from 0.2667 to 0.2214, pressure rollout from 0.4101 to 0.2888, and RHS error from 0.3568 to 0.3008.

## Attractor-Level Metrics

| Case | Attractor | n Re | rollout a | rollout b | RHS | pressure energy drift |
|---|---|---:|---:|---:|---:|---:|
| V15_3 Strong | steady | 4 | 0.20257 | 0.64811 | 0.30094 | 0.97398 |
| V15_3 Strong | hopf | 3 | 1.4983 | 1.9064 | 1.2217 | 3.4775 |
| V15_3 Strong | periodic | 4 | 0.05980 | 0.07391 | 0.20967 | 0.03961 |
| V15_3 HopfAmp | steady | 4 | 0.19678 | 0.44690 | 0.30078 | 0.30294 |
| V15_3 HopfAmp | hopf | 3 | 1.0192 | 0.84250 | 1.7648 | 1.4038 |
| V15_3 HopfAmp | periodic | 4 | 0.04668 | 0.06026 | 0.21018 | 0.04155 |
| V15_3 Weighted | steady | 4 | 0.19270 | 0.64168 | 0.31353 | 0.91810 |
| V15_3 Weighted | hopf | 3 | 0.63182 | 0.54597 | 0.61191 | 0.36306 |
| V15_3 Weighted | periodic | 4 | 0.06690 | 0.07675 | 0.20870 | 0.02085 |
| V16 FullRegimeLoss | steady | 4 | 0.20322 | 0.43732 | 0.26486 | 0.60214 |
| V16 FullRegimeLoss | hopf | 3 | 0.47808 | 0.39676 | 0.46751 | 0.39291 |
| V16 FullRegimeLoss | periodic | 4 | 0.04697 | 0.05933 | 0.21177 | 0.01201 |
| V16 AttractorConditioned | steady | 4 | 0.19168 | 0.41841 | 0.28770 | 0.24606 |
| V16 AttractorConditioned | hopf | 3 | 0.59923 | 0.58919 | 0.69975 | 0.46918 |
| V16 AttractorConditioned | periodic | 4 | 0.07943 | 0.10522 | 0.26176 | 0.07064 |

The attractor-specific losses improve all three attractor groups relative to V15_3 WeightedRollout on the mean rollout metrics. The largest practical gain is in the Hopf attractor group, where V16_FullRegimeLoss reduces mean velocity rollout to 0.4781 and pressure rollout to 0.3968.

## Per-Re Rollout

| Re | Attractor | V15 Weighted a | V16 Full a | V16 Cond a | V15 Weighted b | V16 Full b | V16 Cond b |
|---:|---|---:|---:|---:|---:|---:|---:|
| 24.630 | steady | 0.29484 | 0.30552 | 0.29651 | 1.0212 | 0.64169 | 0.56891 |
| 32.740 | steady | 0.19887 | 0.21198 | 0.19849 | 0.86447 | 0.51134 | 0.54599 |
| 39.685 | steady | 0.15523 | 0.16604 | 0.15330 | 0.43017 | 0.34409 | 0.32170 |
| 45.143 | steady | 0.12188 | 0.12934 | 0.11844 | 0.25083 | 0.25215 | 0.23702 |
| 47.081 | hopf | 0.44077 | 0.62793 | 0.48663 | 0.32213 | 0.26705 | 0.45574 |
| 49.022 | hopf | 0.19973 | 0.30533 | 0.22266 | 0.35119 | 0.29525 | 0.44938 |
| 51.786 | hopf | 1.2550 | 0.50098 | 1.0884 | 0.96460 | 0.62799 | 0.86245 |
| 70.315 | periodic | 0.12929 | 0.10226 | 0.15234 | 0.14545 | 0.12943 | 0.22624 |
| 100.352 | periodic | 0.05067 | 0.02076 | 0.07700 | 0.05871 | 0.02756 | 0.09838 |
| 149.059 | periodic | 0.03594 | 0.03372 | 0.03703 | 0.04034 | 0.04146 | 0.04642 |
| 189.862 | periodic | 0.05172 | 0.03113 | 0.05136 | 0.06251 | 0.03888 | 0.04983 |

## Hopf Critical Points

| Re | Case | velocity rollout | pressure rollout | r_true_mean | r_pred_mean | amplitude L2 | overshoot mean | phase MAE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 47.081 | V15_3 Strong | 0.80696 | 1.0780 | 0.00000 | 0.00003 | 20.841 | 20.127 | 1.8581 |
| 47.081 | V15_3 HopfAmp | 0.55872 | 0.40070 | 0.00000 | 0.00010 | 71.028 | 73.211 | 1.6033 |
| 47.081 | V15_3 Weighted | 0.44077 | 0.32213 | 0.00000 | 0.00004 | 31.445 | 32.519 | 1.6296 |
| 47.081 | V16 FullRegimeLoss | 0.62793 | 0.26705 | 0.00000 | 0.00002 | 16.413 | 15.081 | 1.3613 |
| 47.081 | V16 AttractorConditioned | 0.48663 | 0.45574 | 0.00000 | 0.00002 | 17.566 | 18.125 | 1.5721 |
| 49.022 | V15_3 Strong | 0.61354 | 0.91664 | 0.00000 | 0.00003 | 9.3308 | 9.3245 | 1.0594 |
| 49.022 | V15_3 HopfAmp | 0.36265 | 0.40469 | 0.00000 | 0.00009 | 29.843 | 30.286 | 1.5511 |
| 49.022 | V15_3 Weighted | 0.19973 | 0.35119 | 0.00000 | 0.00005 | 15.865 | 16.218 | 1.5038 |
| 49.022 | V16 FullRegimeLoss | 0.30533 | 0.29525 | 0.00000 | 0.00002 | 4.9471 | 5.2396 | 0.94975 |
| 49.022 | V16 AttractorConditioned | 0.22266 | 0.44938 | 0.00000 | 0.00002 | 7.0233 | 7.5071 | 1.2451 |
| 51.786 | V15_3 Strong | 3.0745 | 3.7244 | 0.00002 | 0.00004 | 1.0894 | 1.9330 | 0.48767 |
| 51.786 | V15_3 HopfAmp | 2.1361 | 1.7221 | 0.00002 | 0.00009 | 2.9713 | 4.2589 | 1.2694 |
| 51.786 | V15_3 Weighted | 1.2550 | 0.96460 | 0.00002 | 0.00006 | 1.8619 | 2.8251 | 0.91550 |
| 51.786 | V16 FullRegimeLoss | 0.50098 | 0.62799 | 0.00002 | 0.00004 | 0.82738 | 1.6663 | 0.39253 |
| 51.786 | V16 AttractorConditioned | 1.0884 | 0.86245 | 0.00002 | 0.00003 | 0.66105 | 1.4133 | 0.70090 |

At Re=51.786, V16_FullRegimeLoss gives the best velocity rollout (0.5010) and the best pressure rollout (0.6280). Both V16 cases remain above the 10% target at this hardest near-Hopf point, but V16_FullRegimeLoss improves substantially over the V15_3 weighted rollout baseline.

## Attractor Router And Adapter Diagnostics

| Re | True attractor | acc | pi_steady | pi_hopf | pi_periodic | adapter delta/h norm |
|---:|---|---:|---:|---:|---:|---:|
| 24.630 | steady | 0.00000 | 0.44818 | 0.53296 | 0.01886 | 0.32911 |
| 32.740 | steady | 0.00000 | 0.44680 | 0.53354 | 0.01966 | 0.32368 |
| 39.685 | steady | 0.00000 | 0.44557 | 0.53372 | 0.02071 | 0.32131 |
| 45.143 | steady | 0.00000 | 0.44463 | 0.53375 | 0.02162 | 0.32072 |
| 47.081 | hopf | 1.0000 | 0.44409 | 0.53457 | 0.02134 | 0.31838 |
| 49.022 | hopf | 1.0000 | 0.44371 | 0.53456 | 0.02173 | 0.31859 |
| 51.786 | hopf | 1.0000 | 0.44293 | 0.53521 | 0.02186 | 0.31680 |
| 70.315 | periodic | 1.0000 | 0.00166 | 0.00281 | 0.99553 | 0.31942 |
| 100.352 | periodic | 1.0000 | 0.00009 | 0.00016 | 0.99976 | 0.25150 |
| 149.059 | periodic | 1.0000 | 0.00043 | 0.00068 | 0.99889 | 0.23222 |
| 189.862 | periodic | 1.0000 | 0.00039 | 0.00057 | 0.99905 | 0.24763 |

The explicit Attractor Router is partially meaningful: it cleanly separates mature periodic cases, but it merges the steady and near-Hopf boundary, assigning the steady held-out cases slightly more Hopf probability than steady probability. This explains why the conditioned framework is interpretable but not yet a better predictor. The adapter/prototype path likely regularizes pressure-energy drift, but it also weakens the base operator fit and one-step metrics.

## Answers To V16 Questions

1. Attractor-specific losses are sufficient to improve the current model family. V16_FullRegimeLoss improves overall, Steady, Hopf, and Periodic rollout metrics relative to V15_3 WeightedRollout while preserving the existing HPRS-MoE topology.
2. The explicit Attractor Router + lightweight adapters are interpretable, but they are not the best performing framework in this run. The router strongly identifies mature periodic cases but does not cleanly separate steady from near-Hopf attractors, and the extra latent adapter/prototype constraints make the optimization less effective than the simpler loss-only version.
3. The Attractor Database does support a dedicated AttractorMoE-ROM: V16_FullRegimeLoss is the best model so far on mean rollout and RHS. But the remaining Re=51.786 error is still far above 10%, so a future Hopf transient database would still be useful if the goal is onset/growth/saturation dynamics rather than stable retained-attractor windows.

## Recommendation

Use `V16_AttractorMoEROM_FullRegimeLoss32` as the next default baseline. Keep `AttractorConditionedFramework32` as an interpretable diagnostic branch, but do not promote it as the default until adapter/prototype weights are retuned or made less restrictive.

## Artifacts

- Light JSON summary: `reports/data/v16_attractor_summary_light.json`
- Aggregate comparison CSV: `reports/data/v16_attractor_aggregate_comparison.csv`
- Per-Re comparison CSV: `reports/data/v16_attractor_per_re_comparison.csv`
- Attractor-level CSV: `reports/data/v16_attractor_regime_metrics.csv`
- Hopf diagnostics CSV: `reports/data/v16_hopf_critical_diagnostics.csv`
- Attractor router CSV: `reports/data/v16_attractor_router_diagnostics.csv`
- Remote raw outputs remain under `/root/moe/V16_AttractorMoEROM/test_results_v16/results/...` on the cluster.
