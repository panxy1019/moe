# V15_3 HopfFocused Interim Report: StrongBaseline32 vs HopfAmpEnvelopeLoss32

Generated from the first two completed V15_3 runs on 2026-07-06. The third case, `V15_3_HopfAmpEnvelopeLoss32_WithWeightedRollout`, was still training when this interim report was prepared, so it is intentionally excluded and will be reported separately after completion.

## Scope

- Dataset: Re=20-200 Physics-Generalizable database, dense time sampling.
- Held-out Re split: 24.630, 32.740, 39.685, 45.143, 47.081, 49.022, 51.786, 70.315, 100.352, 149.059, 189.862.
- Both completed cases keep HPRS-MoE, Shared Encoder, routers, experts, Galerkin ROM, RK4, Pressure Poisson Surrogate, optimizer, learning rate, batch size, and evaluation protocol fixed.
- Metrics are relative L2 unless otherwise stated. `rollout` means autonomous 24-step rollout from the generated evaluator.

## Case Definitions

| Case | Main setting | Hopf amplitude loss | Rollout curriculum | Runtime |
|---|---|---|---|---:|
| StrongBaseline32 | ru=32/rp=32 + Regime-balanced sampling + modal AdaptiveGate | log=0, energy=0, overshoot=0 | `[4, 8, 12, 16]` | 11.21 h |
| HopfAmpEnvelopeLoss32 | ru=32/rp=32 + Regime-balanced sampling + modal AdaptiveGate | log=0.05, energy=0.05, overshoot=0.02 | `[4, 8, 12, 16]` | 11.23 h |

## Aggregate Held-out Results

| Metric | Strong mean | HopfAmp mean | HopfAmp change vs Strong | Interpretation |
|---|---:|---:|---:|---|
| `rhs_l2` | 0.51887 | 0.66710 | -28.568% | RHS fitting became worse with Hopf envelope loss. |
| `pressure_head_l2` | 0.25740 | 0.16284 | 36.737% | Pressure one-step improved. |
| `one_step_a_l2` | 0.09107 | 0.11251 | -23.531% | Velocity one-step became worse. |
| `one_step_b_l2` | 0.26306 | 0.18576 | 29.384% | Pressure one-step improved. |
| `rollout_a_l2` | 0.50404 | 0.36648 | 27.291% | Autonomous velocity rollout improved. |
| `rollout_b_l2` | 0.78247 | 0.41419 | 47.065% | Autonomous pressure rollout improved strongly. |
| `one_step_pressure_energy_error` | 0.19475 | 0.15975 | 17.967% | Pressure energy one-step improved. |
| `rollout_pressure_energy_error` | 1.3170 | 0.50811 | 61.418% | Pressure rollout energy drift improved strongly. |

Key result: adding the Hopf amplitude/energy envelope loss reduces mean autonomous rollout error despite slightly degrading RHS and one-step velocity fitting. Mean velocity rollout improves from 0.5040 to 0.3665, and mean pressure rollout improves from 0.7825 to 0.4142.

## Per-Re Rollout Comparison

| Re | Regime | Strong a-rollout | HopfAmp a-rollout | a improvement | Strong b-rollout | HopfAmp b-rollout | b improvement |
|---:|---|---:|---:|---:|---:|---:|---:|
| 24.630 | Steady | 0.31008 | 0.29380 | 5.2496% | 0.95848 | 0.55767 | 41.818% |
| 32.740 | Steady | 0.20938 | 0.20385 | 2.6408% | 0.71036 | 0.61098 | 13.991% |
| 39.685 | Steady | 0.16244 | 0.16119 | 0.77009% | 0.50557 | 0.35878 | 29.034% |
| 45.143 | Hopf | 0.12838 | 0.12827 | 0.08165% | 0.41802 | 0.26016 | 37.762% |
| 47.081 | Hopf | 0.80696 | 0.55872 | 30.762% | 1.0780 | 0.40070 | 62.831% |
| 49.022 | Hopf | 0.61354 | 0.36265 | 40.892% | 0.91664 | 0.40469 | 55.850% |
| 51.786 | Hopf | 3.0745 | 2.1361 | 30.521% | 3.7244 | 1.7221 | 53.761% |
| 70.315 | Periodic | 0.13936 | 0.12303 | 11.717% | 0.16806 | 0.16318 | 2.9011% |
| 100.352 | Periodic | 0.02288 | 0.02121 | 7.3030% | 0.02566 | 0.02517 | 1.9250% |
| 149.059 | Periodic | 0.02462 | 0.02303 | 6.4448% | 0.02949 | 0.02705 | 8.2907% |
| 189.862 | Periodic | 0.05235 | 0.01943 | 62.878% | 0.07242 | 0.02565 | 64.576% |

The largest absolute gains are concentrated exactly where V15_2 diagnosed the failure: Re=47.081, 49.022, and 51.786. Periodic high-Re cases remain already strong in both models; HopfAmp does not obviously break the periodic regime.

## Hopf Critical Re Diagnostics

| Re | Case | a-rollout | b-rollout | r_true_mean | r_pred_mean | amp L2 | log-amp MAE | overshoot mean | overshoot max | phase MAE | freq MAE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 47.081 | StrongBaseline32 | 0.80696 | 1.0780 | 0.00000 | 0.00003 | 20.841 | 2.8423 | 20.127 | 57.326 | 1.8581 | 0.65721 |
| 47.081 | HopfAmpEnvelopeLoss32 | 0.55872 | 0.40070 | 0.00000 | 0.00010 | 71.028 | 4.2563 | 73.211 | 142.89 | 1.6033 | 0.73537 |
| 49.022 | StrongBaseline32 | 0.61354 | 0.91664 | 0.00000 | 0.00003 | 9.3308 | 2.0559 | 9.3245 | 20.617 | 1.0594 | 0.50435 |
| 49.022 | HopfAmpEnvelopeLoss32 | 0.36265 | 0.40469 | 0.00000 | 0.00009 | 29.843 | 3.3792 | 30.286 | 52.551 | 1.5511 | 0.73233 |
| 51.786 | StrongBaseline32 | 3.0745 | 3.7244 | 0.00002 | 0.00004 | 1.0894 | 0.61396 | 1.9330 | 4.2434 | 0.48767 | 0.29282 |
| 51.786 | HopfAmpEnvelopeLoss32 | 2.1361 | 1.7221 | 0.00002 | 0.00009 | 2.9713 | 1.3436 | 4.2589 | 12.186 | 1.2694 | 0.73763 |

At the hardest Hopf point Re=51.786, HopfAmp cuts autonomous velocity rollout from 3.0745 to 2.1361 and pressure rollout from 3.7244 to 1.7221. The error is still far above the desired 10% target, but the direction is correct: directly constraining the `(a0,a1)` envelope reduces long-horizon drift more effectively than simply relying on the stronger ru=32 baseline.

The raw amplitude ratios should be interpreted carefully. For Re=47.081 and Re=49.022, the true `(a0,a1)` radius is very close to zero, so log-amplitude and overshoot ratios become ill-conditioned. In this interim comparison the Hopf envelope loss improves trajectory-level rollout L2 but does not yet produce a cleaner modal-radius match; that is the remaining gap the weighted rollout case is intended to test.

## Router and Expert Behavior

| Re band | Strong group selection | HopfAmp group selection | Notes |
|---|---|---|---|
| Steady/Hopf up to Re=51.786 | group 0 dominates | group 0 dominates | Router consistently identifies the low-Re/Hopf group. |
| Re=70.315 | mostly group 0, with some transition leakage | group 0 dominates | HopfAmp is slightly cleaner at the transition boundary. |
| Re=100.352-149.059 | group 1 dominates | group 1 dominates | Periodic mid-Re regime is stable. |
| Re=189.862 | group 2 mostly dominates | group 2 mostly dominates, with more group-0 mixing | High-Re routing remains interpretable but not perfectly sharp. |

Expert collapse diagnostics favor HopfAmp: StrongBaseline flags collapse at Re=149.059 and 189.862, while HopfAmp has no collapse flags in the completed summary. However, both cases still report many dead experts under the 1% threshold because group-top-k routing activates a narrow subset of the full expert pool.

## Interpretation

The completed two-case comparison supports three conclusions:

1. The Hopf envelope loss is useful for autonomous stability. It improves rollout metrics even though it worsens RHS and one-step velocity averages, which means the added loss is changing the learned dynamics rather than merely improving local regression.
2. Pressure benefits substantially from stabilizing the velocity envelope. Since pressure is closure-based on the evolved velocity state, the pressure rollout gain is larger than the pressure one-step gain.
3. Re=51.786 remains the bottleneck. HopfAmp reduces the drift but does not solve the near-critical amplitude problem; this is exactly why the still-running WeightedRollout case matters.

## Provisional Recommendation

Do not promote StrongBaseline32 as the V16 default yet. Among the two completed runs, HopfAmpEnvelopeLoss32 is the better candidate because it directly improves the target failure mode: autonomous Hopf rollout drift. The final decision should wait for `V15_3_HopfAmpEnvelopeLoss32_WithWeightedRollout`, because that case tests whether longer weighted Hopf rollout supervision can further reduce Re=51.786 without sacrificing periodic-regime performance.

## Artifacts

- Aggregate comparison CSV: `reports/data/v15_3_interim_aggregate_two_cases.csv`
- Per-Re comparison CSV: `reports/data/v15_3_interim_per_re_two_cases.csv`
- Hopf amplitude diagnostics CSV: `reports/data/v15_3_interim_hopf_amplitude_two_cases.csv`
- Remote raw summaries: generated under `/root/moe/V15_3_HopfFocused/test_results_v15_3/results/...` on the A40 cluster.

## Pending

- `V15_3_HopfAmpEnvelopeLoss32_WithWeightedRollout` is still running and should receive a separate addendum when complete.
- Possible V15_4 tests can be compared against this interim report without blocking the pending WeightedRollout addendum.
