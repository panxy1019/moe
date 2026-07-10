# V16_2 AttractorStabilityMoE Aggregate Report

Baseline is `V16_1_SteadyPressureAnchor32`. V16_2 cases are independent: steady contraction, Hopf log-radius normal-form, and regime-grouped routing.

## Overall Means

| Case | 1-step u | 1-step p | 24-step u | 24-step p | RHS | p energy | alpha | active experts | dead experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `V16_1_SteadyPressureAnchor32` | 0.03994 | 0.1229 | 0.2257 | 0.2967 | 0.3105 | 0.3143 | 0.2421 | 3.512 | 15.64 |
| `V16_2_SteadyContractivePressureROM32` | 0.05324 | 0.1205 | 0.2625 | 0.4512 | 0.3597 | 0.683 | 0.2456 | 4.166 | 14.09 |
| `V16_2_HopfLogRadiusNormalForm32` | 0.3443 | 0.8527 | 1.089 | 2.191 | 2.1 | 7.326 | 0.4575 | 4.792 | 14.18 |
| `V16_2_RegimeGroupedMoE32` | 0.04052 | 0.2574 | 0.2847 | 0.4764 | 0.3299 | 0.6105 | 0.2203 | 12.63 | 7.273 |

## Regime Means

### Steady

| Case | 1-step u | 1-step p | 24-step u | 24-step p | p energy | active experts |
|---|---:|---:|---:|---:|---:|---:|
| `V16_1_SteadyPressureAnchor32` | 0.01546 | 0.1516 | 0.1998 | 0.4071 | 0.5686 | 3 |
| `V16_2_SteadyContractivePressureROM32` | 0.01661 | 0.1414 | 0.1937 | 0.5409 | 0.2477 | 3.922 |
| `V16_2_HopfLogRadiusNormalForm32` | 0.09625 | 0.6359 | 0.6543 | 2.194 | 5.984 | 5 |
| `V16_2_RegimeGroupedMoE32` | 0.01414 | 0.2561 | 0.1752 | 0.7132 | 0.7101 | 13 |

### Hopf

| Case | 1-step u | 1-step p | 24-step u | 24-step p | p energy | active experts |
|---|---:|---:|---:|---:|---:|---:|
| `V16_1_SteadyPressureAnchor32` | 0.07662 | 0.1825 | 0.4953 | 0.4627 | 0.3747 | 3 |
| `V16_2_SteadyContractivePressureROM32` | 0.1271 | 0.1948 | 0.6307 | 0.8354 | 2.123 | 3.962 |
| `V16_2_HopfLogRadiusNormalForm32` | 1.014 | 1.925 | 2.619 | 4.305 | 18.27 | 5 |
| `V16_2_RegimeGroupedMoE32` | 0.09943 | 0.5602 | 0.7507 | 0.7157 | 1.262 | 13 |

### Periodic

| Case | 1-step u | 1-step p | 24-step u | 24-step p | p energy | active experts |
|---|---:|---:|---:|---:|---:|---:|
| `V16_1_SteadyPressureAnchor32` | 0.03691 | 0.04951 | 0.04933 | 0.06178 | 0.01485 | 4.407 |
| `V16_2_SteadyContractivePressureROM32` | 0.03449 | 0.0437 | 0.05526 | 0.07347 | 0.03798 | 4.562 |
| `V16_2_HopfLogRadiusNormalForm32` | 0.08998 | 0.265 | 0.3774 | 0.603 | 0.4597 | 4.429 |
| `V16_2_RegimeGroupedMoE32` | 0.02273 | 0.03153 | 0.04458 | 0.06026 | 0.0222 | 12 |

## Output Files

- `v16_2_summary_metrics.json`
- `v16_2_per_re_metrics.csv`
- `v16_2_hopf_near_onset_diagnostics.csv`
- `v16_2_steady_pressure_drift.csv`
- `v16_2_periodic_degradation.csv`
- `v16_2_router_diagnostics.csv`
