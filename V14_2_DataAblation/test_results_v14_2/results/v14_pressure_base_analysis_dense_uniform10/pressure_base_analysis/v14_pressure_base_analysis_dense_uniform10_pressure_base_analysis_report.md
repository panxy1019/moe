# Pressure Base Analysis Report

## Scope

This diagnostic keeps the V14/V14_2 HPRS-MoE structure and training setup fixed. Only the pressure readout is switched at evaluation time:

- BaseOnly: `b_pred = b_base`
- ResidualOnly(State): `b_pred = pressure_head`
- Closure(Current): `b_pred = b_base + pressure_head`

Source metrics JSON: `/root/moe/V14_2_DataAblation/test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/v14_pressure_base_analysis_dense_uniform10_metrics.json`

Checkpoint: `/root/moe/V14_2_DataAblation/test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/v14_pressure_base_analysis_dense_uniform10_Re_50p000000_checkpoint.pt`

## Poisson Surrogate Notes

- - 计算 Re 数量：`100`
- 不可压缩动量方程取散度后采用压力泊松形式：
- - `L.shape = (80, 80)`
- - `H_tilde.shape = (80, 80, 80)`
- b(t) = c_tilde + A_tilde a(t) + H_tilde(a(t),a(t))

## Aggregate Findings

| Phase | BaseOnly pressure L2 | ResidualOnly pressure L2 | Closure pressure L2 | Closure improvement vs Base | Residual/Base | Contribution |
|---|---:|---:|---:|---:|---:|---:|
| one_step | 0.509786 | 1.03521 | 0.121132 | 86.7% | 0.480756 | 0.313085 |
| rollout | 0.573008 | 1.06439 | 0.164159 | 80.7% | 0.564784 | 0.354618 |

- Poisson base judgement: BaseOnly is not accurate enough; the pressure head is compensating for a large base error.
- Pressure head role: mean residual/base = 0.4808, mean contribution ratio = 0.3131; this behaves as a small-to-moderate correction.
- Low-Re diagnosis: Base error is already large at low Re; residual learning cannot fully repair it.
- High-Re diagnosis: High-Re pressure is mainly limited by the base/residual readout rather than rollout drift alone.

## Per-Re Comparison

| Re | Base one-step | ResidualOnly one-step | Closure one-step | Improve % | Base rollout | ResidualOnly rollout | Closure rollout | Roll improve % | Residual/Base | Dominant? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 1.33211 | 1.53721 | 0.970201 | 27.2% | 1.40978 | 1.66465 | 1.1591 | 17.8% | 1.03254 | True |
| 78.0906 | 0.851837 | 1.25047 | 0.0561346 | 93.4% | 1.05348 | 1.49232 | 0.0796697 | 92.4% | 0.703012 | False |
| 105.983 | 0.554848 | 1.07861 | 0.0171477 | 96.9% | 0.605975 | 1.03588 | 0.0298593 | 95.1% | 0.521879 | False |
| 132.743 | 0.426108 | 1.03016 | 0.015303 | 96.4% | 0.471628 | 0.961354 | 0.0481885 | 89.8% | 0.414411 | False |
| 160.785 | 0.340339 | 0.989226 | 0.0200866 | 94.1% | 0.372811 | 0.929136 | 0.0546321 | 85.3% | 0.348164 | False |
| 187.285 | 0.306344 | 0.952666 | 0.0178647 | 94.2% | 0.333324 | 0.907645 | 0.0537741 | 83.9% | 0.32569 | False |
| 215.256 | 0.325245 | 0.911835 | 0.0183753 | 94.4% | 0.354439 | 0.900965 | 0.0353857 | 90% | 0.358821 | False |
| 244.354 | 0.302435 | 0.882227 | 0.0206708 | 93.2% | 0.352826 | 0.89497 | 0.0318397 | 91% | 0.344769 | False |
| 274.377 | 0.321265 | 0.87527 | 0.0240637 | 92.5% | 0.38047 | 0.928245 | 0.0429471 | 88.7% | 0.37026 | False |
| 300 | 0.337331 | 0.844405 | 0.051472 | 84.7% | 0.39534 | 0.928743 | 0.106186 | 73.1% | 0.388012 | False |

## Direct Answers

1. Poisson surrogate enough precision? BaseOnly is not accurate enough; the pressure head is compensating for a large base error. Mean one-step BaseOnly pressure L2 is 0.5098; mean rollout BaseOnly pressure L2 is 0.573.

2. Is the head a small residual? It is a small-to-moderate correction. Mean one-step residual/base is 0.4808, and contribution ratio is 0.3131.

3. Low-Re failure source: Base error is already large at low Re; residual learning cannot fully repair it. Low-Re mean base/closure one-step errors are 1.092/0.5132.

4. High-Re pressure source: High-Re pressure is mainly limited by the base/residual readout rather than rollout drift alone. High-Re mean closure one-step/rollout errors are 0.03207/0.06032.

5. Current bottleneck: judge from the table above. If Closure only marginally improves Base while residual/base is large, the bottleneck is the Poisson surrogate plus base-residual coupling. If Base is good but Closure degrades, residual learning is the main issue. If one-step is good but rollout is poor, pressure is mainly limited by autonomous trajectory drift.

## Artifacts

- Comparison CSV: `/root/moe/V14_2_DataAblation/test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_pressure_mode_comparison.csv`
- one_step_error_vs_re: `/root/moe/V14_2_DataAblation/test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_one_step_pressure_error_vs_re.svg`
- rollout_error_vs_re: `/root/moe/V14_2_DataAblation/test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_rollout_pressure_error_vs_re.svg`
- base_error_residual_magnitude_vs_re: `/root/moe/V14_2_DataAblation/test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_base_error_residual_magnitude_vs_re.svg`
- one_step_error_distribution: `/root/moe/V14_2_DataAblation/test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_one_step_pressure_error_distribution.svg`
- rollout_error_distribution: `/root/moe/V14_2_DataAblation/test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_rollout_pressure_error_distribution.svg`

Runtime: 1134.49 s.
