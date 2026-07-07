# Pressure Base Analysis Dense Uniform10 Report

本实验在当前 `--pressure-target=closure` 框架下完成，保持 HPRS-MoE、Galerkin、RK4、loss、router、expert、超参数和训练流程不变，只新增诊断评估。训练数据组织使用 V14 原始密集时间采样：`train_time_stride=1`、`train_re_stride=1`，10 个 held-out Re 为 `50.0, 78.0906, 105.983, 132.743, 160.785, 187.285, 215.256, 244.354, 274.377, 300.0`。

模型训练跑满 240 epoch，最佳 checkpoint 选在 epoch 200。训练后对同一个 checkpoint 进行三种压力读出评估：

- `BaseOnly`: `b_pred = b_base`
- `ResidualOnly(State)`: `b_pred = pressure_head`
- `Closure(Current)`: `b_pred = b_base + pressure_head`

## 核心结论

1. 当前 Poisson surrogate 本身不够精确，不能单独作为压力预测器。BaseOnly 的平均 one-step pressure L2 为 `50.98%`，平均 24-step rollout pressure L2 为 `57.30%`。
2. Pressure Head 不是完整 state predictor。ResidualOnly(State) 平均 one-step/rollout pressure L2 为 `103.52% / 106.44%`，说明 head 单独关闭 base 后基本失效。
3. Closure 组合是有效的。Closure 平均 one-step/rollout pressure L2 降到 `12.11% / 16.42%`；若去掉 Re=50 低 Re 失效点，中高 Re 的压力误差大多进入 10% 以内。
4. Pressure Head 不是纯“小修小补”，但也不是整体主导项。平均 `Residual-to-Base Ratio=0.481`，平均 contribution ratio 为 `0.313`；只有 Re=50 下 residual/base `>1`，head 变成主导补偿。
5. Low-Re 压力失效主要来自 Base 误差过大和 residual-base 耦合难以修复。Re=50 的 BaseOnly one-step/rollout 为 `133.21% / 140.98%`，Closure 后仍为 `97.02% / 115.91%`。
6. High-Re 压力不是单纯 Poisson base 限制。Re=300 的 BaseOnly one-step 为 `33.73%`，Closure one-step 降到 `5.15%`，但 rollout 升到 `10.62%`，说明高 Re 后期更多受自主滚动漂移影响。

## Aggregate Metrics

| Phase | BaseOnly pressure L2 | ResidualOnly pressure L2 | Closure pressure L2 | Closure improvement vs Base | Residual/Base | Contribution |
|---|---:|---:|---:|---:|---:|---:|
| one-step | 0.5098 | 1.0352 | 0.1211 | 86.69% | 0.4808 | 0.3131 |
| 24-step rollout | 0.5730 | 1.0644 | 0.1642 | 80.71% | 0.5648 | 0.3546 |

## Per-Re Pressure Comparison

| Re | Base one-step | ResidualOnly one-step | Closure one-step | Base rollout | ResidualOnly rollout | Closure rollout | Residual/Base | Head dominant |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50.0000 | 1.3321 | 1.5372 | 0.9702 | 1.4098 | 1.6646 | 1.1591 | 1.0325 | true |
| 78.0906 | 0.8518 | 1.2505 | 0.0561 | 1.0535 | 1.4923 | 0.0797 | 0.7030 | false |
| 105.9831 | 0.5548 | 1.0786 | 0.0171 | 0.6060 | 1.0359 | 0.0299 | 0.5219 | false |
| 132.7430 | 0.4261 | 1.0302 | 0.0153 | 0.4716 | 0.9614 | 0.0482 | 0.4144 | false |
| 160.7854 | 0.3403 | 0.9892 | 0.0201 | 0.3728 | 0.9291 | 0.0546 | 0.3482 | false |
| 187.2852 | 0.3063 | 0.9527 | 0.0179 | 0.3333 | 0.9076 | 0.0538 | 0.3257 | false |
| 215.2556 | 0.3252 | 0.9118 | 0.0184 | 0.3544 | 0.9010 | 0.0354 | 0.3588 | false |
| 244.3544 | 0.3024 | 0.8822 | 0.0207 | 0.3528 | 0.8950 | 0.0318 | 0.3448 | false |
| 274.3767 | 0.3213 | 0.8753 | 0.0241 | 0.3805 | 0.9282 | 0.0429 | 0.3703 | false |
| 300.0000 | 0.3373 | 0.8444 | 0.0515 | 0.3953 | 0.9287 | 0.1062 | 0.3880 | false |

## 问题回答

### Poisson surrogate 是否已有足够精度？

没有。`pressure_poisson_surrogate_tensors_allRe100_weightedL2_ru80_rp80.npz` 的代数形式

```text
b_base = c_tilde(Re) + A_tilde(Re) a_next + H_tilde(a_next, a_next)
```

提供了有用的物理 base，但 BaseOnly 平均误差仍然很高：one-step `50.98%`、rollout `57.30%`。它不能单独满足 10% 压力目标。

### Pressure Head 是小幅 residual correction，还是承担大部分预测？

整体看是中等幅度 residual correction，不是完整压力预测器。平均 contribution ratio 为 `31.31%`，平均 residual/base 为 `48.08%`。但在 Re=50，residual/base 达到 `1.03`，pressure head 已经变成主导补偿项，说明低 Re 处 closure 语义被迫偏离“小修正”。

### Low-Re 压力失效来自 Base 误差还是 Residual 学习失败？

主要来自 Base 误差过大，并伴随 residual 学习无法完全修复。Re=50 的 BaseOnly one-step/rollout 已经是 `133.21% / 140.98%`，Closure 只能降到 `97.02% / 115.91%`。ResidualOnly 也超过 `150%`，说明 head 自身没有学成独立 state predictor。

### High-Re Pressure 主要受 Base 限制还是 Rollout Drift 影响？

高 Re 的 Base 仍不准，但 Closure 能在 one-step 大幅修正。Re=300 的 BaseOnly one-step 为 `33.73%`，Closure one-step 为 `5.15%`；24-step rollout 则升到 `10.62%`。因此高 Re 的最终压力误差更多体现为 autonomous rollout drift 叠加，而不是 base 完全不可修。

### 当前 Pressure 分支真正瓶颈是什么？

瓶颈不是单一因素：

- `Poisson Surrogate 本身`: BaseOnly 误差过高，是压力分支的基础瓶颈。
- `Residual Learning`: 中高 Re residual 学得有效，但 Re=50 失效，说明低 Re residual 学习能力不足。
- `二者耦合方式`: Closure 依赖 base 与 residual 的抵消关系；ResidualOnly 失效表明 head 没有稳健 state 语义。后续若追求低 Re 和长期 rollout 稳定性，应优先重构 Pressure Poisson base 或引入更强的 pressure-consistency/Poisson residual 约束，而不是单纯增大 head。

## Artifacts

- 自动报告：[pressure_base_analysis_report.md](test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_pressure_base_analysis_report.md)
- 完整 JSON：[pressure_base_analysis.json](test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_pressure_base_analysis.json)
- 对比表 CSV：[pressure_mode_comparison.csv](test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_pressure_mode_comparison.csv)
- 误差分布 CSV：[pressure_error_distribution.csv](test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_pressure_error_distribution.csv)
- One-step Error vs Re：[one_step_pressure_error_vs_re.svg](test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_one_step_pressure_error_vs_re.svg)
- Rollout Error vs Re：[rollout_pressure_error_vs_re.svg](test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_rollout_pressure_error_vs_re.svg)
- Base Error and Residual Magnitude vs Re：[base_error_residual_magnitude_vs_re.svg](test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_base_error_residual_magnitude_vs_re.svg)
- One-step Error Distribution：[one_step_pressure_error_distribution.svg](test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_one_step_pressure_error_distribution.svg)
- Rollout Error Distribution：[rollout_pressure_error_distribution.svg](test_results_v14_2/results/v14_pressure_base_analysis_dense_uniform10/pressure_base_analysis/v14_pressure_base_analysis_dense_uniform10_rollout_pressure_error_distribution.svg)
