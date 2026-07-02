# V14_3 PressureInputAblation 技术报告

## 1. 实验目的

本实验在 V14 HPRS-MoE 框架上验证 Pressure Head 的输入设计是否是 Low-Re 压力泛化失败的主要原因。实验只改变 Pressure Head 输入变量，保持以下内容完全一致：HPRS-MoE 分层专家结构、Router、Galerkin RHS、RK4 推进、loss 配置、优化器、训练轮数、dense V14 数据组织方式和 `--pressure-target=closure` 压力预测框架。

压力预测仍为：

```text
b_base = pressure_surrogate(a_next, Re)
b_pred = b_base + pressure_head(...)
```

对比三种输入：

- `PressureOnly`：V14 当前基线，保持现有压力相关输入逻辑不变。
- `VelocityOnly`：Pressure Head 输入改为 `[a_next, 0]`，只允许使用下一步速度 POD 系数。
- `Hybrid`：Pressure Head 输入改为 `[a_next, b_base]`，同时使用速度模态和 Poisson pressure base。

三种模式均保持 pressure expert 输入维度为 `r_u + r_p`，因此 Linear、Low-rank Quadratic 和 FFN 的参数化形式一致。

## 2. 评测协议

训练数据沿用 V14 原始 dense 时间采样，不采用 V14_2 的时间稀疏或 Re 稀疏训练。评测使用统一 10 个 Held-out Reynolds 数：

```text
50.0, 78.0906, 105.983, 132.743, 160.785,
187.285, 215.256, 244.354, 274.377, 300.0
```

每个模式统计 one-step velocity、one-step pressure、24-step rollout velocity、24-step rollout pressure、RHS 和 pressure energy 指标，并额外汇总 Low-Re (`Re <= 80`) 与 High-Re (`Re >= 240`) 分组。

## 3. 总体结果

| Mode | 1-step velocity | 1-step pressure | 24-step velocity | 24-step pressure | RHS | pressure energy 1-step | pressure energy rollout |
|---|---:|---:|---:|---:|---:|---:|---:|
| PressureOnly | 0.023002 | 0.121132 | 0.073183 | 0.164159 | 0.092572 | 0.044522 | 0.072131 |
| VelocityOnly | 0.025900 | 0.126876 | 0.099898 | 0.208693 | 0.091676 | 0.044907 | 0.066951 |
| Hybrid | 0.034230 | 0.133363 | 0.185170 | 0.292430 | 0.093774 | 0.059801 | 0.150939 |

从均值看，`PressureOnly` 仍是综合最优方案。`VelocityOnly` 的 rollout pressure 比基线更差，说明仅依赖速度模态并没有解决压力残差泛化问题。`Hybrid` 在 one-step 和 rollout 上均显著劣化，尤其 rollout velocity/pressure 都变差，说明在当前共享训练和相同容量压力专家下，直接拼接 `a_next` 与 `b_base` 会加重闭环耦合误差，而不是稳定利用 Poisson prior。

## 4. Pressure 指标分布

| Mode | 1-step pressure mean | std | min | max | 24-step pressure mean | std | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PressureOnly | 0.121132 | 0.283365 | 0.015303 | 0.970201 | 0.164159 | 0.332395 | 0.029859 | 1.159100 |
| VelocityOnly | 0.126876 | 0.290914 | 0.013987 | 0.998312 | 0.208693 | 0.361565 | 0.038131 | 1.285700 |
| Hybrid | 0.133363 | 0.291404 | 0.020652 | 1.006270 | 0.292430 | 0.297750 | 0.135839 | 1.179790 |

压力误差均值和标准差主要被 `Re=50` 拉高。除最低 Re 外，多数工况 one-step pressure 已经接近或低于 10%，但 Low-Re 极端点仍是决定整体泛化质量的瓶颈。

## 5. Low-Re 与 High-Re 分析

| Mode | Low-Re 1-step pressure | Low-Re 24-step pressure | Low-Re pressure energy | High-Re 1-step pressure | High-Re 24-step pressure | High-Re 24-step velocity |
|---|---:|---:|---:|---:|---:|---:|
| PressureOnly | 0.513168 | 0.619387 | 0.187532 | 0.032069 | 0.060324 | 0.052508 |
| VelocityOnly | 0.526232 | 0.721061 | 0.177031 | 0.038754 | 0.096665 | 0.093293 |
| Hybrid | 0.531471 | 0.699098 | 0.230862 | 0.041412 | 0.167404 | 0.147696 |

Low-Re 结论很清楚：`VelocityOnly` 和 `Hybrid` 都没有改善低 Re 压力泛化。`VelocityOnly` 的 Low-Re pressure energy 略低，但压力 L2 更差；`Hybrid` 的 Low-Re pressure energy 和 rollout 均明显更差。因此，当前 Low-Re pressure failure 不能简单归因于“Pressure Head 没有看到速度模态”。

High-Re 区间同样是 `PressureOnly` 最稳。`VelocityOnly` 和 `Hybrid` 的 High-Re rollout drift 增加，尤其 `Hybrid` 将 24-step pressure 从 0.0603 拉高到 0.1674，说明将 `b_base` 直接作为 head 输入没有自动带来更强的物理先验利用，反而可能放大闭环状态误差。

## 6. 结论

本次消融支持以下判断：

1. 当前 Pressure Head 输入设计不是 Low-Re pressure failure 的主要瓶颈。把输入改成速度模态或速度加 Poisson base 都没有改善 Re=50/78 的压力误差。
2. `PressureOnly` 仍应作为当前 V14/V14_3 的默认压力输入基线，因为它在 mean one-step pressure、mean rollout pressure、High-Re pressure 和 velocity rollout 上都最好。
3. `VelocityOnly` 没有证明“压力残差主要应依赖速度模态”。它的 one-step pressure 与 baseline 接近，但 rollout pressure 更差。
4. `Hybrid` 直接拼接 `[a_next, b_base]` 在当前结构下不稳定，说明 Poisson prior 需要更温和的注入方式，例如 detached/base-gated residual、单独 pressure encoder、per-regime pressure calibration 或显式 base-confidence weighting。
5. 下一阶段更值得优先检查 Poisson surrogate 本身、Low-Re pressure base 误差、pressure residual 的 Re 加权学习，以及 pressure branch 与 shared encoder/router 的梯度耦合，而不是继续简单替换 pressure head 输入。

## 7. 产物

- 自动聚合报告：`test_results_v14_3/results/v14_3_pressure_input_ablation_summary/TECHNICAL_REPORT_V14_3_PRESSURE_INPUT_ABLATION.md`
- 聚合 JSON：`test_results_v14_3/results/v14_3_pressure_input_ablation_summary/v14_3_pressure_input_ablation_aggregate.json`
- 汇总 CSV：`test_results_v14_3/results/v14_3_pressure_input_ablation_summary/v14_3_pressure_input_ablation_combined.csv`
- 曲线图：
  - `test_results_v14_3/results/v14_3_pressure_input_ablation_summary/v14_3_one_step_pressure_vs_re.svg`
  - `test_results_v14_3/results/v14_3_pressure_input_ablation_summary/v14_3_rollout_pressure_vs_re.svg`
  - `test_results_v14_3/results/v14_3_pressure_input_ablation_summary/v14_3_pressure_energy_vs_re.svg`
