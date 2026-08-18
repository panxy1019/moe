# V15 测试误差与专家激活专项整理

生成日期：2026-07-05。数据来源：三组 V15 训练完成后的 raw metrics JSON，经压缩整理为 `test_results_v15/results/V15_summary/v15_error_expert_activation.csv`。

## 误差是不是相对误差？

是。当前 V15 报告和 CSV 里的 `rhs_l2`、`one_step_velocity_l2`、`one_step_pressure_l2`、`rollout_velocity_l2`、`rollout_pressure_l2` 都是 relative L2，而不是未归一化 L2。代码定义为：

```text
relative_l2 = ||prediction - truth||_2 / (||truth||_2 + EPS)
```

因此数值可以直接乘以 100% 理解：`0.05` 约等于 5% 相对误差，`1.0` 约等于 100% 相对误差。`pressure_energy` 是相对能量误差，不是 L2。

字段对应关系：

| 报告字段 | 代码来源 | 含义 |
|---|---|---|
| `rhs_rel_l2` | `deep_moe.rhs_relative_l2` | learned RHS/operator 相对 L2 |
| `one_step_velocity_rel_l2` | `one_step_autonomous_pressure.a_relative_l2` | RK4 自主推进一步后的速度 POD 系数相对 L2 |
| `one_step_pressure_rel_l2` | `one_step_autonomous_pressure.b_relative_l2` | 一步后 pressure closure 输出的压力 POD 系数相对 L2 |
| `rollout24_velocity_rel_l2_mean` | `rollout_autonomous_pressure.a_relative_l2_mean` | 24-step autonomous rollout 窗口的速度相对 L2 均值 |
| `rollout24_pressure_rel_l2_mean` | `rollout_autonomous_pressure.b_relative_l2_mean` | 24-step autonomous rollout 窗口的压力相对 L2 均值 |

这里的 24-step rollout 是滚动预测：模型每一步把自己的 `a,b` 预测反馈给下一步，再连续 RK4 推 24 步；不是“预测 24 步后再喂一个真实值”。

## 专家激活字段怎么读

- V15 HPRS-MoE 有 3 个 physics groups、21 个专家。每组 7 个专家，其中第一个是 shared expert：`g0=e0-e6`，`g1=e7-e13`，`g2=e14-e20`；shared expert 分别是 `e0/e7/e14`。
- `dominant_group` 是该 Re 测试样本上 group router 平均 load 最大的 group。
- `dominant_expert` 是 velocity/pressure router 平均后，21 个专家中 mean gate load 最大的 expert。
- `active_experts_mean` 统计 gate > 1e-6 的专家个数均值；由于速度和压力 router 共享 regime 但各自有 Top-k，均值通常在 4-5 左右。
- `top_experts_by_load` 给出平均权重最高的专家；`top_experts_by_top1` 给出被选为 top1 最频繁的专家。

## Case 级误差总览

| Case | 1-step u | 1-step p | 24-step u | 24-step p | RHS | active experts | dominant behavior |
|---|---:|---:|---:|---:|---:|---:|---|
| V15_Base | 0.23345 ± 0.44965 | 1.6423 ± 2.3615 | 0.74618 ± 1.2323 | 3.5577 ± 6.9453 | 1.4795 | 4.2083 | most frequent g2/e0 |
| V15_LargeROM | 0.10283 ± 0.16740 | 0.60357 ± 0.61255 | 0.45314 ± 0.69708 | 1.4827 ± 2.6478 | 0.63563 | 4.5650 | most frequent g2/e14 |
| V15_BalancedTraining | 0.18406 ± 0.38390 | 0.59478 ± 0.66109 | 0.47304 ± 0.70224 | 1.0598 ± 1.6278 | 0.88903 | 4.4123 | most frequent g2/e14 |

## Per-Re One-step / Rollout / Expert Activation

### V15_Base

| Re | Regime | 1-step u | 1-step p | 24-step u | 24-step p | Dominant group | Dominant expert | Active experts | Top experts by load |
|---:|---|---:|---:|---:|---:|---|---|---:|---|
| 24.630 | steady_wake | 9.47% | 445.70% | 62.64% | 467.73% | g2 (1.0000) | e14 (0.54054) | 4.0000 | e14:0.540541, e19:0.228302, e20:0.226092 |
| 32.740 | steady_wake | 5.38% | 255.92% | 43.10% | 325.10% | g2 (1.0000) | e14 (0.54054) | 4.0000 | e14:0.540541, e19:0.228752, e20:0.2264 |
| 39.685 | steady_wake | 3.79% | 97.21% | 31.20% | 116.97% | g2 (0.93443) | e14 (0.50510) | 4.0000 | e14:0.505095, e19:0.213987, e20:0.211731 |
| 45.143 | pre_hopf_steady | 3.13% | 71.32% | 22.47% | 73.78% | g2 (0.70492) | e14 (0.38104) | 4.0000 | e14:0.381037, e19:0.161527, e20:0.159806 |
| 47.081 | hopf_transition | 33.62% | 43.80% | 102.54% | 199.81% | g0 (0.88608) | e0 (0.47896) | 4.0000 | e0:0.47896, e1:0.203591, e5:0.203519 |
| 49.022 | hopf_transition | 24.75% | 79.24% | 74.36% | 174.41% | g0 (0.93038) | e0 (0.50291) | 4.0127 | e0:0.502908, e1:0.213771, e5:0.213693 |
| 51.786 | hopf_transition | 162.14% | 789.36% | 452.31% | 2506.69% | g0 (0.64557) | e0 (0.34896) | 4.0316 | e0:0.348956, e14:0.191584, e1:0.148337 |
| 70.315 | developing_periodic_shedding | 5.56% | 9.71% | 18.55% | 27.59% | g0 (0.85443) | e0 (0.46185) | 4.4810 | e0:0.461854, e2:0.131871, e5:0.102602 |
| 100.352 | mature_periodic_shedding | 3.84% | 6.10% | 4.35% | 8.73% | g1 (0.97468) | e7 (0.52686) | 4.6646 | e7:0.526856, e8:0.206349, e12:0.163184 |
| 149.059 | mature_periodic_shedding | 2.60% | 4.06% | 3.64% | 5.49% | g1 (0.85443) | e7 (0.46185) | 4.4494 | e7:0.461854, e12:0.177943, e8:0.175409 |
| 189.862 | high_re_2d_periodic_near_modeA | 2.50% | 4.15% | 5.63% | 7.19% | g1 (0.58861) | e7 (0.31817) | 4.6519 | e7:0.318166, e14:0.222374, e12:0.131332 |

### V15_LargeROM

| Re | Regime | 1-step u | 1-step p | 24-step u | 24-step p | Dominant group | Dominant expert | Active experts | Top experts by load |
|---:|---|---:|---:|---:|---:|---|---|---:|---|
| 24.630 | steady_wake | 4.51% | 167.29% | 42.55% | 150.48% | g2 (0.59016) | e14 (0.31901) | 5.0000 | e14:0.319008, e7:0.212672, e17:0.135569 |
| 32.740 | steady_wake | 3.39% | 44.24% | 25.10% | 169.55% | g2 (0.54098) | e14 (0.29242) | 5.0000 | e14:0.292424, e0:0.177226, e17:0.124276 |
| 39.685 | steady_wake | 2.74% | 34.00% | 18.12% | 63.74% | g0 (0.49180) | e0 (0.26584) | 5.0000 | e0:0.26584, e14:0.212672, e2:0.112979 |
| 45.143 | pre_hopf_steady | 2.19% | 40.28% | 13.51% | 32.20% | g2 (0.47541) | e14 (0.25698) | 5.0000 | e14:0.256978, e0:0.230394, e17:0.109214 |
| 47.081 | hopf_transition | 13.61% | 113.05% | 71.60% | 147.44% | g2 (0.70253) | e14 (0.37975) | 4.8734 | e14:0.379746, e17:0.16139, e18:0.161389 |
| 49.022 | hopf_transition | 10.11% | 63.39% | 48.88% | 72.01% | g2 (0.77848) | e14 (0.42080) | 4.6835 | e14:0.4208, e17:0.178839, e18:0.178836 |
| 51.786 | hopf_transition | 62.07% | 177.96% | 255.67% | 963.97% | g2 (0.78481) | e14 (0.42422) | 4.2152 | e14:0.424221, e17:0.180297, e18:0.180287 |
| 70.315 | developing_periodic_shedding | 6.07% | 9.24% | 13.08% | 16.94% | g0 (0.95570) | e0 (0.51659) | 4.2785 | e0:0.516592, e4:0.210144, e2:0.116494 |
| 100.352 | mature_periodic_shedding | 3.27% | 4.58% | 3.31% | 5.12% | g1 (0.86076) | e7 (0.46527) | 4.2025 | e7:0.465275, e11:0.150921, e9:0.143345 |
| 149.059 | mature_periodic_shedding | 2.23% | 4.07% | 2.94% | 4.32% | g1 (0.94937) | e7 (0.51317) | 4.0127 | e7:0.513171, e9:0.227623, e8:0.15131 |
| 189.862 | high_re_2d_periodic_near_modeA | 2.93% | 5.83% | 3.70% | 5.20% | g1 (0.75949) | e7 (0.41054) | 3.9494 | e7:0.410536, e9:0.165881, e8:0.163541 |

### V15_BalancedTraining

| Re | Regime | 1-step u | 1-step p | 24-step u | 24-step p | Dominant group | Dominant expert | Active experts | Top experts by load |
|---:|---|---:|---:|---:|---:|---|---|---:|---|
| 24.630 | steady_wake | 3.60% | 187.19% | 25.69% | 81.39% | g2 (0.80328) | e14 (0.43420) | 4.1967 | e14:0.434205, e16:0.206553, e20:0.162488 |
| 32.740 | steady_wake | 2.81% | 101.88% | 22.31% | 108.83% | g2 (0.77049) | e14 (0.41648) | 4.2295 | e14:0.416482, e16:0.194215, e20:0.159777 |
| 39.685 | steady_wake | 2.70% | 29.55% | 19.61% | 59.26% | g2 (0.50820) | e14 (0.27470) | 4.4918 | e14:0.274701, e0:0.239256, e16:0.126539 |
| 45.143 | pre_hopf_steady | 2.69% | 28.71% | 22.84% | 56.14% | g0 (0.90164) | e0 (0.48737) | 4.9016 | e0:0.487373, e6:0.207129, e4:0.207114 |
| 47.081 | hopf_transition | 21.76% | 52.40% | 85.06% | 128.99% | g2 (0.62658) | e14 (0.33869) | 4.3734 | e14:0.338693, e0:0.181321, e16:0.153425 |
| 49.022 | hopf_transition | 18.29% | 48.42% | 66.02% | 96.31% | g2 (0.55063) | e14 (0.29764) | 4.4494 | e14:0.297639, e0:0.222374, e16:0.134877 |
| 51.786 | hopf_transition | 138.03% | 186.21% | 255.04% | 603.05% | g0 (0.67722) | e0 (0.36606) | 4.7342 | e0:0.366062, e6:0.155565, e4:0.155556 |
| 70.315 | developing_periodic_shedding | 4.31% | 7.05% | 9.27% | 13.33% | g1 (0.48101) | e7 (0.26001) | 4.2342 | e7:0.260007, e0:0.246322, e6:0.148598 |
| 100.352 | mature_periodic_shedding | 3.57% | 4.93% | 4.75% | 6.23% | g1 (0.96835) | e7 (0.52343) | 4.5696 | e7:0.523435, e9:0.154419, e11:0.128032 |
| 149.059 | mature_periodic_shedding | 2.36% | 3.87% | 4.50% | 5.60% | g1 (0.79747) | e7 (0.43106) | 4.2595 | e7:0.431063, e10:0.127681, e14:0.102634 |
| 189.862 | high_re_2d_periodic_near_modeA | 2.35% | 4.05% | 5.24% | 6.67% | g1 (0.62025) | e7 (0.33527) | 4.0949 | e7:0.335272, e14:0.205269, e9:0.116744 |

## 关键观察

- `V15_LargeROM` 的速度/RHS 仍是最强：平均 1-step velocity 相对误差 10.28%，24-step velocity 45.31%，RHS 63.56%。
- `V15_BalancedTraining` 的压力 rollout 最强：平均 24-step pressure 相对误差 105.98%，比 `V15_Base` 的 355.77% 低很多，但还没有达到 10% 以内。
- 成熟周期流区间最稳定：Re≈100/149/190 的 24-step pressure 在 `V15_LargeROM` 下约 5.12%/4.32%/5.20%。
- Hopf 临界附近仍是主要失败点：Re≈51.786 的 24-step pressure 即使在 BalancedTraining 中仍为 603.05%，说明长期漂移和压力闭合在临界振荡区仍不稳。
- 专家分工有明显 Re 依赖：低 Re/steady 常在 g2 与 g0 之间切换，Hopf 过渡点会从 g2 转向 g0，成熟周期流多数转向 g1；这符合 HPRS-MoE 用 group router 表示 physics regime 的设计初衷。
- 没有专家函数塌缩：本 CSV 的 `collapse_flag=False`，且报告中最大 expert cosine < 0.95；但 dead expert 数仍偏高，说明后续需要更强的 group 内负载均衡或更软的 group routing。
