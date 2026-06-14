# V6 完整技术报告：Autonomous-Pressure RK4 MoE-ROM

## 1. 任务目标

V6 以 V5 中表现最稳定的 `v5_r8_rk4_b2` 为技术基线，继续做两类升级：

1. 将 pressure/context 纳入自主推进。V5 的 rollout 使用真实 pressure coefficients `b_t` 作为 teacher-forced 输入；V6 新增 pressure-next head，在 rollout 中用模型预测的 `b_{t+1}` 更新下一步上下文，减少对真实 pressure 序列的依赖。
2. 在更高截断阶数上测试容量收益。分别测试 `r_u=16,r_p=16` 和 `r_u=32,r_p=32`，比较 2-block 稳定基线与 3-block 深层 MoE，判断低阶截断是否掩盖深层模型收益。

## 2. 数据与路径

运行数据沿用 V4 质量加权 POD 与半侵入式 Galerkin 张量库：

```text
/root/moe/V4/data
```

关键文件：

```text
global_velocity_pod_weighted_l2.npz
global_pressure_pod_weighted_l2.npz
pod_snapshot_index.csv
semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz
```

仓库内对应说明：

```text
V4/data/WEIGHTED_L2_POD_REPORT.md
V4/data/SEMI_INTRUSIVE_GALERKIN_TENSORS_COMPACT_README.md
V4/data/SEMI_INTRUSIVE_GALERKIN_TENSORS_allRe30_weightedL2_ru80_rp80.md
```

数据集概要：

| 项目 | 数值 |
|---|---:|
| Re 范围 | 500-1000 |
| Re 个数 | 30 |
| 每个 Re 保留快照 | 201 |
| 总快照数 | 6030 |
| V6 有效样本 | 5940 |
| 测试 Re index | `[12, 29]` |
| 测试 Re | `Re_706p896552`, `Re_1000p000000` |

POD 截断能量：

| 截断 | Velocity energy | Pressure energy |
|---|---:|---:|
| `r=16` | 0.8458909513 | 0.9078771065 |
| `r=32` | 0.9448681228 | 0.9730947061 |

## 3. 半侵入式方程

基础 Galerkin RHS：

```text
R_g(a_t,b_t,Re) = c_Re + A_Re a_t + H(a_t,a_t) + P b_t
```

V6 学习 correction：

```text
Delta_R(t) = adot_true(t) - R_g(a_t,b_t,Re)
R_model(t) = R_g(a_t,b_t,Re) + f_theta(x_t)
```

速度系数用 RK4 推进：

```text
k1 = R_model(a_t, b_t)
k2 = R_model(a_t + 0.5 dt k1, b_t)
k3 = R_model(a_t + 0.5 dt k2, b_t)
k4 = R_model(a_t + dt k3, b_t)
a_{t+1} = a_t + dt/6 * (k1 + 2k2 + 2k3 + k4)
```

V6 当前 RK4 stage 内 pressure `b_t` 保持为当前 stage 上下文，下一步 pressure 由 pressure head 预测：

```text
b_{t+1,pred} = g_theta(x_t)
```

## 4. V6 模型结构

V6 网络：

```text
PhysicalContextEncoder
  -> SharedRoutedMoEBlock x N
  -> alpha_next_head
  -> rhs_correction_head
  -> pressure_next_head
```

输入特征包含：

```text
Re_norm, 1000/Re,
sin/cos phase harmonics,
a_t, b_t, R_g(t),
||a_low||, ||a_mid||, ||a_high||, ||b_t||, ||R_g(t)||,
history window features
```

历史窗口 `history_len=3`，包含过去两步：

```text
a_{t-h}, b_{t-h}, R_g(t-h),
a_t-a_{t-h}, b_t-b_{t-h}, R_g(t)-R_g(t-h)
```

与 V5 的关键差异：

| 模块 | V5 | V6 |
|---|---|---|
| 输出头 | alpha-next, RHS correction | alpha-next, RHS correction, pressure-next |
| rollout pressure | teacher-forced true `b_t` | autonomous predicted `b_t` |
| rollout history | 真实历史 pressure/RHS | 预测 `a,b,R_g` 滚动历史 |
| 评估 | teacher-forced pressure rollout | 同时输出 teacher-forced 与 autonomous-pressure rollout |

说明：Re、time index、phase 仍按已知测试序列推进；本轮主要移除 pressure coefficients 和 pressure history 的 teacher forcing。

## 5. 损失函数

V6 总损失：

```text
L = L_coeff
  + L_dyn
  + lambda_pressure * L_pressure
  + lambda_recon * L_recon
  + lambda_consistency * L_consistency
  + lambda_router_balance * L_router_balance
  + lambda_router_entropy * L_router_entropy
  + lambda_router_smooth * L_router_smooth
  + lambda_rollout * L_rollout_autonomous
```

新增项：

| Loss | 说明 |
|---|---|
| `L_pressure` | pressure-next head 预测 `b_{t+1}` 的 MSE |
| `L_rollout_autonomous` | rollout 中同时惩罚 `a` 误差和 `b` 误差 |

Autonomous rollout 训练损失：

```text
L_rollout_autonomous =
  MSE((a_pred - a_true) / alpha_scale)
  + lambda_pressure_rollout * MSE((b_pred - b_true) / pressure_scale)
```

本轮为控制成本，正式实验使用轻量 rollout curriculum：

```text
curriculum_steps = [1, 2, 4, 8]
train_rollout_steps = 8
eval_rollout_steps = 16
rollout_batch = 1
rollout_every_batches = 3
```

备注：最初尝试每个 batch 都做更重的 autonomous RK4 rollout，第一组运行超过 10 分钟仍未完成，因此本报告采用上述轻量预算。所有正式实验使用相同预算，组间可比。

## 6. 实验配置

| 实验 | r | Blocks | Experts | Hidden | Expert hidden | Epochs | LR | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v6_r16_rk4_b2_autop` | 16 | 2 | 6 | 128 | 192 | 45 | 0.00055 | 222.88 s |
| `v6_r16_rk4_b3_deep_autop` | 16 | 3 | 8 | 160 | 224 | 45 | 0.00045 | 263.61 s |
| `v6_r32_rk4_b2_autop` | 32 | 2 | 6 | 160 | 224 | 40 | 0.00045 | 238.36 s |
| `v6_r32_rk4_b3_deep_autop` | 32 | 3 | 8 | 192 | 256 | 40 | 0.00035 | 286.24 s |

共同设置：

| 参数 | 值 |
|---|---:|
| Integrator | RK4 |
| top_k | 2 |
| temperature | 0.8 |
| history_len | 3 |
| batch_size r16 | 512 |
| batch_size r32 | 384 |
| recon_dim | 1024 |
| pressure loss weight | 0.60 |
| pressure rollout weight | 0.35 |
| router balance | 0.02 |
| router entropy | 0.002 |

## 7. 结果一：Autonomous Pressure 是否可行

V6 同时输出：

- `TF rollout`: teacher-forced pressure rollout，使用真实 `b_t`。
- `Auto rollout`: autonomous-pressure rollout，使用模型预测的 `b_t` 和预测历史。

| 实验 | Re | TF rollout a mean | Auto rollout a mean | Auto rollout b mean | Auto a one-step | Auto b one-step |
|---|---:|---:|---:|---:|---:|---:|
| r16 b2 | 706.8966 | 0.114264 | 0.082170 | 0.234134 | 0.012887 | 0.222601 |
| r16 b2 | 1000.0000 | 0.094289 | 0.084550 | 0.343437 | 0.014910 | 0.378840 |
| r16 b3 deep | 706.8966 | 0.100626 | 0.078395 | 0.204854 | 0.013368 | 0.241275 |
| r16 b3 deep | 1000.0000 | 0.073489 | 0.067527 | 0.232963 | 0.015725 | 0.388222 |
| r32 b2 | 706.8966 | 0.086063 | 0.073630 | 0.241874 | 0.019142 | 0.332467 |
| r32 b2 | 1000.0000 | 0.079429 | 0.073476 | 0.368746 | 0.016006 | 0.460991 |
| r32 b3 deep | 706.8966 | 0.086851 | 0.064829 | 0.236345 | 0.019079 | 0.281481 |
| r32 b3 deep | 1000.0000 | 0.071151 | 0.067171 | 0.227352 | 0.014820 | 0.426688 |

判断：

- pressure 自主推进没有导致速度 rollout 立即发散。所有 `Auto rollout a mean` 均在 `0.0648-0.0846` 范围。
- pressure coefficients 自身仍较难预测，`Auto rollout b mean` 在 `0.2049-0.3687` 范围，是 V6 后续主要误差源。
- 多数组合中 autonomous-pressure velocity rollout 甚至低于 teacher-forced pressure rollout。原因可能是 pressure head 预测较平滑，短窗口内对速度 RHS 有正则化效果；这不代表 pressure 预测已经完全准确。

## 8. 结果二：r16 下深层模型收益

| Re | 模型 | RHS L2 | Pressure-next L2 | Auto rollout a | Auto rollout b | TF rollout a | Load CV | Entropy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 706.8966 | r16 b2 | 0.051156 | 0.222601 | 0.082170 | 0.234134 | 0.114264 | 0.571965 | 0.910739 |
| 706.8966 | r16 b3 deep | 0.054975 | 0.241275 | 0.078395 | 0.204854 | 0.100626 | 0.402917 | 1.339254 |
| 1000.0000 | r16 b2 | 0.071044 | 0.378840 | 0.084550 | 0.343437 | 0.094289 | 0.480212 | 0.958113 |
| 1000.0000 | r16 b3 deep | 0.074866 | 0.388222 | 0.067527 | 0.232963 | 0.073489 | 0.330871 | 1.297341 |

深层相对 b2 的变化：

| Re | Auto rollout a | Auto rollout b | TF rollout a | RHS L2 |
|---:|---:|---:|---:|---:|
| 706.8966 | -4.59% | -12.51% | -11.94% | +7.47% |
| 1000.0000 | -20.13% | -32.17% | -22.06% | +5.38% |

结论：在 `r=16` 下，深层模型没有降低 RHS L2，甚至略差；但它明显改善长期 rollout，尤其 `Re_1000p000000` 的 autonomous velocity rollout 降低约 20%，pressure rollout 降低约 32%。这说明深层容量收益更多体现在多步稳定性，而不是单步 RHS 拟合。

## 9. 结果三：r32 下深层模型收益

| Re | 模型 | RHS L2 | Pressure-next L2 | Auto rollout a | Auto rollout b | TF rollout a | Load CV | Entropy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 706.8966 | r32 b2 | 0.072203 | 0.332467 | 0.073630 | 0.241874 | 0.086063 | 0.417958 | 1.028327 |
| 706.8966 | r32 b3 deep | 0.072778 | 0.281481 | 0.064829 | 0.236345 | 0.086851 | 0.489897 | 1.323993 |
| 1000.0000 | r32 b2 | 0.063484 | 0.460991 | 0.073476 | 0.368746 | 0.079429 | 0.598251 | 0.847409 |
| 1000.0000 | r32 b3 deep | 0.057164 | 0.426688 | 0.067171 | 0.227352 | 0.071151 | 0.290717 | 1.360223 |

深层相对 b2 的变化：

| Re | Auto rollout a | Auto rollout b | TF rollout a | RHS L2 |
|---:|---:|---:|---:|---:|
| 706.8966 | -11.95% | -2.29% | +0.92% | +0.80% |
| 1000.0000 | -8.58% | -38.34% | -10.42% | -9.96% |

结论：在 `r=32` 下，深层模型的收益更稳定地出现在 autonomous rollout 中。`Re_1000p000000` 同时改善 RHS、pressure-next、teacher-forced rollout 和 autonomous rollout；`Re_706p896552` 的 RHS 基本持平，但 autonomous velocity rollout 仍下降约 12%。

## 10. r16/r32 是否说明低阶截断掩盖深层收益

V5 的 `r=8` 测试中，深层 MoE 的收益不稳定：RHS 不一定更好，rollout 有时改善、有时变差。V6 在 `r=16` 和 `r=32` 下观察到：

1. 深层模型在两个截断阶数上都降低了 autonomous velocity rollout mean。
2. 深层模型在 pressure autonomous rollout 上也多数显著改善，尤其高 Re。
3. RHS L2 不是深层收益的充分指标；深层模型可能牺牲一点 RHS 单步拟合，换来更平滑、更稳定的长期推进。
4. `r=32` 的 POD 能量更高，velocity energy 从 `r16` 的 0.8459 提升到 0.9449；此时深层模型对 `Re_1000p000000` 的 RHS 和 rollout 均更优，说明低阶截断确实可能掩盖深层模型的容量收益。

总体判断：**是的，低阶截断会掩盖一部分深层 MoE 的长期预测收益。** 但这种收益主要应通过 autonomous rollout、pressure rollout 和路由诊断判断，而不能只看 RHS L2。

## 11. 专家路由诊断

| 实验 | Re | Mean load | Top-1 fraction | Load CV | Entropy | Dead experts |
|---|---:|---|---|---:|---:|---:|
| r16 b2 | 706.8966 | `[0.185,0.063,0.101,0.137,0.152,0.362]` | `[0.071,0.025,0.106,0.157,0.157,0.485]` | 0.571965 | 0.910739 | 0 |
| r16 b2 | 1000.0000 | `[0.073,0.328,0.109,0.154,0.178,0.158]` | `[0.056,0.293,0.111,0.152,0.207,0.182]` | 0.480212 | 0.958113 | 0 |
| r16 b3 | 706.8966 | `[0.069,0.181,0.157,0.071,0.175,0.112,0.178,0.057]` | `[0.066,0.081,0.071,0.076,0.313,0.172,0.177,0.045]` | 0.402917 | 1.339254 | 0 |
| r16 b3 | 1000.0000 | `[0.197,0.175,0.081,0.113,0.087,0.082,0.149,0.116]` | `[0.172,0.298,0.071,0.076,0.086,0.061,0.111,0.126]` | 0.330871 | 1.297341 | 0 |
| r32 b2 | 706.8966 | `[0.099,0.266,0.255,0.153,0.090,0.136]` | `[0.081,0.343,0.232,0.121,0.111,0.111]` | 0.417958 | 1.028327 | 0 |
| r32 b2 | 1000.0000 | `[0.084,0.341,0.082,0.128,0.263,0.101]` | `[0.091,0.419,0.076,0.116,0.237,0.061]` | 0.598251 | 0.847409 | 0 |
| r32 b3 | 706.8966 | `[0.212,0.076,0.083,0.068,0.089,0.174,0.077,0.220]` | `[0.197,0.066,0.086,0.025,0.081,0.071,0.101,0.374]` | 0.489897 | 1.323993 | 0 |
| r32 b3 | 1000.0000 | `[0.059,0.172,0.127,0.074,0.152,0.138,0.134,0.145]` | `[0.030,0.197,0.131,0.051,0.081,0.167,0.157,0.187]` | 0.290717 | 1.360223 | 0 |

观察：

- 所有实验 dead experts 均为 0，没有 mean-load 级别塌缩。
- 深层模型 entropy 明显更高，说明 gate 更分散，专家协同更多。
- r32 b3 在 `Re_1000p000000` 的 load CV 从 b2 的 0.598 降到 0.291，路由均衡性明显改善，这与 rollout 改善一致。

## 12. 文件索引

```text
V6/
  README.md
  TECHNICAL_REPORT_V6.md
  test_results_v6/
    README.md
    deep_moe_rom_v6.py
    results/
      v6_r16_rk4_b2_autop_metrics.json
      v6_r16_rk4_b2_autop_summary.md
      v6_r16_rk4_b3_deep_autop_metrics.json
      v6_r16_rk4_b3_deep_autop_summary.md
      v6_r32_rk4_b2_autop_metrics.json
      v6_r32_rk4_b2_autop_summary.md
      v6_r32_rk4_b3_deep_autop_metrics.json
      v6_r32_rk4_b3_deep_autop_summary.md
```

## 13. 结论

1. V6 autonomous-pressure rollout 可行：预测 pressure 进入后续上下文后，速度系数 rollout 没有发散，并且在多个配置中优于 teacher-forced pressure rollout。
2. pressure 预测仍是主要瓶颈：pressure rollout 误差明显高于 velocity rollout 误差，后续应加强 pressure dynamics 建模。
3. 更高截断阶数揭示了深层 MoE 的长期收益：r16/r32 下 deep 模型多数改善 autonomous rollout，尤其高 Re。
4. 不能只用 RHS L2 选择模型：V6 中深层模型的 RHS 有时略差，但长期 autonomous rollout 更好。
5. 推荐下一步以 `r32 + RK4 + 3-block MoE + autonomous pressure` 为主线，增加训练预算并加入 pressure dynamics residual 或 pressure Galerkin surrogate。
