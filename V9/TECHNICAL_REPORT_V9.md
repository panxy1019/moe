# V9 技术报告：精度优先的 Re50-300 Pressure-Surrogate MoE-ROM

日期：2026-06-20

代码：`test_results_v9/deep_moe_rom_v9.py`

结果目录：`test_results_v9/results/`

## 1. 目标

V8 在 `Re=50-300` 新数据集上证明了 V7 的 pressure-surrogate residual 架构可以迁移，但精度仍不理想，特别是：

- 低 Re `Re_56p374525` 的 pressure relative error 偏高。
- `r32-b3` 虽然 POD 能量更高，但训练不稳，高阶模态没有转化为更好的 rollout。
- router 在部分低阶配置上出现 dead experts。

V9 的目标是从提高精度出发，保留物理结构，同时调整训练目标和训练预算。

## 2. 保留的 V7/V8 物理结构

V9 没有取消 V7 的刚性压力物理基线，仍然使用：

```text
a_next = RK4(a_t, Galerkin velocity tensors + learned RHS correction)
b_base = c_tilde + A_tilde @ a_next
       + torch.einsum("pij,bi,bj->bp", H_tilde, a_next, a_next)
delta_b = pressure_next_head(x_t)
b_next = b_base + delta_b
```

也就是说，`pressure_next_head` 仍然只学习 pressure Poisson surrogate baseline 上的残差。

## 3. V9 改动

### 3.1 Relative Loss

V8 主要使用 standardized MSE。该损失对低幅值状态不够敏感，导致低 Re 的 relative error 偏高。V9 增加 amplitude-aware relative loss：

```text
L_rel(y, y_hat) = mean_i ||y_hat_i - y_i||^2 / (||y_i||^2 + floor)
```

其中 floor 从训练集自动估计：

```text
floor = percentile_10(||y||^2) * relative_floor_frac
```

本轮设置：

- `lambda_alpha_rel = 0.08`
- `lambda_rhs_rel = 0.08`
- `lambda_pressure_rel = 0.35`
- `relative_floor_frac = 0.05`

### 3.2 Relative Rollout Mix

rollout loss 从纯 standardized MSE 改成混合形式：

```text
L_rollout = (1 - mix) L_std + mix L_rel
```

本轮设置：

- `rollout_relative_mix = 0.35`

### 3.3 Router Regularization

V8 的 `r16-b2` 在低/中 Re 有 dead experts。V9 增强 load balance，并用轻微负 entropy 系数鼓励更均匀路由：

- `num_experts = 8`
- `lambda_router_balance = 0.08`
- `lambda_router_entropy = -0.0015`
- `temperature = 1.15`

### 3.4 训练预算

V9 增大训练预算和模型宽度：

- `hidden_dim = 144`
- `expert_hidden = 224`
- `recon_dim = 2048`
- `train_rollout_steps = 8`
- `rollout_steps = 16`
- `rollout curriculum = 1, 2, 4, 8`
- `rollout_batch = 2`

## 4. 实验设置

数据仍为 V8：

- Re range: 50-300
- Re samples: 100
- Total snapshots: 12869
- Valid samples: 12569
- Velocity/pressure POD rank available: 80/80

测试 Re：

| Test index | Re label | Re value |
|---:|---|---:|
| 10 | `Re_56p374525` | 56.3745 |
| 59 | `Re_120p000000` | 120.0000 |
| 99 | `Re_300p000000` | 300.0000 |

V9 正式实验：

| Config | `r_u` | `r_p` | Blocks | Experts | Epochs | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `v9_r16_rk4_b2_relmod` | 16 | 16 | 2 | 8 | 70 | 589.56 s |
| `v9_r24_rk4_b3_relmod` | 24 | 24 | 3 | 8 | 60 | 603.33 s |

POD 能量：

| Config | Velocity energy | Pressure energy |
|---|---:|---:|
| `v9_r16_rk4_b2_relmod` | 0.9431 | 0.9450 |
| `v9_r24_rk4_b3_relmod` | 0.9676 | 0.9733 |

## 5. V9 指标

| Config | Test Re | Base pressure L2 | Final pressure L2 | RHS L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | Auto a rollout mean | Auto b rollout mean | Entropy | Load CV | Dead experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r16-b2 | 56.37 | 2.7265 | 0.3849 | 0.1027 | 0.0693 | 0.0693 | 0.3950 | 0.3675 | 0.5456 | 1.0609 | 1.1717 | 2 |
| r16-b2 | 120.00 | 0.4817 | 0.0325 | 0.0292 | 0.0915 | 0.0601 | 0.0733 | 0.3527 | 0.3130 | 1.1435 | 0.2890 | 0 |
| r16-b2 | 300.00 | 0.3379 | 0.0919 | 0.0216 | 0.0997 | 0.0838 | 0.1093 | 0.5415 | 0.6145 | 0.9514 | 0.4063 | 0 |
| r24-b3 | 56.37 | 5.3191 | 1.2425 | 0.1486 | 0.0840 | 0.0924 | 1.2437 | 0.4220 | 1.2493 | 1.3173 | 1.0516 | 0 |
| r24-b3 | 120.00 | 0.4755 | 0.0475 | 0.0748 | 0.1192 | 0.0796 | 0.0910 | 0.4160 | 0.4340 | 1.4238 | 0.2123 | 0 |
| r24-b3 | 300.00 | 0.2898 | 0.0988 | 0.0765 | 0.1572 | 0.1306 | 0.1527 | 0.4866 | 0.6300 | 1.3244 | 0.3431 | 0 |

## 6. 与 V8 的定量对比

### 6.1 V8 r16-b2 到 V9 r16-b2

负值表示误差下降。

| Test Re | Final pressure | Auto pressure one-step | Auto velocity rollout | Auto pressure rollout | RHS |
|---:|---:|---:|---:|---:|---:|
| 56.37 | -58.6% | -58.8% | -21.9% | -53.9% | -19.4% |
| 120.00 | -48.8% | -23.3% | +0.2% | -5.0% | -9.3% |
| 300.00 | -4.1% | -8.0% | +19.7% | +3.0% | -22.1% |

三点平均：

| Metric | V8 r16-b2 | V9 r16-b2 | Change |
|---|---:|---:|---:|
| Final pressure L2 | 0.3633 | 0.1698 | -53.3% |
| Auto pressure one-step L2 | 0.3911 | 0.1925 | -50.8% |
| Auto velocity rollout mean | 0.4250 | 0.4206 | -1.0% |
| Auto pressure rollout mean | 0.7035 | 0.4910 | -30.2% |
| RHS L2 | 0.0624 | 0.0512 | -18.1% |

### 6.2 V8 r32-b3 到 V9 r24-b3

| Test Re | Final pressure | Auto pressure one-step | Auto velocity rollout | Auto pressure rollout | RHS |
|---:|---:|---:|---:|---:|---:|
| 56.37 | -40.2% | -40.2% | -48.3% | -53.3% | -46.6% |
| 120.00 | -33.9% | -17.8% | +29.8% | +14.2% | -11.0% |
| 300.00 | -16.9% | -12.3% | +1.3% | -5.2% | -13.2% |

三点平均：

| Metric | V8 r32-b3 | V9 r24-b3 | Change |
|---|---:|---:|---:|
| Final pressure L2 | 0.7566 | 0.4629 | -38.8% |
| Auto pressure one-step L2 | 0.7887 | 0.4958 | -37.1% |
| Auto velocity rollout mean | 0.5393 | 0.4415 | -18.1% |
| Auto pressure rollout mean | 1.2404 | 0.7711 | -37.8% |
| RHS L2 | 0.1501 | 0.1000 | -33.4% |

## 7. 结论

V9 相对 V8 有明确精度提升，尤其是压力预测：

- 最推荐的 V9 baseline 是 `v9_r16_rk4_b2_relmod`。
- `r16-b2` 的三点平均 final pressure L2 从 0.3633 降到 0.1698，下降 53.3%。
- `r16-b2` 的三点平均 autonomous pressure rollout 从 0.7035 降到 0.4910，下降 30.2%。
- 低 Re `Re_56p374525` 是提升最明显的点：final pressure 下降 58.6%，pressure rollout 下降 53.9%。
- `r24-b3` 也明显优于 V8 的 `r32-b3`，说明 V8 中直接提高到 `r32` 确实带来了欠训练/高阶模态难学问题；`r24` 是更稳的中间阶数。

代价：

- `r16-b2` 在 `Re_300` 的 velocity rollout 变差 19.7%，pressure rollout 小幅变差 3.0%。
- `r16-b2` 在低 Re 仍有 2 个 dead experts，说明 router 没有完全均衡。
- V9 的训练时间约为 V8 的 1.6 倍。

总体判断：V9 的 relative-loss 训练目标有效，能显著改善 V8 最不满意的低 Re 压力精度，并改善平均压力 rollout。下一步如果继续追求高 Re velocity rollout，需要对 `Re_300` 加入分段权重或单独的 velocity-rollout loss annealing。
