# V8 技术报告：Re50-300 数据集上的 V7 架构复测

日期：2026-06-20

代码：`test_results_v8/deep_moe_rom_v8.py`

结果目录：`test_results_v8/results/`

## 1. 目标

V8 的目标是在新的 `Re=50-300`、`100` 个 Reynolds 数 weighted-L2 POD 数据集上，复测 V7 的 pressure-surrogate residual MoE-ROM 架构。V7 的核心压力推进方式保持不变：

```text
a_next = RK4(a_t, Galerkin velocity tensors + learned RHS correction)
b_base = c_tilde + A_tilde @ a_next
       + torch.einsum("pij,bi,bj->bp", H_tilde, a_next, a_next)
delta_b = pressure_next_head(x_t)
b_next = b_base + delta_b
```

其中 `pressure_next_head` 不直接预测 `b_next`，只预测压力 Poisson surrogate baseline 之上的残差修正。

## 2. 新数据集

集群路径：`/root/moe/V8/data`

数据说明文档：

- `RE50_300_WEIGHTED_POD_RUN_REPORT.md`
- `WEIGHTED_L2_POD_OPERATIONS.md`
- `V8_TENSOR_BUILD_AND_DATA_MANIFEST.md`
- `SEMI_INTRUSIVE_GALERKIN_TENSORS_allRe100_weightedL2_ru80_rp80.md`
- `PRESSURE_POISSON_SURROGATE_TENSORS_allRe100_weightedL2_ru80_rp80.md`

数据规模：

| Item | Value |
|---|---:|
| Re range | 50-300 |
| Re samples | 100 |
| Total snapshots | 12869 |
| Valid training/eval samples after derivative/history filtering | 12569 |
| Mesh points | 97368 |
| Velocity POD modes | 80 |
| Pressure POD modes | 80 |
| Retained velocity energy at 80 modes | 0.9970816818 |
| Retained pressure energy at 80 modes | 0.9987816277 |

本数据集覆盖了刚过 Hopf 分岔后的低 Re 周期尾流，也覆盖了二维模拟意义下的高 Re 区间。报告中的 `Re>~188` 结果应理解为二维 Navier-Stokes/ROM benchmark，而不是真实三维实验尾流的定量预测。

## 3. V8 相对 V7 脚本的修改

架构不变，主要修改如下：

- 数据路径改到 `/root/moe/V8/data/Global_POD_Weighted_L2`。
- Galerkin 张量默认读取 `semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_slim.npz`。
- 兼容 V8 slim Galerkin 布局：共享 `H/P/G_u`，每个 Re 使用 `c_all[q]`、`A_all[q]`。
- Pressure surrogate 读取 `pressure_poisson_surrogate_tensors_allRe100_weightedL2_ru80_rp80.npz`。
- Re 特征缩放从 V7 的 500-1000 区间改为 V8 的 50-300 区间：

```text
re_norm = (Re - 175) / 125
inv_re = 300 / Re
```

该改动只改变输入特征尺度，不改变模型结构。

## 4. 实验设置

测试 Re 选择低/中/高三个代表点：

| Test index | Re label | Re value |
|---:|---|---:|
| 10 | `Re_56p374525` | 56.3745 |
| 59 | `Re_120p000000` | 120.0000 |
| 99 | `Re_300p000000` | 300.0000 |

实验配置：

| Config | `r_u` | `r_p` | MoE blocks | Experts | top-k | hidden_dim | expert_hidden | Epoch budget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v8_r16_rk4_b2_surres` | 16 | 16 | 2 | 6 | 2 | 128 | 192 | 28 |
| `v8_r32_rk4_b3_surres` | 32 | 32 | 3 | 6 | 2 | 128 | 192 | 24 |

共同设置：

- Integrator: RK4。
- History length: 3。
- Train rollout steps: 6。
- Eval rollout steps: 12。
- Rollout curriculum: 1, 2, 4, 6。
- Reconstruction sampled columns: 1024。
- Batch size: 768。
- `lambda_pressure=0.60`。
- `lambda_pressure_rollout=0.35`。
- `lambda_router_smooth=0.05`。
- 训练设备：NVIDIA A100 80GB PCIe，PyTorch `2.11.0+cu126`。

POD 能量：

| Rank | Velocity energy | Pressure energy |
|---|---:|---:|
| 16 | 0.9431 | 0.9450 |
| 32 | 0.9810 | 0.9853 |

## 5. 主要结果

| Config | Test Re | Base pressure L2 | Final pressure L2 | Pressure reduction vs base | RHS L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | Auto a rollout mean | Auto b rollout mean | Entropy | Load CV | Dead experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r16-b2 | 56.37 | 2.7265 | 0.9304 | 65.9% | 0.1275 | 0.0786 | 0.0918 | 0.9590 | 0.4706 | 1.1845 | 0.9231 | 1.2481 | 3 |
| r16-b2 | 120.00 | 0.4817 | 0.0636 | 86.8% | 0.0322 | 0.0821 | 0.0736 | 0.0955 | 0.3521 | 0.3294 | 0.7821 | 0.9820 | 3 |
| r16-b2 | 300.00 | 0.3379 | 0.0958 | 71.6% | 0.0277 | 0.1018 | 0.0881 | 0.1188 | 0.4522 | 0.5967 | 1.0977 | 0.4289 | 0 |
| r32-b3 | 56.37 | 5.3064 | 2.0790 | 60.8% | 0.2781 | 0.1241 | 0.1447 | 2.0812 | 0.8169 | 2.6764 | 1.0951 | 1.0261 | 1 |
| r32-b3 | 120.00 | 0.4274 | 0.0719 | 83.2% | 0.0841 | 0.1041 | 0.0927 | 0.1107 | 0.3205 | 0.3801 | 1.1222 | 0.5609 | 0 |
| r32-b3 | 300.00 | 0.2512 | 0.1189 | 52.7% | 0.0881 | 0.1492 | 0.1353 | 0.1742 | 0.4804 | 0.6647 | 1.2143 | 0.4557 | 0 |

运行时间：

| Config | Runtime seconds | Best epochs |
|---|---:|---|
| `v8_r16_rk4_b2_surres` | 371.65 | 20, 20, 20 |
| `v8_r32_rk4_b3_surres` | 380.86 | 20, 20, 20 |

## 6. 结果分析

### 6.1 Pressure surrogate residual 仍然有效

两组配置下，残差头都能显著降低 pressure surrogate baseline 的误差：

- `r16-b2`: pressure one-step 相对 baseline 降低 65.9%-86.8%。
- `r32-b3`: pressure one-step 相对 baseline 降低 52.7%-83.2%。

这说明 V7 的“刚性物理基线 + residual correction”机制迁移到 `Re=50-300` 数据集后仍然成立。

### 6.2 V8 上 `r16-b2` 比 `r32-b3` 更稳

虽然 `r32` 的 POD 能量更高，但本轮预算下 `r16-b2` 的 one-step 和 rollout 更稳：

- `Re_56p374525`: `r32-b3` final pressure L2 比 `r16-b2` 高 123.5%，pressure rollout 高 125.9%。
- `Re_120p000000`: `r32-b3` velocity rollout 略好 9.0%，但 pressure rollout 高 15.4%。
- `Re_300p000000`: `r32-b3` final pressure 高 24.1%，pressure rollout 高 11.4%。

可能原因：

- 低 Re 接近涡脱落起始区，压力/速度主振荡幅值较小，relative L2 对高阶低能模态误差更敏感。
- `r32` 增加了更多高阶压力/速度自由度，但本轮只给 24 epoch，残差头和 RHS correction 对高阶模态尚未充分收敛。
- V8 覆盖 50-300 的物理区间更宽，单一 MoE 容量和 router 正则需要更细调；直接沿用 V7 超参并不一定最优。

### 6.3 Router 分工

Router 没有完全塌缩，但低阶配置存在专家闲置：

- `r16-b2` 在低/中 Re 上 dead experts = 3，load CV 偏高。
- `r16-b2` 在 `Re_300` 上 dead experts = 0，load CV = 0.4289，路由更健康。
- `r32-b3` 在中/高 Re 上 dead experts = 0，entropy 约 1.12-1.21，专家使用更均匀。

这说明更深 `r32-b3` 的 router 分工更充分，但预测精度未同步改善。V8 下一步应把 router regularization 和 loss 权重与 rank/depth 联合调参，而不是只增加截断阶数。

## 7. 与 V7 数据集现象的区别

V7 的 `Re=500-1000` 数据集处于强周期涡街区，`r32-b3` 更容易通过高阶模态提升 pressure rollout。V8 的 `Re=50-300` 覆盖从刚过 Hopf 分岔到二维高 Re benchmark 的宽区间，低 Re 处的相对误差非常敏感。

因此，V8 目前的最佳轻量配置是 `r16-b2`，而不是 V7 中表现更强的 `r32-b3`。

## 8. 建议

后续 V8 可以继续做三类增强：

1. 对 `r32-b3` 增加 epoch 和 early stopping patience，观察高阶模型是否只是欠训练。
2. 增强低 Re 分段建模，例如按 Re 区间增加 router context 或引入 Re-band auxiliary loss。
3. 调高 router load-balance loss，减少 `r16-b2` 在低/中 Re 上的 dead experts。

当前结论：V7 pressure-surrogate residual 架构可以迁移到 V8 新数据集；在当前轻量训练预算下，`r16-b2` 是更稳定的 V8 baseline。
