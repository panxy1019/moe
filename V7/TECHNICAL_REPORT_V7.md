# V7 技术报告：Pressure-Surrogate Residual MoE-ROM

日期：2026-06-16

代码：`test_results_v7/deep_moe_rom_v7.py`

结果目录：`test_results_v7/results/`

## 1. 目标

V7 以 V6 的 RK4 autonomous-pressure MoE-ROM 为基础，引入刚性的 Pressure Poisson surrogate 物理基线，将 `pressure_next_head` 降级为残差修正网络。核心目标是减少压力自主推进时对纯神经网络压力头的依赖，提升 one-step pressure 和 multi-step pressure rollout 的稳定性。

V7 的压力更新不再直接学习：

```text
b_next = pressure_next_head(x_t)
```

而是学习：

```text
b_next = b_base(a_next) + delta_b_theta(x_t)
```

其中 `b_base` 由半侵入式压力代理张量给出，`delta_b_theta` 只负责补偿截断误差、数据误差和模型不可表达部分。

## 2. 数据与张量

集群数据路径：

- `/root/moe/V7/data/Global_POD_Weighted_L2/global_velocity_pod_weighted_l2.npz`
- `/root/moe/V7/data/Global_POD_Weighted_L2/global_pressure_pod_weighted_l2.npz`
- `/root/moe/V7/data/Global_POD_Weighted_L2/pod_snapshot_index.csv`
- `/root/moe/V7/data/semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz`
- `/root/moe/V7/data/pressure_poisson_surrogate_tensors_allRe30_weightedL2_ru80_rp80.npz`
- `/root/moe/V7/data/PRESSURE_POISSON_SURROGATE_TENSORS_allRe30_weightedL2_ru80_rp80.md`

Pressure surrogate 文档说明了压力 Poisson 弱式离线张量：

```text
L b = c^p + A^p a + H^p(a,a)
```

NPZ 中已经保存了伪逆映射后的张量：

```text
c_tilde = L_pinv c^p
A_tilde = L_pinv A^p
H_tilde[:,j,k] = L_pinv H^p[:,j,k]
```

V7 对每个 Re 使用对应的 `c_tilde` 与 `A_tilde`，共享 `H_tilde`，并按截断阶数裁剪到 `r_u, r_p`。

## 3. V7 推进流程

`rollout_step` 和训练期短 rollout 的逻辑按以下四步重构。

1. RK4 速度推进

```text
a_next = RK4(a_t, b_t, Re, phase, velocity Galerkin tensors, rhs_correction_head)
```

速度 RHS 仍由半侵入式 Galerkin velocity operator 加神经网络 RHS correction 组成。

2. Pressure surrogate 物理基线

```text
b_base = c_tilde + A_tilde @ a_next
       + torch.einsum("pij,bi,bj->bp", H_tilde, a_next, a_next)
```

3. 残差学习

```text
delta_b = pressure_next_head(x_t)
```

训练标签改为：

```text
pressure_residual = b_true_next - b_base_true_next
```

其中 `b_base_true_next` 使用数据中的真实 `a_next` 离线计算，用于稳定监督。

4. 状态组合

```text
b_next = b_base + delta_b
```

自主 rollout 时 `b_base` 使用模型预测得到的 `a_next` 计算，因此评估不再依赖 teacher-forced pressure。

## 4. 模型结构

V7 延续 V6 深度 MoE 框架：

- `PhysicalContextEncoder` 输入历史窗口、当前速度系数、当前压力系数、Galerkin RHS、Re 和相位谐波特征。
- Shared-routed MoE block：共享 MLP 分支 + top-k routed expert 分支。
- 三个输出头：
  - `alpha_next_head`: 辅助预测下一步速度系数。
  - `rhs_correction_head`: 修正 Galerkin velocity RHS。
  - `pressure_next_head`: V7 中改为压力残差头，只预测 `delta_b`。
- Router 正则：
  - load-balance loss。
  - entropy regularization。
  - temporal smoothness loss。
- Dynamic loss：
  - coefficient loss。
  - sampled reconstruction loss。
  - dynamic residual loss。
  - short rollout loss。
  - pressure surrogate residual loss。
  - alpha/RHS consistency loss。

## 5. 实验设置

测试 Re：

- `Re_706p896552`
- `Re_1000p000000`

四组配置：

| Config | `r_u` | `r_p` | MoE blocks | Experts | top-k | hidden_dim | expert_hidden | rollout curriculum |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| r16-b2 | 16 | 16 | 2 | 6 | 2 | 128 | 192 | 1,2,4,8 |
| r16-b3 | 16 | 16 | 3 | 6 | 2 | 128 | 192 | 1,2,4,8 |
| r32-b2 | 32 | 32 | 2 | 6 | 2 | 128 | 192 | 1,2,4,8 |
| r32-b3 | 32 | 32 | 3 | 6 | 2 | 128 | 192 | 1,2,4,8 |

共同设置：

- Integrator: RK4。
- History length: 3。
- Train rollout steps: 8。
- Eval rollout steps: 16。
- Reconstruction sampled columns: 1024。
- `lambda_pressure=0.60`。
- `lambda_pressure_rollout=0.35`。
- 训练设备：PyTorch GPU 环境，RTX 3090。

POD 能量：

| Rank | Velocity energy | Pressure energy |
|---|---:|---:|
| 16 | 0.8459 | 0.9079 |
| 32 | 0.9449 | 0.9731 |

## 6. V7 结果

| Config | Test Re | Base pressure L2 | Final pressure L2 | Residual target L2 | RHS L2 | TF one-step L2 | Auto a rollout mean | Auto b rollout mean | Entropy | Load CV | Dead experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r16-b2 | Re706.9 | 0.4492 | 0.1791 | 0.3986 | 0.0495 | 0.0152 | 0.0863 | 0.1537 | 0.8008 | 0.7752 | 0 |
| r16-b2 | Re1000 | 0.2578 | 0.1230 | 0.4771 | 0.0699 | 0.0179 | 0.0650 | 0.0921 | 0.9618 | 0.4708 | 0 |
| r16-b3 | Re706.9 | 0.4492 | 0.1655 | 0.3683 | 0.0553 | 0.0161 | 0.0741 | 0.1313 | 1.3261 | 0.4422 | 0 |
| r16-b3 | Re1000 | 0.2578 | 0.1125 | 0.4366 | 0.0745 | 0.0175 | 0.0599 | 0.0831 | 1.1858 | 0.5719 | 0 |
| r32-b2 | Re706.9 | 0.2083 | 0.0949 | 0.4556 | 0.0740 | 0.0210 | 0.0756 | 0.0998 | 1.0476 | 0.3783 | 0 |
| r32-b2 | Re1000 | 0.1581 | 0.0878 | 0.5554 | 0.0611 | 0.0170 | 0.0639 | 0.0779 | 0.8541 | 0.6249 | 0 |
| r32-b3 | Re706.9 | 0.2083 | 0.1026 | 0.4923 | 0.0734 | 0.0215 | 0.0655 | 0.0861 | 1.2295 | 0.5953 | 0 |
| r32-b3 | Re1000 | 0.1581 | 0.0805 | 0.5091 | 0.0568 | 0.0169 | 0.0633 | 0.0761 | 1.3759 | 0.2553 | 0 |

结论：

- Pressure surrogate baseline 本身已经给出可用的压力近似，`r32` 下 base pressure L2 降到 0.1581-0.2083。
- 残差头进一步显著降低压力误差：`r32-b3` 在 `Re1000` 上 final pressure L2 为 0.0805，是本轮最好压力 one-step 结果。
- 所有配置均无 dead expert，说明 MoE 没有发生完全专家塌缩。
- `r32-b3` 的 router entropy 达到 1.3759，load CV 降到 0.2553，说明更高阶截断和更深 MoE 下专家使用更均衡。

## 7. 与 V6 的直接对比

V6 是直接压力头 autonomous-pressure RK4 MoE-ROM；V7 是 pressure surrogate residual 版本。下表为相对 V6 的变化，负值表示误差降低。

| Config | Test Re | Pressure one-step delta | Auto pressure rollout delta | Auto velocity rollout delta |
|---|---:|---:|---:|---:|
| r16-b2 | Re706.9 | -19.6% | -34.4% | +5.0% |
| r16-b2 | Re1000 | -67.5% | -73.2% | -23.1% |
| r16-b3 | Re706.9 | -31.4% | -35.9% | -5.5% |
| r16-b3 | Re1000 | -71.0% | -64.3% | -11.3% |
| r32-b2 | Re706.9 | -71.4% | -58.7% | +2.6% |
| r32-b2 | Re1000 | -81.0% | -78.9% | -13.0% |
| r32-b3 | Re706.9 | -63.6% | -63.6% | +1.1% |
| r32-b3 | Re1000 | -81.1% | -66.5% | -5.8% |

主要判断：

- V7 对压力预测和压力 rollout 的收益非常明确，尤其在 `Re1000` 上 pressure one-step 降低 67.5%-81.1%，autonomous pressure rollout 降低 64.3%-78.9%。
- 速度 rollout 的收益较温和，`Re1000` 全部改善；`Re706.9` 的部分配置有 1.1%-5.0% 的轻微退化。这说明压力基线显著稳定了压力子系统，但速度方程的长期误差仍主要受 RHS correction、RK4 rollout loss 和截断阶数影响。
- V7 的最佳综合配置是 `r32-b3`：压力 one-step 和 pressure rollout 最低，同时 router load balance 最好。

## 8. 深层 MoE block 的影响

| Rank | Test Re | Metric | b2 | b3 deep | Delta |
|---|---:|---|---:|---:|---:|
| r16 | Re706.9 | final pressure L2 | 0.1791 | 0.1655 | -7.6% |
| r16 | Re706.9 | auto a rollout | 0.0863 | 0.0741 | -14.2% |
| r16 | Re706.9 | auto b rollout | 0.1537 | 0.1313 | -14.6% |
| r16 | Re706.9 | RHS L2 | 0.0495 | 0.0553 | +11.8% |
| r16 | Re1000 | final pressure L2 | 0.1230 | 0.1125 | -8.5% |
| r16 | Re1000 | auto a rollout | 0.0650 | 0.0599 | -7.8% |
| r16 | Re1000 | auto b rollout | 0.0921 | 0.0831 | -9.8% |
| r16 | Re1000 | RHS L2 | 0.0699 | 0.0745 | +6.7% |
| r32 | Re706.9 | final pressure L2 | 0.0949 | 0.1026 | +8.0% |
| r32 | Re706.9 | auto a rollout | 0.0756 | 0.0655 | -13.3% |
| r32 | Re706.9 | auto b rollout | 0.0998 | 0.0861 | -13.7% |
| r32 | Re1000 | final pressure L2 | 0.0878 | 0.0805 | -8.3% |
| r32 | Re1000 | auto a rollout | 0.0639 | 0.0633 | -1.0% |
| r32 | Re1000 | auto b rollout | 0.0779 | 0.0761 | -2.3% |

判断：

- 增加到 3 个 MoE block 对 autonomous rollout 基本有益，尤其 `r16` 和 `r32/Re706.9`。
- 深层模型并不总是降低 one-step RHS 或 one-step pressure；例如 `r16` RHS L2 变差，`r32/Re706.9` final pressure one-step 变差。
- 这符合 V5/V6 中观察到的现象：更深 MoE 更擅长长期展开稳定性，但需要更强正则或更长训练来同时优化单步误差。

## 9. 专家分工与路由

V7 四组实验均无 dead expert。更深模型通常带来更高 entropy：

- `r16-b2`: entropy 0.8008/0.9618，load CV 0.7752/0.4708。
- `r16-b3`: entropy 1.3261/1.1858，load CV 0.4422/0.5719。
- `r32-b2`: entropy 1.0476/0.8541，load CV 0.3783/0.6249。
- `r32-b3`: entropy 1.2295/1.3759，load CV 0.5953/0.2553。

结论：

- V7 没有出现 top-k 路由完全塌缩。
- `r32-b3/Re1000` 的 load CV 最低，为 0.2553，说明专家负载最均衡。
- `r16-b2/Re706.9` load CV 仍偏高，为 0.7752，说明该配置中有一个专家承担了较多样本，后续可提高 load-balance loss 或调高 temperature。

## 10. 运行时间

| Config | Runtime seconds | Best epochs |
|---|---:|---|
| r16-b2 | 239.76 | 40, 40 |
| r16-b3 | 298.63 | 40, 40 |
| r32-b2 | 252.53 | 30, 40 |
| r32-b3 | 302.04 | 30, 40 |

V7 的额外计算主要来自 pressure surrogate einsum 和残差标签构造。相对深层 MoE 本身，这部分开销可接受。

## 11. 总结

V7 的核心改动有效：把压力预测从纯神经网络直接预测改为 Pressure Poisson surrogate baseline + residual correction 后，压力 one-step 和 autonomous pressure rollout 都显著改善。最佳配置 `r32-b3` 在两个测试 Re 上均取得最低或接近最低的压力 rollout，并且路由没有塌缩。

代价是速度 rollout 并未在所有 Re 上同步改善，说明压力子系统被稳定后，速度长期误差瓶颈转移到 RHS correction 和训练期 rollout objective。下一步建议保留 V7 pressure surrogate residual 结构，并继续调大 rollout curriculum、增强 velocity RHS consistency loss，或采用两阶段训练：先锁定 surrogate residual pressure，再联合优化 velocity rollout。
