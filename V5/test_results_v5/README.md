# V5: RK4 / Deep MoE Block / Expert Routing Diagnostics

本目录记录基于 V4 质量加权 POD 数据与半侵入式 Galerkin 张量库的 V5 增强测试。目标是验证三项改动是否真正提升模型预测精度与长期稳定性：

1. 将时间推进从单步 Euler 形式扩展为 RK4。
2. 增加 shared-routed MoE block 深度与专家容量。
3. 分析专家分工是否真实有效，而不只看最终 loss。

数据路径保持为集群 `/root/moe/V4/data`，半侵入式张量为 `semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz`。本轮为控制计算成本，固定 `r_u=8, r_p=8`，测试留出 Re index 为 `[12, 29]`，即 `Re_706p896552` 与 `Re_1000p000000`。

## 文件

- `deep_moe_rom_v5.py`: V5 完整 PyTorch 脚本，包含 RK4 推进、可调 MoE block 深度、专家路由诊断。
- `results/v5_r8_euler_b2_metrics.json`: 2-block Euler 基线增强模型。
- `results/v5_r8_rk4_b2_metrics.json`: 2-block RK4 对比模型。
- `results/v5_r8_euler_b4_deep_metrics.json`: 4-block/8-expert 深层容量模型。
- `results/*_summary.md`: 每组实验自动生成摘要。

## 共同网络与训练设置

所有实验均使用：

- PhysicalContextEncoder + shared-routed MoE blocks + dual heads。
- 双输出头：`alpha_next_head` 预测下一步 POD 系数，`rhs_correction_head` 预测半侵入式 Galerkin RHS correction。
- 复合损失：coefficient、sampled reconstruction、dynamic residual、short rollout、alpha/RHS consistency、router load-balance、entropy、router temporal smoothness。
- 历史上下文：`history_len=3`。
- rollout curriculum：`[2, 4, 8, 16]`，训练最大 rollout step 为 16，评估 rollout step 为 20。
- GPU 环境：`cuda`，PyTorch `2.11.0+cu126`。

## 实验配置

| 实验 | Integrator | Blocks | Experts | Hidden | Expert hidden | Epochs | Loss rollout | Router smooth | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_r8_euler_b2` | Euler | 2 | 6 | 96 | 128 | 180 | 0.25 | 0.05 | 346.91 s |
| `v5_r8_rk4_b2` | RK4 | 2 | 6 | 96 | 128 | 150 | 0.18 | 0.05 | 838.12 s |
| `v5_r8_euler_b4_deep` | Euler | 4 | 8 | 128 | 192 | 150 | 0.22 | 0.06 | 360.26 s |

说明：尝试过 `4-block + 8-expert + RK4`，但单次训练成本过高，已中止，未作为有效对比结果纳入本报告。

## 1. RK4 与 Euler 对比

这里保持模型容量一致，只改变积分器。`Euler one-step L2` 是用 Euler 形式评估的单步误差，`Integrator one-step L2` 是当前实验积分器自己的单步误差；RK4 行中该列即 RK4 one-step。

| Re | Integrator | RHS L2 | Euler one-step L2 | Integrator one-step L2 | Rollout mean L2 | Rollout p90 | Rollout max |
|---:|---|---:|---:|---:|---:|---:|---:|
| 706.8966 | Euler | 0.042659 | 0.041716 | 0.043711 | 0.561969 | 1.125558 | 1.452137 |
| 706.8966 | RK4 | 0.045510 | 0.042023 | 0.017475 | 0.384591 | 0.648420 | 0.679377 |
| 1000.0000 | Euler | 0.092335 | 0.040037 | 0.041950 | 0.582850 | 0.908235 | 1.031214 |
| 1000.0000 | RK4 | 0.090753 | 0.039524 | 0.021912 | 0.311117 | 0.669114 | 1.018012 |

相对 Euler 的变化：

| Re | RK4 one-step 降幅 | Rollout mean 降幅 | Rollout p90 降幅 | Rollout max 降幅 |
|---:|---:|---:|---:|---:|
| 706.8966 | 60.02% | 31.56% | 42.39% | 53.21% |
| 1000.0000 | 47.77% | 46.62% | 26.33% | 1.28% |

结论：RK4 对 RHS correction 本身没有明显增益，甚至在 `Re_706p896552` 上 RHS L2 略高；但对真正时间推进误差和长期 rollout 稳定性收益非常明确。代价是运行时间从 346.91 s 增至 838.12 s，约为 2.42 倍。

当前 RK4 实现中，中间 stage 使用同一时刻的 pressure/phase context，这是本轮为了和 V4 数据流对齐的折中；后续若要完全自主长期预测，需要同时推进或预测 pressure/context。

## 2. 增加 MoE Block 深度

这里比较 2-block/6-expert Euler 与 4-block/8-expert Euler。加深模型带来更高容量，但也更容易在小截断阶数下过拟合或学到更软的路由。

| Re | Model | RHS L2 | Rollout mean L2 | Rollout p90 | Rollout max | Load CV | Entropy | Dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 706.8966 | 2-block / 6 expert | 0.042659 | 0.561969 | 1.125558 | 1.452137 | 0.428881 | 0.982304 | 0 |
| 706.8966 | 4-block / 8 expert | 0.044790 | 0.584439 | 1.061592 | 1.212869 | 0.357596 | 1.544306 | 0 |
| 1000.0000 | 2-block / 6 expert | 0.092335 | 0.582850 | 0.908235 | 1.031214 | 0.264922 | 0.949287 | 0 |
| 1000.0000 | 4-block / 8 expert | 0.088531 | 0.414290 | 0.750376 | 0.767568 | 0.315309 | 1.491132 | 0 |

结论：

- `Re_1000p000000` 上深层模型有效，RHS L2 从 0.092335 降至 0.088531，rollout mean 从 0.582850 降至 0.414290。
- `Re_706p896552` 上深层模型没有提升 RHS 与 rollout mean，但降低了 p90 和 max，说明极端发散有所缓解。
- 深层模型 entropy 明显增大，路由更分散；所有专家 mean load 都大于 1%，没有 dead expert。
- 深层模型两个测试 Re 的 best validation epoch 都在 20 左右，提示容量增加后存在验证集过拟合或早停敏感性，后续应配合更强正则、更多 Re 留出或更长但更平滑的学习率调度。

## 3. 专家分工诊断

测试集路由统计如下。`Load CV` 越小表示专家平均负载越均匀；`Entropy` 越高表示 gate 更软、更分散；`Top-1` 表示每个专家作为最大权重专家的比例。

| 实验 | Re | Mean load | Top-1 fraction | Load CV | Entropy | Dead experts |
|---|---:|---|---|---:|---:|---:|
| Euler b2 | 706.8966 | `[0.301, 0.191, 0.085, 0.095, 0.173, 0.154]` | `[0.379, 0.192, 0.061, 0.126, 0.076, 0.167]` | 0.428881 | 0.982304 | 0 |
| Euler b2 | 1000.0000 | `[0.161, 0.220, 0.121, 0.112, 0.158, 0.227]` | `[0.217, 0.273, 0.091, 0.101, 0.136, 0.182]` | 0.264922 | 0.949287 | 0 |
| RK4 b2 | 706.8966 | `[0.284, 0.196, 0.080, 0.103, 0.185, 0.152]` | `[0.318, 0.212, 0.081, 0.136, 0.106, 0.146]` | 0.400917 | 1.005819 | 0 |
| RK4 b2 | 1000.0000 | `[0.157, 0.218, 0.129, 0.130, 0.139, 0.228]` | `[0.217, 0.273, 0.126, 0.091, 0.136, 0.157]` | 0.245405 | 0.927541 | 0 |
| Euler b4 | 706.8966 | `[0.203, 0.133, 0.065, 0.081, 0.170, 0.127, 0.080, 0.140]` | `[0.293, 0.212, 0.071, 0.101, 0.253, 0.015, 0.005, 0.051]` | 0.357596 | 1.544306 | 0 |
| Euler b4 | 1000.0000 | `[0.066, 0.153, 0.068, 0.152, 0.124, 0.102, 0.176, 0.158]` | `[0.061, 0.146, 0.081, 0.086, 0.121, 0.086, 0.237, 0.182]` | 0.315309 | 1.491132 | 0 |

相位分箱上的主导专家也不同：

| 实验 | Re | Phase [0,0.25) | Phase [0.25,0.5) | Phase [0.5,0.75) | Phase [0.75,1.0) |
|---|---:|---|---|---|---|
| Euler b2 | 706.8966 | E0 / 0.422 | E0 / 0.320 | E0 / 0.260 | E0 / 0.509 |
| Euler b2 | 1000.0000 | E1 / 0.271 | E1 / 0.360 | E0 / 0.280 | E0 / 0.320 |
| RK4 b2 | 706.8966 | E0 / 0.356 | E0 / 0.300 | E1 / 0.260 | E0 / 0.396 |
| RK4 b2 | 1000.0000 | E1 / 0.271 | E1 / 0.300 | E1 / 0.320 | E0 / 0.340 |
| Euler b4 | 706.8966 | E0 / 0.333 | E0 / 0.440 | E4 / 0.300 | E4 / 0.245 |
| Euler b4 | 1000.0000 | E7 / 0.188 | E7 / 0.240 | E6 / 0.240 | E6 / 0.400 |

结论：

- 没有出现 mean-load 级别的专家塌缩，三组实验 dead experts 都为 0。
- 2-block 模型在 `Re_706p896552` 上有明显 E0 主导倾向；在 `Re_1000p000000` 上 E1/E0 随相位变化切换，说明路由已捕捉到部分 Re 与 phase 差异。
- 4-block 模型的相位分工更清晰，例如 `Re_706p896552` 从 E0 转到 E4，`Re_1000p000000` 从 E7 转到 E6；但 `Re_706p896552` 上 E5/E6 的 Top-1 比例仅 0.015/0.005，说明虽然 mean load 没死，硬路由意义上的专家使用仍不均匀。
- Top-k 组合计数与完整 phase-bin 统计已保存在 metrics JSON 中。

## 总体判断

1. RK4 是本轮最明确有效的改动：对 one-step integrator error 与 multi-step rollout error 的改善显著，适合作为后续长期预测默认推进器；代价是训练/评估成本显著上升。
2. 增加 block 深度的收益是局部的：高 Re 上改善更明显，低/中 Re 上未必降低平均 rollout error；它更像提高容量与稳定性上界，而不是稳定的免费增益。
3. 专家分工是真实存在的，但仍不够理想：没有全局塌缩，且不同 Re/phase 会激活不同专家；深层模型中仍有 Top-1 低使用专家，后续应继续调 load-balance、top-k 稀疏、temperature 或增加专家多样性约束。

推荐下一步以 `v5_r8_rk4_b2` 作为稳定基线，再做两条优化：一是降低 RK4 成本，例如缓存 Galerkin stage 或混合精度；二是在 RK4 下小步增加容量，而不是直接上 4-block/8-expert。
