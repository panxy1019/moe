# 测试 v3：History Context + Router Temporal Smoothness

日期：2026-06-13

本次 v3 在 `测试v2` 的 PyTorch 深度 Shared-Routed MoE-ROM 上继续增强两点：

1. **Router 物理连续性**：增加相邻时间步的 gate smoothness loss。
2. **时序历史输入**：把过去 2 步的 POD/pressure/RHS 信息拼接进 PhysicalContextEncoder 输入。

目标是提高局部 RHS/单步预测精度，并测试这种物理连续性约束是否能改善长期 rollout 稳定性。

## 1. 实验环境

集群：

```text
ssh root@10.210.22.206 -p 32062
```

数据目录：

```text
/root/Cylinder_Results_Re500_1000_POD_data
```

运行环境：

```text
GPU: NVIDIA GeForce RTX 3090, 24 GB
Python: /root/miniconda3/envs/pt_env/bin/python
PyTorch: 2.11.0+cu126
CUDA: available
```

运行时需要：

```bash
LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:$LD_LIBRARY_PATH
```

## 2. V3 实现改动

### 2.1 History Context

v2 的输入是当前状态：

```text
x_t = [Re, phase features, a_t, b_t, R_galerkin(t), energy features]
```

v3 使用 `history_len = 3`，即：

```text
[t, t-1, t-2]
```

新增历史特征包括：

```text
a_{t-1}, b_{t-1}, R_{t-1}
a_{t-2}, b_{t-2}, R_{t-2}
a_t - a_{t-1}, b_t - b_{t-1}, R_t - R_{t-1}
a_t - a_{t-2}, b_t - b_{t-2}, R_t - R_{t-2}
```

这让 `PhysicalContextEncoder` 同时看到当前相位、当前 reduced state、历史惯性和局部变化趋势。

### 2.2 Router Temporal Smoothness

新增 router 平滑损失：

```text
L_smooth = mean_l || gate_l(t) - gate_l(t-1) ||_2^2
```

其中 `l` 是 MoE block 层号。训练时对同一 Re 的相邻时间步样本计算该项。

总损失变为：

```text
L = L_coeff
  + L_dyn
  + lambda_recon L_recon
  + lambda_rollout L_rollout
  + lambda_consistency L_consistency
  + lambda_balance L_router_balance
  + lambda_entropy L_router_entropy
  + lambda_smooth L_smooth
```

主实验中：

```text
lambda_smooth = 0.05
history_len = 3
epochs = 300-360
patience = 75-85
```

## 3. 数据划分

保持 v2 设置：

```text
r_u = 16
r_p = 16
```

测试划分：

- `Re=700`：插值测试，训练用其他 Re。
- `Re=1000`：外推测试，训练用 `Re=500,600,700,800,900`。

注意：由于需要 `t-1,t-2`，前两个可用样本会被排除，因此 Galerkin baseline 的数值与 v2 有极小差异。

## 4. 主结果：多层 MoE + History + Smooth

主配置：

```text
config = h160_b3_e8_hist3_smooth
hidden_dim = 160
num_moe_blocks = 3
num_experts = 8
top_k = 2
expert_hidden = 224
history_len = 3
lambda_router_smooth = 0.05
epochs = 360
patience = 85
train_rollout_steps = 5
lambda_rollout = 0.14
runtime = 308.78 s
```

| Test Re | Galerkin RHS L2 | V3 RHS L2 | One-step L2 | Alpha-head L2 | Gate smooth MSE | 20-step rollout L2 | RHS improvement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 700 | 0.203349 | 0.0839706 | 0.0446767 | 0.195786 | 0.00802101 | 0.487919 | 58.7061% |
| 1000 | 0.179599 | 0.0798829 | 0.0433860 | 0.341423 | 0.00939704 | 0.408768 | 55.5217% |

结论：主 V3 配置显著提升 RHS/单步预测。`Re=700` 的 RHS 改善从 v2 主模型的 49.88% 提升到 58.71%；`Re=1000` 从 54.54% 提升到 55.52%。

## 5. Rollout 强化配置

为改善长期积分，额外测试了小容量但更强 rollout loss 的配置：

```text
config = h96_b1_e4_hist3_smooth_rollheavy
hidden_dim = 96
num_moe_blocks = 1
num_experts = 4
history_len = 3
lambda_router_smooth = 0.05
train_rollout_steps = 8
lambda_rollout = 0.35
epochs = 320
patience = 85
runtime = 243.20 s
```

| Test Re | Galerkin RHS L2 | V3 RHS L2 | One-step L2 | Alpha-head L2 | Gate smooth MSE | 20-step rollout L2 | RHS improvement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 700 | 0.203349 | 0.109822 | 0.0471754 | 0.227408 | 0.00943511 | 0.480399 | 45.9934% |
| 1000 | 0.179599 | 0.0766095 | 0.0430989 | 0.329851 | 0.00662815 | 0.360704 | 57.3443% |

这个配置在 `Re=1000` 上给出本次最佳 RHS L2 和最佳 rollout L2；但在 `Re=700` 上不如多层主配置。

## 6. Smoothness Ablation

为了验证 `L_smooth` 的作用，测试了同一小模型但 `lambda_router_smooth = 0` 的对照：

| Config | Test Re | RHS L2 | Gate smooth MSE | 20-step rollout L2 | RHS improvement |
|---|---:|---:|---:|---:|---:|
| `h96_b1_e4_hist3_smooth` | 700 | 0.100156 | 0.0117745 | 0.536718 | 50.7466% |
| `h96_b1_e4_hist3_nosmooth` | 700 | 0.0993524 | 0.0116240 | 0.533656 | 51.1419% |
| `h96_b1_e4_hist3_smooth` | 1000 | 0.0793811 | 0.00746105 | 0.409190 | 55.8011% |
| `h96_b1_e4_hist3_nosmooth` | 1000 | 0.0798238 | 0.00857703 | 0.410794 | 55.5545% |

观察：

- `Re=1000` 外推上，smooth loss 降低了 gate smooth MSE，并略微改善 RHS/rollout。
- `Re=700` 插值上，smooth loss 没有带来收益，甚至略差。
- 说明 `L_smooth` 对外推 regime 更有价值，但权重需要按目标调参；它不是越大越好。

## 7. V2/V3 对比

| Config | Test Re | RHS L2 | One-step L2 | 20-step rollout L2 | RHS improvement |
|---|---:|---:|---:|---:|---:|
| V2 `h160_b3_e8` | 700 | 0.101948 | 0.0485415 | 0.457562 | 49.8839% |
| V3 `h160_b3_e8_hist3_smooth` | 700 | 0.0839706 | 0.0446767 | 0.487919 | 58.7061% |
| V2 `h160_b3_e8` | 1000 | 0.0816702 | 0.0437758 | 0.361526 | 54.5368% |
| V3 `h160_b3_e8_hist3_smooth` | 1000 | 0.0798829 | 0.0433860 | 0.408768 | 55.5217% |
| V2 `h96_b1_e4` | 1000 | 0.0849238 | 0.0445818 | 0.356669 | 52.7256% |
| V3 `h96_b1_e4_hist3_smooth_rollheavy` | 1000 | 0.0766095 | 0.0430989 | 0.360704 | 57.3443% |

结论：

- 从 **RHS/单步预测精度** 看，v3 成功提升，尤其 `Re=700`。
- 从 **20 步 rollout** 看，v3 没有全面超过 v2；只有 `Re=1000` 的 rollout-heavy 配置接近 v2 最优。
- 当前长期 rollout 的瓶颈仍然是 teacher-forced pressure/history 设定和显式 Euler 积分误差，而不只是 router 不连续。

## 8. 推荐使用结论

如果目标是最强 RHS residual correction：

```text
推荐 h160_b3_e8_hist3_smooth
```

如果目标是 `Re=1000` 外推和 rollout 稳定性：

```text
推荐 h96_b1_e4_hist3_smooth_rollheavy
```

如果目标是论文叙事：

```text
V3 证明 history context 和 router smoothness 能提升局部物理闭合精度；
但长期 autonomous rollout 还需要 pressure head、自回归 history buffer 和更高阶积分器。
```

## 9. 复现命令

主配置：

```bash
mkdir -p /root/moe_rom_v3_h160_b3_e8_hist3_smooth
LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:$LD_LIBRARY_PATH \
/root/miniconda3/envs/pt_env/bin/python /root/deep_moe_rom_v3.py \
  --experiment-name h160_b3_e8_hist3_smooth \
  --hidden-dim 160 \
  --expert-hidden 224 \
  --num-blocks 3 \
  --num-experts 8 \
  --top-k 2 \
  --epochs 360 \
  --patience 85 \
  --batch-size 128 \
  --lr 6e-4 \
  --dropout 0.05 \
  --train-rollout-steps 5 \
  --rollout-batch 20 \
  --recon-dim 2048 \
  --history-len 3 \
  --lambda-router-smooth 0.05 \
  --lambda-rollout 0.14 \
  --output-dir /root/moe_rom_v3_h160_b3_e8_hist3_smooth
```

Rollout-heavy 配置：

```bash
mkdir -p /root/moe_rom_v3_h96_b1_e4_hist3_smooth_rollheavy
LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:$LD_LIBRARY_PATH \
/root/miniconda3/envs/pt_env/bin/python /root/deep_moe_rom_v3.py \
  --experiment-name h96_b1_e4_hist3_smooth_rollheavy \
  --hidden-dim 96 \
  --expert-hidden 128 \
  --num-blocks 1 \
  --num-experts 4 \
  --top-k 2 \
  --epochs 320 \
  --patience 85 \
  --batch-size 128 \
  --lr 6e-4 \
  --dropout 0.04 \
  --train-rollout-steps 8 \
  --rollout-batch 24 \
  --recon-dim 1024 \
  --history-len 3 \
  --lambda-router-smooth 0.05 \
  --lambda-rollout 0.35 \
  --output-dir /root/moe_rom_v3_h96_b1_e4_hist3_smooth_rollheavy
```

## 10. 文件说明

```text
测试v3/
  README.md
  deep_moe_rom_v3.py
  results/
    h160_b3_e8_hist3_smooth_metrics.json
    h160_b3_e8_hist3_smooth_summary.md
    h96_b1_e4_hist3_smooth_metrics.json
    h96_b1_e4_hist3_smooth_summary.md
    h96_b1_e4_hist3_smooth_rollheavy_metrics.json
    h96_b1_e4_hist3_smooth_rollheavy_summary.md
    h96_b1_e4_hist3_nosmooth_metrics.json
    h96_b1_e4_hist3_nosmooth_summary.md
    h160_b3_e8_metrics.json
    h96_b1_e4_metrics.json
```

## 11. 下一步

要继续提高长期预测精度，建议优先做：

- **Autonomous pressure head**：不要再 rollout 时使用真实 `b(t)`。
- **自回归 history buffer**：rollout 中历史 `a_{t-1},a_{t-2}` 应来自模型预测，而不是 teacher-forced 历史。
- **更稳定积分器**：把 Euler 换成 RK4 / learned integrator。
- **phase smoothness + energy regularization**：不仅平滑 gate，也约束 modal energy drift。
- **rollout curriculum**：训练步长从 2、4、8、16 逐渐增加。
