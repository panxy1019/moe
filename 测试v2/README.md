# 测试 v2：半侵入式 Galerkin + 深度 Shared-Routed MoE-ROM

日期：2026-06-13

本次 v2 在 v1 的“纯 numpy Ridge-MoE closure”基础上，完成了深度学习框架升级：使用 PyTorch/CUDA 实现 `PhysicalContextEncoder + 多层 Shared-Routed MoE + dual output heads`，并把 coefficient / reconstruction / dynamic residual / rollout / router losses 纳入端到端训练。

## 1. 实验环境

集群数据目录：

```text
/root/Cylinder_Results_Re500_1000_POD_data
```

半侵入式 Galerkin 张量：

```text
/root/Cylinder_Results_Re500_1000_POD_data/semi_intrusive_galerkin_tensors_allRe_ru80_rp80.npz
```

运行环境：

```text
GPU: NVIDIA GeForce RTX 3090, 24 GB
Python: /root/miniconda3/envs/pt_env/bin/python
PyTorch: 2.11.0+cu126
CUDA available: true
```

由于 `pt_env` 需要使用 conda 环境内的 `libstdc++`，运行命令中显式设置：

```bash
LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:$LD_LIBRARY_PATH
```

## 2. 模型结构

物理骨架仍使用 v1 中的半侵入式 Galerkin RHS：

```text
R_galerkin(a,b;Re) = c(Re) + A(Re)a + H(a,a) + P(Re)b
```

v2 神经网络学习：

```text
adot_hat = R_galerkin(a,b;Re) + C_MoE(x)
alpha_next_hat = AlphaHead(x)
```

整体结构：

```text
PhysicalContextEncoder
  -> Shared-Routed MoE Block x L
  -> alpha_next_head
  -> rhs_correction_head
```

输入特征 `x` 包含：

- `Re`、`1/Re`
- phase Fourier features：`sin(k theta), cos(k theta), k=1..4`
- 当前速度 POD 系数 `a`
- 当前压力 POD 系数 `b`
- 半侵入式 Galerkin RHS `R_galerkin`
- 低/中/高阶模态能量范数
- 压力系数范数与 RHS 范数

MoE block：

- 每层包含 1 个 always-on shared expert
- 多个 routed MLP experts
- learned router 输出 expert logits
- top-k sparse routing，本次 `top_k=2`
- residual connection + LayerNorm + nonlinear FFN

Dual heads：

- `alpha_next_head`：预测下一步速度 POD 系数
- `rhs_correction_head`：预测 `adot_true - R_galerkin`

## 3. 复合损失函数

训练目标不是单一 RHS correction，而是端到端复合损失：

```text
L = lambda_coeff * L_coeff
  + lambda_dyn * L_dyn
  + lambda_recon * L_recon
  + lambda_rollout * L_rollout
  + lambda_consistency * L_consistency
  + lambda_router_balance * L_router_balance
  + lambda_router_entropy * L_router_entropy
```

各项含义：

- `L_coeff`：`alpha_next_head` 对下一步 POD 系数的监督损失。
- `L_dyn`：`rhs_correction_head` 对有限差分 residual `adot_true - R_galerkin` 的监督损失。
- `L_recon`：通过 POD basis 的固定随机列子采样做 velocity reconstruction loss。主模型使用 `recon_dim=2048`，避免每个 batch 重构全场导致训练过重。
- `L_rollout`：训练时 4 步 Euler rollout loss。
- `L_consistency`：约束 `alpha_next_head` 与 `a + dt * (R_galerkin + C_MoE)` 一致。
- `L_router_balance`：鼓励 expert utilization 不塌缩。
- `L_router_entropy`：配合 top-k 稀疏路由，抑制过高路由熵。

评价 rollout 为 20 步 Euler rollout，使用真实压力 POD 系数和已知 phase 作为上下文。因此它仍是 teacher-forced pressure rollout，不是完全 autonomous pressure-coupled ROM。

## 4. 数据划分

使用 `r_u = 16, r_p = 16`，与 v1 最稳定配置一致。

POD 能量覆盖：

```text
velocity first 16: 0.912204
pressure first 16: 0.951839
```

测试划分：

- `Re=700`：插值测试，训练使用其他 Re。
- `Re=1000`：外推测试，训练使用 `Re=500,600,700,800,900`。

每个测试 Re 单独训练一个模型；训练 Re 内部按时间尾段切出验证集，用验证集选择 best epoch。

## 5. 主模型结果

主模型采用完整多层 MoE 配置 `h160_b3_e8`：

```text
hidden_dim = 160
num_moe_blocks = 3
num_routed_experts = 8
top_k = 2
expert_hidden = 224
dropout = 0.05
lr = 7e-4
weight_decay = 1e-4
epochs = 220
patience = 55
batch_size = 128
train_rollout_steps = 4
eval_rollout_steps = 20
recon_dim = 2048
```

| Test Re | Model | RHS relative L2 | one-step relative L2 | alpha-head relative L2 | 20-step rollout mean L2 | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|
| 700 | Galerkin only | 0.203423 | 0.0650337 | - | - | 0% |
| 700 | Deep shared-routed MoE | 0.101948 | 0.0485415 | 0.221672 | 0.457562 | 49.8839% |
| 1000 | Galerkin only | 0.179640 | 0.0575312 | - | - | 0% |
| 1000 | Deep shared-routed MoE | 0.0816702 | 0.0437758 | 0.311927 | 0.361526 | 54.5368% |

结论：完整 v2 多层深度 MoE 在 RHS relative L2 上，相比半侵入式 Galerkin-only 分别降低约 49.9% 和 54.5%。它也优于 v1 Ridge-MoE 的 42.2% / 45.1% 改善。

## 6. 超参数对照

| Config | Blocks | Experts | Hidden | Expert hidden | Runtime | Re=700 RHS improvement | Re=1000 RHS improvement | Re=700 rollout | Re=1000 rollout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `h96_b1_e4` | 1 | 4 | 96 | 128 | 75.37 s | 49.3278% | 52.7256% | 0.442404 | 0.356669 |
| `h128_b2_e6` | 2 | 6 | 128 | 192 | 95.40 s | 45.6851% | 48.7389% | 0.446292 | 0.365528 |
| `h160_b3_e8` | 3 | 8 | 160 | 224 | 131.41 s | 49.8839% | 54.5368% | 0.457562 | 0.361526 |

解释：

- 小模型 `h96_b1_e4` 的 rollout 最稳，说明当前数据量下小容量模型很有竞争力。
- 完整多层模型 `h160_b3_e8` 获得最佳 RHS residual 校正，符合 v2 对“多层非线性 MoE”的目标。
- `h128_b2_e6` 作为中间容量配置没有超过小模型或深模型，可能处在容量和正则的尴尬区间。
- 深模型虽然 RHS 最优，但 rollout 并未同步显著最优，后续需要加入更强 rollout loss 或 autonomous pressure head。

## 7. 与 v1 Ridge-MoE 对比

| Test Re | v1 Ridge-MoE RHS relative L2 | v2 Deep MoE RHS relative L2 | v1 improvement | v2 improvement |
|---:|---:|---:|---:|---:|
| 700 | 0.117520 | 0.101948 | 42.2285% | 49.8839% |
| 1000 | 0.0985543 | 0.0816702 | 45.1380% | 54.5368% |

v2 的收益主要来自：

- 非线性专家网络可表达比 ridge 更复杂的 Re/phase/state closure。
- learned router 相比物理固定 router 更灵活。
- dual heads 和 consistency loss 把“状态推进”和“RHS correction”绑在一起。
- reconstruction/rollout loss 给了模型更接近 ROM 真实使用方式的训练信号。

## 8. 复现命令

主模型：

```bash
mkdir -p /root/moe_rom_v2_h160_b3_e8
LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:$LD_LIBRARY_PATH \
/root/miniconda3/envs/pt_env/bin/python /root/deep_moe_rom_v2.py \
  --experiment-name h160_b3_e8 \
  --hidden-dim 160 \
  --expert-hidden 224 \
  --num-blocks 3 \
  --num-experts 8 \
  --top-k 2 \
  --epochs 220 \
  --patience 55 \
  --batch-size 128 \
  --lr 7e-4 \
  --dropout 0.05 \
  --train-rollout-steps 4 \
  --rollout-batch 16 \
  --recon-dim 2048 \
  --output-dir /root/moe_rom_v2_h160_b3_e8
```

小模型对照：

```bash
LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:$LD_LIBRARY_PATH \
/root/miniconda3/envs/pt_env/bin/python /root/deep_moe_rom_v2.py \
  --experiment-name h96_b1_e4 \
  --hidden-dim 96 \
  --expert-hidden 128 \
  --num-blocks 1 \
  --num-experts 4 \
  --top-k 2 \
  --epochs 180 \
  --patience 45 \
  --batch-size 128 \
  --lr 9e-4 \
  --dropout 0.04 \
  --train-rollout-steps 3 \
  --rollout-batch 16 \
  --recon-dim 1024 \
  --output-dir /root/moe_rom_v2_h96_b1_e4
```

中模型对照：

```bash
LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:$LD_LIBRARY_PATH \
/root/miniconda3/envs/pt_env/bin/python /root/deep_moe_rom_v2.py \
  --experiment-name h128_b2_e6 \
  --hidden-dim 128 \
  --expert-hidden 192 \
  --num-blocks 2 \
  --num-experts 6 \
  --top-k 2 \
  --epochs 220 \
  --patience 55 \
  --batch-size 128 \
  --lr 8e-4 \
  --dropout 0.04 \
  --train-rollout-steps 4 \
  --rollout-batch 16 \
  --recon-dim 2048 \
  --output-dir /root/moe_rom_v2_h128_b2_e6
```

## 9. 文件说明

```text
测试v2/
  README.md
  deep_moe_rom_v2.py
  results/
    h96_b1_e4_metrics.json
    h96_b1_e4_summary.md
    h128_b2_e6_metrics.json
    h128_b2_e6_summary.md
    h160_b3_e8_metrics.json
    h160_b3_e8_summary.md
    v1_ridge_moe_r16_metrics.json
```

## 10. 局限与下一步

- 当前 pressure coefficient `b(t)` 在 RHS 和 rollout 中仍使用真实数据，下一步应增加 pressure head 或 pressure closure，使模型能 autonomous rollout。
- reconstruction loss 使用 POD basis 随机列子采样，不是每个 batch 的完整全场重构；完整重构可以作为离线评价或低频训练项加入。
- router 已经是 learned top-k router，但还没有加入 phase-neighbor smoothness loss。后续可加入 `||gate_t - gate_{t+1}||` 让路由更符合涡脱落相位连续性。
- 当前只跑了 3 个容量配置，后续建议系统扫描 `lambda_rollout`、`lambda_router_balance`、`top_k` 和 `num_experts`。
- 20 步 rollout 仍明显高于单步误差，说明单步 RHS 准确不等于长期稳定。后续主攻方向应是更强 rollout training 与 energy regularization。
