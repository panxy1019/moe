# 测试 V4：Weighted-L2 30Re 数据集 + Curriculum MoE-ROM

日期：2026-06-14

本次 V4 基于仓库已有的新数据流：

```text
V4/data/
  global_velocity_pod_weighted_l2.npz
  global_pressure_pod_weighted_l2.npz
  pod_snapshot_index.csv
  semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz
```

核心升级：

- 使用 30 个 Reynolds number，范围 `Re=500-1000`。
- 使用 mass-weighted / weighted-L2 POD 系数。
- 使用 compact 半侵入式 Galerkin 张量库：per-Re `c/A` + shared `H/P`。
- 测试截断阶数 `r_u=r_p=8` 和 `r_u=r_p=16`。
- 训练 rollout curriculum：`2 -> 4 -> 8 -> 16`。

## 1. 数据与方程

V4 的 POD 模态是 raw physical space 中的 weighted-L2 POD 模态，满足质量矩阵正交：

```text
Phi^T M Phi ~= I
```

半侵入式 Galerkin ROM 使用 compact 张量：

```text
da/dt = c(Re) + A(Re) a + H(a,a) + P b
```

其中：

- `c/A`：每个 `Re_label` 单独存储。
- `H/P`：对 30 个 Re 共享。
- `a`：速度 POD 系数。
- `b`：压力 POD 系数。

本次脚本通过 `pod_snapshot_index.csv` 中的 `Re_label` 精确匹配 tensor key，例如：

```text
Re_706p896552_c
Re_706p896552_A
```

## 2. 模型与训练

模型沿用 V3 的深度物理 MoE-ROM：

```text
PhysicalContextEncoder
  -> Shared-Routed MoE blocks
  -> alpha_next_head
  -> rhs_correction_head
```

输入特征包括：

- `Re`、`1/Re`
- phase Fourier features，`k=1..4`
- 当前 `a_t, b_t, R_galerkin(t)`
- history context：`[t, t-1, t-2]`
- 历史差分：`a_t-a_{t-1}`、`b_t-b_{t-1}`、`R_t-R_{t-1}` 等
- reduced energy / norm features

复合损失：

```text
L = L_coeff
  + L_dyn
  + lambda_recon L_recon
  + lambda_rollout L_rollout
  + lambda_consistency L_consistency
  + lambda_balance L_router_balance
  + lambda_entropy L_router_entropy
  + lambda_smooth L_router_smooth
```

新增 rollout curriculum：

```text
curriculum_steps = [2, 4, 8, 16]
```

训练 epoch 被分成四段，逐步增加 rollout loss 的展开步长。

## 3. 测试划分

默认测试两个 Re：

| Re index | Re label | Re value | 用途 |
|---:|---|---:|---|
| 12 | `Re_706p896552` | 706.896552 | 中间 Re 泛化 |
| 29 | `Re_1000p000000` | 1000.0 | 高 Re 端点泛化 |

每个测试 Re 都采用 leave-one-Re-out：

```text
train = other 29 Re
test  = held-out Re
```

每个训练 Re 内部按时间尾段切出 validation set。

## 4. r=8 结果

配置：

```text
experiment = v4_r8_curriculum
r_u = 8
r_p = 8
hidden_dim = 96
num_blocks = 2
num_experts = 6
top_k = 2
expert_hidden = 128
epochs = 260
min_epochs = 200
batch_size = 512
lr = 7e-4
history_len = 3
curriculum_steps = [2, 4, 8, 16]
lambda_rollout = 0.25
lambda_router_smooth = 0.05
runtime = 547.31 s
```

POD energy:

```text
velocity_first_8 = 0.684963
pressure_first_8 = 0.768992
```

| Test Re | Model | RHS relative L2 | One-step relative L2 | Alpha-head relative L2 | Gate smooth MSE | 20-step rollout L2 | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|---:|
| 706.896552 | Galerkin only | 0.153112 | 0.0587276 | - | - | - | 0% |
| 706.896552 | Deep MoE + curriculum | 0.0460573 | 0.0420849 | 0.0896233 | 0.00573116 | 0.647908 | 69.9193% |
| 1000.0 | Galerkin only | 0.178619 | 0.0521943 | - | - | - | 0% |
| 1000.0 | Deep MoE + curriculum | 0.0907608 | 0.0397504 | 0.265987 | 0.00398087 | 0.396159 | 49.1874% |

## 5. r=16 结果

配置：

```text
experiment = v4_r16_curriculum
r_u = 16
r_p = 16
hidden_dim = 128
num_blocks = 2
num_experts = 8
top_k = 2
expert_hidden = 192
epochs = 230
min_epochs = 180
batch_size = 512
lr = 6e-4
history_len = 3
curriculum_steps = [2, 4, 8, 16]
lambda_rollout = 0.22
lambda_router_smooth = 0.05
runtime = 558.27 s
```

POD energy:

```text
velocity_first_16 = 0.845891
pressure_first_16 = 0.907877
```

| Test Re | Model | RHS relative L2 | One-step relative L2 | Alpha-head relative L2 | Gate smooth MSE | 20-step rollout L2 | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|---:|
| 706.896552 | Galerkin only | 0.211125 | 0.0630250 | - | - | - | 0% |
| 706.896552 | Deep MoE + curriculum | 0.0487769 | 0.0372165 | 0.204281 | 0.00396189 | 0.419239 | 76.8966% |
| 1000.0 | Galerkin only | 0.131844 | 0.0439991 | - | - | - | 0% |
| 1000.0 | Deep MoE + curriculum | 0.0711912 | 0.0361546 | 0.309629 | 0.00501848 | 0.380274 | 46.0034% |

## 6. 结论

V4 的 weighted-L2 数据和 compact Galerkin 张量使物理骨架更干净，神经 closure 的收益也更清晰。

主要观察：

- `r=8` 在中间 Re 上把 RHS relative L2 从 `0.153112` 降到 `0.0460573`，改善 `69.92%`。
- `r=16` 在中间 Re 上把 RHS relative L2 从 `0.211125` 降到 `0.0487769`，改善 `76.90%`。
- `r=16` 的 one-step relative L2 最优：`0.0372165` at `Re_706p896552`，`0.0361546` at `Re_1000p000000`。
- `Re=1000` 上，`r=8` 和 `r=16` 都有稳定改善，但 RHS 降幅低于中间 Re，说明高 Re 端点仍然更难。
- 20-step rollout 仍然是主要短板，尤其 `r=8` 中间 Re rollout L2 达到 `0.647908`；这说明单步 RHS 准确性和长期积分稳定性仍不是同一个问题。

推荐：

```text
如果关注单步/RHS 精度：优先使用 r=16。
如果关注低阶成本和 Re=1000 rollout：r=8 与 r=16 差距不大。
下一步应加入 autonomous pressure head、RK4 integrator、energy regularization。
```

## 7. 复现命令

r=8:

```bash
LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:$LD_LIBRARY_PATH \
/root/miniconda3/envs/pt_env/bin/python /root/deep_moe_rom_v4.py \
  --experiment-name v4_r8_curriculum \
  --r-u 8 \
  --r-p 8 \
  --hidden-dim 96 \
  --expert-hidden 128 \
  --num-blocks 2 \
  --num-experts 6 \
  --top-k 2 \
  --epochs 260 \
  --min-epochs 200 \
  --patience 70 \
  --batch-size 512 \
  --lr 7e-4 \
  --dropout 0.04 \
  --history-len 3 \
  --curriculum-steps 2,4,8,16 \
  --train-rollout-steps 16 \
  --rollout-batch 24 \
  --lambda-rollout 0.25 \
  --lambda-router-smooth 0.05 \
  --recon-dim 1024 \
  --output-dir /root/moe/V4/results_v4_r8_curriculum
```

r=16:

```bash
LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:$LD_LIBRARY_PATH \
/root/miniconda3/envs/pt_env/bin/python /root/deep_moe_rom_v4.py \
  --experiment-name v4_r16_curriculum \
  --r-u 16 \
  --r-p 16 \
  --hidden-dim 128 \
  --expert-hidden 192 \
  --num-blocks 2 \
  --num-experts 8 \
  --top-k 2 \
  --epochs 230 \
  --min-epochs 180 \
  --patience 60 \
  --batch-size 512 \
  --lr 6e-4 \
  --dropout 0.05 \
  --history-len 3 \
  --curriculum-steps 2,4,8,16 \
  --train-rollout-steps 16 \
  --rollout-batch 20 \
  --lambda-rollout 0.22 \
  --lambda-router-smooth 0.05 \
  --recon-dim 2048 \
  --output-dir /root/moe/V4/results_v4_r16_curriculum
```

## 8. 文件说明

```text
V4/test_results_v4/
  README.md
  deep_moe_rom_v4.py
  results/
    v4_r8_curriculum_metrics.json
    v4_r8_curriculum_summary.md
    v4_r16_curriculum_metrics.json
    v4_r16_curriculum_summary.md
```

## 9. 下一步

V4 证明 weighted-L2 + 30Re 数据显著改善了训练条件。下一步为了继续提升长期预测，应优先做：

- pressure head：rollout 中不再使用真实压力系数 `b(t)`。
- RK4 / implicit-stabilized integrator：减少 Euler 长期积分误差。
- energy regularization：约束 modal energy drift。
- longer rollout curriculum：从 `2,4,8,16` 扩展到 `2,4,8,16,32`。
- Re interpolation grid analysis：测试更多 held-out Re，而不只 `Re_706p896552` 和 `Re_1000p000000`。
