# V5 完整技术报告：物理一致半侵入式 MoE-ROM 圆柱绕流测试

## 1. 目标与结论概览

V5 在 V4 的质量加权 POD 数据和半侵入式 Galerkin 张量库基础上，继续验证三个增强方向：

1. 将时间推进器从 Euler 单步推进扩展为 RK4。
2. 增加 shared-routed MoE block 深度和专家容量。
3. 系统分析专家分工，包括 load balance、gate entropy、top-k 路由分布、Re/phase 激活模式。

本轮测试固定截断阶数为 `r_u=8, r_p=8`，采用 30 个雷诺数数据中的 leave-one-Re-out 方式，在 `Re_706p896552` 与 `Re_1000p000000` 上做外推/泛化测试。三组有效实验如下：

| 实验 | 时间推进 | MoE blocks | Experts | 参数量 | 主要目的 |
|---|---|---:|---:|---:|---|
| `v5_r8_euler_b2` | Euler | 2 | 6 | 429,692 | V5 基线增强模型 |
| `v5_r8_rk4_b2` | RK4 | 2 | 6 | 429,692 | 单独验证高阶积分收益 |
| `v5_r8_euler_b4_deep` | Euler | 4 | 8 | 1,989,296 | 单独验证更深 MoE 容量 |

核心结果：

- RK4 是本轮最明确有效的改动。相同网络容量下，RK4 将 integrator one-step error 降低 `47.77% - 60.02%`，将 20-step rollout mean error 降低 `31.56% - 46.62%`。
- 加深 MoE block 的收益具有 Re 依赖性。4-block/8-expert 在 `Re_1000p000000` 明显降低 rollout mean，但在 `Re_706p896552` 没有降低平均 rollout error，只改善了 p90/max 极端误差。
- 专家没有出现 mean-load 级别塌缩，三组实验 dead experts 均为 0；不同 Re 和不同相位段的主导专家不同，说明路由分工真实存在，但深层模型中仍有少数专家 Top-1 使用率偏低。

代码与结果文件位于：

```text
V5/test_results_v5/deep_moe_rom_v5.py
V5/test_results_v5/results/v5_r8_euler_b2_metrics.json
V5/test_results_v5/results/v5_r8_rk4_b2_metrics.json
V5/test_results_v5/results/v5_r8_euler_b4_deep_metrics.json
```

## 2. 数据集与文件说明

### 2.1 运行时数据路径

集群测试使用的数据目录为：

```text
/root/moe/V4/data
```

V5 脚本默认读取：

```text
/root/moe/V4/data/global_velocity_pod_weighted_l2.npz
/root/moe/V4/data/global_pressure_pod_weighted_l2.npz
/root/moe/V4/data/pod_snapshot_index.csv
/root/moe/V4/data/semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz
```

仓库中与数据相关的说明和已上传文件位于：

```text
V4/data/README_weighted_l2.txt
V4/data/WEIGHTED_L2_POD_REPORT.md
V4/data/SEMI_INTRUSIVE_GALERKIN_TENSORS_COMPACT_README.md
V4/data/SEMI_INTRUSIVE_GALERKIN_TENSORS_allRe30_weightedL2_ru80_rp80.md
V4/data/mesh_l2_point_weights.npz
V4/data/pod_snapshot_index.csv
V4/data/pod_weighted_l2_metadata.json
V4/data/semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz
```

### 2.2 快照数据

数据集包含：

| 项目 | 数值 |
|---|---:|
| Re 范围 | 500 到 1000 |
| Re 个数 | 30 |
| 每个 Re 原始帧数 | 241 |
| 每个 Re 保留帧数 | 201 |
| 总快照数 | 6030 |
| V5 有效训练样本数 | 5940 |
| 网格点数 | 97368 |
| 速度维度 | 194736，即 `[u, v]` 拼接 |
| 压力维度 | 97368 |

V4 重新生成了质量加权 L2 POD。速度和压力分别用点控制体积构造离散 L2 内积：

```text
<u, v>_M = sum_i V_i * (u_i v_i)
```

速度场对两个分量 `[u, v]` 使用相同点体积权重。加权 POD 输出最大阶数为 80：

| POD | 文件 | 主要数组 | V5 使用 |
|---|---|---|---|
| Velocity | `global_velocity_pod_weighted_l2.npz` | `coeff_uv`, `phi_uv`, `mean_uv_by_Re`, `cumulative_energy_uv` | `coeff_uv[:, :r_u]`, `phi_uv[:r_u]` |
| Pressure | `global_pressure_pod_weighted_l2.npz` | `coeff_p`, `phi_p`, `mean_p_by_Re`, `cumulative_energy_p` | `coeff_p[:, :r_p]` |
| Snapshot index | `pod_snapshot_index.csv` | `Re`, `Re_label`, `time`, `phase` | 序列切分、相位特征、历史窗口 |

本轮固定 `r_u=8, r_p=8`，对应能量：

| 截断 | 累计能量 |
|---|---:|
| Velocity first 8 | 0.6849627599953964 |
| Pressure first 8 | 0.7689923606465794 |

说明：`r=8` 是轻量测试配置，速度 POD 能量只有约 68.5%，因此绝对 reconstruction/alpha 指标不能代表高阶 ROM 上限；本轮重点比较不同 V5 改动的相对收益。

### 2.3 半侵入式 Galerkin 张量

V5 使用 V4 离线生成的 compact 张量：

```text
semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz
```

共享数组：

| 数组 | Shape | 说明 |
|---|---:|---|
| `G_u` | `(80, 80)` | 速度质量/Gram 相关矩阵 |
| `H` | `(80, 80, 80)` | 二次对流张量，Re-independent |
| `P` | `(80, 80)` | 压力耦合张量，Re-independent |
| `H_raw` | `(80, 80, 80)` | raw 版本 |
| `P_raw` | `(80, 80)` | raw 版本 |

每个 Re 独立数组：

| 数组 | Shape | 说明 |
|---|---:|---|
| `{Re_label}_c` | `(80,)` | 常数项 |
| `{Re_label}_A` | `(80, 80)` | 线性项 |
| `{Re_label}_c_raw` | `(80,)` | raw 版本 |
| `{Re_label}_A_raw` | `(80, 80)` | raw 版本 |

半侵入式 ROM 基础方程为：

```text
da_i/dt = c_i(Re)
        + sum_j A_ij(Re) a_j
        + sum_j sum_k H_ijk a_j a_k
        + sum_m P_im b_m
```

V5 模型学习的是 Galerkin RHS 的 closure/correction：

```text
R_galerkin(a_t, b_t, Re) = c_Re + A_Re a_t + H(a_t, a_t) + P b_t
R_true(t)                = centered_derivative(a_t)
Delta_R(t)               = R_true(t) - R_galerkin(t)
R_model(t)               = R_galerkin(t) + f_theta(x_t)
```

## 3. V5 模型结构

### 3.1 输入特征

基础输入 `base_x` 包含：

```text
[ Re_norm,
  1000/Re,
  sin(k*phase), cos(k*phase), k=1..phase_harmonics,
  a_t,
  b_t,
  R_galerkin(t),
  ||a_low||, ||a_mid||, ||a_high||,
  ||b_t||,
  ||R_galerkin(t)|| ]
```

其中：

```text
Re_norm = (Re - 750) / 250
phase_harmonics = 4
```

V5 继续使用历史窗口 `history_len=3`。除当前 `base_x` 外，对过去两步加入：

```text
[a_{t-h}, b_{t-h}, R_galerkin(t-h),
 a_t - a_{t-h}, b_t - b_{t-h}, R_galerkin(t) - R_galerkin(t-h)]
```

在本轮 `r_u=8, r_p=8, phase_harmonics=4, history_len=3` 下：

| 部分 | 维度 |
|---|---:|
| 当前基础特征 | 39 |
| 每个历史步增量 | 48 |
| 历史步数 | 2 |
| 总输入维度 | 135 |

所有输入、RHS residual target、alpha-next target 均在训练集上做 standardization。

### 3.2 网络模块

模型为：

```text
PhysicalContextEncoder
  -> SharedRoutedMoEBlock x N
  -> alpha_next_head
  -> rhs_correction_head
```

`PhysicalContextEncoder`：

```text
Linear(in_dim, hidden_dim)
LayerNorm
SiLU
Dropout
Linear(hidden_dim, hidden_dim)
LayerNorm
SiLU
```

`SharedRoutedMoEBlock` 包含：

```text
LayerNorm
shared MLP
router Linear(hidden_dim, num_experts)
top-k gate with softmax temperature
expert MLP x num_experts
residual update: h = h + shared + routed
post MLP residual block
```

每个专家和 shared branch 都是两层 MLP：

```text
Linear(hidden_dim, expert_hidden)
SiLU
Dropout
Linear(expert_hidden, hidden_dim)
```

Router 使用：

```text
gate = softmax(router(z) / temperature)
top_k = 2
```

如果 `top_k < num_experts`，仅保留 top-k expert 概率并重新归一化。

### 3.3 双输出头

V5 实装了双输出头：

```text
alpha_next_head:       z -> standardized alpha_{t+1}
rhs_correction_head:   z -> standardized Delta_R(t)
```

其中 RHS correction 经反标准化后加入 Galerkin RHS：

```text
R_model(t) = R_galerkin(t) + unstandardize(rhs_correction_head(z))
```

### 3.4 时间推进器

Euler：

```text
a_{t+1} = a_t + dt * R_model(a_t, b_t, Re)
```

RK4：

```text
k1 = R_model(a_t)
k2 = R_model(a_t + 0.5 * dt * k1)
k3 = R_model(a_t + 0.5 * dt * k2)
k4 = R_model(a_t + dt * k3)
a_{t+1} = a_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

当前 RK4 stage 中使用同一时刻的 pressure/phase/context，这是为了与现有 V4 数据流保持一致。该实现适合评估速度系数推进误差，但还不是完全自主的 pressure-context 联合预测。

## 4. 损失函数

训练总损失：

```text
L = lambda_coeff       * L_coeff
  + lambda_dyn         * L_dyn
  + lambda_recon       * L_recon
  + lambda_consistency * L_consistency
  + lambda_router_bal  * L_router_balance
  + lambda_entropy     * L_router_entropy
  + lambda_smooth      * L_router_smooth
  + lambda_rollout     * L_rollout
```

各项含义：

| Loss | 目标 | 实现含义 |
|---|---|---|
| `L_coeff` | 系数预测 | `alpha_next_head` 与标准化 `a_{t+1}` 的 MSE |
| `L_dyn` | RHS correction | `rhs_correction_head` 与标准化 `Delta_R` 的 MSE |
| `L_recon` | 流场重构 | sampled POD reconstruction delta，相对真实重构能量归一化 |
| `L_consistency` | 双头一致性 | alpha head 与 Euler RHS 更新结果保持一致 |
| `L_rollout` | 短期展开 | curriculum rollout 系数误差 |
| `L_router_balance` | 负载均衡 | mean gate 接近均匀分布 |
| `L_router_entropy` | 路由熵 | gate entropy 正则 |
| `L_router_smooth` | 时间平滑 | 相邻时间步 gate 的 MSE |

Router temporal smoothness：

```text
L_smooth = mean_blocks || gate_t - gate_{t-1} ||^2
```

rollout curriculum：

```text
2 -> 4 -> 8 -> 16 steps
```

训练中使用梯度裁剪：

```text
clip_grad_norm = 1.0
```

优化器与调度器：

```text
AdamW + CosineAnnealingLR
```

## 5. 数据切分与评估方法

### 5.1 切分方式

每个测试 Re 单独训练一个 split：

1. 当前测试 Re 的全部有效样本作为 test。
2. 其他 29 个 Re 作为 train/validation 候选。
3. 对每个训练 Re，按时间排序取最后约 12% 作为 validation，至少 10 个样本。
4. 其余样本作为 training。

本轮每个 split 的样本数：

| Split | 样本数 |
|---|---:|
| Train | 5046 |
| Validation | 696 |
| Test | 198 |

### 5.2 评估指标

基础 Galerkin baseline：

```text
RHS relative L2:
||R_true - R_galerkin|| / ||R_true||

Euler one-step relative L2:
||a_{t+1,true} - (a_t + dt*R_galerkin)|| / ||a_{t+1,true}||
```

Deep MoE 指标：

| 指标 | 含义 |
|---|---|
| `rhs_relative_l2` | `R_model` 对 centered derivative 的相对 L2 |
| `alpha_head_relative_l2` | 双头 alpha-next 直接预测误差 |
| `one_step_euler_relative_l2` | 使用 `R_model` 做 Euler 单步误差 |
| `one_step_integrator.relative_l2` | 当前 integrator 的单步误差，Euler 实验为 Euler，RK4 实验为 RK4 |
| `rollout_teacher_forced_pressure` | 20-step rollout，pressure/phase 使用真实序列上下文 |
| `router_temporal_smooth_mse` | 相邻时间步 gate MSE |
| `routing_analysis_test.load_cv` | 专家平均负载变异系数 |
| `routing_analysis_test.entropy_mean` | gate entropy |
| `routing_analysis_test.dead_experts_threshold_1pct` | mean load < 1% 的专家数 |

20-step rollout 统计包含：

```text
relative_l2_mean
relative_l2_median
relative_l2_p90
relative_l2_max
```

## 6. 实验超参数

| 实验 | Integrator | Blocks | Experts | Hidden | Expert hidden | Dropout | Epochs | LR | Patience | Min epochs | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_r8_euler_b2` | Euler | 2 | 6 | 96 | 128 | 0.04 | 180 | 0.0007 | 50 | 130 | 346.91 s |
| `v5_r8_rk4_b2` | RK4 | 2 | 6 | 96 | 128 | 0.04 | 150 | 0.0006 | 45 | 110 | 838.12 s |
| `v5_r8_euler_b4_deep` | Euler | 4 | 8 | 128 | 192 | 0.06 | 150 | 0.0005 | 45 | 100 | 360.26 s |

共同设置：

| 参数 | 值 |
|---|---:|
| `r_u` | 8 |
| `r_p` | 8 |
| `phase_harmonics` | 4 |
| `top_k` | 2 |
| `temperature` | 0.8 |
| `batch_size` | 512 |
| `weight_decay` | 0.0001 |
| `history_len` | 3 |
| `curriculum_steps` | `[2, 4, 8, 16]` |
| `train_rollout_steps` | 16 |
| `eval rollout_steps` | 20 |
| `recon_dim` | 1024 |
| `analysis_bins` | 4 |
| `device` | CUDA |
| `torch_version` | 2.11.0+cu126 |

Loss weights：

| 实验 | coeff | dyn | recon | rollout | consistency | router balance | entropy | router smooth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_r8_euler_b2` | 1.0 | 1.0 | 0.08 | 0.25 | 0.15 | 0.02 | 0.002 | 0.05 |
| `v5_r8_rk4_b2` | 1.0 | 1.0 | 0.08 | 0.18 | 0.15 | 0.02 | 0.002 | 0.05 |
| `v5_r8_euler_b4_deep` | 1.0 | 1.0 | 0.08 | 0.22 | 0.15 | 0.02 | 0.002 | 0.06 |

## 7. 复现实验命令

脚本：

```bash
python3 /root/moe/V5/test_results_v5/deep_moe_rom_v5.py --help
```

2-block Euler：

```bash
python3 /root/moe/V5/test_results_v5/deep_moe_rom_v5.py \
  --data-root /root/moe/V4/data \
  --tensor-path /root/moe/V4/data/semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz \
  --output-dir /root/moe/V5/results_v5_r8_euler_b2 \
  --experiment-name v5_r8_euler_b2 \
  --r-u 8 --r-p 8 \
  --num-blocks 2 --num-experts 6 --hidden-dim 96 --expert-hidden 128 \
  --dropout 0.04 --temperature 0.8 \
  --epochs 180 --min-epochs 130 --patience 50 \
  --batch-size 512 --lr 0.0007 --weight-decay 0.0001 \
  --lambda-rollout 0.25 --lambda-router-smooth 0.05 \
  --recon-dim 1024 --history-len 3 --curriculum-steps 2,4,8,16 \
  --integrator euler --test-re-indices 12 29 --device cuda
```

2-block RK4：

```bash
python3 /root/moe/V5/test_results_v5/deep_moe_rom_v5.py \
  --data-root /root/moe/V4/data \
  --tensor-path /root/moe/V4/data/semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz \
  --output-dir /root/moe/V5/results_v5_r8_rk4_b2 \
  --experiment-name v5_r8_rk4_b2 \
  --r-u 8 --r-p 8 \
  --num-blocks 2 --num-experts 6 --hidden-dim 96 --expert-hidden 128 \
  --dropout 0.04 --temperature 0.8 \
  --epochs 150 --min-epochs 110 --patience 45 \
  --batch-size 512 --lr 0.0006 --weight-decay 0.0001 \
  --lambda-rollout 0.18 --lambda-router-smooth 0.05 \
  --recon-dim 1024 --history-len 3 --curriculum-steps 2,4,8,16 \
  --integrator rk4 --test-re-indices 12 29 --device cuda
```

4-block Euler：

```bash
python3 /root/moe/V5/test_results_v5/deep_moe_rom_v5.py \
  --data-root /root/moe/V4/data \
  --tensor-path /root/moe/V4/data/semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz \
  --output-dir /root/moe/V5/results_v5_r8_euler_b4_deep \
  --experiment-name v5_r8_euler_b4_deep \
  --r-u 8 --r-p 8 \
  --num-blocks 4 --num-experts 8 --hidden-dim 128 --expert-hidden 192 \
  --dropout 0.06 --temperature 0.8 \
  --epochs 150 --min-epochs 100 --patience 45 \
  --batch-size 512 --lr 0.0005 --weight-decay 0.0001 \
  --lambda-rollout 0.22 --lambda-router-smooth 0.06 \
  --recon-dim 1024 --history-len 3 --curriculum-steps 2,4,8,16 \
  --integrator euler --test-re-indices 12 29 --device cuda
```

## 8. 结果一：半侵入式 Galerkin baseline 与 MoE RHS correction

| Re | 模型 | RHS relative L2 | RHS improvement vs Galerkin | Euler one-step L2 | Euler one-step improvement |
|---:|---|---:|---:|---:|---:|
| 706.8966 | Galerkin only | 0.153112 | 0.00% | 0.058728 | 0.00% |
| 706.8966 | Euler b2 | 0.042659 | 72.14% | 0.041716 | 28.97% |
| 706.8966 | RK4 b2 | 0.045510 | 70.28% | 0.042023 | 28.44% |
| 706.8966 | Euler b4 | 0.044790 | 70.75% | 0.041889 | 28.67% |
| 1000.0000 | Galerkin only | 0.178619 | 0.00% | 0.052194 | 0.00% |
| 1000.0000 | Euler b2 | 0.092335 | 48.31% | 0.040037 | 23.29% |
| 1000.0000 | RK4 b2 | 0.090753 | 49.19% | 0.039524 | 24.27% |
| 1000.0000 | Euler b4 | 0.088531 | 50.44% | 0.039508 | 24.31% |

观察：

- 三个 V5 模型都显著改善 Galerkin RHS，说明神经 correction 对半侵入式 ROM 的残差有稳定补偿作用。
- RK4 主要改善时间积分，不直接优化 RHS target，因此 RHS L2 与 Euler b2 接近。
- 深层 Euler b4 在 `Re_1000p000000` 的 RHS 最优，但在 `Re_706p896552` 不如 Euler b2。

## 9. 结果二：RK4 与 Euler 时间推进对比

| Re | Integrator | RHS L2 | Euler one-step L2 | Integrator one-step L2 | Rollout mean L2 | Rollout p90 | Rollout max |
|---:|---|---:|---:|---:|---:|---:|---:|
| 706.8966 | Euler b2 | 0.042659 | 0.041716 | 0.043711 | 0.561969 | 1.125558 | 1.452137 |
| 706.8966 | RK4 b2 | 0.045510 | 0.042023 | 0.017475 | 0.384591 | 0.648420 | 0.679377 |
| 1000.0000 | Euler b2 | 0.092335 | 0.040037 | 0.041950 | 0.582850 | 0.908235 | 1.031214 |
| 1000.0000 | RK4 b2 | 0.090753 | 0.039524 | 0.021912 | 0.311117 | 0.669114 | 1.018012 |

相对 Euler b2 的 RK4 降幅：

| Re | Integrator one-step 降幅 | Rollout mean 降幅 | Rollout p90 降幅 | Rollout max 降幅 |
|---:|---:|---:|---:|---:|
| 706.8966 | 60.02% | 31.56% | 42.39% | 53.21% |
| 1000.0000 | 47.77% | 46.62% | 26.33% | 1.28% |

结论：

- RK4 显著缓解误差累积，尤其在 rollout mean 和 p90 上表现稳定。
- `Re_706p896552` 的 rollout max 从 1.452137 降到 0.679377，长期稳定性提升非常明显。
- 代价是 runtime 从 346.91 s 增至 838.12 s，约 2.42 倍。

## 10. 结果三：增加 MoE block 深度

| Re | 模型 | RHS L2 | Integrator one-step L2 | Rollout mean | Rollout p90 | Rollout max | Best epoch | Runtime |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 706.8966 | Euler b2 | 0.042659 | 0.043711 | 0.561969 | 1.125558 | 1.452137 | 60 | 346.91 s |
| 706.8966 | Euler b4 | 0.044790 | 0.043088 | 0.584439 | 1.061592 | 1.212869 | 20 | 360.26 s |
| 1000.0000 | Euler b2 | 0.092335 | 0.041950 | 0.582850 | 0.908235 | 1.031214 | 20 | 346.91 s |
| 1000.0000 | Euler b4 | 0.088531 | 0.040693 | 0.414290 | 0.750376 | 0.767568 | 20 | 360.26 s |

观察：

- 深层模型在高 Re 上收益更明显，`Re_1000p000000` 的 rollout mean 从 0.582850 降到 0.414290。
- `Re_706p896552` 的平均 rollout 没有改善，但 p90 和 max 降低，说明深层容量可能改善极端窗口稳定性。
- 深层模型 best epoch 较早，提示容量增加后过拟合风险上升；更深网络需要更细的正则和学习率策略。
- 尝试过 `4-block + 8-expert + RK4`，但计算成本过高，中止后未纳入有效结果。

## 11. 结果四：训练、验证与泛化情况

| 实验 | Re | Train RHS L2 | Val RHS L2 | Test RHS L2 | Train alpha L2 | Val alpha L2 | Best val score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Euler b2 | 706.8966 | 0.009489 | 0.148632 | 0.042659 | 0.030564 | 0.456598 | 0.605230 |
| Euler b2 | 1000.0000 | 0.018158 | 0.146738 | 0.092335 | 0.056461 | 0.459634 | 0.606372 |
| RK4 b2 | 706.8966 | 0.010684 | 0.149146 | 0.045510 | 0.034348 | 0.456233 | 0.605380 |
| RK4 b2 | 1000.0000 | 0.014548 | 0.144846 | 0.090753 | 0.043529 | 0.464352 | 0.609198 |
| Euler b4 | 706.8966 | 0.017073 | 0.144583 | 0.044790 | 0.051368 | 0.483104 | 0.627687 |
| Euler b4 | 1000.0000 | 0.017295 | 0.140676 | 0.088531 | 0.050713 | 0.483072 | 0.623748 |

解释：

- Train RHS L2 明显低于 validation/test，存在跨 Re 泛化难度。
- Validation alpha L2 较高，与 `r_u=8` 截断能量较低有关；V5 更适合用 RHS、one-step integrator、rollout 稳定性判断相对改动有效性。
- RK4 并不显著降低 validation RHS，但能显著降低 integrator/rollout 误差，说明主要收益来自数值积分误差控制。

## 12. 结果五：专家分工与路由分析

### 12.1 Load balance 与 entropy

| 实验 | Re | Mean load | Top-1 fraction | Load CV | Entropy | Dead experts |
|---|---:|---|---|---:|---:|---:|
| Euler b2 | 706.8966 | `[0.301, 0.191, 0.085, 0.095, 0.173, 0.154]` | `[0.379, 0.192, 0.061, 0.126, 0.076, 0.167]` | 0.428881 | 0.982304 | 0 |
| Euler b2 | 1000.0000 | `[0.161, 0.220, 0.121, 0.112, 0.158, 0.227]` | `[0.217, 0.273, 0.091, 0.101, 0.136, 0.182]` | 0.264922 | 0.949287 | 0 |
| RK4 b2 | 706.8966 | `[0.284, 0.196, 0.080, 0.103, 0.185, 0.152]` | `[0.318, 0.212, 0.081, 0.136, 0.106, 0.146]` | 0.400917 | 1.005819 | 0 |
| RK4 b2 | 1000.0000 | `[0.157, 0.218, 0.129, 0.130, 0.139, 0.228]` | `[0.217, 0.273, 0.126, 0.091, 0.136, 0.157]` | 0.245405 | 0.927541 | 0 |
| Euler b4 | 706.8966 | `[0.203, 0.133, 0.065, 0.081, 0.170, 0.127, 0.080, 0.140]` | `[0.293, 0.212, 0.071, 0.101, 0.253, 0.015, 0.005, 0.051]` | 0.357596 | 1.544306 | 0 |
| Euler b4 | 1000.0000 | `[0.066, 0.153, 0.068, 0.152, 0.124, 0.102, 0.176, 0.158]` | `[0.061, 0.146, 0.081, 0.086, 0.121, 0.086, 0.237, 0.182]` | 0.315309 | 1.491132 | 0 |

结论：

- 所有实验的 `dead_experts_threshold_1pct = 0`，没有全局专家塌缩。
- 深层 Euler b4 entropy 更高，表示 gate 更软、更分散。
- 深层 Euler b4 在 `Re_706p896552` 上 expert 5/6 的 Top-1 比例只有 0.015/0.005，说明 mean-load 没死不等于 hard routing 完全均匀。

### 12.2 相位分箱主导专家

| 实验 | Re | Phase [0,0.25) | Phase [0.25,0.5) | Phase [0.5,0.75) | Phase [0.75,1.0) |
|---|---:|---|---|---|---|
| Euler b2 | 706.8966 | E0 / 0.422 | E0 / 0.320 | E0 / 0.260 | E0 / 0.509 |
| Euler b2 | 1000.0000 | E1 / 0.271 | E1 / 0.360 | E0 / 0.280 | E0 / 0.320 |
| RK4 b2 | 706.8966 | E0 / 0.356 | E0 / 0.300 | E1 / 0.260 | E0 / 0.396 |
| RK4 b2 | 1000.0000 | E1 / 0.271 | E1 / 0.300 | E1 / 0.320 | E0 / 0.340 |
| Euler b4 | 706.8966 | E0 / 0.333 | E0 / 0.440 | E4 / 0.300 | E4 / 0.245 |
| Euler b4 | 1000.0000 | E7 / 0.188 | E7 / 0.240 | E6 / 0.240 | E6 / 0.400 |

解释：

- 不同 Re 的主导专家不同，说明 router 使用 Re 和 phase 上下文区分流动状态。
- 深层模型中 phase 后半段专家发生切换，例如 `Re_706p896552` 从 E0 转向 E4，`Re_1000p000000` 从 E7 转向 E6。
- 完整 top-k 组合计数保存在各 `*_metrics.json` 的 `routing_analysis_test.topk_set_counts` 中。

## 13. V5 限制与风险

1. `r_u=8, r_p=8` 是快速验证配置，POD 能量较低；后续应在 `r_u=16/32` 上复测关键结论。
2. RK4 当前使用 teacher-forced pressure/phase context，不是完全自主 pressure-ROM。
3. 深层 MoE 在 `Re_706p896552` 未提升平均 rollout，说明单纯增大容量不能保证泛化。
4. Validation alpha error 偏高，可能来自低阶截断、跨 Re 泛化难度和 alpha direct head/rollout 目标之间的张力。
5. 深层 RK4 成本较高，本轮没有完成有效深层 RK4 对比。

## 14. 推荐后续方向

1. 以 `v5_r8_rk4_b2` 作为当前稳定基线，因为它在 one-step integrator 和 rollout 上收益最明确。
2. 在 RK4 下先尝试小幅增加容量，例如 3 blocks / 6 experts，而不是直接 4 blocks / 8 experts。
3. 增强 router 约束：提高 load-balance 权重、使用 top-k diversity loss 或对 hard top-1 usage 加约束。
4. 将 pressure/context 也纳入自主推进，减少 teacher-forced pressure 对 rollout 评估的依赖。
5. 在 `r_u=16, r_p=16` 上复测 RK4 和 router 结论，判断低阶截断是否掩盖深层模型收益。

## 15. 文件索引

```text
V5/
  TECHNICAL_REPORT_V5.md                  # 本完整技术报告
  test_results_v5/
    README.md                             # V5 快速摘要
    deep_moe_rom_v5.py                    # 完整 PyTorch 实验脚本
    results/
      v5_r8_euler_b2_metrics.json
      v5_r8_euler_b2_summary.md
      v5_r8_rk4_b2_metrics.json
      v5_r8_rk4_b2_summary.md
      v5_r8_euler_b4_deep_metrics.json
      v5_r8_euler_b4_deep_summary.md
```
