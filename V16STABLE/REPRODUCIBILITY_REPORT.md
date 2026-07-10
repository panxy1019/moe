# V16_1_SteadyPressureAnchor32 Reproducibility Report

本文档面向 ClaudeCode 复现 `V16_1_SteadyPressureAnchor32` 单组实验。它只解释这一组 case 的数据、网络、loss、训练与评估流程；横向对比三组 V16_1 的结论见 `V16_1_TrainStableAttractorMoE_REPORT.md`。

## 1. 实验定位

`V16_1_SteadyPressureAnchor32` 是 V16_1 TrainStableAttractorMoE 系列中推荐作为下一轮默认 baseline 的版本。它沿用 V16 的 `FullRegimeLoss32` 主体，不修改 HPRS-MoE-ROM 主干、Galerkin、RK4、Poisson Pressure Base、modal AdaptiveGate、POD/ROM 维数、优化器和训练流程，只在 steady attractor 的 closed-loop rollout 中增加压力 anchor 约束，用来抑制低 Re steady/pre-Hopf 区域的 pressure residual drift。

固定主干：

- 数据库：Re=20-200 Physics-Generalizable Attractor Database。
- ROM 维数：`ru=32`, `rp=32`。
- 速度：Galerkin base + operator-space MoE residual closure + RK4。
- 压力：Poisson surrogate base + Pressure MoE residual + modal AdaptiveGate。
- 训练 rollout curriculum：`4 -> 8 -> 12 -> 16`。
- 主评估：24-step autonomous rollout。

这一组 case 的唯一新增项：

```text
lambda_v16_1_steady_p_state = 0.02
lambda_v16_1_steady_p_mean = 0.02
lambda_v16_1_steady_p_delta = 0.01
lambda_v16_1_steady_residual_damp = 0.01
lambda_v16_1_steady_p_energy = 0.01
v16_1_loss_warmup_epochs = 0
```

Hopf V16_1 growth/floor losses在本 case 中关闭：

```text
lambda_v16_1_hopf_growth = 0.0
lambda_v16_1_hopf_false_growth = 0.0
lambda_v16_1_hopf_floor_rel = 0.0
```

## 2. 代码入口与产物路径

整理后的仓库分支：

```bash
git checkout codex/v16stable
```

核心文件：

```text
V16STABLE/train_v16_1_steady_pressure_anchor32.py
V16STABLE/run_train.sh
V16STABLE/eval_checkpoint.sh
```

远端训练输出目录：

```text
/root/moe/V16STABLE/results/V16_1_SteadyPressureAnchor32_ru32_rp32
```

本次已完成 run 的关键产物：

```text
V16_1_SteadyPressureAnchor32_ru32_rp32_metrics.json
V16_1_SteadyPressureAnchor32_ru32_rp32_summary.md
V16_1_SteadyPressureAnchor32_ru32_rp32_error_vs_re.csv
V16_1_SteadyPressureAnchor32_ru32_rp32_error_vs_re.svg
V16_1_SteadyPressureAnchor32_ru32_rp32_pressure_anchor_stats.npz
V16_1_SteadyPressureAnchor32_ru32_rp32_Re_24p630436_checkpoint.pt
run.log
```

SwanLab run：

```text
https://swanlab.cn/@panxy1019/V16_1_TrainStableAttractorMoE/runs/s94sbgtx
```

## 3. 数据结构

数据根目录：

```text
/root/moe/ROM_PhysicsGeneralizable/data/Global_POD_AreaWeighted_L2
```

输入文件：

```text
pod_snapshot_index.csv
global_velocity_pod_area_weighted_l2.npz
global_pressure_pod_area_weighted_l2.npz
```

ROM/pressure tensor：

```text
/root/moe/ROM_PhysicsGeneralizable/data/semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz
/root/moe/ROM_PhysicsGeneralizable/data/pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz
```

`build_arrays(args)` 读取并构造如下数组：

| 字段 | 形状含义 | 用途 |
|---|---|---|
| `a` | `[N, ru]` | velocity POD 系数，取 `coeff_uv[:, :32]` |
| `b` | `[N, rp]` | pressure POD 系数，取 `coeff_p[:, :32]` |
| `rhs_g` | `[N, ru]` | Galerkin RHS base |
| `adot` | `[N, ru]` | velocity POD 的中心差分时间导数 |
| `residual` | `[N, ru]` | `adot - rhs_g`，速度 MoE 的 residual target |
| `a_next` | `[N, ru]` | 下一时刻 velocity POD |
| `b_next` | `[N, rp]` | 下一时刻 pressure POD |
| `pressure_base_next` | `[N, rp]` | `pressure_surrogate(a_next, Re)` |
| `pressure_residual` | `[N, rp]` | `b_next - pressure_base_next` |
| `re` | `[N]` | Reynolds 数 |
| `label_id` | `[N]` | Re label index |
| `regime` | `[N]` | 原始物理 regime |
| `attractor_id` | `[N]` | steady/hopf/periodic 三类吸引子标签 |
| `phase` | `[N]` | 相位特征 |
| `time` | `[N]` | 物理时间 |
| `prev_idx`, `next_idx` | `[N]` | 同一 Re 轨迹的前后索引 |
| `hist_idx` | `[N, history_len]` | 当前及历史状态索引，`history_len=3` |
| `x` | `[N, input_dim]` | encoder 输入特征 |

### 3.1 Attractor 标签

本数据库在 V16 后被解释为 Attractor Database，而不是 Hopf transient database。所有样本统一 `stage_label=attractor`，再由原始 `regime` 映射为三类 attractor：

- steady：`steady_wake`, `pre_hopf_steady`
- hopf：`hopf_transition`
- periodic：`developing_periodic_shedding`, `mature_periodic_shedding`, `high_re_2d_periodic_near_modeA`

### 3.2 输入特征 `x`

当前时刻基础特征 `base_x` 由 `make_features_np(a,b,rhs,re,phase)` 构造：

```text
[Re_norm, inv_Re,
 sin/cos phase harmonics k=1..4,
 a_t, b_t, rhs_g(t),
 ||a_low||, ||a_mid||, ||a_high||,
 ||b||, ||rhs_g||,
 ||a||^2, ||b||^2, total_energy,
 low_mode_energy_fraction, high_mode_energy_fraction,
 ||b|| / ||a||]
```

随后 `make_history_features_np` 拼接历史状态，对 `history_len=3` 使用：

```text
x_t = [
  base_x(t),
  a(t-1), b(t-1), rhs_g(t-1), a(t)-a(t-1), b(t)-b(t-1), rhs_g(t)-rhs_g(t-1),
  a(t-2), b(t-2), rhs_g(t-2), a(t)-a(t-2), b(t)-b(t-2), rhs_g(t)-rhs_g(t-2)
]
```

所有 `x` 在训练前用训练集均值/尺度标准化。

## 4. 数据划分

测试 Re 采用 `--test-re-selection regime_default`，从 100 个 Re 中选取覆盖 steady、Hopf near-onset 和 periodic 的 11 个 held-out Re。它们完全不参与训练：

```text
24.630, 32.740, 39.685, 45.143,
47.081, 49.022, 51.786,
70.315, 100.352, 149.059, 189.862
```

实际 split 统计：

| 项 | 数值 |
|---|---:|
| 总 Re 数 | 100 |
| 训练 Re 数 | 89 |
| 测试 Re 数 | 11 |
| 总 snapshots | 14167 |
| valid samples | 13867 |
| dense train samples | 10970 |
| kept train samples | 10970 |
| validation samples | 1547 |
| test samples | 1350 |
| train time stride | 1 |
| train Re stride | 1 |
| compression ratio vs dense train | 1.0 |

训练样本按 attractor-balanced sampling 抽样，使 steady/hopf/periodic 三类在期望 epoch 权重上接近 `1/3, 1/3, 1/3`：

| Split | Steady | Hopf | Periodic |
|---|---:|---:|---:|
| train | 813 | 1949 | 8208 |
| val | 160 | 266 | 1121 |
| rollout_pool | 813 | 1949 | 8208 |
| test | 244 | 474 | 632 |

注意：这里不做 V14_2 式时间稀疏，也不做 Re 稀疏训练。

## 5. 网络架构

模型类：`OperatorSpaceMoEROM`。

### 5.1 Shared Encoder

`PhysicalContextEncoder(in_dim, hidden_dim=224, dropout=0.04)`：

```text
Linear(in_dim -> 224)
LayerNorm
SiLU
Dropout(0.04)
Linear(224 -> 224)
LayerNorm
SiLU
Dropout(0.04)
Linear(224 -> 224)
LayerNorm
SiLU
```

再接 `num_blocks - 1 = 2` 个 residual refine block：

```text
h = h + [LayerNorm -> Linear -> SiLU -> Dropout -> Linear](h)
```

本 case `attractor_conditioned=False`，因此没有启用 Attractor Router/Adapter。

### 5.2 分层 Router

Router 输入为：

```text
router_in = concat(h, x_standardized)
```

两级结构：

1. Group Router：从 3 个 physics-regime group 中选组。
2. Group 内 Velocity/Pressure Router：在选中 group 内做 top-2 routing。

参数：

```text
num_regime_groups = 3
experts_per_group = 6
shared_experts_per_group = 1
group_top_k = 1
top_k = 2
temperature = 0.95
group_temperature = 0.90
gate_floor = 0.0
group_gate_floor = 0.0
```

Group Router 是硬 top-1 稀疏选择，因此 `group_entropy=0` 是预期现象。

### 5.3 Expert 结构

每个 group 包含：

- 1 个 shared velocity expert
- 6 个 routed velocity experts
- 1 个 shared pressure expert
- 6 个 routed pressure experts

总计按 velocity/pressure 各自计算为 `3 * (1 + 6) = 21` 个专家。

Expert 类：`PhysicsAwareExpert`。每个 expert 是结构化 operator block，不是普通纯 MLP：

```text
input = concat(h, state)
z = LayerNorm -> Linear -> GELU
z = 3 x ExpandedFFNBlock(z)
out = LinearHead(z) + Linear(state) + 0.05 * LowRankQuadratic(state, state)
```

其中：

```text
expert_hidden = 768
expert_blocks = 3
quadratic_rank = 4
quadratic_scale = 0.05
```

Velocity expert：

```text
state_dim = ru = 32
out_dim = ru = 32
```

Pressure expert：

```text
state_dim = ru + rp = 64
out_dim = rp = 32
```

`pressure_input_mode=pressure_only` 表示 Pressure Head 使用当前标准化 state slice，即当前实现中的 `[a_t, b_t]` 压力相关输入；不是 `a_next`，也不是 `[a_next, b_base]`。

### 5.4 Shared/Routed Mixing

在选中的 group 内：

```text
shared_part = shared_scale / (shared_scale + routed_scale)
routed_part = routed_scale / (shared_scale + routed_scale)
group_out = shared_part * SharedExpert(h,state)
          + routed_part * sum_i top2_gate_i * RoutedExpert_i(h,state)
```

本 case：

```text
shared_scale = 1.0
routed_scale = 0.85
```

所以每个被选中 group 的 shared expert 始终参与输出。这也是本模型里“共享专家负责通用知识”的实际实现方式：不是全局共享 expert，而是每个 physics-regime group 内有一个 always-active shared expert。

## 6. Velocity 动力学与 RK4

速度分支预测 operator-space closure residual，而不是直接黑盒输出下一步状态。

对当前状态：

```text
rhs_g = GalerkinRHS(a_t, b_t, Re)
rhs_residual = VelocityMoE(h, a_t)
f_u = rhs_g + rhs_residual
```

因为 `--rhs-target residual`，训练 target 是 `adot - rhs_g`，推理时恢复为完整 RHS：

```text
f_u = rhs_g + standardized_inverse(rhs_std)
```

时间推进使用 RK4：

```text
k1 = f(a_t)
k2 = f(a_t + 0.5 dt k1)
k3 = f(a_t + 0.5 dt k2)
k4 = f(a_t + dt k3)
a_next = a_t + dt/6 * (k1 + 2k2 + 2k3 + k4)
```

Pressure 在每一步使用 RK4 推出的 `a_next` 计算 Poisson base，然后叠加 pressure residual closure。

## 7. Pressure Closure

压力基线来自 Poisson surrogate：

```text
b_base = pressure_surrogate(a_next, Re)
       = c_tilde(Re) + A_tilde(Re) a_next + H_tilde(a_next, a_next)
```

Pressure Head 输出 residual：

```text
pressure_residual = PressureMoE(h, [a_t, b_t])
```

本 case 使用 modal AdaptiveGate：

```text
alpha = sigmoid(MLP(h))      # shape: [batch, rp=32]
beta = 0
b_pred = alpha * b_base + pressure_residual
```

Confidence Head：

```text
Linear(224 -> 64)
GELU
Linear(64 -> 32)
Sigmoid
```

该 gate 是每个 pressure POD 模态独立的 32 维向量，不是单一标量。

## 8. Loss 设计

总 loss 包含原 V16 FullRegimeLoss 主体与 V16_1 steady pressure anchor。核心权重如下：

```text
lambda_coeff = 0.75
lambda_dyn = 0.90
lambda_pressure = 0.95
lambda_rollout = 0.45
lambda_pressure_rollout = 0.45
lambda_energy = 0.05
lambda_trajectory_consistency = 0.18
lambda_router_balance = 0.06
lambda_router_entropy = -0.002
lambda_group_balance = 0.04
lambda_group_supervision = 0.04
lambda_router_smooth = 0.04
lambda_expert_diversity = 0.006
lambda_regime_router = 0.004
lambda_alpha_rel = 0.04
lambda_rhs_rel = 0.06
lambda_pressure_rel = 0.70
rollout_relative_mix = 0.35
relative_floor_frac = 0.05
```

V16 attractor-specific loss 保留：

```text
lambda_steady_rhs = 0.02
lambda_steady_state = 0.01
lambda_attractor_hopf_radius = 0.02
lambda_attractor_hopf_overshoot = 0.02
lambda_attractor_hopf_onset = 0.01
lambda_periodic_energy = 0.02
lambda_periodic_radius = 0.01
```

V16_1 steady pressure anchor 只在 `attractor_id == steady` 的 rollout step 上激活：

```text
L_steady_p_state = mean(((b_pred_next - b_true_next) / pressure_state_scale)^2)
L_steady_p_mean  = mean(((b_pred_next - b_mean(Re)) / pressure_state_scale)^2)
L_steady_p_delta = mean(((b_pred_next - b_cur) / pressure_state_scale)^2)
L_steady_residual_damp = mean((pressure_residual / pressure_scale)^2)
L_steady_p_energy = Huber(
  (||b_pred_next||^2 - ||b_true_next||^2)
  / (||b_true_next||^2 + pressure_rel_floor)
)
```

对应权重：

```text
0.02 * L_steady_p_state
+ 0.02 * L_steady_p_mean
+ 0.01 * L_steady_p_delta
+ 0.01 * L_steady_residual_damp
+ 0.01 * L_steady_p_energy
```

`b_mean(Re)` 由训练前保存的 `pressure_mean_by_label` 提供，并写入：

```text
V16_1_SteadyPressureAnchor32_ru32_rp32_pressure_anchor_stats.npz
```

## 9. 训练配置

主要训练参数：

```text
batch_size = 256
epochs = 240
min_epochs = 130
patience = 70
eval_every = 5
eval_routing_every = 20
lr = 5.5e-4
weight_decay = 1.5e-4
seed = 1600
optimizer = AdamW
allow_tf32 = true
device = cuda
torch_version = 2.11.0+cu126
```

rollout 训练：

```text
train_rollout_steps = 16
curriculum_steps = 4,8,12,16
rollout_batch = 2
rollout_every_batches = 1
scheduled_sampling_start = 0.0
scheduled_sampling_end = 0.85
scheduled_sampling_warmup_frac = 0.70
```

Scheduled sampling 的含义：

```text
a_feed = p * a_pred + (1-p) * a_true
b_feed = p * b_pred + (1-p) * b_true
```

`p` 从 0 逐渐上升到 0.85，使训练从 teacher forcing 过渡到更多使用模型自身预测作为下一步输入。

## 10. 复现命令

在集群上执行：

```bash
cd /root/moe
git fetch origin
git checkout codex/v16stable
```

检查数据：

```bash
test -f /root/moe/ROM_PhysicsGeneralizable/data/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv
test -f /root/moe/ROM_PhysicsGeneralizable/data/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz
test -f /root/moe/ROM_PhysicsGeneralizable/data/Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz
test -f /root/moe/ROM_PhysicsGeneralizable/data/semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz
test -f /root/moe/ROM_PhysicsGeneralizable/data/pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz
```

启动单组训练：

```bash
cd /root/moe/V16STABLE
export SWANLAB_TRACKING_MODE=online
export SWANLAB_TRACKING_PROJECT=V16_1_TrainStableAttractorMoE
export SWANLAB_TRACKING_GROUP=v16-1-train-stable-attractor-moe
CUDA_VISIBLE_DEVICES=0 ./run_train.sh
```

如果只想离线复现结果写盘，可禁用 SwanLab：

```bash
export SWANLAB_TRACKING_MODE=disabled
CUDA_VISIBLE_DEVICES=0 ./run_train.sh
```

不要把 SwanLab API key、SSH 密码或 GitHub token 写进脚本和报告；需要登录时用环境变量或本机已登录状态。

### 10.1 Eval-only checkpoint

如已有 checkpoint，只跑终评：

```bash
cd /root/moe/V16STABLE
export SWANLAB_TRACKING_MODE=disabled
CUDA_VISIBLE_DEVICES=0 ./eval_checkpoint.sh \
  /root/moe/V16STABLE/results/V16_1_SteadyPressureAnchor32_ru32_rp32/V16_1_SteadyPressureAnchor32_ru32_rp32_Re_24p630436_checkpoint.pt
```

## 11. 评估协议

每个 held-out Re 输出：

- one-step velocity relative L2
- one-step pressure relative L2
- 24-step autonomous rollout velocity relative L2
- 24-step autonomous rollout pressure relative L2
- RHS relative L2
- pressure energy error
- pressure base relative L2
- residual-only pressure error
- adaptive gate `alpha` mean/std
- base/residual contribution ratio
- group router usage
- expert top1/top-k/mean-load
- Hopf pair `(a0,a1)` amplitude diagnostics

注意：报告中的误差都是 relative L2 或相对能量误差，不是绝对 RMSE。JSON 内同时保留部分 RMSE 字段。

## 12. 复现核对指标

本次 run 的 best epoch：

```text
best_epoch = 225
best_val_score = 0.482090595504269
```

Overall held-out Re 均值：

| 指标 | mean | std | min | max |
|---|---:|---:|---:|---:|
| RHS relative L2 | 0.3105 | 0.1422 | 0.1852 | 0.6404 |
| one-step velocity relative L2 | 0.03994 | 0.03304 | 0.01099 | 0.1317 |
| one-step pressure relative L2 | 0.1229 | 0.08510 | 0.03246 | 0.2882 |
| 24-step rollout velocity relative L2 | 0.2257 | 0.1983 | 0.02547 | 0.6063 |
| 24-step rollout pressure relative L2 | 0.2967 | 0.2156 | 0.03235 | 0.6603 |
| rollout pressure energy error | 0.3143 | 0.3882 | 0.001423 | 1.270 |
| pressure base relative L2 | 83.02 | 161.67 | 0.7147 | 588.07 |
| alpha mean | 0.2421 | 0.2382 | 0.06175 | 0.5729 |

按 attractor 分组均值：

| Regime | one u | one p | roll u | roll p | RHS | p energy | alpha | active experts | dead experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Steady | 0.01546 | 0.1516 | 0.1998 | 0.4071 | 0.2797 | 0.5686 | 0.06278 | 3.000 | 18.0 |
| Hopf | 0.07662 | 0.1825 | 0.4953 | 0.4627 | 0.4856 | 0.3747 | 0.06188 | 3.000 | 18.0 |
| Periodic | 0.03691 | 0.04951 | 0.04933 | 0.06178 | 0.2101 | 0.01485 | 0.5566 | 4.407 | 11.5 |

逐 Re 核对表：

| Re | regime | one u | one p | roll u | roll p | RHS | alpha | active exp | top1 experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 24.63 | Steady | 0.02317 | 0.2874 | 0.3061 | 0.6274 | 0.4274 | 0.0633 | 3 | e0:1.000 |
| 32.74 | Steady | 0.01506 | 0.1295 | 0.2068 | 0.4680 | 0.2712 | 0.06269 | 3 | e0:1.000 |
| 39.69 | Steady | 0.01261 | 0.09475 | 0.1609 | 0.3034 | 0.2246 | 0.06255 | 3 | e0:1.000 |
| 45.14 | Steady | 0.01100 | 0.09467 | 0.1256 | 0.2295 | 0.1954 | 0.06260 | 3 | e0:1.000 |
| 47.08 | Hopf | 0.05613 | 0.1234 | 0.5925 | 0.3550 | 0.6404 | 0.06192 | 3 | e0:1.000 |
| 49.02 | Hopf | 0.04199 | 0.1360 | 0.2872 | 0.3729 | 0.5010 | 0.06196 | 3 | e0:1.000 |
| 51.79 | Hopf | 0.1317 | 0.2882 | 0.6063 | 0.6603 | 0.3154 | 0.06175 | 3 | e0:1.000 |
| 70.31 | Periodic | 0.06108 | 0.08664 | 0.1019 | 0.1272 | 0.2081 | 0.5130 | 4.386 | e0:0.728; e14:0.190; e7:0.082 |
| 100.35 | Periodic | 0.02901 | 0.03246 | 0.02834 | 0.03664 | 0.2313 | 0.5698 | 4.620 | e7:0.981; e14:0.019 |
| 149.06 | Periodic | 0.02565 | 0.03487 | 0.02547 | 0.03235 | 0.2158 | 0.5707 | 4.316 | e7:0.747; e14:0.222; e0:0.032 |
| 189.86 | Periodic | 0.03191 | 0.04406 | 0.04161 | 0.05099 | 0.1852 | 0.5729 | 4.304 | e14:0.677; e7:0.297; e0:0.025 |

## 13. 结果解读

`V16_1_SteadyPressureAnchor32` 的主要收益是整体 one-step 和 periodic rollout 都比较稳，且 overall 24-step velocity rollout 是 V16_1 三组中最低。steady pressure anchor 确实约束了 residual/head 的漂移倾向，但低 Re pressure rollout 仍偏高，尤其 `Re=24.63` 的 24-step pressure relative L2 为 `0.6274`，说明该 loss 还不能单独解决 steady pressure 长期漂移。

Pressure Base 的 held-out relative L2 均值为 `83.02`，远高于最终 closure pressure error。AdaptiveGate 在 steady/Hopf 区域的 alpha 只有约 `0.062`，而 periodic 区域约 `0.557`，说明模型自动学会了在低 Re/Hopf 区域不信任 Poisson base，在周期区更多使用 base。当前压力预测主要依赖 residual/head 和 gate 抑制错误 base 的注入。

Router 仍有 group/expert 使用退化：steady/Hopf 的 top1 全部集中到 `e0`，active experts mean 为 3；periodic 区域会分流到 `e7/e14` 等专家。复现实验时不能只看 active expert count，还要看 top1、top-k set 和 mean-load。

## 14. ClaudeCode 复现检查清单

1. 确认分支为 `codex/v16stable`。
2. 确认 5 个数据/tensor 文件均存在。
3. 确认使用 `V16STABLE/run_train.sh` 的冻结配置。
4. 确认 `ru=32`, `rp=32`, `closure-mode=adaptive_gate`, `pressure-input-mode=pressure_only`, `pressure-base-mode=static`。
5. 确认只启用 steady pressure anchor 五个 V16_1 loss，Hopf V16_1 loss 为 0。
6. 用 `CUDA_VISIBLE_DEVICES=0 ./run_train.sh` 启动。
7. 训练结束后检查 best epoch 是否接近 225，overall 指标是否接近第 12 节。
8. 若只做评估，用 `--eval-only-checkpoint` 指向 best checkpoint。
9. 解析 JSON 时使用嵌套字段：`deep_moe`、`one_step_integrator`、`one_step_autonomous_pressure`、`rollout_autonomous_pressure`、`routing_analysis_test`，不要假设指标平铺。

## 15. 后续风险

- `pressure_base_l2` 极高，说明 Poisson surrogate base 对部分 held-out Re 尤其 Hopf/low-Re 仍不可靠。
- Steady/Hopf 的 group routing/top1 expert 几乎坍缩到单一路径，后续若继续沿用此 baseline，应加入 top-2 usage、mean load 和 group 内专家退化诊断。
- Steady pressure anchor 不应与 Hopf growth loss 直接叠加为 Combined；已有实验显示二者存在负迁移。
- 若 V16_2 继续以本 case 为 baseline，建议优先做 pressure residual state-space contraction、pressure mean projection 或 regime-aware pressure base，而不是单纯加大网络。
