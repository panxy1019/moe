# V11 技术报告：Operator-Space MoE-ROM

日期：2026-06-23

代码：`test_results_v11/deep_moe_rom_v11.py`

数据：`/root/moe/V8/data/Global_POD_Weighted_L2`

## 1. 目标

V11 按方案 B 将 V10 的 residual-correction MoE 改为 operator-space MoE：

- expert 不再只作为 hidden residual patch；
- 每个 routed expert 输出一个局部 ROM velocity RHS operator；
- router 根据当前 ROM 状态、Re、phase、能量/范数等物理描述符选择动力学算子；
- 速度推进保留 RK4；
- 压力和速度共享同一个 regime router。

## 2. 模型结构

输入特征由 V10 的状态/历史特征扩展，包含：

- velocity POD 系数 `a_t`；
- pressure POD 系数 `b_t`；
- `Re`、`1/Re`、phase harmonics；
- Galerkin RHS descriptor；
- 低/中/高模态范数、velocity/pressure 能量、总能量、能量比例、pressure/velocity 范数比；
- 2-step history 以及状态差分。

核心网络：

```text
features x_t
  -> PhysicalContextEncoder
  -> latent refinement blocks
  -> router pi_i = router([h_t, x_t])
  -> operator experts f_i(h_t)
  -> f = sum_i pi_i f_i
  -> RK4(a_t, f)
```

每个 expert 输出：

- 标准化完整 velocity RHS `da/dt`；
- 一个 pressure branch，共享同一个 router。

V11 实现了两种 pressure target：

- `closure`: expert 输出 pressure surrogate closure，`b_next = b_base(a_next) + closure`；
- `state`: expert 直接输出 `b_next`，用于测试低 Re pressure 是否受 surrogate residual target 限制。

## 3. Loss 与诊断

保留并改造的损失：

- one-step coefficient loss：由当前 operator 推进一步后与 `a_{t+1}` 比较；
- full RHS dynamic loss：直接拟合中心差分 `adot`；
- pressure loss；
- sampled reconstruction loss；
- short rollout loss；
- relative RHS / pressure / one-step losses；
- router load-balance；
- router entropy regularization；
- router temporal smoothness；
- expert diversity regularization；
- weak Re-regime router separation。

新增诊断：

- 每个 expert 的 mean load 和 top-1 fraction；
- low/mid/high Re 分组 expert 使用分布；
- per-expert one-step 与 rollout error；
- expert 输出 pairwise cosine，用于检查 expert collapse。

## 4. 实验设置

主 truncation 与 V10 一致：

| item | value |
|---|---:|
| `r_u`, `r_p` | 16, 16 |
| velocity POD energy | 0.943109 |
| pressure POD energy | 0.944973 |
| held-out Re indices | 10, 59, 99 |
| held-out Re values | 56.3745, 120.0, 300.0 |

主实验：

- `v11_r16_operator_space_b2`: pressure closure target, 3 held-out Re。
- `v11_r16_operator_space_state_b2`: direct pressure state target, 3 held-out Re。

Low-Re ablations：

- `v11_r16_operator_space_state_lowRe`: state target, only Re=56 split。
- `v11_r16_operator_space_state_amp_lowRe`: state target + pressure amplitude weighting, only Re=56 split。

## 5. 结果

### 5.1 Closure-Pressure V11

Experiment: `v11_r16_operator_space_b2`

| Test Re | Best epoch | RHS L2 | Pressure L2 | Auto a one-step | Auto b one-step | Auto a rollout | Auto b rollout | Load CV | Entropy | Dead experts | Max expert cos | Collapse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 56.3745 | 140 | 0.211674 | 0.362618 | 0.091935 | 0.386260 | 0.736995 | 1.075368 | 1.646366 | 1.003098 | 0 | 0.908752 | false |
| 120.0 | 65 | 0.287706 | 0.063967 | 0.114243 | 0.135455 | 0.534633 | 0.925962 | 1.351615 | 0.979842 | 0 | 0.707414 | false |
| 300.0 | 145 | 0.199877 | 0.064081 | 0.110078 | 0.099287 | 1.029878 | 0.839126 | 0.703721 | 0.865097 | 0 | 0.732109 | false |

Closure target keeps Re=120 and Re=300 one-step pressure below 10%, but low Re pressure worsens versus V10.

### 5.2 State-Pressure V11

Experiment: `v11_r16_operator_space_state_b2`

| Test Re | Best epoch | RHS L2 | Pressure L2 | Auto a one-step | Auto b one-step | Auto a rollout | Auto b rollout | Load CV | Entropy | Dead experts | Max expert cos | Collapse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 56.3745 | 140 | 0.226883 | 0.301817 | 0.110884 | 0.301817 | 0.513716 | 0.668784 | 1.890811 | 0.876136 | 0 | 0.666638 | false |
| 120.0 | 120 | 0.253182 | 0.061540 | 0.073694 | 0.061540 | 0.354867 | 0.260632 | 1.488867 | 0.915719 | 0 | 0.663731 | false |
| 300.0 | 115 | 0.202380 | 0.109804 | 0.121841 | 0.109805 | 0.827936 | 0.522854 | 1.047082 | 0.865211 | 0 | 0.723322 | false |

State target improves low-Re pressure and rollout versus closure target, and greatly improves Re=120 rollout. Re=300 pressure becomes slightly worse than the 10% target.

### 5.3 Low-Re Ablations

| Experiment | Pressure target | Extra weighting | Pressure L2 | Auto a one-step | Auto b rollout | Comment |
|---|---|---|---:|---:|---:|---|
| `v11_r16_operator_space_b2` | closure | none | 0.362618 | 0.091935 | 1.075368 | baseline V11 closure |
| `v11_r16_operator_space_state_lowRe` | state | none | 0.292125 | 0.110974 | 0.712607 | best low-Re V11 pressure |
| `v11_r16_operator_space_state_amp_lowRe` | state | pressure amplitude weighting | 0.325060 | 0.113814 | 1.037625 | weighting worsened low Re |

Amplitude weighting did not help. It increased low-Re pressure error and rollout error.

## 6. 与 V10 对比

V10 main:

| Test Re | V10 Pressure L2 | V10 Auto a one-step | V10 Auto b one-step | V10 Auto a rollout | V10 Auto b rollout |
|---:|---:|---:|---:|---:|---:|
| 56.3745 | 0.283692 | 0.066361 | 0.289178 | 0.403320 | 0.543825 |
| 120.0 | 0.020780 | 0.054604 | 0.063671 | 0.327383 | 0.311903 |
| 300.0 | 0.064957 | 0.079498 | 0.087281 | 0.496450 | 0.587781 |

V11 state target:

| Test Re | V11 Pressure L2 | V11 Auto a one-step | V11 Auto b one-step | V11 Auto a rollout | V11 Auto b rollout |
|---:|---:|---:|---:|---:|---:|
| 56.3745 | 0.301817 | 0.110884 | 0.301817 | 0.513716 | 0.668784 |
| 120.0 | 0.061540 | 0.073694 | 0.061540 | 0.354867 | 0.260632 |
| 300.0 | 0.109804 | 0.121841 | 0.109805 | 0.827936 | 0.522854 |

V11 does not beat V10 in one-step pressure accuracy. Its main value is that it satisfies the operator-space architecture requirement and exposes clear regime routing diagnostics. The state-pressure branch improves pressure rollout at Re=120 and Re=300 compared with closure V11, but not enough to beat V10 everywhere.

## 7. Expert Routing

Observed routing behavior:

- No dead experts in all V11 runs under the 1% mean-load threshold.
- No expert-output collapse: maximum absolute pairwise expert cosine stayed below 0.95.
- Low Re consistently routes to a concentrated top-1 expert, even though gate floor keeps all experts nonzero.
- High Re uses a healthier multi-expert distribution, especially in closure target and state target.

Representative top-1 fractions:

| Experiment | Re | Top-1 expert fractions |
|---|---:|---|
| closure | 56.3745 | `[0,0,0,0,0,1,0,0]` |
| closure | 300.0 | `[0,0.159,0.079,0.333,0.238,0,0.190,0]` |
| state | 56.3745 | `[0,1,0,0,0,0,0,0]` |
| state | 300.0 | `[0,0,0.286,0.254,0.429,0,0.032,0]` |

## 8. Conclusion

V11 implemented the requested operator-space MoE minimal version:

- shared encoder + router + multiple operator experts;
- expert outputs full local velocity RHS operators;
- weighted expert operator composition;
- RK4 time integration retained;
- velocity/pressure share one regime router;
- load-balance, entropy, diversity, Re-regime losses and diagnostics added.

Accuracy target:

- The all-Re `<10%` pressure target was not reached.
- Best V11 low-Re pressure was `0.292125`, still far above 10%.
- State-pressure V11 gets Re=120 pressure below 10% and closure-pressure V11 gets Re=300 pressure below 10%, but no single V11 run gets all three below 10%.

Interpretation:

The low-Re pressure issue appears to be a data/target sensitivity bottleneck rather than a simple router-collapse problem. Increasing pressure relative weight, switching closure to state target, and amplitude weighting did not break the 29% low-Re floor. Historical V8/V9 reports also show that increasing rank to r24/r32 worsened low-Re pressure under similar budgets.

Next suggested step:

Use operator-space V11 as the architecture baseline, but add a more structured pressure operator, such as a pressure-specific algebraic operator expert conditioned on `a_next`, or a local Re-band pressure surrogate instead of a generic MLP pressure branch.
