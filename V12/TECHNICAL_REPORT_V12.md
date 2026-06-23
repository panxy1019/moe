# V12 技术报告：Shared-Expert Operator-Space MoE-ROM

日期：2026-06-23

代码：`test_results_v12/deep_moe_rom_v12.py`

数据：`/root/moe/V8/data/Global_POD_Weighted_L2`

## 1. 结论先行

这版 V12 重点回答两个问题：

1. V11 当前主实验有没有共享专家？
2. 如果借鉴大模型 MoE 架构，后续怎么把速度/压力误差继续往 10% 内压？

结论：

- V11 最终主实验没有启用 learned shared experts，`num_shared_experts=0`。V10 有 residual-correction 风格的 shared expert 设计，但 V11 为了做纯 operator-space 最小版本时关掉了。
- V12 已经重新加入共享专家：默认 `num_shared_experts=2`，两个 learned shared operator experts 总是参与；同时 Galerkin RHS 作为物理共享 operator 底座。
- V12 采用 `Galerkin RHS + learned closure` 的 operator-space 形式。网络输出 residual operator closure，但每个 expert 表示的是完整局部算子 `f_i = f_Galerkin + c_i`，最终仍用 RK4 推进速度。
- V12 主模型 `v12_r16_shared_operator_closure_b3` 把三个测试 Re 的速度 autonomous one-step error 都压到 10% 内：`9.13% / 7.48% / 8.33%`。
- 压力 direct one-step 在 Re=120 和 Re=300 达到 10% 内：`2.61% / 5.88%`；低 Re=56.3745 仍为 `29.45%`，低 Re pressure-focused ablation 最好也只有 `28.65%`。
- 因此 V12 没有实现“速度和压力全 Re 都低于 10%”，但相比 V11 closure 版明显改善，并证明瓶颈不是简单 expert collapse，而是低 Re pressure 的跨 Re 泛化/目标建模问题。

## 2. 借鉴的大模型 MoE 设计

我检索并参考了几类成熟 MoE 设计：

- GShard 使用 top-2 gating 和 auxiliary load-balance loss 来做稀疏专家选择：[GShard, arXiv 2006.16668](https://arxiv.org/abs/2006.16668)。
- Switch Transformer 将路由简化为 top-1 expert，并强调路由负载均衡对稳定训练的重要性：[Switch Transformer, JMLR](https://jmlr.org/papers/v23/21-0998.html)。
- Mixtral 每层选择 top-2 experts 并加权组合输出，是稀疏 MoE 在大模型里的典型实现：[Mixtral, arXiv 2401.04088](https://arxiv.org/abs/2401.04088)。
- DeepSeekMoE 明确提出 shared expert isolation，让 always-on shared experts 承担通用知识，routed experts 承担更细粒度的专门知识：[DeepSeekMoE, arXiv 2401.06066](https://arxiv.org/abs/2401.06066)。

V12 采用了这些思想的 ROM 版本：

- routed experts：按当前物理状态/Re/phase/模态能量选择局部动力学闭合；
- shared experts：始终激活，承担跨 Re 的通用 closure；
- top-k routing：主实验用 `top_k=3`；
- load-balance/entropy/diversity loss：避免所有状态只走单个 expert；
- shared/routed 分离：shared expert 不进入稀疏竞争，和 DeepSeekMoE 的 shared expert isolation 语义一致。

## 3. V12 模型结构

输入特征仍保持 V11 operator-space 接口：

- velocity POD 系数 `a_t`；
- pressure POD 系数 `b_t`；
- `Re`、`1/Re`、phase harmonics；
- Galerkin RHS descriptor；
- 低/中/高模态范数、速度/压力能量、总能量、能量比例、pressure/velocity 范数比；
- 3-step history 与状态差分。

主结构：

```text
[a_t, b_t, Re, phase, physical descriptors, history]
  -> shared encoder h_t
  -> routed regime router pi_i(h_t, x_t)
  -> routed local operator experts c_i(h_t)
  -> always-on shared operator experts s_j(h_t)
  -> learned closure c = mix(shared) + sum_i pi_i c_i
  -> f = Galerkin RHS + c
  -> RK4 velocity advance
  -> pressure branch shares the same regime router
```

两个 pressure target：

- `closure`: 预测 pressure surrogate residual，`b_next = b_base(a_next) + closure`；
- `state`: 直接预测 `b_next`，用于检查低 Re pressure 是否受 surrogate residual target 限制。

主实验采用 `closure`，因为它在 Re=120/300 pressure 和速度稳定性上最好。

## 4. 实验设置

| item | value |
|---|---:|
| `r_u`, `r_p` | 16, 16 |
| velocity POD energy | 0.943109 |
| pressure POD energy | 0.944973 |
| held-out Re indices | 10, 59, 99 |
| held-out Re values | 56.3745, 120.0, 300.0 |
| main routed experts | 12 |
| main shared experts | 2 |
| main top-k | 3 |
| integrator | RK4 |
| RHS target | residual closure added to Galerkin RHS |

运行脚本：

- `run_v12_shared_operator_smoke.sh`: smoke test；
- `run_v12_shared_operator.sh`: closure-pressure 主实验；
- `run_v12_shared_operator_state.sh`: state-pressure 对照；
- `run_v12_shared_operator_lowre_pressure.sh`: low-Re pressure-focused ablation。

## 5. 主结果

### 5.1 Closure-Pressure 主模型

Experiment: `v12_r16_shared_operator_closure_b3`

| Test Re | Best epoch | RHS L2 | Pressure L2 | Auto a one-step | Auto b one-step | Auto a rollout | Auto b rollout | Load CV | Dead experts | Max expert cos | Shared mixer mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 56.3745 | 135 | 0.117531 | 0.294464 | 0.091258 | 0.307380 | 0.439722 | 0.713620 | 1.938936 | 9 | 0.848474 | `[0.606, 0.394]` |
| 120.0 | 100 | 0.115111 | 0.026072 | 0.074845 | 0.122918 | 0.371371 | 0.477895 | 1.468916 | 7 | 0.392102 | `[0.377, 0.623]` |
| 300.0 | 105 | 0.064694 | 0.058756 | 0.083334 | 0.114990 | 0.529503 | 0.751965 | 0.794420 | 4 | 0.473445 | `[0.689, 0.311]` |

解释：

- 速度 autonomous one-step 已经全部低于 10%。
- pressure direct one-step 在 Re=120/300 低于 10%，低 Re 仍高。
- shared expert mixer 的权重随 Re 改变，说明 shared experts 不是静态常数项，而是在不同 regime 中承担不同通用 closure。
- max expert cosine 都低于 0.95，没有出现函数输出完全塌缩。

### 5.2 State-Pressure 对照

Experiment: `v12_r16_shared_operator_state_b3`

| Test Re | Best epoch | RHS L2 | Pressure L2 | Auto a one-step | Auto b one-step | Auto a rollout | Auto b rollout | Max expert cos | Shared mixer mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 56.3745 | 135 | 0.110391 | 0.345813 | 0.092384 | 0.345813 | 0.526315 | 0.622502 | 0.516628 | `[0.201, 0.799]` |
| 120.0 | 160 | 0.113775 | 0.037474 | 0.070443 | 0.037474 | 0.364321 | 0.310578 | 0.285770 | `[0.597, 0.403]` |
| 300.0 | 135 | 0.065358 | 0.106880 | 0.084827 | 0.106880 | 0.523343 | 0.528297 | 0.505724 | `[0.772, 0.228]` |

State target 改善了 pressure rollout，尤其 Re=120/300，但低 Re direct pressure 更差，Re=300 direct pressure 略高于 10%。所以它不是 V12 的 best main。

### 5.3 Low-Re Pressure Ablation

Experiment: `v12_r16_shared_operator_lowre_pressure`

| Test Re | Best epoch | Routed experts | Top-k | Pressure L2 | Auto a one-step | Auto b one-step | Auto b rollout | Dead experts | Max expert cos | Shared mixer mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 56.3745 | 120 | 6 | 2 | 0.286531 | 0.096173 | 0.307844 | 0.580900 | 0 | 0.694107 | `[0.917, 0.083]` |

这个 ablation 只把低 Re pressure 从 `0.294464` 降到 `0.286531`，仍远高于 10%。它说明仅靠调 routed expert 数量和加大 pressure loss，无法突破低 Re pressure floor。

## 6. 与 V10/V11 对比

### 6.1 V12 vs V11 closure

| Test Re | V11 closure Pressure | V12 closure Pressure | V11 Auto a one-step | V12 Auto a one-step | V11 Auto b rollout | V12 Auto b rollout |
|---:|---:|---:|---:|---:|---:|---:|
| 56.3745 | 0.362618 | 0.294464 | 0.091935 | 0.091258 | 1.075368 | 0.713620 |
| 120.0 | 0.063967 | 0.026072 | 0.114243 | 0.074845 | 0.925962 | 0.477895 |
| 300.0 | 0.064081 | 0.058756 | 0.110078 | 0.083334 | 0.839126 | 0.751965 |

V12 的 shared-expert physical-anchor 设计明显优于 V11 closure：压力、速度 one-step 和 pressure rollout 基本都改善。

### 6.2 V12 closure vs V10 main

| Test Re | V10 Pressure | V12 Pressure | V10 Auto a one-step | V12 Auto a one-step | V10 Auto b one-step | V12 Auto b one-step |
|---:|---:|---:|---:|---:|---:|---:|
| 56.3745 | 0.283692 | 0.294464 | 0.066361 | 0.091258 | 0.289178 | 0.307380 |
| 120.0 | 0.020780 | 0.026072 | 0.054604 | 0.074845 | 0.063671 | 0.122918 |
| 300.0 | 0.064957 | 0.058756 | 0.079498 | 0.083334 | 0.087281 | 0.114990 |

V12 在架构语义上更符合 operator-space MoE，但一阶精度还没有全面超过 V10 residual-correction。高 Re pressure 略优于 V10；低/中 Re 和 autonomous pressure 仍落后。

## 7. 对“共享专家”的回答

V12 有共享专家，而且有两层共享：

- 物理共享 operator：Galerkin RHS，是所有 Re/regime 都使用的底座；
- learned shared experts：2 个 always-on shared operator experts，输出通用 closure，与 routed experts 加权融合。

V12 的 shared expert 不是“所有样本固定同一个输出”。它们读同一个 latent `h_t`，再由 shared mixer 输出权重。主实验中 shared mixer weights 随 Re 有明显变化：

- Re=56.3745: `[0.606, 0.394]`
- Re=120.0: `[0.377, 0.623]`
- Re=300.0: `[0.689, 0.311]`

这说明 shared experts 承担的是跨 regime 的通用动力学闭合，而 routed experts 更像 regime-local correction。

## 8. 为什么压力还没有到 10%

低 Re pressure 的问题不像 router collapse：

- low-Re pressure-focused ablation 中 routed expert cosine max 只有 `0.694`，没有 expert 函数塌缩；
- shared expert 明确参与，且 low Re 使用第一个 shared expert 权重大约 `0.917`；
- 训练/验证 pressure 在训练 Re 上已经很低，但 held-out Re=56.3745 仍约 28%-29%，说明主要是跨 Re 泛化和 pressure target 本身的问题。

我认为瓶颈在这几个地方：

- 低 Re pressure coefficient 范数较小，relative error 对相位/幅值偏差很敏感；
- pressure surrogate residual 在低 Re 附近变化可能比 velocity RHS 更尖锐，普通 MLP expert 不足以表达；
- velocity 和 pressure 强制共享同一个 router，有利于耦合，但 pressure 的 regime 边界可能并不完全等同于 velocity；
- 当前 pressure branch 只输出系数，不显式利用 Poisson/algebraic pressure operator 的结构。

## 9. 后续 V13 建议

我建议下一版不要只继续调 loss，而是改 pressure operator 结构：

1. 保留 V12 的 shared-expert operator backbone  
   继续使用 `Galerkin RHS + learned closure + RK4`，因为它让速度 one-step 已经进 10%。

2. 增加 pressure-specific structured operator experts  
   让 pressure expert 读 `a_t, b_t, a_{t+1}, Re, phase`，输出结构化 pressure residual，例如线性项 + 二次项 + 小 MLP closure，而不是单纯 MLP 系数头。

3. 允许 velocity/pressure 共享 regime encoder，但使用两个 router head  
   共享 latent 和部分专家，pressure router 加弱一致性 loss，而不是完全绑死同一个 router。这能保留速度压力耦合，同时允许 pressure 有自己的 regime 切分。

4. 增加 Re-band local pressure surrogate  
   对低 Re / 中 Re / 高 Re 建局部 surrogate 或 spline/interpolation anchor，再由 MoE 学 residual。低 Re 只靠全局 MLP 泛化不够稳。

5. 加入 router logit z-loss / temperature annealing  
   参考 ST-MoE/Switch 系列的路由稳定化思路，控制 logits 爆炸，同时避免 top-k 在单个 Re 上过早僵化。

6. 做多 rank pressure correction  
   保持 velocity `r_u=16`，pressure 可以尝试 `r_p=24` 的只压力 correction head，避免整体高 rank ROM 变差。

下一步最可能把 pressure 推到 10% 内的不是“再加专家”，而是 pressure-specific structured operator + local Re-band pressure anchor。

## 10. 文件与产物

- V12 code: `test_results_v12/deep_moe_rom_v12.py`
- V12 scripts:
  - `test_results_v12/run_v12_shared_operator_smoke.sh`
  - `test_results_v12/run_v12_shared_operator.sh`
  - `test_results_v12/run_v12_shared_operator_state.sh`
  - `test_results_v12/run_v12_shared_operator_lowre_pressure.sh`
- Result JSON/summary:
  - `test_results_v12/results/v12_r16_shared_operator_closure_b3/`
  - `test_results_v12/results/v12_r16_shared_operator_state_b3/`
  - `test_results_v12/results/v12_r16_shared_operator_lowre_pressure/`

总体评价：V12 是比 V11 更合理的 shared-expert operator-space MoE baseline；速度目标基本达成，但压力 10% 目标还需要更结构化的 pressure operator。
