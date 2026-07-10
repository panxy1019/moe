# V16_2 AttractorStabilityMoE 技术报告

## 1. 实验目标

V16_2 基于 `V16_1_SteadyPressureAnchor32` 创建独立版本 `V16_2_AttractorStabilityMoE`，不覆盖 V16_1 的代码、checkpoint、metrics 和报告。统一基线保持：

- Re=20-200 Physics-Generalizable Attractor Database
- `ru=32`, `rp=32`
- HPRS-MoE, Shared Encoder, Galerkin, RK4
- Pressure Poisson Surrogate + modal AdaptiveGate
- `pressure_target=closure`, `pressure_input_mode=pressure_only`
- rollout curriculum `4 -> 8 -> 12 -> 16`
- 主评估为 24-step autonomous rollout
- held-out Re 与 V16_1 一致

本轮只做三组互相独立的实验，不做 Combined：

| Case | 唯一新增方向 | 核心目标 |
|---|---|---|
| `V16_2_SteadyContractivePressureROM32` | steady equilibrium / contraction / pressure damping loss | 降低 steady pressure drift |
| `V16_2_HopfLogRadiusNormalForm32` | Hopf log-radius / overshoot / phase / normal-form auxiliary loss | 抑制 Hopf near-onset overshoot |
| `V16_2_RegimeGroupedMoE32` | 4 组 regime-aware grouped router + expert balance | 减少 top1 collapse 与 dead experts |

## 2. 产物路径

集群结果目录：

```text
/root/moe/V16_2_AttractorStabilityMoE/test_results_v16_2/results
```

GitHub compact 结果目录：

```text
V16_2_AttractorStabilityMoE/test_results_v16_2/results
```

Raw metrics 每个约 86 MB，仅保留在集群，不进入 Git：

```text
results/V16_2_SteadyContractivePressureROM32/..._metrics.json
results/V16_2_HopfLogRadiusNormalForm32/..._metrics.json
results/V16_2_RegimeGroupedMoE32/..._metrics.json
```

已提交 compact 文件：

- `results/aggregate/v16_2_summary_metrics.json`
- `results/aggregate/v16_2_per_re_metrics.csv`
- `results/aggregate/v16_2_hopf_near_onset_diagnostics.csv`
- `results/aggregate/v16_2_steady_pressure_drift.csv`
- `results/aggregate/v16_2_periodic_degradation.csv`
- `results/aggregate/v16_2_router_diagnostics.csv`
- 每个 case 的 `summary.md`, `error_vs_re.csv`, `error_vs_re.svg`, `run.log`

SwanLab：

- SteadyContractive: https://swanlab.cn/@panxy1019/V16_2_AttractorStabilityMoE/runs/ug6w98xl
- HopfLogRadiusNormalForm: https://swanlab.cn/@panxy1019/V16_2_AttractorStabilityMoE/runs/b4l00vst
- RegimeGroupedMoE: https://swanlab.cn/@panxy1019/V16_2_AttractorStabilityMoE/runs/chlhe9cf

## 3. Overall 结果

所有误差均为 relative L2 或相对能量误差。

| Case | 1-step u | 1-step p | 24-step u | 24-step p | RHS | p energy drift | alpha | active experts | dead experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `V16_1_SteadyPressureAnchor32` | 0.0399 | 0.1229 | 0.2257 | 0.2967 | 0.3105 | 0.3143 | 0.2421 | 3.51 | 15.64 |
| `V16_2_SteadyContractivePressureROM32` | 0.0532 | 0.1205 | 0.2625 | 0.4512 | 0.3597 | 0.6830 | 0.2456 | 4.17 | 14.09 |
| `V16_2_HopfLogRadiusNormalForm32` | 0.3443 | 0.8527 | 1.0894 | 2.1912 | 2.1000 | 7.3258 | 0.4575 | 4.79 | 14.18 |
| `V16_2_RegimeGroupedMoE32` | 0.0405 | 0.2574 | 0.2847 | 0.4764 | 0.3299 | 0.6105 | 0.2203 | 12.63 | 7.27 |

结论：三组都没有在 overall 上超过 `V16_1_SteadyPressureAnchor32`。其中 `RegimeGroupedMoE32` 在 expert usage 和 periodic 区域最有价值，但压力分支显著退化；`HopfLogRadiusNormalForm32` 明显失稳，不建议继续沿用当前权重。

## 4. Steady Pressure Drift

Steady 组平均：

| Case | 24-step u | 24-step p | pressure energy drift | active experts |
|---|---:|---:|---:|---:|
| `V16_1` | 0.1998 | 0.4071 | 0.5686 | 3.00 |
| `SteadyContractive` | 0.1937 | 0.5409 | 0.2477 | 3.92 |
| `HopfLogNF` | 0.6543 | 2.1937 | 5.9844 | 5.00 |
| `RegimeGrouped` | 0.1752 | 0.7132 | 0.7101 | 13.00 |

重点 Re：

| Re | Baseline p-rollout | SteadyContractive p-rollout | Baseline p-energy | SteadyContractive p-energy |
|---:|---:|---:|---:|---:|
| 24.630 | 0.6274 | 0.6748 | 1.2705 | 0.3490 |
| 32.740 | 0.4680 | 0.7044 | 0.6665 | 0.4497 |
| 39.685 | 0.3034 | 0.4512 | 0.3068 | 0.1919 |
| 45.143 | 0.2295 | 0.3330 | 0.0306 | 0.0003 |

判断：

- SteadyContractive 的 fixed-point/pressure contraction 确实把 steady pressure energy drift 明显压低，平均从 `0.5686` 降到 `0.2477`。
- 但核心目标 steady 24-step pressure relative L2 没有下降，反而从 `0.4071` 升到 `0.5409`。
- 说明当前 contraction/energy anchor 更像是在约束压力模态能量，而不是约束完整压力向量方向。能量对了，压力 POD 相位/模态方向仍漂。
- periodic 区域 24-step velocity 只恶化 `+12.0%`，满足不超过 15% 的要求；但 periodic pressure 从 `0.0618` 到 `0.0735`，恶化 `+18.9%`，略超阈值。

结论：`SteadyContractivePressureROM32` 不达成主成功标准，但提供了一个有用信号：steady energy drift 可通过 contraction 类 loss 被压住，下一步应改成 pressure vector anchor 或 mean-projection，而不是继续只压能量。

## 5. Hopf Near-Onset Overshoot

Hopf 组平均：

| Case | 24-step u | 24-step p | pressure energy drift |
|---|---:|---:|---:|
| `V16_1` | 0.4953 | 0.4627 | 0.3747 |
| `SteadyContractive` | 0.6307 | 0.8354 | 2.1234 |
| `HopfLogNF` | 2.6189 | 4.3055 | 18.2691 |
| `RegimeGrouped` | 0.7507 | 0.7157 | 1.2620 |

关键 overshoot：

| Re | Case | overshoot mean | overshoot >2 | amplitude rel L2 | 24-step u | 24-step p |
|---:|---|---:|---:|---:|---:|---:|
| 47.081 | V16_1 | 9.91 | 99.17% | 9.14 | 0.5925 | 0.3550 |
| 47.081 | SteadyContractive | 7.29 | 94.17% | 7.00 | 0.3964 | 0.1556 |
| 47.081 | HopfLogNF | 409.94 | 100.00% | 416.00 | 1.6750 | 2.7509 |
| 47.081 | RegimeGrouped | 6.57 | 95.83% | 5.88 | 0.3925 | 0.2731 |
| 49.022 | V16_1 | 4.24 | 90.00% | 3.64 | 0.2872 | 0.3729 |
| 49.022 | RegimeGrouped | 2.88 | 68.33% | 2.29 | 0.2147 | 0.3023 |
| 51.786 | V16_1 | 1.92 | 45.00% | 1.00 | 0.6063 | 0.6603 |
| 51.786 | RegimeGrouped | 1.49 | 9.17% | 0.57 | 1.6451 | 1.5718 |

判断：

- `HopfLogRadiusNormalForm32` 完全失败。它没有把 overshoot 压到 2-3x，反而在 Re=47.081 和 49.022 出现 100x 到 400x 量级的过冲。
- 失败原因大概率是 auxiliary normal-form head 的梯度与主 RK4/MoE operator 梯度冲突，且 `lambda_logr=0.25`, `lambda_over=0.15`, `lambda_nf=0.10` 对 near-zero radius 样本过强，使模型用错误的尺度解释 Hopf pair。
- 反而 `RegimeGroupedMoE32` 在 Re=49.022 和 51.786 上自然降低 overshoot，说明 Hopf 误差有一部分来自专家/路由分流，而不只是 loss 形式。
- `SteadyContractive` 也在 Re=47.081 改善 velocity/pressure rollout，但在 Re=51.786 明显恶化，说明 steady contraction 对临界附近存在非局部影响。

结论：当前 Hopf normal-form 版本不建议作为 V16 后续路线。若继续做 Hopf loss，应先去掉 normal-form head，只保留很小权重的 floor-normalized radius damping，并对 Re=47-49 单独做 curriculum。

## 6. RegimeGroupedMoE 分析

`RegimeGroupedMoE32` 的核心收益是专家使用显著改善：

| Metric | V16_1 | RegimeGrouped |
|---|---:|---:|
| overall active experts | 3.51 | 12.63 |
| overall dead experts | 15.64 | 7.27 |
| periodic active experts | 4.41 | 12.00 |
| periodic dead experts | 11.50 | 4.75 |
| group entropy | 0.00 | 0.58 |

Periodic 结果：

| Case | periodic 24-step u | periodic 24-step p |
|---|---:|---:|
| V16_1 | 0.0493 | 0.0618 |
| RegimeGrouped | 0.0446 | 0.0603 |

判断：

- RegimeGrouped 是本轮唯一在 periodic velocity 和 pressure rollout 上同时不退化、甚至略改善的 V16_2 方案。
- 它确实缓解了 active expert count 误导问题，dead experts 从 `15.64` 降到 `7.27`。
- 但 steady/Hopf pressure 明显退化：steady pressure rollout `0.4071 -> 0.7132`，Hopf pressure rollout `0.4627 -> 0.7157`。
- 当前版本把 V16_1 的每组 shared expert 改成 4 组纯 routed experts，这改善了覆盖率，但削弱了原来 group-shared expert 对 pressure residual 的稳定支撑。

结论：RegimeGroupedMoE 是最值得保留的结构方向，但不能直接替代 V16_1 baseline。下一版应采用 hybrid grouped design：保留每个 regime group 的 shared expert，同时增加 explicit grouped prior，而不是把 shared expert 完全移除。

## 7. 成功标准逐项判断

### SteadyContractivePressureROM32

- 目标：降低 Re=24.630/32.740/39.685/45.143 的 steady 24-step pressure drift。
- 结果：pressure energy drift 降低，但 pressure relative L2 升高。
- 判定：未达成主目标，部分达成能量稳定目标。

### HopfLogRadiusNormalForm32

- 目标：Re=47.081/49.022 overshoot mean 从 10x-20x 压到 2x-3x。
- 结果：Re=47.081 overshoot 到 409.94x，Re=49.022 到 158.12x。
- 判定：失败，应暂停该方案。

### RegimeGroupedMoE32

- 目标：减少 top1 collapse/dead experts，保持 rollout 不明显劣于 baseline。
- 结果：dead experts 明显下降，periodic rollout 改善；steady/Hopf pressure 退化。
- 判定：专家可解释性目标达成，精度目标未达成。

## 8. 推荐路线

1. V16_3 不建议继续 Hopf normal-form head。先移除 `mu/omega/beta` 辅助头，只做低权重、warm-up 更慢的 `log(r+r_floor)` 或 floor-normalized radius damping。
2. Steady pressure 方向保留 energy/contraction 思想，但改成直接约束 `b_pred -> b_mean(Re)` 的向量投影和 residual contraction。当前能量对齐不等价于压力模态向量对齐。
3. RegimeGroupedMoE 值得继续，但应恢复 shared expert。建议结构为 `4 groups x (1 shared + routed experts)`，并降低 group prior 权重，避免 pressure residual 失去共享稳定项。
4. 后续默认 baseline 仍应是 `V16_1_SteadyPressureAnchor32`，不是任一 V16_2 case。
5. 若只选一个 V16_2 继续迭代，选择 `RegimeGroupedMoE32` 的路由思想，而不是它当前的完整实现。

## 9. 最终结论

V16_2 证明了三个事实：

- steady pressure drift 的 energy 分量可以被 contraction loss 抑制，但 pressure relative L2 需要更强的模态方向约束；
- Hopf near-onset 不能简单叠加强 normal-form auxiliary loss，否则会破坏共享 operator 表征；
- expert collapse 确实可以通过 grouped routing 缓解，且 periodic 区域可受益，但 pressure branch 需要 shared expert 或更温和的 grouped prior 来保持稳定。

因此，V16_2 不产生新的默认精度 baseline，但给 V16_3 提供了清晰方向：以 `V16_1_SteadyPressureAnchor32` 为精度基线，吸收 `RegimeGroupedMoE32` 的专家分组诊断和部分 router prior，同时重构 steady pressure vector anchor，暂时放弃当前 Hopf normal-form head。
