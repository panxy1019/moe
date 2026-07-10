# V16_1 TrainStableAttractorMoE 技术报告

**版本**: `V16_1_TrainStableAttractorMoE`

**基线动机**: V16 FullRegimeLoss32 的 one-step 表现已经较好，但训练 Re autonomous rollout 仍暴露两个稳定性问题：Hopf near-onset 的 false oscillation/false growth，以及 low-Re steady pressure residual closed-loop drift。因此 V16_1 只修改 closed-loop attractor stability losses，不改 HPRS-MoE、Shared Encoder、Router、Expert、Galerkin、RK4、Pressure Poisson Surrogate、modal AdaptiveGate、数据集、POD/ROM 维数和优化超参数。

## 结论先行

- **推荐下一版默认 baseline: `V16_1_SteadyPressureAnchor32`**。它取得最低 overall one-step velocity `0.0399`、one-step pressure `0.1229`、24-step velocity `0.2257`，pressure rollout `0.2967` 仅略高于 HopfOnset 的 `0.2877`。
- `V16_1_HopfOnsetGrowthLoss32` 对 pressure rollout 均值略有优势，但 Hopf near-onset false oscillation 没有被根治：Re=47.081 的 rollout overshoot mean 仍约 `22.7x`，Re=49.022 约 `8.36x`。
- `V16_1_TrainStableCombined32` 不应作为默认方案：组合 loss 产生明显负迁移，overall 24-step pressure relative L2 升至 `1.147`，Hopf 组 pressure rollout 均值升至 `2.804`，Re=51.786 pressure rollout 达 `6.375`。
- Pressure Base 本身仍不是低 Re 主要可用信号：held-out pressure base relative L2 overall 均值为 `83.0`，steady alpha 仍只有约 `0.063-0.088`，模型主要依赖 residual/head。Steady pressure anchor 能降低 drift，但没有解决 Poisson base 跨 regime 精度问题。
- Router/Expert 使用仍有退化迹象：Hopf 测试点 top1 常集中到 expert 0，而 active experts 仍可显示 3-5 个，说明后续报告必须同时看 top1、mean load 和 top-k set，不能只看 active expert count。

## 实验配置

| 实验 | 唯一变化 | 目的 |
|---|---|---|
| `V16_1_HopfOnsetGrowthLoss32` | Hopf growth consistency + false-growth penalty + floor-normalized rollout；Hopf radius 降到 0.01 | 抑制 Re≈47-52 near-onset false oscillation/growth |
| `V16_1_SteadyPressureAnchor32` | Steady pressure state/mean/delta/residual damping/energy anchor | 抑制 low-Re steady pressure residual drift |
| `V16_1_TrainStableCombined32` | 上述两类 loss 同时启用，新增 loss 30 epoch warm-up | 验证是否能作为统一稳定 baseline |

固定项：Re=20-200 Physics-Generalizable Attractor Database，ru=32/rp=32，HPRS-MoE + Galerkin + RK4 + Pressure Poisson Surrogate + modal AdaptiveGate，rollout curriculum 4->8->12->16，主评估为 24-step autonomous rollout。

## SwanLab 与产物路径

| Case | SwanLab | 远端原始 metrics | 远端 checkpoint |
|---|---|---|---|
| `HopfOnsetGrowth` | https://swanlab.cn/@panxy1019/V16_1_TrainStableAttractorMoE/runs/crdm77ak | `/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1/results/V16_1_HopfOnsetGrowthLoss32/V16_1_HopfOnsetGrowthLoss32_ru32_rp32/V16_1_HopfOnsetGrowthLoss32_ru32_rp32_metrics.json` | `/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1/results/V16_1_HopfOnsetGrowthLoss32/V16_1_HopfOnsetGrowthLoss32_ru32_rp32/V16_1_HopfOnsetGrowthLoss32_ru32_rp32_Re_24p630436_checkpoint.pt` |
| `SteadyPressureAnchor` | https://swanlab.cn/@panxy1019/V16_1_TrainStableAttractorMoE/runs/s94sbgtx | `/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1/results/V16_1_SteadyPressureAnchor32/V16_1_SteadyPressureAnchor32_ru32_rp32/V16_1_SteadyPressureAnchor32_ru32_rp32_metrics.json` | `/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1/results/V16_1_SteadyPressureAnchor32/V16_1_SteadyPressureAnchor32_ru32_rp32/V16_1_SteadyPressureAnchor32_ru32_rp32_Re_24p630436_checkpoint.pt` |
| `TrainStableCombined` | https://swanlab.cn/@panxy1019/V16_1_TrainStableAttractorMoE/runs/sqb7a5zy | `/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1/results/V16_1_TrainStableCombined32/V16_1_TrainStableCombined32_ru32_rp32/V16_1_TrainStableCombined32_ru32_rp32_metrics.json` | `/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1/results/V16_1_TrainStableCombined32/V16_1_TrainStableCombined32_ru32_rp32/V16_1_TrainStableCombined32_ru32_rp32_Re_24p630436_checkpoint.pt` |

> 说明：HopfOnset 的 online SwanLab run 在 epoch 240 后卡在最后一次上传，未进入终评写盘。已用新增的 `--eval-only-checkpoint` 从 best checkpoint epoch 220 离线补跑 evaluator，未重训，生成完整 metrics。

### Overall Mean Metrics

| Case | 1-step u | 1-step p | 24-step u | 24-step p | RHS | p energy drift | alpha | active experts | dead experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HopfOnsetGrowth` | 0.0493 | 0.1444 | 0.2420 | 0.2877 | 0.3175 | 0.3444 | 0.2504 | 4.772 | 15.5 |
| `SteadyPressureAnchor` | 0.0399 | 0.1229 | 0.2257 | 0.2967 | 0.3105 | 0.3143 | 0.2421 | 3.512 | 15.6 |
| `TrainStableCombined` | 0.0777 | 0.2537 | 0.3389 | 1.147 | 0.5015 | 3.025 | 0.2396 | 4.377 | 13.1 |

### Steady Mean Metrics

| Case | 1-step u | 1-step p | 24-step u | 24-step p | RHS | p energy drift | alpha | active experts | dead experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HopfOnsetGrowth` | 0.0153 | 0.1558 | 0.1979 | 0.3831 | 0.2727 | 0.3522 | 0.0875 | 5.000 | 18 |
| `SteadyPressureAnchor` | 0.0155 | 0.1516 | 0.1998 | 0.4071 | 0.2797 | 0.5686 | 0.0628 | 3.000 | 18 |
| `TrainStableCombined` | 0.0178 | 0.2463 | 0.2217 | 0.9260 | 0.3200 | 0.9744 | 0.0658 | 4.373 | 15 |

### Hopf Mean Metrics

| Case | 1-step u | 1-step p | 24-step u | 24-step p | RHS | p energy drift | alpha | active experts | dead experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HopfOnsetGrowth` | 0.1009 | 0.2541 | 0.5415 | 0.4430 | 0.5182 | 0.7601 | 0.0865 | 5.000 | 18 |
| `SteadyPressureAnchor` | 0.0766 | 0.1825 | 0.4953 | 0.4627 | 0.4856 | 0.3747 | 0.0619 | 3.000 | 18 |
| `TrainStableCombined` | 0.2048 | 0.5318 | 0.8293 | 2.804 | 1.129 | 9.666 | 0.0648 | 4.118 | 15 |

### Periodic Mean Metrics

| Case | 1-step u | 1-step p | 24-step u | 24-step p | RHS | p energy drift | alpha | active experts | dead experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HopfOnsetGrowth` | 0.0445 | 0.0507 | 0.0615 | 0.0757 | 0.2116 | 0.0248 | 0.5363 | 4.372 | 11.2 |
| `SteadyPressureAnchor` | 0.0369 | 0.0495 | 0.0493 | 0.0618 | 0.2101 | 0.0149 | 0.5566 | 4.407 | 11.5 |
| `TrainStableCombined` | 0.0421 | 0.0524 | 0.0881 | 0.1259 | 0.2127 | 0.0939 | 0.5447 | 4.574 | 9.750 |

## Hopf Near-Onset 重点诊断

| Case | Re | r_true mean | r_pred mean | amp rel L2 | overshoot mean | overshoot >2 | phase err | freq err | 24-step u | 24-step p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HopfOnsetGrowth` | 47.081 | 1.356e-06 | 3.001e-05 | 21.5 | 22.7 | 100.00% | 1.457 | 0.7217 | 0.4523 | 0.1636 |
| `HopfOnsetGrowth` | 49.022 | 3.132e-06 | 2.605e-05 | 7.755 | 8.356 | 100.00% | 1.340 | 0.7381 | 0.1883 | 0.1830 |
| `HopfOnsetGrowth` | 51.786 | 2.324e-05 | 4.878e-05 | 1.306 | 2.115 | 51.67% | 0.5049 | 0.3674 | 0.9839 | 0.9824 |
| `SteadyPressureAnchor` | 47.081 | 1.356e-06 | 1.297e-05 | 9.140 | 9.912 | 99.17% | 1.376 | 0.8180 | 0.5925 | 0.3550 |
| `SteadyPressureAnchor` | 49.022 | 3.132e-06 | 1.320e-05 | 3.635 | 4.236 | 90.00% | 0.6902 | 0.3852 | 0.2872 | 0.3729 |
| `SteadyPressureAnchor` | 51.786 | 2.324e-05 | 4.470e-05 | 1.004 | 1.919 | 45.00% | 0.2603 | 0.1519 | 0.6063 | 0.6603 |
| `TrainStableCombined` | 47.081 | 1.356e-06 | 3.565e-05 | 28.4 | 27.1 | 100.00% | 1.593 | 0.8777 | 0.9185 | 1.152 |
| `TrainStableCombined` | 49.022 | 3.132e-06 | 2.905e-05 | 9.207 | 9.392 | 97.50% | 1.374 | 0.8120 | 0.5587 | 0.8847 |
| `TrainStableCombined` | 51.786 | 2.324e-05 | 5.114e-05 | 1.379 | 2.285 | 58.33% | 0.5807 | 0.4035 | 1.011 | 6.375 |

解读：HopfOnset loss 确实没有让 Re=51.786 进一步恶化，且 pressure rollout 比 Combined 稳定很多；但 Re=47.081/49.022 的真实半径极小，预测半径仍保持 1e-5 量级，overshoot >2 的比例为 100%。这说明当前 loss 仍在“减轻错误”，还没有学到 near-onset weak attractor 的吸引域尺度。

## Steady Pressure 重点诊断

| Case | Re | Regime | 1-step u | 1-step p | 24-step u | 24-step p | p energy | alpha | experts top1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `HopfOnsetGrowth` | 24.630 | steady_wake | 0.0223 | 0.3448 | 0.3123 | 0.5128 | 0.7932 | 0.0884 | e0:1.000 |
| `HopfOnsetGrowth` | 32.740 | steady_wake | 0.0149 | 0.1385 | 0.2038 | 0.4640 | 0.3340 | 0.0873 | e0:1.000 |
| `HopfOnsetGrowth` | 39.685 | steady_wake | 0.0128 | 0.0635 | 0.1551 | 0.3236 | 0.2279 | 0.0870 | e0:1.000 |
| `HopfOnsetGrowth` | 45.143 | pre_hopf_steady | 0.0113 | 0.0762 | 0.1205 | 0.2318 | 0.0538 | 0.0874 | e0:1.000 |
| `SteadyPressureAnchor` | 24.630 | steady_wake | 0.0232 | 0.2874 | 0.3061 | 0.6274 | 1.270 | 0.0633 | e0:1.000 |
| `SteadyPressureAnchor` | 32.740 | steady_wake | 0.0151 | 0.1295 | 0.2068 | 0.4680 | 0.6665 | 0.0627 | e0:1.000 |
| `SteadyPressureAnchor` | 39.685 | steady_wake | 0.0126 | 0.0948 | 0.1609 | 0.3034 | 0.3068 | 0.0625 | e0:1.000 |
| `SteadyPressureAnchor` | 45.143 | pre_hopf_steady | 0.0110 | 0.0947 | 0.1256 | 0.2295 | 0.0306 | 0.0626 | e0:1.000 |
| `TrainStableCombined` | 24.630 | steady_wake | 0.0255 | 0.3942 | 0.3313 | 1.360 | 2.831 | 0.0667 | e14:0.754; e0:0.246 |
| `TrainStableCombined` | 32.740 | steady_wake | 0.0179 | 0.2251 | 0.2323 | 0.9634 | 0.3263 | 0.0657 | e14:0.574; e0:0.426 |
| `TrainStableCombined` | 39.685 | steady_wake | 0.0149 | 0.1800 | 0.1810 | 0.7424 | 0.2854 | 0.0654 | e14:0.590; e0:0.410 |
| `TrainStableCombined` | 45.143 | pre_hopf_steady | 0.0130 | 0.1861 | 0.1423 | 0.6380 | 0.4553 | 0.0653 | e14:0.590; e0:0.410 |

解读：SteadyPressureAnchor 在 low-Re steady 的 one-step pressure 与 overall velocity 上最稳，但 24-step pressure drift 仍偏高，尤其 Re=24.63 的 rollout pressure 为 0.627。Combined 中 steady pressure anchor 与 Hopf growth loss 同时作用后，steady pressure 反而大幅变差，说明两个新增闭环约束在共享 latent/router/expert 上存在梯度冲突。

## Periodic 退化检查

| Case | Re | Regime | 1-step u | 1-step p | 24-step u | 24-step p | p energy | alpha | experts top1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `HopfOnsetGrowth` | 70.315 | developing_periodic_shedding | 0.0587 | 0.0697 | 0.1082 | 0.1240 | 0.0429 | 0.5157 | e0:0.835; e7:0.139; e14:0.025 |
| `HopfOnsetGrowth` | 100.352 | mature_periodic_shedding | 0.0444 | 0.0433 | 0.0379 | 0.0452 | 0.0066 | 0.5470 | e7:0.956; e14:0.044 |
| `HopfOnsetGrowth` | 149.059 | mature_periodic_shedding | 0.0328 | 0.0385 | 0.0522 | 0.0652 | 0.0144 | 0.5416 | e7:0.646; e14:0.354 |
| `HopfOnsetGrowth` | 189.862 | high_re_2d_periodic_near_modeA | 0.0422 | 0.0515 | 0.0477 | 0.0685 | 0.0354 | 0.5408 | e14:0.513; e7:0.487 |
| `SteadyPressureAnchor` | 70.315 | developing_periodic_shedding | 0.0611 | 0.0866 | 0.1019 | 0.1272 | 0.0139 | 0.5130 | e0:0.728; e14:0.190; e7:0.082 |
| `SteadyPressureAnchor` | 100.352 | mature_periodic_shedding | 0.0290 | 0.0325 | 0.0283 | 0.0366 | 0.0014 | 0.5698 | e7:0.981; e14:0.019 |
| `SteadyPressureAnchor` | 149.059 | mature_periodic_shedding | 0.0257 | 0.0349 | 0.0255 | 0.0323 | 0.0043 | 0.5707 | e7:0.747; e14:0.222; e0:0.032 |
| `SteadyPressureAnchor` | 189.862 | high_re_2d_periodic_near_modeA | 0.0319 | 0.0441 | 0.0416 | 0.0510 | 0.0398 | 0.5729 | e14:0.677; e7:0.297; e0:0.025 |
| `TrainStableCombined` | 70.315 | developing_periodic_shedding | 0.0740 | 0.0969 | 0.2286 | 0.3233 | 0.3340 | 0.5182 | e0:0.823; e7:0.101; e14:0.076 |
| `TrainStableCombined` | 100.352 | mature_periodic_shedding | 0.0378 | 0.0413 | 0.0535 | 0.0807 | 0.0339 | 0.5464 | e7:0.930; e14:0.070 |
| `TrainStableCombined` | 149.059 | mature_periodic_shedding | 0.0272 | 0.0312 | 0.0381 | 0.0537 | 0.0023 | 0.5562 | e7:0.766; e0:0.127; e14:0.108 |
| `TrainStableCombined` | 189.862 | high_re_2d_periodic_near_modeA | 0.0296 | 0.0403 | 0.0323 | 0.0458 | 0.0055 | 0.5579 | e7:0.468; e0:0.266; e14:0.266 |

Periodic 区域整体没有灾难性退化，三组的 velocity rollout 都远低于 Hopf 区域；但 Combined 的 periodic pressure rollout 和 energy drift 也高于另外两组，说明组合 loss 的负迁移不是只发生在 Hopf。

## 图表索引

- `test_results_v16_1/results/aggregate/one_step_velocity_l2.svg`
- `test_results_v16_1/results/aggregate/one_step_pressure_l2.svg`
- `test_results_v16_1/results/aggregate/rollout_velocity_l2.svg`
- `test_results_v16_1/results/aggregate/rollout_pressure_l2.svg`
- `test_results_v16_1/results/aggregate/rollout_pressure_energy_error.svg`
- `test_results_v16_1/results/aggregate/hopf_overshoot_ratio_mean.svg`

## 详细建议

1. **V16_2 默认从 `SteadyPressureAnchor32` 出发**，不要用 Combined。Combined 的 warm-up 没有解决多目标冲突，反而显著破坏 Hopf pressure rollout。
2. **Hopf near-onset 要改成 attractor-scale normalization，而不是继续加普通 loss**。建议把 Re=47-52 的半径预测改为 floor-aware bounded target，例如预测 `log(r+r_floor)` 或 `r/r_floor`，并对 near-zero true radius 加强 damping/fixed-point-like 约束；当前 HopfOnset 仍允许预测保持 10x-20x 过冲。
3. **Steady pressure drift 应继续 anchor residual，而不是提高 Poisson base 权重**。低 Re alpha 很低，说明模型已经学会不信任 base；真正问题是 residual 在 autonomous rollout 中积累漂移。下一步可尝试 residual spectral damping、pressure residual state-space contraction 或 pressure mean-projection。
4. **Router 需要 top-k 使用约束或 regime-supervised group prior**。Hopf held-out 的 top1 常集中到单专家，dead experts 很多；建议增加 group 内 top2 load-balance、expert dropout 或 entropy floor，并在报告中固定输出 top1/top2/mean-load 三套统计。
5. **Pressure Base 仍应单独升级**。当前 base relative L2 在 held-out 上约 83，Closure 主要靠 residual/head 扛住。后续若要把压力压到更低，建议回到 V15_1 的 pressure base evolution 路线：regime-aware Poisson base 或重新构造 steady/Hopf/periodic base tensor。
6. **评估协议建议补 train Re autonomous diagnostics 到 V16_1**。当前三组报告完整覆盖 held-out Re；但本轮动机来自 train Re false growth/drift，后续应把 Re≈47.7 和 low-Re train Re 的 per-Re train autonomous 诊断纳入自动报告。

## 文件清单

GitHub 提交 compact 结果、报告和脚本；原始 86MB metrics JSON 与 195MB checkpoint 保留在集群远端路径，避免仓库膨胀。

- `V16_1_EXPERIMENT_CONFIG.md`: 实验设计与诊断依据。
- `test_results_v16_1/train_v16_1_train_stable_attractor_moe.py`: V16_1 loss 与 eval-only checkpoint 支持。
- `test_results_v16_1/run_v16_1_one.sh`, `run_v16_1_all.sh`: 三组启动脚本。
- `test_results_v16_1/results/aggregate/v16_1_per_re_metrics.csv`: 逐 Re compact 指标。
- `test_results_v16_1/results/aggregate/v16_1_summary_metrics.json`: overall/regime 统计。
- `test_results_v16_1/results/aggregate/*.svg`: 对比曲线。

## 最终判断

`V16_1_SteadyPressureAnchor32` 是最适合作为 V16_2 起点的版本；`V16_1_HopfOnsetGrowthLoss32` 的 Hopf loss 设计有价值但还不够；`V16_1_TrainStableCombined32` 暂时应废弃或重做 loss 权重解耦。当前瓶颈已经从“有没有闭环物理约束”转向“不同 attractor 约束在共享 MoE 表征中的冲突”和“pressure base 精度不足导致 residual 长期承担主预测”。
