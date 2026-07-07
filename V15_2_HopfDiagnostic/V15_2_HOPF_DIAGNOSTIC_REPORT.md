# V15_2 Hopf Diagnostic 技术报告

## 摘要

本实验只对已经训练完成的 `V15_1_AdaptiveGate` checkpoint 做诊断，不修改网络结构、不重新训练、不改变 HPRS-MoE、Router、Expert、Galerkin、RK4 或 loss。诊断目标是回答 Hopf 难点 `Re=51.786` 的长期误差到底来自幅值、相位、频率，还是 POD 投影/截断本身。

核心结论：

- 主振荡 POD pair 自动识别为 `(a0, a1)`。它在 held-out Hopf/Periodic 真实轨迹中的 oscillatory energy score 为 `0.0695`，远高于 `(a2,a3)` 的 `0.00668`。
- `Re=51.786` 不是单纯相位错。该点 24-step velocity L2 与 phase error 的相关性为 `-0.464`，但与 amplitude error 的相关性为 `0.975`，说明窗口间误差主要跟振荡幅值/能量过冲一致。
- `Re=51.786` 的代表性 24-step window：velocity L2=`2.44`，pressure L2=`1.81`，modal amplitude relative error=`5.39`，mean phase error=`1.87 rad`，frequency/Strouhal error=`44.7%`。跨 5 个窗口平均：velocity L2=`3.07`，pressure L2=`2.52`，amplitude error=`5.93`，phase error=`1.28 rad`，frequency error=`20.8%`。
- POD 投影/截断不是无关因素。`Re=51.786` 在 `ru=16/rp=16` 下的 velocity tail relative L2=`0.418`，pressure tail relative L2=`0.382`，是 Hopf 区域中最高的一档；这说明 V15_Base 的 ROM 空间在 Hopf 分岔附近确实偏紧。
- 所以 `Re=51.786` 的主瓶颈排序是：**幅值/能量过冲为主，POD 截断为重要底层限制，相位漂移次之，频率误差中等但不是首要解释**。

## 诊断方法

- 加载 checkpoint：
  `/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/results/V15_1_AdaptiveGate/.../Re_24p630436_checkpoint.pt`
- 使用同一套 V15_1 数据、Galerkin tensor、Pressure Poisson surrogate 和 checkpoint scalers。
- 对 11 个 held-out Reynolds 数运行 24-step autonomous RK4 rolling windows，共 `39` 个窗口。
- 对每个 Re 输出一个代表性 median-L2 window 的诊断图：
  - phase portrait: `a_i vs a_j`
  - amplitude: `r(t)=sqrt(a_i^2+a_j^2)`
  - phase: `theta(t)=unwrap(atan2(a_j,a_i))`
  - frequency: `omega(t)=d theta/dt`
  - amplitude error
  - wrapped phase error
- 对所有 24-step windows 统计：
  - velocity/pressure relative L2
  - amplitude error
  - wrapped phase error
  - frequency/Strouhal relative error
  - 24-step velocity L2 与 phase/amplitude error 的相关性
  - POD tail energy beyond `ru/rp=16`

原始 ROM_PhysicsGeneralizable retained POD 数据库没有保存真实 Lift/Drag coefficient time series。已经在数据库目录中搜索 `lift/drag/force/CL/CD` 相关文件，未发现可用于逐时刻对齐的力系数。因此本报告不伪造 Lift/Drag 代理指标，Lift/Drag 的幅值、相位、频率误差标记为 unavailable。

## Per-Re 结果

| Re | Regime | windows | 24-step u L2 | 24-step p L2 | amplitude error | phase error rad | Strouhal/freq error | L2-phase corr | L2-amp corr | POD u tail | POD p tail |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630 | Steady | 1 | 0.2436 | 1.400 | 1.06e3 | 1.311 | 0.577 | nan | nan | 0.475 | 0.765 |
| 32.740 | Steady | 1 | 0.1834 | 0.947 | 961 | 1.455 | 0.424 | nan | nan | 0.459 | 0.764 |
| 39.685 | Steady | 1 | 0.1546 | 0.717 | 876 | 1.473 | 0.326 | nan | nan | 0.455 | 0.728 |
| 45.143 | Steady | 1 | 0.1265 | 0.621 | 787 | 1.438 | 0.309 | nan | nan | 0.453 | 0.695 |
| 47.081 | Hopf | 5 | 0.6197 | 0.563 | 172 | 1.388 | 0.993 | 0.573 | -0.796 | 0.406 | 0.366 |
| 49.022 | Hopf | 5 | 0.4277 | 0.553 | 44.1 | 1.422 | 0.896 | -0.441 | -0.810 | 0.408 | 0.376 |
| 51.786 | Hopf | 5 | 3.067 | 2.518 | 5.93 | 1.282 | 0.208 | -0.464 | 0.975 | 0.418 | 0.382 |
| 70.315 | Periodic | 5 | 0.0942 | 0.108 | 0.0253 | 0.0675 | 0.0079 | -0.851 | 0.648 | 0.178 | 0.269 |
| 100.352 | Periodic | 5 | 0.0267 | 0.030 | 0.0167 | 0.0186 | 0.0005 | 0.946 | 0.973 | 0.0655 | 0.140 |
| 149.059 | Periodic | 5 | 0.0179 | 0.0238 | 0.0052 | 0.0084 | 0.0004 | 0.905 | 0.864 | 0.0486 | 0.0625 |
| 189.862 | Periodic | 5 | 0.0259 | 0.0351 | 0.0114 | 0.0120 | 0.0011 | 0.473 | -0.581 | 0.0554 | 0.0718 |

注意：Steady 区域真实振荡幅值接近 0，`|r_pred-r_true|/|r_true|` 会被极小分母放大，因此 Steady 的 amplitude/frequency 数值不应按周期流解释，只能说明模型产生了不该有的微弱振荡。

## Hopf 区域解读

### Re=47.081 和 Re=49.022

这两个点处于 Hopf onset 附近，真实主振荡模态幅值极小：

- `Re=47.081`: true `r_mean` 约 `4.8e-7` 到 `1.4e-6`，pred `r_mean` 约 `1.2e-4` 到 `1.4e-4`。
- `Re=49.022`: true `r_mean` 约 `2.6e-6` 到 `3.3e-6`，pred `r_mean` 约 `1.3e-4`。

因此 amplitude relative error 非常大，且 frequency error 接近 90%-100%。这不是普通的相位滞后，而是模型在 Hopf onset 处没有正确保持“接近零但正在萌发”的小幅振荡状态，预测轨迹带入了过强的 limit-cycle-like 模态。

### Re=51.786

`Re=51.786` 是最关键难点。逐窗口结果显示：

| window start | velocity L2 | pressure L2 | amp error | phase error rad | freq error | r_true mean | r_pred mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1936.79 | 5.639 | 4.566 | 10.09 | 0.928 | 0.108 | 1.40e-5 | 1.57e-4 |
| 2447.60 | 2.440 | 1.810 | 5.39 | 1.867 | 0.447 | 1.76e-5 | 1.12e-4 |
| 2958.40 | 1.957 | 1.645 | 3.72 | 1.959 | 0.267 | 2.21e-5 | 1.04e-4 |
| 3469.20 | 3.199 | 2.732 | 6.87 | 0.844 | 0.117 | 2.77e-5 | 2.19e-4 |
| 3979.99 | 2.099 | 1.839 | 3.56 | 0.811 | 0.100 | 3.47e-5 | 1.62e-4 |

这里有三件事同时发生：

- 幅值显著过冲：predicted `r_mean` 是 true `r_mean` 的数倍到十几倍。
- 相位确实有误差，平均约 `1.28 rad`，代表性窗口可到 `1.87-1.96 rad`。
- 频率误差平均 `20.8%`，个别窗口到 `44.7%`，但不是所有窗口都大。

决定性证据是相关性：`velocity L2` 与 `amplitude error` 的相关性为 `0.975`，与 `phase error` 的相关性为 `-0.464`。因此从 24-step L2 的窗口变化看，Re=51.786 的 drift 更主要是幅值/能量错误，而不是相位错误。

## POD 投影/截断判断

若只看 Periodic 区域，`ru=16/rp=16` 已经比较够用：

- `Re=100.352`: velocity tail `0.0655`，pressure tail `0.140`
- `Re=149.059`: velocity tail `0.0486`，pressure tail `0.0625`
- `Re=189.862`: velocity tail `0.0554`，pressure tail `0.0718`

但 Hopf 区域明显不同：

- `Re=47.081`: velocity tail `0.406`，pressure tail `0.366`
- `Re=49.022`: velocity tail `0.408`，pressure tail `0.376`
- `Re=51.786`: velocity tail `0.418`，pressure tail `0.382`

这说明 Hopf transition 附近的动力学不是很好地压进前 16 个 velocity/pressure POD modes。POD 投影本身不是唯一错误源，但它给模型留下了一个很硬的下限：在真实轨迹尚处于小幅振荡、模态能量分散的区域，`ru=16/rp=16` 对幅值和压力相位都不够友好。

## Lift/Drag 说明

本次无法给出真实 Lift/Drag 的相位、幅值、频率误差，原因是当前集群上的 retained ROM 数据只包含 POD coefficients、POD basis、Galerkin tensors、Pressure Poisson surrogate 和 snapshot index。数据库构造报告提到 OpenFOAM 生成流程中曾用 lift/drag 做稳定性监控，但这些 force histories 没有随 ROM_PhysicsGeneralizable retained data 一起保存。

因此：

- 没有输出真实 CL/CD phase/amplitude/frequency error。
- 没有用 POD mode 代理 Lift/Drag，因为那会混淆物理力系数和模态坐标。
- 后续若要补齐第 8 项，需要从原始 OpenFOAM case 的 forceCoeffs 或 postProcessing 目录导出 `Cl(t), Cd(t)`，再与 POD snapshot index 按时间对齐。

## 结论

`Re=51.786` 的问题不是单一的 phase drift，也不是单一 frequency mismatch。更准确的判断是：

1. **幅值/能量过冲是直接主因。** 24-step L2 与 amplitude error 强相关 `0.975`，预测主振荡 pair 的半径明显大于真实值。
2. **相位误差存在但不是 L2 的主导解释。** phase error 约 `1.28 rad`，但与 L2 的相关性为负。
3. **频率/Strouhal 有中等误差。** 平均 `20.8%`，代表性窗口可到 `44.7%`，但不如幅值错误稳定地解释 drift。
4. **POD 截断是底层瓶颈之一。** Hopf 三个点的 velocity/pressure tail 都在 `0.36-0.42` 左右，远高于成熟周期流。

后续建议：

- 在 V16 或 V15_3 中优先做 `AdaptiveGate + Hopf-balanced training`，并对 Hopf onset 区域加 amplitude/energy envelope loss。
- 对 `Re=47-52` 使用更高 ROM 维数或局部 Hopf POD basis 做对照，验证 ru/rp=16 的截断下限。
- 如果继续追求物理可解释性，需要补齐真实 Lift/Drag time series，并把 CL/CD phase/frequency 作为独立诊断目标。

## 输出文件

- `results/v15_2_per_re_summary.csv`
- `results/v15_2_per_window_metrics.csv`
- `results/v15_2_pod_projection_summary.csv`
- `results/v15_2_hopf_diagnostic_summary.json`
- `results/figures/Re_51p786450_hopf_diagnostic.svg`
- `results/timeseries/Re_51p786450_representative.csv`
