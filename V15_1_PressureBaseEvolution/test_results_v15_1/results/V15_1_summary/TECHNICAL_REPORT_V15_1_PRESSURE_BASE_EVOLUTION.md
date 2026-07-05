# V15_1 Pressure Base Evolution 技术报告

## 摘要

- 三组 V15_1 实验已完成，均使用 V15_Base 的 Re=20-200 Physics-Generalizable 数据库、同一 Held-out Re 划分、HPRS-MoE、Expert/Router、Loss、优化器、batch size、epoch schedule 和 RK4 closed-loop 训练流程。
- `V15_1_AdaptiveGate` 是本轮唯一稳定增益方案：平均 one-step velocity=0.08981, one-step pressure=0.20867, 24-step velocity=0.45334, 24-step pressure=0.68339。相对 V15_Base 的 24-step pressure=3.558 降低 80.8%，相对 V15_BalancedTraining 的 1.06 继续降低 35.5%。
- `V15_1_FiLMBase` 和 `V15_1_RegimeAwareROM` 没有带来可用提升。二者的 teacher-forced closure pressure L2 仍为 0.33665 / 0.34048，但 autonomous one-step 与 24-step pressure 分别升至 133 / 223 和 68.41 / 95.56，说明新 base 与预测速度状态耦合后非常不稳定。
- AdaptiveGate 的 gate 明确学到了物理规律：Steady 平均 alpha=0.13365, Hopf alpha=0.12980, Periodic alpha=0.75434，与 Re 的相关系数为 0.84379。低 Re/Hopf 自动降低对 Poisson base 的信任，高 Re 周期流提高 base 权重。

## 实验设置

| Case | 唯一改动 | Closure/Base 形式 | 训练时间 |
|---|---|---|---:|
| V15_1_AdaptiveGate | Pressure base 与 residual 之间加入按 pressure POD 模态输出的 gate | `b_pred = g(h) * b_base + residual` | 11.03 h |
| V15_1_FiLMBase | Pressure Poisson base 加 FiLM/Re 条件校准 | `b_pred = b_base_film(a,Re) + residual` | 11.42 h |
| V15_1_RegimeAwareROM | Steady/Hopf/Periodic 三套 velocity ROM 与 pressure base 经 RegimeGate 混合 | `ROM_base = sum pi_r * ROM_r; b_pred = base + residual` | 12.32 h |

保持不变项：V15_Base dense training split、11 个 Held-out Re、ru=16、rp=16、Shared Encoder、Group Router、Velocity/Pressure Router、Physics-aware Expert、Linear + Low-rank Quadratic + FFN Expert、Galerkin + RK4、pressure_target=closure、loss 权重、AdamW、学习率、batch size、epoch/patience、scheduled sampling。

SwanLab runs:

- AdaptiveGate: https://swanlab.cn/@panxy1019/V15_1_AdaptiveGate/runs/yy8izpg6
- FiLMBase: https://swanlab.cn/@panxy1019/V15_1_FiLMBase/runs/udwlmzzx
- RegimeAwareROM: https://swanlab.cn/@panxy1019/V15_1_RegimeAwareROM/runs/hjiyqyul

## Overall 指标对比

所有误差均为相对 L2。`pressure closure` 是 teacher-forced/true a_next 下的最终压力 closure 误差，`one-step pressure` 和 `24-step pressure` 是 autonomous RK4 推进后的压力误差。

| Case | one-step u | one-step p | 24-step u | 24-step p | RHS | p base | pressure closure | active experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V15_Base | 0.23345 | 1.642 | 0.74618 | 3.558 | 1.48 | 85.95 | 1.572 | 4.208 |
| V15_LargeROM | 0.10283 | 0.60357 | 0.45314 | 1.483 | 0.63563 | 83.02 | 0.57301 | 4.565 |
| V15_BalancedTraining | 0.18406 | 0.59478 | 0.47304 | 1.06 | 0.88903 | 85.95 | 0.55108 | 4.412 |
| V15_1_AdaptiveGate | 0.08981 | 0.20867 | 0.45334 | 0.68339 | 0.51030 | 85.95 | 0.20554 | 4.691 |
| V15_1_FiLMBase | 0.10207 | 133 | 3.651 | 223 | 0.51436 | 217 | 0.33665 | 3.48 |
| V15_1_RegimeAwareROM | 0.30754 | 68.41 | 2.325 | 95.56 | 0.72654 | 25.43 | 0.34048 | 4.454 |

关键读数：

- AdaptiveGate 同时超过 V15_LargeROM 的 one-step velocity (0.08981 vs 0.10283)，并接近其 24-step velocity (0.45334 vs 0.45314).
- AdaptiveGate 的 pressure rollout 是目前最强：0.68339，低于 V15_BalancedTraining 的 1.06。
- FiLMBase 的 `pressure closure`=0.33665 看起来不坏，但 autonomous pressure=133 / 223，说明问题发生在 base 对 predicted a_next 的闭环敏感性，而不是压力 residual head 单独拟合能力。
- RegimeAwareROM 的 pressure base L2 从 V15_Base 的 85.95 降到 25.43，但 velocity rollout 升至 2.325，压力 rollout 升至 95.56。这说明更准的 BaseOnly 还不足以保证闭环稳定，velocity ROM base 和 pressure base 的坐标/动态耦合更关键。

## Regime-aware 结果

| Regime | Case | one-step u | one-step p | 24-step u | 24-step p | p base |
|---|---|---:|---:|---:|---:|---:|
| Steady | V15_Base | 0.05442 | 2.175 | 0.39852 | 2.459 | 57.48 |
| Steady | V15_BalancedTraining | 0.02949 | 0.86833 | 0.22612 | 0.76405 | 57.48 |
| Steady | V15_1_AdaptiveGate | 0.01869 | 0.30670 | 0.17702 | 0.92155 | 57.48 |
| Steady | V15_1_FiLMBase | 0.02412 | 172 | 1.006 | 277 | 228 |
| Steady | V15_1_RegimeAwareROM | 0.10379 | 43.79 | 1.082 | 60.96 | 25.77 |
| Hopf | V15_Base | 0.73506 | 3.041 | 2.097 | 9.603 | 237 |
| Hopf | V15_BalancedTraining | 0.59360 | 0.95680 | 1.354 | 2.761 | 237 |
| Hopf | V15_1_AdaptiveGate | 0.27507 | 0.32599 | 1.371 | 1.211 | 237 |
| Hopf | V15_1_FiLMBase | 0.22726 | 258 | 11.78 | 449 | 489 |
| Hopf | V15_1_RegimeAwareROM | 0.89851 | 192 | 5.923 | 268 | 57.97 |
| Periodic | V15_Base | 0.03628 | 0.06002 | 0.08043 | 0.12252 | 0.97316 |
| Periodic | V15_BalancedTraining | 0.03146 | 0.04972 | 0.05942 | 0.07956 | 0.97316 |
| Periodic | V15_1_AdaptiveGate | 0.02200 | 0.02264 | 0.04115 | 0.04930 | 0.97316 |
| Periodic | V15_1_FiLMBase | 0.08613 | 0.45708 | 0.20228 | 0.53065 | 1.324 |
| Periodic | V15_1_RegimeAwareROM | 0.06805 | 0.32614 | 0.86928 | 0.94323 | 0.70357 |

Regime 观察：

- Periodic 是三组中最稳定的区域。AdaptiveGate 在 Periodic 上 24-step pressure=0.0493，已经进入 5% 左右；velocity rollout=0.0412。
- Steady 的原始 pressure base 误差很大，AdaptiveGate 通过 alpha≈0.13 把 pressure rollout 控制到 0.922，但仍未达到 10%。这说明 Low-Re pressure 的瓶颈主要是 base 信任与 residual 校正强度，而不是需要更大 MoE。
- Hopf 仍是最难区域，尤其 Re=51.786。AdaptiveGate 把该点 pressure rollout 从 V15_Base 的 25.07 明显降到 2.52，但仍显著高于其它 Re，表明分岔附近的振荡幅值/相位恢复仍是主要瓶颈。
- FiLMBase 和 RegimeAwareROM 在 Steady/Hopf 的 autonomous pressure 均出现数量级发散，说明不加约束地改 base 会把 closed-loop exposure bias 放大。

## AdaptiveGate 诊断

| Regime | alpha mean | alpha std | alpha min | alpha max |
|---|---:|---:|---:|---:|
| Steady | 0.13365 | 0.00212 | 0.13176 | 0.13716 |
| Hopf | 0.12980 | 0.00027 | 0.12942 | 0.13004 |
| Periodic | 0.75434 | 0.02413 | 0.71329 | 0.77365 |

- alpha 与 Re 的相关系数为 0.84379，与 raw pressure base error 的相关系数为 -0.39841。这符合预期：Re 越高、base 越可靠，模型越信任 Poisson base；base error 越高，模型越压低 base。
- 低 Re/Hopf 的 alpha 基本稳定在 0.13 左右，相当于显式 suppress 约 87% raw base contribution；Periodic alpha 上升到 0.71-0.77，使 pressure base 重新成为有效 prior。
- 这个结果直接回答 V15_1 的核心问题：当前压力瓶颈首先不是 MoE expert 容量，而是固定 `b_base + residual` 的信任机制过硬。modal base confidence 是有效路线。

## Held-out Re 明细

| Case | Re | Regime | one-step u | one-step p | 24-step u | 24-step p | p base | alpha |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| V15_1_AdaptiveGate | 24.630 | Steady | 0.02573 | 0.45430 | 0.24358 | 1.4 | 54.87 | 0.13716 |
| V15_1_AdaptiveGate | 32.740 | Steady | 0.01937 | 0.30532 | 0.18336 | 0.94733 | 68.99 | 0.13350 |
| V15_1_AdaptiveGate | 39.685 | Steady | 0.01603 | 0.23269 | 0.15462 | 0.71749 | 56.04 | 0.13219 |
| V15_1_AdaptiveGate | 45.143 | Steady | 0.01364 | 0.23449 | 0.12653 | 0.62141 | 50.04 | 0.13176 |
| V15_1_AdaptiveGate | 47.081 | Hopf | 0.13887 | 0.20789 | 0.61973 | 0.56277 | 66.45 | 0.12995 |
| V15_1_AdaptiveGate | 49.022 | Hopf | 0.10047 | 0.21925 | 0.42768 | 0.55276 | 53.89 | 0.13004 |
| V15_1_AdaptiveGate | 51.786 | Hopf | 0.58587 | 0.55084 | 3.067 | 2.518 | 591 | 0.12942 |
| V15_1_AdaptiveGate | 70.315 | Periodic | 0.03907 | 0.03536 | 0.09421 | 0.10811 | 1.439 | 0.71329 |
| V15_1_AdaptiveGate | 100.352 | Periodic | 0.01794 | 0.01597 | 0.02666 | 0.03012 | 0.93649 | 0.76104 |
| V15_1_AdaptiveGate | 149.059 | Periodic | 0.01274 | 0.01441 | 0.01788 | 0.02383 | 0.80067 | 0.76940 |
| V15_1_AdaptiveGate | 189.862 | Periodic | 0.01824 | 0.02481 | 0.02585 | 0.03513 | 0.71597 | 0.77365 |
| V15_1_FiLMBase | 24.630 | Steady | 0.03299 | 289 | 1.158 | 422 | 343 | 1 |
| V15_1_FiLMBase | 32.740 | Steady | 0.02487 | 210 | 1.047 | 357 | 278 | 1 |
| V15_1_FiLMBase | 39.685 | Steady | 0.02068 | 113 | 0.95385 | 200 | 168 | 1 |
| V15_1_FiLMBase | 45.143 | Steady | 0.01793 | 74.57 | 0.86661 | 130 | 123 | 1 |
| V15_1_FiLMBase | 47.081 | Hopf | 0.10958 | 90.27 | 6.882 | 195 | 155 | 1 |
| V15_1_FiLMBase | 49.022 | Hopf | 0.08757 | 65.76 | 5.12 | 143 | 118 | 1 |
| V15_1_FiLMBase | 51.786 | Hopf | 0.48461 | 618 | 23.33 | 1.01e+03 | 1.19e+03 | 1 |
| V15_1_FiLMBase | 70.315 | Periodic | 0.09037 | 0.82680 | 0.40981 | 0.98358 | 2.123 | 1 |
| V15_1_FiLMBase | 100.352 | Periodic | 0.09855 | 0.56751 | 0.16502 | 0.70197 | 1.37 | 1 |
| V15_1_FiLMBase | 149.059 | Periodic | 0.08805 | 0.28935 | 0.08491 | 0.20355 | 0.97800 | 1 |
| V15_1_FiLMBase | 189.862 | Periodic | 0.06754 | 0.14467 | 0.14940 | 0.23349 | 0.82674 | 1 |
| V15_1_RegimeAwareROM | 24.630 | Steady | 0.18369 | 38.88 | 2.629 | 53.4 | 39.11 | 1 |
| V15_1_RegimeAwareROM | 32.740 | Steady | 0.11306 | 51.71 | 0.94319 | 72.28 | 31.24 | 1 |
| V15_1_RegimeAwareROM | 39.685 | Steady | 0.07295 | 44.19 | 0.45498 | 61.83 | 18.82 | 1 |
| V15_1_RegimeAwareROM | 45.143 | Steady | 0.04545 | 40.37 | 0.30178 | 56.34 | 13.9 | 1 |
| V15_1_RegimeAwareROM | 47.081 | Hopf | 0.74158 | 53.56 | 3.857 | 91.24 | 17.9 | 1 |
| V15_1_RegimeAwareROM | 49.022 | Hopf | 0.38679 | 43.59 | 2.997 | 74.78 | 13.8 | 1 |
| V15_1_RegimeAwareROM | 51.786 | Hopf | 1.567 | 479 | 10.91 | 638 | 142 | 1 |
| V15_1_RegimeAwareROM | 70.315 | Periodic | 0.11069 | 0.58031 | 1.419 | 1.442 | 0.92782 | 1 |
| V15_1_RegimeAwareROM | 100.352 | Periodic | 0.07163 | 0.31115 | 0.88745 | 0.94877 | 0.68252 | 1 |
| V15_1_RegimeAwareROM | 149.059 | Periodic | 0.04954 | 0.22273 | 0.61808 | 0.70780 | 0.62162 | 1 |
| V15_1_RegimeAwareROM | 189.862 | Periodic | 0.04036 | 0.19036 | 0.55304 | 0.67419 | 0.58232 | 1 |

## 结论与 V16 建议

1. 保留 AdaptiveGate，并把它作为新的压力分支默认策略。建议下一步从 V15_1_AdaptiveGate checkpoint 继续，不要回到固定 `b_base + residual`。
2. FiLMBase 目前不应直接作为主线。若继续研究，需要先加 base-identity 正则、base delta 范数约束、teacher-forced base pretraining、再逐步打开 closed-loop，而不是让校准 base 从一开始参与 RK4 闭环。
3. RegimeAwareROM 的方向有价值，因为它显著降低了 BaseOnly pressure error，但当前 mixed ROM 破坏了 velocity rollout 稳定性。后续应先单独验证 global-to-regime-to-global POD 投影误差、mean offset、RHS consistency，再训练 MoE residual；也可以先只替换 pressure base，不同时替换 velocity ROM。
4. 当前最强瓶颈排序：固定 base 信任机制 > Hopf 分岔附近 rollout drift > Regime ROM 坐标/闭环耦合。不是简单增加专家数量或隐藏层规模。
5. 如果目标是把所有 Re 的 velocity/pressure rollout 都压到 10% 内，下一步应该重点处理 Re=51.786 Hopf 边界：更长 rollout curriculum、phase-aware loss、Hopf 样本重加权，以及 AdaptiveGate + BalancedTraining 组合实验。

## 产物

- `test_results_v15_1/results/V15_1_summary/v15_1_per_re_metrics.csv`
- `test_results_v15_1/results/V15_1_summary/v15_1_overall_comparison.csv`
- `test_results_v15_1/results/V15_1_summary/v15_1_regime_comparison.csv`
- `test_results_v15_1/results/V15_1_summary/v15_1_rollout_pressure_vs_re.svg`
- `test_results_v15_1/results/V15_1_summary/v15_1_rollout_velocity_vs_re.svg`
- `test_results_v15_1/results/V15_1_summary/v15_1_one_step_pressure_vs_re.svg`
- `test_results_v15_1/results/V15_1_summary/v15_1_pressure_base_vs_re.svg`
- `test_results_v15_1/results/V15_1_summary/v15_1_adaptive_gate_alpha_vs_re.svg`

原始 80MB 级 metrics JSON、checkpoint 和 SwanLab cache 保留在集群，不纳入 Git。
