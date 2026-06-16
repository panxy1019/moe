# V7 Test Results

本目录保存 V7 pressure-surrogate residual MoE-ROM 的可复现实验代码和原始结果。

## Files

- `deep_moe_rom_v7.py`: V7 训练、评估和报告生成脚本。
- `results/v7_r16_rk4_b2_surres_metrics.json`: `r_u=16, r_p=16`, 2 个 shared-routed MoE block。
- `results/v7_r16_rk4_b2_surres_summary.md`: 上述实验摘要。
- `results/v7_r16_rk4_b3_deep_surres_metrics.json`: `r_u=16, r_p=16`, 3 个 shared-routed MoE block。
- `results/v7_r16_rk4_b3_deep_surres_summary.md`: 上述实验摘要。
- `results/v7_r32_rk4_b2_surres_metrics.json`: `r_u=32, r_p=32`, 2 个 shared-routed MoE block。
- `results/v7_r32_rk4_b2_surres_summary.md`: 上述实验摘要。
- `results/v7_r32_rk4_b3_deep_surres_metrics.json`: `r_u=32, r_p=32`, 3 个 shared-routed MoE block。
- `results/v7_r32_rk4_b3_deep_surres_summary.md`: 上述实验摘要。

## Data Inputs on Cluster

实验读取集群数据：

- `/root/moe/V7/data/Global_POD_Weighted_L2/global_velocity_pod_weighted_l2.npz`
- `/root/moe/V7/data/Global_POD_Weighted_L2/global_pressure_pod_weighted_l2.npz`
- `/root/moe/V7/data/Global_POD_Weighted_L2/pod_snapshot_index.csv`
- `/root/moe/V7/data/semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz`
- `/root/moe/V7/data/pressure_poisson_surrogate_tensors_allRe30_weightedL2_ru80_rp80.npz`

GitHub 的 `V7/data/` 已包含 pressure surrogate 文档、元数据和 surrogate NPZ；全局 POD 与 Galerkin compact 大文件仍按集群路径引用。报告和 JSON 中记录了完整路径、截断阶数、模型超参数和测试结果。

## V7 Pressure Update

V7 将 `pressure_next_head` 从直接压力预测头改成 residual correction head：

```text
a_next = RK4(a_t, Galerkin velocity tensors + learned RHS correction)
b_base = c_tilde + A_tilde @ a_next + einsum(H_tilde, a_next, a_next)
delta_b = pressure_next_head(x_t)
b_next = b_base + delta_b
```

完整技术报告见 `../TECHNICAL_REPORT_V7.md`。
