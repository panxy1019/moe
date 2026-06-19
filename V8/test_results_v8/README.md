# V8 Test Results

本目录保存 V8 在 `Re=50-300` 新数据集上复测 V7 pressure-surrogate residual MoE-ROM 架构的代码和结果。

## Files

- `deep_moe_rom_v8.py`: V8 训练、评估和 summary 生成脚本。
- `run_v8_experiments_light.sh`: 正式实验运行脚本。
- `results/v8_r16_rk4_b2_surres/`: `r_u=16, r_p=16`, 2 个 shared-routed MoE blocks。
- `results/v8_r32_rk4_b3_surres/`: `r_u=32, r_p=32`, 3 个 shared-routed MoE blocks。

## Cluster Data Inputs

实验读取新集群数据：

- `/root/moe/V8/data/Global_POD_Weighted_L2/global_velocity_pod_weighted_l2.npz`
- `/root/moe/V8/data/Global_POD_Weighted_L2/global_pressure_pod_weighted_l2.npz`
- `/root/moe/V8/data/Global_POD_Weighted_L2/pod_snapshot_index.csv`
- `/root/moe/V8/data/semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_slim.npz`
- `/root/moe/V8/data/pressure_poisson_surrogate_tensors_allRe100_weightedL2_ru80_rp80.npz`

`slim` Galerkin 张量使用共享 `H/P/G_u` 与 per-Re `c_all/A_all` 布局；V8 脚本已兼容该布局。
