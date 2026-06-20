# V9 Test Results

本目录保存 V9 精度优先实验代码和结果。

## Files

- `deep_moe_rom_v9.py`: V9 训练、评估和 summary 生成脚本。
- `run_v9_experiments_precision.sh`: 正式实验运行脚本。
- `results/v9_r16_rk4_b2_relmod/`: `r_u=16, r_p=16`, 2 个 MoE blocks，relative-loss 精调。
- `results/v9_r24_rk4_b3_relmod/`: `r_u=24, r_p=24`, 3 个 MoE blocks，relative-loss 精调。

## Data

V9 继续使用 V8 数据：

- `/root/moe/V8/data/Global_POD_Weighted_L2/global_velocity_pod_weighted_l2.npz`
- `/root/moe/V8/data/Global_POD_Weighted_L2/global_pressure_pod_weighted_l2.npz`
- `/root/moe/V8/data/Global_POD_Weighted_L2/pod_snapshot_index.csv`
- `/root/moe/V8/data/semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_slim.npz`
- `/root/moe/V8/data/pressure_poisson_surrogate_tensors_allRe100_weightedL2_ru80_rp80.npz`
