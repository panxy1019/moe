# V9(2) Test Results

本目录保存 V9(2) 长 epoch / 稳定早停实验代码和结果。

## Files

- `deep_moe_rom_v9_2.py`: V9(2) 训练、评估和 summary 生成脚本。
- `run_v9_2_longstable.sh`: 正式长训练运行脚本。
- `results/v9_2_r16_rk4_b2_longstable/`: `r_u=16, r_p=16`, 2 个 MoE blocks，长 epoch + stability early stopping。

## Data

V9(2) 继续使用 V8 数据：

- `/root/moe/V8/data/Global_POD_Weighted_L2/global_velocity_pod_weighted_l2.npz`
- `/root/moe/V8/data/Global_POD_Weighted_L2/global_pressure_pod_weighted_l2.npz`
- `/root/moe/V8/data/Global_POD_Weighted_L2/pod_snapshot_index.csv`
- `/root/moe/V8/data/semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_slim.npz`
- `/root/moe/V8/data/pressure_poisson_surrogate_tensors_allRe100_weightedL2_ru80_rp80.npz`
