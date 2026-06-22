# V10 Test Results

## Reproducible Commands

主实验：

```bash
cd /root/moe/V10/test_results_v10
./run_v10_shared_expert.sh
```

低 Re pressure-focused 消融：

```bash
cd /root/moe/V10/test_results_v10
./run_v10_lowRe_pressure_focus.sh
```

## Environment

- Cluster path: `/root/moe/V10/test_results_v10`
- Data root: `/root/moe/V8/data/Global_POD_Weighted_L2`
- Galerkin tensor: `/root/moe/V8/data/semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_slim.npz`
- Pressure surrogate tensor: `/root/moe/V8/data/pressure_poisson_surrogate_tensors_allRe100_weightedL2_ru80_rp80.npz`
- GPU: NVIDIA GeForce RTX 3090
- PyTorch: `2.11.0+cu126`

## Result Directories

- `results/v10_r16_rk4_b2_shared_floor`: final V10 main experiment.
- `results/v10_r16_lowRe_pressure_focus`: low-Re pressure loss ablation.
- `smoke`: short r=8 smoke test.
