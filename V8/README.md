# V8 Re50-300 Pressure-Surrogate Residual MoE-ROM

V8 在新的 `Re=50-300`、`100` 个 Reynolds 数 weighted-L2 POD 数据集上复测 V7 架构。

核心架构保持 V7：

```text
a_next = RK4(a_t, Galerkin velocity tensors + learned RHS correction)
b_base = c_tilde + A_tilde @ a_next + einsum(H_tilde, a_next, a_next)
delta_b = pressure_next_head(x_t)
b_next = b_base + delta_b
```

完整报告：

- `TECHNICAL_REPORT_V8.md`

代码与实验结果：

- `test_results_v8/deep_moe_rom_v8.py`
- `test_results_v8/run_v8_experiments_light.sh`
- `test_results_v8/results/`
