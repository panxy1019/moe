# V7 Pressure-Surrogate Residual MoE-ROM

完整技术报告：

- `TECHNICAL_REPORT_V7.md`

代码与原始实验结果：

- `test_results_v7/deep_moe_rom_v7.py`
- `test_results_v7/results/`

V7 在 V6 autonomous-pressure RK4 MoE-ROM 上引入刚性 Pressure Poisson surrogate baseline，将 `pressure_next_head` 降级为 residual correction head：

```text
b_next = c_tilde + A_tilde a_next + H_tilde(a_next,a_next) + delta_b_theta(x_t)
```
