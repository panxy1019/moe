# V6 Autonomous-Pressure MoE-ROM

完整技术报告：

- `TECHNICAL_REPORT_V6.md`

代码与原始实验结果：

- `test_results_v6/deep_moe_rom_v6.py`
- `test_results_v6/results/`

V6 基于 V5 稳定 RK4 MoE-ROM，新增 pressure-next head，并在 rollout 中使用模型预测的 pressure coefficients 更新上下文，减少 teacher-forced pressure 依赖。
