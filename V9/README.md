# V9 Accuracy-Oriented Re50-300 MoE-ROM

V9 基于 V8 的 `Re=50-300` 数据集继续优化 V7/V8 pressure-surrogate residual MoE-ROM 架构，目标是提高低 Re 和跨 Re 区间的预测精度。

核心改动：

- 保留 RK4 + pressure Poisson surrogate baseline + residual pressure head。
- 增加 alpha/RHS/pressure 的 amplitude-aware relative loss。
- rollout loss 中混入 relative rollout error。
- 增强 router load-balance，并用轻微负 entropy 系数鼓励更均匀的专家使用。
- 增大训练预算、专家数、hidden width、重构采样列数和 rollout curriculum。

完整报告：

- `TECHNICAL_REPORT_V9.md`

代码与结果：

- `test_results_v9/deep_moe_rom_v9.py`
- `test_results_v9/run_v9_experiments_precision.sh`
- `test_results_v9/results/`
