# V10 Shared-Expert MoE-ROM

本目录记录 V10 测试：在 V9(2) 稳定基线 `v9_2_r16_rk4_b2_longstable` 上加入显式 shared expert 机制，并用 gate floor 抑制 dead experts。

## 文件

- `TECHNICAL_REPORT_V10.md`: 完整技术报告、超参数、对比表和结论。
- `test_results_v10/deep_moe_rom_v10.py`: V10 训练与评估代码。
- `test_results_v10/run_v10_shared_expert.sh`: 主实验脚本。
- `test_results_v10/run_v10_lowRe_pressure_focus.sh`: 低 Re pressure-focused 消融脚本。
- `test_results_v10/results/v10_r16_rk4_b2_shared_floor/`: V10 主实验 JSON、summary 和日志。
- `test_results_v10/results/v10_r16_lowRe_pressure_focus/`: 低 Re 消融 JSON、summary 和日志。
- `test_results_v10/smoke/`: 集群 smoke test 输出。

## 核心结论

V10 主配置消除了 V9/V9(2) 在低 Re 上的 dead experts：三个测试 Re 的 dead expert 数均为 0。

精度上，V10 相对 V9(2) 将平均 pressure relative L2 从 0.1520 降到 0.1231，平均 autonomous velocity rollout 从 0.4260 降到 0.4091。高 Re=300 恢复最明显：pressure L2 为 0.06496，autonomous velocity rollout 为 0.49645，均优于 V9(2)。

目标“全 Re pressure relative error < 10%”尚未完全达成：Re=120 和 Re=300 已低于 10%，但低 Re=56 仍为 0.28369。追加的 low-Re pressure-focused 消融没有改善该点。
