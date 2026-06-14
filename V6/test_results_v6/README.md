# V6 Test Results

本目录包含 V6 autonomous-pressure MoE-ROM 测试代码与四组正式实验结果。

## 文件

- `deep_moe_rom_v6.py`: V6 PyTorch 实验脚本。
- `results/v6_r16_rk4_b2_autop_metrics.json`: r16 2-block 稳定基线。
- `results/v6_r16_rk4_b3_deep_autop_metrics.json`: r16 3-block 深层模型。
- `results/v6_r32_rk4_b2_autop_metrics.json`: r32 2-block 稳定基线。
- `results/v6_r32_rk4_b3_deep_autop_metrics.json`: r32 3-block 深层模型。

对应 `*_summary.md` 为自动摘要。完整解释见 `../TECHNICAL_REPORT_V6.md`。
