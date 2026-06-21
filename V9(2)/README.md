# V9(2) Long-Stability Accuracy Run

V9(2) 在 V9 的精度优先配置上延长 epoch，并加入更明确的 loss 稳定早停规则。

核心变化：

- 最大 epoch 从 70 延长到 180。
- 每 5 epoch 验证一次。
- 只有 `val_score` 改善超过 `8e-4` 才刷新 best checkpoint。
- `min_epochs=100`，`patience=35`，loss 稳定后自动停止。

完整报告：

- `TECHNICAL_REPORT_V9_2.md`

代码与结果：

- `test_results_v9_2/deep_moe_rom_v9_2.py`
- `test_results_v9_2/run_v9_2_longstable.sh`
- `test_results_v9_2/results/`
