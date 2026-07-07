# V14_3 Test Scripts

- `train_v14_3.py`: V14/V14_2 training framework plus `--pressure-input-mode`.
- `run_v14_3_pressure_input_one.sh`: run one ablation mode.
- `run_v14_3_pressure_input_ablation_all.sh`: run all three modes and aggregate.
- `aggregate_pressure_input_ablation.py`: merge metrics into CSV, SVG curves, and report.

The intended full command is:

```bash
bash run_v14_3_pressure_input_ablation_all.sh
```

Set `ALLOW_SHARED_GPU=0` to wait for a fully idle GPU. By default the runner may
share a low-utilization GPU with one existing job, but it still runs the three
V14_3 modes sequentially.
