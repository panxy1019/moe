# V14 HPRS-MoE Test Results

Scripts:

- `train_v14.py`: HPRS-MoE-ROM trainer/evaluator.
- `evaluate.py`: summarize one or more metrics JSON files.
- `monitor_routing.py`: print group/router active-expert usage.
- `run_v14_hprs_smoke.sh`: 2-epoch smoke test.
- `run_v14_hprs_closed_loop.sh`: closure-pressure closed-loop rollout run.
- `run_v14_hprs_state_pressure.sh`: state-pressure comparison.
- `run_v14_hprs_lowre_pressure.sh`: low-Re pressure-focused ablation.
- `run_v14_hprs_lowre_state_pressure.sh`: low-Re state-pressure targeted run.

Result directories are produced under `smoke/` and `results/` after running the
cluster scripts.

The main V14 report and framework diagrams live one directory up:

- `../TECHNICAL_REPORT_V14.md`
- `../docs/v14_training_flow.svg`
- `../docs/v14_inference_flow.svg`
