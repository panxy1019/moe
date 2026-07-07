# V13 Test Results

Scripts:

- `train_v13.py`: hierarchical physics-aware MoE-ROM trainer/evaluator.
- `evaluate.py`: summarize one or more metrics JSON files.
- `monitor_routing.py`: print velocity/pressure router and shared-expert usage.
- `run_v13_shared_operator_smoke.sh`: 2-epoch smoke test.
- `run_v13_shared_operator.sh`: 24 routed + 4 shared closure-pressure run.
- `run_v13_shared_operator_state.sh`: 24 routed + 4 shared state-pressure run.
- `run_v13_shared_operator_lowre_pressure.sh`: low-Re pressure-focused ablation.

Result directories are produced under `smoke/` and `results/` after running the
cluster scripts.
