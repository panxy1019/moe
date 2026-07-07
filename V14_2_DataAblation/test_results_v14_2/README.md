# V14_2 DataAblation Test Results

Scripts:

- `train_v14_2.py`: data-ablation HPRS-MoE-ROM trainer/evaluator.
- `evaluate.py`: compact metrics reader.
- `monitor_routing.py`: group/router activation reader.
- `run_v14_2_smoke.sh`: 2-epoch smoke test.
- `run_v14_2_test1_time_sparse.sh`: time-sparse training, full-Re held-out evaluation.
- `run_v14_2_test2_time_re_sparse.sh`: time-sparse plus Re-sparse training, full-Re held-out evaluation.

Each full run trains one model and evaluates every uniformly selected held-out
Re. Results are written under `results/` with:

- `*_metrics.json`
- `*_summary.md`
- `*_error_vs_re.csv`
- `*_error_vs_re.svg`
- `run.log`
