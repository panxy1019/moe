# V14_2 DataAblation

V14_2 keeps the V14 HPRS-MoE-ROM architecture and training objective fixed,
then changes only data organization to test whether dense consecutive time
samples limit cross-Re generalization.

Unchanged from V14:

- HPRS-MoE shared encoder, group router, group-local Top-2 experts, and
  group-shared experts.
- Physics-aware expert block: linear state term, low-rank quadratic term, and
  residual FFN.
- Galerkin RHS + learned closure + RK4 integration.
- Loss weights, scheduled sampling, rollout curriculum, router losses, and
  expert diversity regularization.
- Main hyperparameters from V14 closure run.

Data ablations:

- Test1: sparse time sampling only, default one training sample every 5 time
  steps per training Re.
- Test2: sparse time sampling plus sparse training Re sampling, default one
  training Re every 2 non-test Re values.
- Both tests use a uniformly selected 10-Re held-out set across the full
  Re=50-300 range. All held-out Re are excluded from training.
- One model is trained per ablation and evaluated on every held-out Re.

Main files:

- `test_results_v14_2/train_v14_2.py`: V14_2 trainer/evaluator.
- `test_results_v14_2/run_v14_2_smoke.sh`: remote smoke test.
- `test_results_v14_2/run_v14_2_test1_time_sparse.sh`: Test1 full run.
- `test_results_v14_2/run_v14_2_test2_time_re_sparse.sh`: Test2 full run.
- `TECHNICAL_REPORT_V14_2_DATA_ABLATION.md`: final report.

Result artifacts include metrics JSON, summary markdown, error-vs-Re CSV, and
error-vs-Re SVG. Large checkpoints stay on the cluster and are not committed.
