# V13 Hierarchical Physics-Aware MoE-ROM

V13 implements the supplied research plan on top of V12:

- enlarged shared encoder and expanded expert FFN blocks;
- separate velocity and pressure routers;
- Top-2 sparse routed experts;
- always-on shared experts;
- physics-aware expert heads with linear state terms and low-rank quadratic
  state interactions;
- RK4 velocity integration retained.

Main files:

- `test_results_v13/train_v13.py`: V13 training/evaluation script.
- `test_results_v13/evaluate.py`: compact metrics aggregator.
- `test_results_v13/monitor_routing.py`: routed/shared expert utilization reader.
- `test_results_v13/run_v13_shared_operator_smoke.sh`: smoke test.
- `test_results_v13/run_v13_shared_operator.sh`: closure-pressure main run.
- `test_results_v13/run_v13_shared_operator_state.sh`: state-pressure comparison.
- `test_results_v13/run_v13_shared_operator_lowre_pressure.sh`: low-Re pressure ablation.
- `TECHNICAL_REPORT_V13.md`: final experiment report.

Best result:

- Main model: `v13_r16_hier_closure_24x4_top2`.
- Velocity autonomous one-step relative L2 is below 10% for all held-out Re.
- Pressure direct relative L2 is `27.37% / 1.88% / 4.27%` at
  Re=`56.3745 / 120 / 300`.
- Large checkpoint files are kept on the cluster under
  `/root/moe/V13/test_results_v13/results/*/*_checkpoint.pt` and are not
  committed to GitHub because each is hundreds of MB.
