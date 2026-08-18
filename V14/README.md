# V14 HPRS-MoE-ROM

V14 upgrades V13 into a Hierarchical Physics-Regime Sparse Mixture of Experts
(HPRS-MoE) ROM:

- shared encoder + Galerkin + RK4 backbone retained;
- shared group router selects Low-Re / Transition / High-Re style regimes;
- each regime group owns one shared expert and several group-local routed
  experts;
- velocity and pressure share the same regime router while keeping separate
  in-group Top-2 routers;
- physics-aware expert heads with linear state terms and low-rank quadratic
  state interactions;
- closed-loop 8-16 step rollout training with scheduled sampling;
- energy/trajectory consistency losses and expert activation diagnostics.

Main files:

- `docs/v14_training_flow.svg`: HPRS-MoE closed-loop training framework.
- `docs/v14_inference_flow.svg`: autonomous inference/rollout framework.
- `docs/v14_training_flow.mmd`: Mermaid source for the training diagram.
- `docs/v14_inference_flow.mmd`: Mermaid source for the inference diagram.
- `test_results_v14/train_v14.py`: V14 training/evaluation script.
- `test_results_v14/evaluate.py`: compact metrics aggregator.
- `test_results_v14/monitor_routing.py`: group/router utilization reader.
- `test_results_v14/run_v14_hprs_smoke.sh`: smoke test.
- `test_results_v14/run_v14_hprs_closed_loop.sh`: closure-pressure main run.
- `test_results_v14/run_v14_hprs_state_pressure.sh`: state-pressure comparison.
- `test_results_v14/run_v14_hprs_lowre_pressure.sh`: low-Re pressure ablation.
- `test_results_v14/run_v14_hprs_lowre_state_pressure.sh`: low-Re
  state-pressure targeted run.
- `TECHNICAL_REPORT_V14.md`: final experiment report.

Large checkpoint files are kept on the cluster under
`/root/moe/V14/test_results_v14/results/*/*_checkpoint.pt` and are not
committed to GitHub because each can be hundreds of MB.
