# V11 Operator-Space MoE-ROM

V11 converts the V10 residual-correction MoE into a minimal operator-space MoE.
The shared encoder and regime router read the current ROM state, Reynolds
number, phase, physics descriptors, and short history. Routed experts output
local velocity RHS operators directly, and RK4 remains the velocity time
integrator.

Main files:

- `test_results_v11/deep_moe_rom_v11.py`: V11 training/evaluation script.
- `test_results_v11/run_v11_operator_space.sh`: closure-pressure main run.
- `test_results_v11/run_v11_operator_space_state.sh`: state-pressure main run.
- `test_results_v11/results/v11_r16_operator_space_b2/`: closure-pressure results.
- `test_results_v11/results/v11_r16_operator_space_state_b2/`: state-pressure results.
- `TECHNICAL_REPORT_V11.md`: experiment report and comparison with V10.

Summary:

- The implementation runs end-to-end with RK4 and records expert load, Re-group
  routing, per-expert errors, and expert-output cosine diversity.
- No dead experts were observed under the 1% load threshold.
- Expert-output collapse was not observed (`max |cos| < 0.95` in all runs).
- The `<10%` pressure target was not reached for low Re. Best low-Re pressure
  relative L2 in V11 was `0.2921` in the single-split state-pressure ablation.
