# V12 Shared-Expert Operator-Space MoE-ROM

V12 upgrades the V11 operator-space MoE into a shared-expert physical-anchor
MoE-ROM. The model keeps RK4 time integration and uses the Galerkin RHS as a
shared physical operator. Learned shared experts are always active and capture
common closure; routed experts capture local flow-regime operators.

Main files:

- `test_results_v12/deep_moe_rom_v12.py`: V12 training/evaluation script.
- `test_results_v12/run_v12_shared_operator_smoke.sh`: smoke test.
- `test_results_v12/run_v12_shared_operator.sh`: closure-pressure main run.
- `test_results_v12/run_v12_shared_operator_state.sh`: state-pressure run.
- `test_results_v12/run_v12_shared_operator_lowre_pressure.sh`: low-Re pressure ablation.
- `TECHNICAL_REPORT_V12.md`: experiment report and V13 recommendations.

Best V12 main result:

- `v12_r16_shared_operator_closure_b3` enables 12 routed experts, 2 shared
  experts, top-k=3 routing, and residual operator closure added to Galerkin RHS.
- Velocity autonomous one-step relative L2 reaches `<10%` at all held-out Re.
- Pressure direct relative L2 reaches `<10%` at Re=120 and Re=300, but low
  Re=56.3745 remains at `29.45%`.

The report concludes that the remaining bottleneck is low-Re pressure
generalization, not expert collapse. The recommended next step is a
pressure-specific structured operator with a local Re-band pressure anchor.
