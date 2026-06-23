# V12 Test Results

Scripts:

- `deep_moe_rom_v12.py`: shared-expert operator-space MoE-ROM trainer/evaluator.
- `run_v12_shared_operator_smoke.sh`: 2-epoch smoke test.
- `run_v12_shared_operator.sh`: closure-pressure three-split main experiment.
- `run_v12_shared_operator_state.sh`: state-pressure three-split experiment.
- `run_v12_shared_operator_lowre_pressure.sh`: low-Re pressure-focused ablation.

Result directories:

- `smoke/`: smoke test output.
- `results/v12_r16_shared_operator_closure_b3/`: best V12 main experiment.
- `results/v12_r16_shared_operator_state_b3/`: state-pressure comparison.
- `results/v12_r16_shared_operator_lowre_pressure/`: low-Re pressure ablation.

Recommended V12 reference:

- Use `v12_r16_shared_operator_closure_b3` as the main baseline. It gives the
  best all-Re balance and puts velocity autonomous one-step error below 10%.
- Use `v12_r16_shared_operator_state_b3` only when pressure rollout is more
  important than direct one-step pressure.
