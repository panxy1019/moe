# V11 Test Results

Scripts:

- `deep_moe_rom_v11.py`: operator-space MoE-ROM trainer/evaluator.
- `run_v11_operator_space_smoke.sh`: 2-epoch smoke test.
- `run_v11_operator_space.sh`: closure-pressure three-split run.
- `run_v11_operator_space_state.sh`: state-pressure three-split run.

Result directories:

- `smoke/`: smoke test output.
- `results/v11_r16_operator_space_b2/`: closure-pressure main experiment.
- `results/v11_r16_operator_space_state_b2/`: state-pressure main experiment.
- `results/v11_r16_operator_space_state_lowRe/`: low-Re state-pressure ablation.
- `results/v11_r16_operator_space_state_amp_lowRe/`: low-Re amplitude-weighted ablation.

Recommended V11 reference:

- Use `v11_r16_operator_space_state_b2` when pressure rollout is more important.
- Use `v11_r16_operator_space_b2` when high-Re one-step pressure is more important.
