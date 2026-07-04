# V15 Physics-Generalizable

V15 runs the V14 best HPRS-MoE + Galerkin + RK4 pressure-closure framework on
the new `ROM_PhysicsGeneralizable` Re=20-200 database.

The three experiments are isolated under:

- `V15_Base`: ru=16, rp=16, dense training.
- `V15_LargeROM`: ru=32, rp=32, same training and hyperparameters.
- `V15_BalancedTraining`: ru=16, rp=16, same hyperparameters with regime-balanced mini-batch sampling.

Default held-out Re values are selected by nearest database Re to:

```text
24, 32, 40, 45, 47, 49, 52, 70, 100, 150, 190
```

This covers steady wake, pre-Hopf steady, Hopf transition, developing periodic
shedding, mature periodic shedding, and high-Re 2D periodic regimes.

Run all three cases on the cluster:

```bash
export SWANLAB_API_KEY=...
bash /root/moe/V15_PhysicsGeneralizable/test_results_v15/run_v15_all.sh
```

SwanLab project: `V15_PhysicsGeneralizable`.

The final aggregate report is written after all three cases finish:

```text
test_results_v15/results/V15_summary/TECHNICAL_REPORT_V15_PHYSICS_GENERALIZABLE.md
```

Raw checkpoints, full metrics JSON, logs, and SwanLab cache are ignored by Git.
