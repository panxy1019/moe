# V15_2 Hopf Diagnostic

V15_2 is a diagnostic-only experiment. It does not modify the HPRS-MoE
architecture or retrain any weights. It loads the finished
`V15_1_AdaptiveGate` checkpoint and analyzes 24-step autonomous RK4 rollouts on
the held-out Reynolds numbers.

The goal is to identify whether the hard Hopf case near `Re=51.786` is dominated
by amplitude error, phase error, frequency error, or insufficient POD projection
capacity.

Run on the cluster:

```bash
cd /root/moe/V15_2_HopfDiagnostic
./run_v15_2_hopf_diagnostic.sh
```

Outputs are written under `results/`:

- `v15_2_per_re_summary.csv`
- `v15_2_per_window_metrics.csv`
- `v15_2_pod_projection_summary.csv`
- `figures/*_hopf_diagnostic.svg`
- `timeseries/*_representative.csv`
- `V15_2_HOPF_DIAGNOSTIC_REPORT.md`

Raw lift/drag coefficient time series are not present in the retained
ROM_PhysicsGeneralizable POD database, so the report explicitly marks true
Lift/Drag phase/amplitude/frequency errors as unavailable rather than deriving
unverifiable proxies.
