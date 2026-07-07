# V16 AttractorMoE-ROM

V16 redefines the Re=20-200 Physics-Generalizable retained data as a
Physics-Generalizable Attractor Database. The retained windows contain steady
fixed-point, weak Hopf, developing periodic, and mature periodic attractors,
rather than complete Hopf onset-to-saturation transients.

## Experiments

- `V16_AttractorMoEROM_FullRegimeLoss32`: ru=32/rp=32, modal AdaptiveGate
  pressure closure, attractor-balanced sampling, and Steady/Hopf/Periodic
  attractor-specific losses. The HPRS-MoE-ROM topology is unchanged.
- `V16_AttractorMoEROM_AttractorConditionedFramework32`: adds an explicit
  Attractor Router and three lightweight latent adapters after the shared
  encoder, then enables the same attractor-specific losses plus prototype
  energy/radius constraints.

Both cases keep Galerkin ROM, RK4, Pressure Poisson Surrogate, pressure head,
velocity/pressure experts, optimizer, batch size, epoch schedule, and held-out
Re split aligned with V15_3.

## Run

```bash
cd /root/moe/V16_AttractorMoEROM/test_results_v16
export SWANLAB_API_KEY=...
export SWANLAB_TRACKING_MODE=online
export SWANLAB_TRACKING_PROJECT=V16_AttractorMoEROM
V16_WAIT_FOR_IDLE=1 ./run_v16_all.sh
```

`V16_WAIT_FOR_IDLE=1` waits for the current V15_3 training process to finish
before launching the two V16 cases in parallel.
