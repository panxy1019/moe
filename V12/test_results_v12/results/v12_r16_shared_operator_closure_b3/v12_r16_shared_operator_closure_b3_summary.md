# Deep MoE-ROM v12 Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=160, routed_experts=12, shared_operator_experts=2, top_k=3, expert_hidden=256.

Shared/routed scales: 0.55 / 1; routed gate floor: 0.06.

Routed and shared experts output `residual` velocity operator targets plus a pressure `closure` branch using one shared regime router. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, short rollout, pressure closure, RHS/pressure relative terms, router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime router separation.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep operator-space MoE | 0.117531 | 2.72647 | 0.294464 | 0.142074 | 0.091258 | 0.30738 | 1.18442 | 0.439722 | 0.71362 | 1.93894 | 1.19077 | 9 |
| 120.0 | Galerkin only | 0.173703 | 0.481705 | - | 0.290683 | - | - | - | - | - | - | - | - |
| 120.0 | Deep operator-space MoE | 0.115111 | 0.481705 | 0.0260723 | 0.110839 | 0.0748447 | 0.122918 | 0.426 | 0.371371 | 0.477895 | 1.46892 | 1.14549 | 7 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - |
| 300.0 | Deep operator-space MoE | 0.0646944 | 0.337899 | 0.0587561 | 0.108003 | 0.0833336 | 0.11499 | 0.580056 | 0.529503 | 0.751965 | 0.79442 | 0.936202 | 4 |

## Expert Diagnostics

| Test Re | shared always active | shared mixer mean weights | shared mixer entropy |
|---:|---|---|---:|
| 56.3745231628418 | True | [0.606, 0.394] | 0.663773 |
| 120.0 | True | [0.377, 0.623] | 0.572904 |
| 300.0 | True | [0.689, 0.311] | 0.436026 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.848474 | False | low_Re_lt_80: e11; mid_Re_80_160: e3; high_Re_ge_160: e7 |
| 120.0 | 0.392102 | False | low_Re_lt_80: e8; mid_Re_80_160: e2; high_Re_ge_160: e5 |
| 300.0 | 0.473445 | False | low_Re_lt_80: e9; mid_Re_80_160: e6; high_Re_ge_160: e0 |

Runtime: 929.57 s.
