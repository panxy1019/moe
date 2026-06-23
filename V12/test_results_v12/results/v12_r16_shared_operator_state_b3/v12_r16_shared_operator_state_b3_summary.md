# Deep MoE-ROM v12 Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=160, routed_experts=12, shared_operator_experts=2, top_k=3, expert_hidden=256.

Shared/routed scales: 0.55 / 1; routed gate floor: 0.06.

Routed and shared experts output `residual` velocity operator targets plus a pressure `state` branch using one shared regime router. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, short rollout, pressure closure, RHS/pressure relative terms, router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime router separation.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep operator-space MoE | 0.110391 | 2.72647 | 0.345813 | 0.0903983 | 0.0923836 | 0.345813 | 0.564489 | 0.526315 | 0.622502 | 2.42631 | 0.92703 | 9 |
| 120.0 | Galerkin only | 0.173703 | 0.481705 | - | 0.290683 | - | - | - | - | - | - | - | - |
| 120.0 | Deep operator-space MoE | 0.113775 | 0.481705 | 0.0374743 | 0.11312 | 0.0704434 | 0.0374743 | 0.399801 | 0.364321 | 0.310578 | 1.39863 | 1.13365 | 7 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - |
| 300.0 | Deep operator-space MoE | 0.0653583 | 0.337899 | 0.10688 | 0.107053 | 0.0848267 | 0.10688 | 0.584913 | 0.523343 | 0.528297 | 1.17298 | 0.917956 | 4 |

## Expert Diagnostics

| Test Re | shared always active | shared mixer mean weights | shared mixer entropy |
|---:|---|---|---:|
| 56.3745231628418 | True | [0.201, 0.799] | 0.500515 |
| 120.0 | True | [0.597, 0.403] | 0.453604 |
| 300.0 | True | [0.772, 0.228] | 0.439068 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.516628 | False | low_Re_lt_80: e4; mid_Re_80_160: e1; high_Re_ge_160: e3 |
| 120.0 | 0.28577 | False | low_Re_lt_80: e4; mid_Re_80_160: e8; high_Re_ge_160: e10 |
| 300.0 | 0.505724 | False | low_Re_lt_80: e0; mid_Re_80_160: e0; high_Re_ge_160: e9 |

Runtime: 1127.09 s.
