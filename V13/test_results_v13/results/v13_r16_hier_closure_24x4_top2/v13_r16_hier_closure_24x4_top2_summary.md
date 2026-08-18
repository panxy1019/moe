# Deep MoE-ROM v13 Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=256, routed_experts=24, shared_operator_experts=4, top_k=2, expert_hidden=1024, expert_blocks=4, quadratic_rank=4.

Shared/routed scales: 0.65 / 1; routed gate floor: 0.

Routed and shared experts output `residual` velocity operator targets plus a pressure `closure` branch using separate velocity/pressure regime routers. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, short rollout, pressure closure, RHS/pressure relative terms, router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime router separation.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep operator-space MoE | 0.119795 | 2.72647 | 0.273652 | 0.116275 | 0.0996054 | 0.300589 | 0.75123 | 0.597859 | 0.714219 | 2.87372 | 1.01877 | 21 |
| 120.0 | Galerkin only | 0.173703 | 0.481705 | - | 0.290683 | - | - | - | - | - | - | - | - |
| 120.0 | Deep operator-space MoE | 0.113894 | 0.481705 | 0.0188271 | 0.0915829 | 0.0825531 | 0.125244 | 0.410528 | 0.418137 | 0.462014 | 2.15276 | 1.29579 | 17 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - |
| 300.0 | Deep operator-space MoE | 0.0661398 | 0.337899 | 0.0426584 | 0.101143 | 0.0846297 | 0.0950158 | 0.586044 | 0.372106 | 0.432742 | 0.968157 | 1.1775 | 8 |

## Expert Diagnostics

| Test Re | shared always active | shared mixer mean weights | shared mixer entropy |
|---:|---|---|---:|
| 56.3745231628418 | True | [0.429, 0.163, 0.364, 0.044] | 1.15642 |
| 120.0 | True | [0.274, 0.186, 0.273, 0.267] | 1.25494 |
| 300.0 | True | [0.414, 0.143, 0.120, 0.324] | 1.10377 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.89672 | False | low_Re_lt_80: e1; mid_Re_80_160: e1; high_Re_ge_160: e20 |
| 120.0 | 0.856976 | False | low_Re_lt_80: e6; mid_Re_80_160: e2; high_Re_ge_160: e11 |
| 300.0 | 0.445853 | False | low_Re_lt_80: e8; mid_Re_80_160: e12; high_Re_ge_160: e20 |

Runtime: 7257.92 s.
