# Deep MoE-ROM v13 Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=256, routed_experts=16, shared_operator_experts=4, top_k=2, expert_hidden=1024, expert_blocks=4, quadratic_rank=4.

Shared/routed scales: 0.8 / 0.9; routed gate floor: 0.

Routed and shared experts output `residual` velocity operator targets plus a pressure `closure` branch using separate velocity/pressure regime routers. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, short rollout, pressure closure, RHS/pressure relative terms, router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime router separation.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep operator-space MoE | 0.120881 | 2.72647 | 0.279935 | 0.115558 | 0.102397 | 0.300686 | 0.745175 | 0.63054 | 0.751022 | 2.0325 | 1.22265 | 12 |

## Expert Diagnostics

| Test Re | shared always active | shared mixer mean weights | shared mixer entropy |
|---:|---|---|---:|
| 56.3745231628418 | True | [0.210, 0.231, 0.105, 0.454] | 1.25122 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.895418 | False | low_Re_lt_80: e15; mid_Re_80_160: e8; high_Re_ge_160: e5 |

Runtime: 1749.34 s.
