# Deep MoE-ROM v13 Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=128, routed_experts=8, shared_operator_experts=2, top_k=2, expert_hidden=512, expert_blocks=2, quadratic_rank=2.

Shared/routed scales: 0.65 / 1; routed gate floor: 0.

Routed and shared experts output `residual` velocity operator targets plus a pressure `closure` branch using separate velocity/pressure regime routers. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, short rollout, pressure closure, RHS/pressure relative terms, router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime router separation.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep operator-space MoE | 0.217775 | 2.72647 | 0.719274 | 0.113368 | 0.114213 | 0.66845 | 0.198127 | 0.200168 | 0.805221 | 1.05036 | 1.3553 | 4 |

## Expert Diagnostics

| Test Re | shared always active | shared mixer mean weights | shared mixer entropy |
|---:|---|---|---:|
| 56.3745231628418 | True | [0.506, 0.494] | 0.690696 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.596103 | False | low_Re_lt_80: e5; mid_Re_80_160: e2; high_Re_ge_160: e6 |

Runtime: 207.52 s.
