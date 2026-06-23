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
| 56.3745231628418 | Deep operator-space MoE | 0.216808 | 2.72647 | 0.717162 | 0.111661 | 0.123844 | 0.777685 | 0.200047 | 0.229603 | 0.798378 | 1.56306 | 1.30561 | 8 |

## Expert Diagnostics

| Test Re | shared always active | shared mixer mean weights | shared mixer entropy |
|---:|---|---|---:|
| 56.3745231628418 | True | [0.570, 0.430] | 0.676864 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.700983 | False | low_Re_lt_80: e3; mid_Re_80_160: e3; high_Re_ge_160: e7 |

Runtime: 168.82 s.
