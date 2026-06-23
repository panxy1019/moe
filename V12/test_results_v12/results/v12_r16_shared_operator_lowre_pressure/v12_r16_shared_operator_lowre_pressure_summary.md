# Deep MoE-ROM v12 Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=160, routed_experts=6, shared_operator_experts=2, top_k=2, expert_hidden=256.

Shared/routed scales: 0.7 / 0.9; routed gate floor: 0.08.

Routed and shared experts output `residual` velocity operator targets plus a pressure `closure` branch using one shared regime router. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, short rollout, pressure closure, RHS/pressure relative terms, router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime router separation.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep operator-space MoE | 0.117205 | 2.72647 | 0.286531 | 0.128763 | 0.0961729 | 0.307844 | 3.01271 | 0.500076 | 0.5809 | 1.36752 | 0.903697 | 0 |

## Expert Diagnostics

| Test Re | shared always active | shared mixer mean weights | shared mixer entropy |
|---:|---|---|---:|
| 56.3745231628418 | True | [0.917, 0.083] | 0.28465 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.694107 | False | low_Re_lt_80: e4; mid_Re_80_160: e1; high_Re_ge_160: e5 |

Runtime: 326.66 s.
