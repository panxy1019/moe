# Deep MoE-ROM v13 Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=256, routed_experts=24, shared_operator_experts=4, top_k=2, expert_hidden=1024, expert_blocks=4, quadratic_rank=4.

Shared/routed scales: 0.65 / 1; routed gate floor: 0.

Routed and shared experts output `residual` velocity operator targets plus a pressure `state` branch using separate velocity/pressure regime routers. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, short rollout, pressure closure, RHS/pressure relative terms, router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime router separation.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep operator-space MoE | 0.116229 | 2.72647 | 0.32707 | 0.108276 | 0.108511 | 0.32707 | 0.684732 | 0.644349 | 0.651632 | 2.62535 | 1.21533 | 20 |
| 120.0 | Galerkin only | 0.173703 | 0.481705 | - | 0.290683 | - | - | - | - | - | - | - | - |
| 120.0 | Deep operator-space MoE | 0.116011 | 0.481705 | 0.0222117 | 0.0910069 | 0.0832528 | 0.0222117 | 0.432816 | 0.427953 | 0.322921 | 1.87272 | 1.32941 | 17 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - |
| 300.0 | Deep operator-space MoE | 0.0665813 | 0.337899 | 0.0440284 | 0.101374 | 0.0966915 | 0.0440284 | 0.613871 | 0.592707 | 0.485447 | 1.22894 | 1.19116 | 8 |

## Expert Diagnostics

| Test Re | shared always active | shared mixer mean weights | shared mixer entropy |
|---:|---|---|---:|
| 56.3745231628418 | True | [0.252, 0.263, 0.296, 0.189] | 1.37119 |
| 120.0 | True | [0.271, 0.360, 0.113, 0.256] | 1.2036 |
| 300.0 | True | [0.684, 0.092, 0.100, 0.124] | 0.879577 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.918273 | False | low_Re_lt_80: e15; mid_Re_80_160: e15; high_Re_ge_160: e22 |
| 120.0 | 0.819322 | False | low_Re_lt_80: e4; mid_Re_80_160: e4; high_Re_ge_160: e19 |
| 300.0 | 0.495568 | False | low_Re_lt_80: e8; mid_Re_80_160: e8; high_Re_ge_160: e20 |

Runtime: 6832.71 s.
