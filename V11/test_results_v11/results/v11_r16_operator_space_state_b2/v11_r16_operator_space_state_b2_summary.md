# Deep MoE-ROM v11 Summary

## Architecture

Shared encoder + 2 latent refinement blocks, hidden_dim=144, routed_experts=8, operator_spaces=0, top_k=2, expert_hidden=224.

Shared/routed scales: 1 / 0.75; routed gate floor: 0.1.

Experts output full standardized velocity RHS operators plus a pressure `state` branch using one shared regime router.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, short rollout, pressure closure, RHS/pressure relative terms, router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime router separation.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep operator-space MoE | 0.226883 | 2.72647 | 0.301817 | 0.36822 | 0.110884 | 0.301817 | 3.76191 | 0.513716 | 0.668784 | 1.89081 | 0.876136 | 0 |
| 120.0 | Galerkin only | 0.173703 | 0.481705 | - | 0.290683 | - | - | - | - | - | - | - | - |
| 120.0 | Deep operator-space MoE | 0.253182 | 0.481705 | 0.0615404 | 0.328482 | 0.0736939 | 0.0615404 | 1.36723 | 0.354867 | 0.260632 | 1.48887 | 0.915719 | 0 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - |
| 300.0 | Deep operator-space MoE | 0.20238 | 0.337899 | 0.109804 | 0.271224 | 0.121841 | 0.109805 | 1.1874 | 0.827936 | 0.522854 | 1.04708 | 0.865211 | 0 |

## Expert Diagnostics

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.666638 | False | low_Re_lt_80: e1; mid_Re_80_160: e5; high_Re_ge_160: e4 |
| 120.0 | 0.663731 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e6 |
| 300.0 | 0.723322 | False | low_Re_lt_80: e7; mid_Re_80_160: e0; high_Re_ge_160: e2 |

Runtime: 897.85 s.
