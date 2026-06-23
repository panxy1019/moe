# Deep MoE-ROM v11 Summary

## Architecture

Shared encoder + 2 latent refinement blocks, hidden_dim=144, routed_experts=8, operator_spaces=0, top_k=2, expert_hidden=224.

Shared/routed scales: 1 / 0.75; routed gate floor: 0.1.

Experts output full standardized velocity RHS operators plus a pressure closure using one shared regime router.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, short rollout, pressure closure, RHS/pressure relative terms, router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime router separation.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep operator-space MoE | 0.211674 | 2.72647 | 0.362618 | 0.364408 | 0.0919348 | 0.38626 | 5.49423 | 0.736995 | 1.07537 | 1.64637 | 1.0031 | 0 |
| 120.0 | Galerkin only | 0.173703 | 0.481705 | - | 0.290683 | - | - | - | - | - | - | - | - |
| 120.0 | Deep operator-space MoE | 0.287706 | 0.481705 | 0.0639671 | 0.34833 | 0.114243 | 0.135455 | 2.92918 | 0.534633 | 0.925962 | 1.35162 | 0.979842 | 0 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - |
| 300.0 | Deep operator-space MoE | 0.199877 | 0.337899 | 0.0640809 | 0.247821 | 0.110078 | 0.0992873 | 1.5663 | 1.02988 | 0.839126 | 0.703721 | 0.865097 | 0 |

## Expert Diagnostics

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.908752 | False | low_Re_lt_80: e5; mid_Re_80_160: e5; high_Re_ge_160: e7 |
| 120.0 | 0.707414 | False | low_Re_lt_80: e0; mid_Re_80_160: e0; high_Re_ge_160: e2 |
| 300.0 | 0.732109 | False | low_Re_lt_80: e0; mid_Re_80_160: e0; high_Re_ge_160: e5 |

Runtime: 856.13 s.
