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
| 56.3745231628418 | Deep operator-space MoE | 0.21451 | 2.72647 | 0.32506 | 0.359222 | 0.113814 | 0.32506 | 4.73641 | 0.850621 | 1.03762 | 1.94605 | 0.842945 | 0 |

## Expert Diagnostics

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.738763 | False | low_Re_lt_80: e1; mid_Re_80_160: e1; high_Re_ge_160: e7 |

Runtime: 307.11 s.
