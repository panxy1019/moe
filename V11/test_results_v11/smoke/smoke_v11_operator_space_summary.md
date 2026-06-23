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
| 56.3745231628418 | Deep operator-space MoE | 1.82172 | 2.72647 | 1.14487 | 1.46381 | 1.4741 | 2.65926 | 4.13574 | 3.95345 | 10.6089 | 1.69687 | 0.974121 | 0 |

## Expert Diagnostics

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.661386 | False | low_Re_lt_80: e5; mid_Re_80_160: e5; high_Re_ge_160: e5 |

Runtime: 165.54 s.
