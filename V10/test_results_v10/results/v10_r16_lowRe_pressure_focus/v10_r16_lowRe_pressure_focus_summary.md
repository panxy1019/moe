# Deep MoE-ROM v10 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=144, routed_experts=8, shared_experts=2, top_k=2, expert_hidden=224.

Shared/routed scales: 1 / 0.55; routed gate floor: 0.12.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep shared-routed MoE | 0.0791339 | 2.72647 | 0.287155 | 0.0973961 | 0.0857984 | 0.301697 | nan | 0.464596 | 0.591496 | 0.702608 | 1.45698 | 0 |

Runtime: 312.53 s.
