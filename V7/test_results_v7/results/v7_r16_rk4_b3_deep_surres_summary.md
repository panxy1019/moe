# Deep MoE-ROM v7 Summary

## Architecture

PhysicalContextEncoder + 3 Shared-Routed MoE blocks, hidden_dim=160, experts=8, top_k=2, expert_hidden=224.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.211125 | 0.449212 | - | 0.063025 | - | - | - | - | - | - | - | - |
| 706.8965454101562 | Deep shared-routed MoE | 0.0553476 | 0.449212 | 0.165456 | 0.0161469 | 0.0133712 | 0.166641 | 0.101507 | 0.0740774 | 0.131273 | 0.442195 | 1.32612 | 0 |
| 1000.0 | Galerkin only | 0.131844 | 0.257762 | - | 0.0439991 | - | - | - | - | - | - | - | - |
| 1000.0 | Deep shared-routed MoE | 0.0745436 | 0.257762 | 0.11253 | 0.0175329 | 0.0157356 | 0.109725 | 0.0746354 | 0.0599284 | 0.0830796 | 0.571889 | 1.18582 | 0 |

Runtime: 298.63 s.
