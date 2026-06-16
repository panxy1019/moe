# Deep MoE-ROM v7 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=160, experts=6, top_k=2, expert_hidden=224.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.168561 | 0.208347 | - | 0.0597901 | - | - | - | - | - | - | - | - |
| 706.8965454101562 | Deep shared-routed MoE | 0.0739683 | 0.208347 | 0.0949244 | 0.0210418 | 0.01986 | 0.0959619 | 0.0828902 | 0.0755705 | 0.0998227 | 0.378273 | 1.04755 | 0 |
| 1000.0 | Galerkin only | 0.134105 | 0.158107 | - | 0.0495973 | - | - | - | - | - | - | - | - |
| 1000.0 | Deep shared-routed MoE | 0.061095 | 0.158107 | 0.0878059 | 0.0169561 | 0.0155968 | 0.087198 | 0.0767453 | 0.0639093 | 0.0779011 | 0.624864 | 0.854112 | 0 |

Runtime: 252.53 s.
