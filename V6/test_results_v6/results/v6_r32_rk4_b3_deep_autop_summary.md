# Deep MoE-ROM v6 Summary

## Architecture

PhysicalContextEncoder + 3 Shared-Routed MoE blocks, hidden_dim=192, experts=8, top_k=2, expert_hidden=256.

Three heads: `alpha_next_head`, `rhs_correction_head`, and `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure prediction, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure-next L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.168561 | - | 0.0597901 | - | - | - | - | - | - | - | - |
| 706.8965454101562 | Deep shared-routed MoE | 0.0727777 | 0.281481 | 0.0206972 | 0.0190788 | 0.281481 | 0.0868511 | 0.0648288 | 0.236345 | 0.489897 | 1.32399 | 0 |
| 1000.0 | Galerkin only | 0.134105 | - | 0.0495973 | - | - | - | - | - | - | - | - |
| 1000.0 | Deep shared-routed MoE | 0.0571636 | 0.426688 | 0.0171249 | 0.0148199 | 0.426688 | 0.0711506 | 0.0671714 | 0.227352 | 0.290717 | 1.36022 | 0 |

Runtime: 286.24 s.
