# Deep MoE-ROM v6 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=160, experts=6, top_k=2, expert_hidden=224.

Three heads: `alpha_next_head`, `rhs_correction_head`, and `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure prediction, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure-next L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.168561 | - | 0.0597901 | - | - | - | - | - | - | - | - |
| 706.8965454101562 | Deep shared-routed MoE | 0.0722033 | 0.332467 | 0.0208776 | 0.0191424 | 0.332467 | 0.0860634 | 0.0736302 | 0.241874 | 0.417958 | 1.02833 | 0 |
| 1000.0 | Galerkin only | 0.134105 | - | 0.0495973 | - | - | - | - | - | - | - | - |
| 1000.0 | Deep shared-routed MoE | 0.063484 | 0.460991 | 0.0175057 | 0.0160056 | 0.460991 | 0.0794293 | 0.0734756 | 0.368746 | 0.598251 | 0.847409 | 0 |

Runtime: 238.36 s.
