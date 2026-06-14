# Deep MoE-ROM v6 Summary

## Architecture

PhysicalContextEncoder + 3 Shared-Routed MoE blocks, hidden_dim=160, experts=8, top_k=2, expert_hidden=224.

Three heads: `alpha_next_head`, `rhs_correction_head`, and `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure prediction, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure-next L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.211125 | - | 0.063025 | - | - | - | - | - | - | - | - |
| 706.8965454101562 | Deep shared-routed MoE | 0.0549753 | 0.241275 | 0.0160328 | 0.0133677 | 0.241275 | 0.100626 | 0.0783949 | 0.204854 | 0.402917 | 1.33925 | 0 |
| 1000.0 | Galerkin only | 0.131844 | - | 0.0439991 | - | - | - | - | - | - | - | - |
| 1000.0 | Deep shared-routed MoE | 0.0748661 | 0.388222 | 0.0174733 | 0.0157253 | 0.388222 | 0.073489 | 0.0675273 | 0.232963 | 0.330871 | 1.29734 | 0 |

Runtime: 263.61 s.
