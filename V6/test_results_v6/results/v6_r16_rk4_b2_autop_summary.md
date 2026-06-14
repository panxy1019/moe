# Deep MoE-ROM v6 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=128, experts=6, top_k=2, expert_hidden=192.

Three heads: `alpha_next_head`, `rhs_correction_head`, and `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure prediction, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure-next L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.211125 | - | 0.063025 | - | - | - | - | - | - | - | - |
| 706.8965454101562 | Deep shared-routed MoE | 0.0511564 | 0.222601 | 0.0153766 | 0.0128874 | 0.222601 | 0.114264 | 0.0821695 | 0.234134 | 0.571965 | 0.910739 | 0 |
| 1000.0 | Galerkin only | 0.131844 | - | 0.0439991 | - | - | - | - | - | - | - | - |
| 1000.0 | Deep shared-routed MoE | 0.0710443 | 0.37884 | 0.0179666 | 0.0149102 | 0.37884 | 0.0942885 | 0.0845496 | 0.343437 | 0.480212 | 0.958113 | 0 |

Runtime: 222.88 s.
