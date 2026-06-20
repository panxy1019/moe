# Deep MoE-ROM v9 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=144, experts=8, top_k=2, expert_hidden=224.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep shared-routed MoE | 0.10274 | 2.72647 | 0.384854 | 0.0692527 | 0.069281 | 0.394992 | 0.467166 | 0.367537 | 0.54555 | 1.17165 | 1.06085 | 2 |
| 120.0 | Galerkin only | 0.173703 | 0.481705 | - | 0.290683 | - | - | - | - | - | - | - | - |
| 120.0 | Deep shared-routed MoE | 0.0292154 | 0.481705 | 0.0325297 | 0.0915049 | 0.0600562 | 0.0732736 | 0.427925 | 0.352717 | 0.312959 | 0.288991 | 1.14347 | 0 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - |
| 300.0 | Deep shared-routed MoE | 0.0215603 | 0.337899 | 0.0919075 | 0.0996797 | 0.0837739 | 0.109275 | 0.524123 | 0.541505 | 0.614456 | 0.406284 | 0.951425 | 0 |

Runtime: 589.56 s.
