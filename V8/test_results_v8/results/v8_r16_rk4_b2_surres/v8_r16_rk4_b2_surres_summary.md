# Deep MoE-ROM v8 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=128, experts=6, top_k=2, expert_hidden=192.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep shared-routed MoE | 0.12747 | 2.72647 | 0.930386 | 0.0786197 | 0.091842 | 0.958997 | 0.351803 | 0.470558 | 1.18452 | 1.24807 | 0.923074 | 3 |
| 120.0 | Galerkin only | 0.173703 | 0.481705 | - | 0.290683 | - | - | - | - | - | - | - | - |
| 120.0 | Deep shared-routed MoE | 0.0322062 | 0.481705 | 0.0635662 | 0.0821359 | 0.0736001 | 0.0955439 | 0.40082 | 0.352074 | 0.329402 | 0.982033 | 0.782132 | 3 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - |
| 300.0 | Deep shared-routed MoE | 0.0276605 | 0.337899 | 0.0958405 | 0.101841 | 0.0881411 | 0.118821 | 0.471606 | 0.452237 | 0.596678 | 0.428857 | 1.09765 | 0 |

Runtime: 371.65 s.
