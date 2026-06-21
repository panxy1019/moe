# Deep MoE-ROM v9(2) Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=144, experts=8, top_k=2, expert_hidden=224.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep shared-routed MoE | 0.0876351 | 2.72647 | 0.362332 | 0.0853662 | 0.0602834 | 0.369096 | 9.13562 | 0.331366 | 0.510604 | 1.2767 | 1.00197 | 2 |
| 120.0 | Galerkin only | 0.173703 | 0.481705 | - | 0.290683 | - | - | - | - | - | - | - | - |
| 120.0 | Deep shared-routed MoE | 0.0250387 | 0.481705 | 0.0220476 | 0.102915 | 0.0567397 | 0.0640145 | 0.48883 | 0.35683 | 0.323518 | 0.396416 | 1.15095 | 0 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - |
| 300.0 | Deep shared-routed MoE | 0.017273 | 0.337899 | 0.071738 | 0.0998564 | 0.0849091 | 0.0935125 | 0.502006 | 0.589806 | 0.656814 | 0.427097 | 0.986829 | 0 |

Runtime: 809.36 s.
