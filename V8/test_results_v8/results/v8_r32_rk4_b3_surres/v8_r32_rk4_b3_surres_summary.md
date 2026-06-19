# Deep MoE-ROM v8 Summary

## Architecture

PhysicalContextEncoder + 3 Shared-Routed MoE blocks, hidden_dim=128, experts=6, top_k=2, expert_hidden=192.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.231476 | 5.30637 | - | 0.375609 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep shared-routed MoE | 0.278137 | 5.30637 | 2.079 | 0.124107 | 0.144675 | 2.08117 | 0.558972 | 0.816873 | 2.67639 | 1.02607 | 1.09505 | 1 |
| 120.0 | Galerkin only | 0.196661 | 0.427395 | - | 0.306466 | - | - | - | - | - | - | - | - |
| 120.0 | Deep shared-routed MoE | 0.0840905 | 0.427395 | 0.0719112 | 0.104145 | 0.0927079 | 0.110728 | 0.364502 | 0.320509 | 0.380124 | 0.560935 | 1.12221 | 0 |
| 300.0 | Galerkin only | 0.289981 | 0.251192 | - | 0.399128 | - | - | - | - | - | - | - | - |
| 300.0 | Deep shared-routed MoE | 0.0881092 | 0.251192 | 0.118922 | 0.149227 | 0.135274 | 0.174156 | 0.537995 | 0.480393 | 0.664667 | 0.455744 | 1.21428 | 0 |

Runtime: 380.86 s.
