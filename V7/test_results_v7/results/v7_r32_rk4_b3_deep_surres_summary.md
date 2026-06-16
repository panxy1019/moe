# Deep MoE-ROM v7 Summary

## Architecture

PhysicalContextEncoder + 3 Shared-Routed MoE blocks, hidden_dim=192, experts=8, top_k=2, expert_hidden=256.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.168561 | 0.208347 | - | 0.0597901 | - | - | - | - | - | - | - | - |
| 706.8965454101562 | Deep shared-routed MoE | 0.0733929 | 0.208347 | 0.102559 | 0.021543 | 0.0192445 | 0.1032 | 0.0835994 | 0.0655373 | 0.0861336 | 0.595258 | 1.22951 | 0 |
| 1000.0 | Galerkin only | 0.134105 | 0.158107 | - | 0.0495973 | - | - | - | - | - | - | - | - |
| 1000.0 | Deep shared-routed MoE | 0.0567902 | 0.158107 | 0.080494 | 0.0168983 | 0.0147655 | 0.0795825 | 0.0702524 | 0.0632755 | 0.0761294 | 0.255263 | 1.3759 | 0 |

Runtime: 302.04 s.
