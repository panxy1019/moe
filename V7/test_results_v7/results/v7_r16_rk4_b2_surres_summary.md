# Deep MoE-ROM v7 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=128, experts=6, top_k=2, expert_hidden=192.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.211125 | 0.449212 | - | 0.063025 | - | - | - | - | - | - | - | - |
| 706.8965454101562 | Deep shared-routed MoE | 0.0495064 | 0.449212 | 0.179063 | 0.0151573 | 0.0126347 | 0.180211 | 0.108807 | 0.0862977 | 0.153664 | 0.77522 | 0.800766 | 0 |
| 1000.0 | Galerkin only | 0.131844 | 0.257762 | - | 0.0439991 | - | - | - | - | - | - | - | - |
| 1000.0 | Deep shared-routed MoE | 0.0698835 | 0.257762 | 0.12299 | 0.0178686 | 0.0149873 | 0.119559 | 0.0932754 | 0.064983 | 0.0921306 | 0.470769 | 0.961806 | 0 |

Runtime: 239.76 s.
