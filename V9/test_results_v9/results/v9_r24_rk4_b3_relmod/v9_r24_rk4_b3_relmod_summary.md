# Deep MoE-ROM v9 Summary

## Architecture

PhysicalContextEncoder + 3 Shared-Routed MoE blocks, hidden_dim=144, experts=8, top_k=2, expert_hidden=224.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.211386 | 5.31912 | - | 0.356961 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep shared-routed MoE | 0.148603 | 5.31912 | 1.24252 | 0.0840075 | 0.0923533 | 1.24375 | 0.391714 | 0.421962 | 1.24932 | 1.05162 | 1.31726 | 0 |
| 120.0 | Galerkin only | 0.198021 | 0.475544 | - | 0.302331 | - | - | - | - | - | - | - | - |
| 120.0 | Deep shared-routed MoE | 0.0748208 | 0.475544 | 0.047507 | 0.119182 | 0.0796344 | 0.0909864 | 0.447312 | 0.415994 | 0.433978 | 0.212306 | 1.4238 | 0 |
| 300.0 | Galerkin only | 0.290992 | 0.289846 | - | 0.396912 | - | - | - | - | - | - | - | - |
| 300.0 | Deep shared-routed MoE | 0.0764767 | 0.289846 | 0.0987859 | 0.157184 | 0.130629 | 0.152746 | 0.548934 | 0.486644 | 0.629951 | 0.343138 | 1.32442 | 0 |

Runtime: 603.33 s.
