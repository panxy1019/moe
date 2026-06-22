# Deep MoE-ROM v10 Summary

## Architecture

PhysicalContextEncoder + 1 Shared-Routed MoE blocks, hidden_dim=32, routed_experts=4, shared_experts=2, top_k=2, expert_hidden=48.

Shared/routed scales: 1 / 0.6; routed gate floor: 0.12.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.233696 | 5.04139 | - | 0.362865 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep shared-routed MoE | 0.251001 | 5.04139 | 1.00586 | 0.111978 | 0.111417 | 1.03784 | 0.259312 | 0.257206 | 1.14576 | 0.907119 | 0.885662 | 0 |

Runtime: 117.22 s.
