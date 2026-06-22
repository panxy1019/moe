# Deep MoE-ROM v10 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=144, routed_experts=8, shared_experts=2, top_k=2, expert_hidden=224.

Shared/routed scales: 1 / 0.55; routed gate floor: 0.12.

Three heads: `alpha_next_head`, `rhs_correction_head`, and residual `pressure_next_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, pressure-surrogate residual, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - |
| 56.3745231628418 | Deep shared-routed MoE | 0.0784657 | 2.72647 | 0.283692 | 0.0844645 | 0.0663608 | 0.289178 | 3.46959e+07 | 0.40332 | 0.543825 | 0.932987 | 1.37395 | 0 |
| 120.0 | Galerkin only | 0.173703 | 0.481705 | - | 0.290683 | - | - | - | - | - | - | - | - |
| 120.0 | Deep shared-routed MoE | 0.0284546 | 0.481705 | 0.0207798 | 0.110854 | 0.0546039 | 0.0636706 | 0.552248 | 0.327383 | 0.311903 | 0.543456 | 1.33234 | 0 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - |
| 300.0 | Deep shared-routed MoE | 0.0169312 | 0.337899 | 0.0649567 | 0.100768 | 0.0794978 | 0.0872806 | 0.481183 | 0.49645 | 0.587781 | 0.321574 | 1.38385 | 0 |

Runtime: 1079.49 s.
