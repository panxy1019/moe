# Deep MoE-ROM v3 Summary

## Architecture

PhysicalContextEncoder + 1 Shared-Routed MoE blocks, hidden_dim=96, experts=4, top_k=2, expert_hidden=128.

Dual heads: `alpha_next_head` and `rhs_correction_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

| Test Re | Model | RHS relative L2 | one-step relative L2 | alpha-head relative L2 | gate smooth MSE | rollout mean L2 | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|---:|
| 700 | Galerkin only | 0.203349 | 0.0650657 | - | - | - | 0% |
| 700 | Deep shared-routed MoE | 0.109822 | 0.0471754 | 0.227408 | 0.00943511 | 0.480399 | 45.9934% |
| 1000 | Galerkin only | 0.179599 | 0.0575607 | - | - | - | 0% |
| 1000 | Deep shared-routed MoE | 0.0766095 | 0.0430989 | 0.329851 | 0.00662815 | 0.360704 | 57.3443% |

Runtime: 243.20 s.
