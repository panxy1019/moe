# Deep MoE-ROM v2 Summary

## Architecture

PhysicalContextEncoder + 1 Shared-Routed MoE blocks, hidden_dim=96, experts=4, top_k=2, expert_hidden=128.

Dual heads: `alpha_next_head` and `rhs_correction_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, alpha/RHS consistency, router load-balance and entropy.

## Metrics

| Test Re | Model | RHS relative L2 | one-step relative L2 | alpha-head relative L2 | rollout mean L2 | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|
| 700 | Galerkin only | 0.203423 | 0.0650337 | - | - | 0% |
| 700 | Deep shared-routed MoE | 0.103079 | 0.0470508 | 0.207483 | 0.442404 | 49.3278% |
| 1000 | Galerkin only | 0.17964 | 0.0575312 | - | - | 0% |
| 1000 | Deep shared-routed MoE | 0.0849238 | 0.0445818 | 0.311819 | 0.356669 | 52.7256% |

Runtime: 75.37 s.
