# Deep MoE-ROM v2 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=128, experts=6, top_k=2, expert_hidden=192.

Dual heads: `alpha_next_head` and `rhs_correction_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, alpha/RHS consistency, router load-balance and entropy.

## Metrics

| Test Re | Model | RHS relative L2 | one-step relative L2 | alpha-head relative L2 | rollout mean L2 | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|
| 700 | Galerkin only | 0.203423 | 0.0650337 | - | - | 0% |
| 700 | Deep shared-routed MoE | 0.110489 | 0.0494735 | 0.2023 | 0.446292 | 45.6851% |
| 1000 | Galerkin only | 0.17964 | 0.0575312 | - | - | 0% |
| 1000 | Deep shared-routed MoE | 0.0920856 | 0.0452443 | 0.345042 | 0.365528 | 48.7389% |

Runtime: 95.40 s.
