# Deep MoE-ROM v2 Summary

## Architecture

PhysicalContextEncoder + 3 Shared-Routed MoE blocks, hidden_dim=160, experts=8, top_k=2, expert_hidden=224.

Dual heads: `alpha_next_head` and `rhs_correction_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, alpha/RHS consistency, router load-balance and entropy.

## Metrics

| Test Re | Model | RHS relative L2 | one-step relative L2 | alpha-head relative L2 | rollout mean L2 | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|
| 700 | Galerkin only | 0.203423 | 0.0650337 | - | - | 0% |
| 700 | Deep shared-routed MoE | 0.101948 | 0.0485415 | 0.221672 | 0.457562 | 49.8839% |
| 1000 | Galerkin only | 0.17964 | 0.0575312 | - | - | 0% |
| 1000 | Deep shared-routed MoE | 0.0816702 | 0.0437758 | 0.311927 | 0.361526 | 54.5368% |

Runtime: 131.41 s.
