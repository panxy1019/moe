# Deep MoE-ROM v4 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=128, experts=8, top_k=2, expert_hidden=192.

Dual heads: `alpha_next_head` and `rhs_correction_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

| Test Re | Model | RHS relative L2 | one-step relative L2 | alpha-head relative L2 | gate smooth MSE | rollout mean L2 | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.211125 | 0.063025 | - | - | - | 0% |
| 706.8965454101562 | Deep shared-routed MoE | 0.0487769 | 0.0372165 | 0.204281 | 0.00396189 | 0.419239 | 76.8966% |
| 1000.0 | Galerkin only | 0.131844 | 0.0439991 | - | - | - | 0% |
| 1000.0 | Deep shared-routed MoE | 0.0711912 | 0.0361546 | 0.309629 | 0.00501848 | 0.380274 | 46.0034% |

Runtime: 558.27 s.
