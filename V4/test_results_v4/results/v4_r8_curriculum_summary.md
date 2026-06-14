# Deep MoE-ROM v4 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=96, experts=6, top_k=2, expert_hidden=128.

Dual heads: `alpha_next_head` and `rhs_correction_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

| Test Re | Model | RHS relative L2 | one-step relative L2 | alpha-head relative L2 | gate smooth MSE | rollout mean L2 | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.153112 | 0.0587276 | - | - | - | 0% |
| 706.8965454101562 | Deep shared-routed MoE | 0.0460573 | 0.0420849 | 0.0896233 | 0.00573116 | 0.647908 | 69.9193% |
| 1000.0 | Galerkin only | 0.178619 | 0.0521943 | - | - | - | 0% |
| 1000.0 | Deep shared-routed MoE | 0.0907608 | 0.0397504 | 0.265987 | 0.00398087 | 0.396159 | 49.1874% |

Runtime: 547.31 s.
