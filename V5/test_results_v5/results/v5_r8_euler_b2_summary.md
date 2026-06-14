# Deep MoE-ROM v5 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=96, experts=6, top_k=2, expert_hidden=128.

Dual heads: `alpha_next_head` and `rhs_correction_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `euler`.

| Test Re | Model | RHS relative L2 | Euler one-step L2 | Integrator one-step L2 | rollout mean L2 | load CV | entropy | dead experts | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.153112 | 0.0587276 | - | - | - | - | - | 0% |
| 706.8965454101562 | Deep shared-routed MoE | 0.0426595 | 0.0417162 | 0.0437115 | 0.561969 | 0.428881 | 0.982304 | 0 | 72.1385% |
| 1000.0 | Galerkin only | 0.178619 | 0.0521943 | - | - | - | - | - | 0% |
| 1000.0 | Deep shared-routed MoE | 0.0923349 | 0.0400369 | 0.0419497 | 0.58285 | 0.264922 | 0.949287 | 0 | 48.3061% |

Runtime: 346.91 s.
