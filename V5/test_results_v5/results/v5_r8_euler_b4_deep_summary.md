# Deep MoE-ROM v5 Summary

## Architecture

PhysicalContextEncoder + 4 Shared-Routed MoE blocks, hidden_dim=128, experts=8, top_k=2, expert_hidden=192.

Dual heads: `alpha_next_head` and `rhs_correction_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `euler`.

| Test Re | Model | RHS relative L2 | Euler one-step L2 | Integrator one-step L2 | rollout mean L2 | load CV | entropy | dead experts | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.153112 | 0.0587276 | - | - | - | - | - | 0% |
| 706.8965454101562 | Deep shared-routed MoE | 0.0447901 | 0.0418887 | 0.0430881 | 0.584439 | 0.357596 | 1.54431 | 0 | 70.7469% |
| 1000.0 | Galerkin only | 0.178619 | 0.0521943 | - | - | - | - | - | 0% |
| 1000.0 | Deep shared-routed MoE | 0.0885313 | 0.0395077 | 0.0406932 | 0.41429 | 0.315309 | 1.49113 | 0 | 50.4356% |

Runtime: 360.26 s.
