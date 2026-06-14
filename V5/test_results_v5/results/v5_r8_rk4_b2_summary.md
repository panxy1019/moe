# Deep MoE-ROM v5 Summary

## Architecture

PhysicalContextEncoder + 2 Shared-Routed MoE blocks, hidden_dim=96, experts=6, top_k=2, expert_hidden=128.

Dual heads: `alpha_next_head` and `rhs_correction_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS relative L2 | Euler one-step L2 | Integrator one-step L2 | rollout mean L2 | load CV | entropy | dead experts | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 706.8965454101562 | Galerkin only | 0.153112 | 0.0587276 | - | - | - | - | - | 0% |
| 706.8965454101562 | Deep shared-routed MoE | 0.0455104 | 0.0420231 | 0.0174754 | 0.384591 | 0.400917 | 1.00582 | 0 | 70.2764% |
| 1000.0 | Galerkin only | 0.178619 | 0.0521943 | - | - | - | - | - | 0% |
| 1000.0 | Deep shared-routed MoE | 0.0907532 | 0.0395244 | 0.0219123 | 0.311117 | 0.245405 | 0.927541 | 0 | 49.1917% |

Runtime: 838.12 s.
