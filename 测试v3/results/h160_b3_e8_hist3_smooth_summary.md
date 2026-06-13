# Deep MoE-ROM v3 Summary

## Architecture

PhysicalContextEncoder + 3 Shared-Routed MoE blocks, hidden_dim=160, experts=8, top_k=2, expert_hidden=224.

Dual heads: `alpha_next_head` and `rhs_correction_head`.

Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, alpha/RHS consistency, router load-balance, entropy and temporal smoothness.

## Metrics

| Test Re | Model | RHS relative L2 | one-step relative L2 | alpha-head relative L2 | gate smooth MSE | rollout mean L2 | RHS improvement |
|---:|---|---:|---:|---:|---:|---:|---:|
| 700 | Galerkin only | 0.203349 | 0.0650657 | - | - | - | 0% |
| 700 | Deep shared-routed MoE | 0.0839706 | 0.0446767 | 0.195786 | 0.00802101 | 0.487919 | 58.7061% |
| 1000 | Galerkin only | 0.179599 | 0.0575607 | - | - | - | 0% |
| 1000 | Deep shared-routed MoE | 0.0798829 | 0.043386 | 0.341423 | 0.00939704 | 0.408768 | 55.5217% |

Runtime: 308.78 s.
