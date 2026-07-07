# HPRS-MoE-ROM V15 Physics-Generalizable Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.85; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

V15 pressure input mode: `pressure_only`. All V15 cases keep the V14 best `[a_t,b_t]` pressure state and differ only by ROM dimension or regime-balanced sampling.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, closed-loop multi-step rollout, energy consistency, trajectory consistency, pressure closure, relative terms, group/router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime supervision.

## Dense Training Split

Test selection: `regime_default`, time stride=1, Re stride=1.

- Train Re count: 89
- Test Re count: 11
- Excluded Re count from Re sparsity: 0
- Dense train samples before time sparsity: 10970
- Kept train samples: 10970
- Validation samples: 1547
- Test samples: 1350
- Compression vs dense train: 1
- Compression vs all non-test candidates: 0.876408

| Re | role | total | dense train | kept train | val | test |
|---:|---|---:|---:|---:|---:|---:|
| 20 | train | 61 | 51 | 51 | 10 | 0 |
| 22.5357 | train | 61 | 51 | 51 | 10 | 0 |
| 24.6304 | test | 61 | 0 | 0 | 0 | 61 |
| 26.6673 | train | 61 | 51 | 51 | 10 | 0 |
| 28.6951 | train | 61 | 51 | 51 | 10 | 0 |
| 30.7204 | train | 61 | 51 | 51 | 10 | 0 |
| 32.7401 | test | 61 | 0 | 0 | 0 | 61 |
| 34.7376 | train | 61 | 51 | 51 | 10 | 0 |
| 36.6578 | train | 60 | 50 | 50 | 10 | 0 |
| 38.3572 | train | 61 | 51 | 51 | 10 | 0 |
| 39.6855 | test | 61 | 0 | 0 | 0 | 61 |
| 40.7115 | train | 61 | 51 | 51 | 10 | 0 |
| 41.5766 | train | 60 | 50 | 50 | 10 | 0 |
| 42.3591 | train | 61 | 51 | 51 | 10 | 0 |
| 43.0939 | train | 61 | 51 | 51 | 10 | 0 |
| 43.7974 | train | 60 | 50 | 50 | 10 | 0 |
| 44.4783 | train | 61 | 51 | 51 | 10 | 0 |
| 45.1427 | test | 61 | 0 | 0 | 0 | 61 |
| 45.7952 | train | 61 | 51 | 51 | 10 | 0 |
| 46.4401 | train | 61 | 51 | 51 | 10 | 0 |
| 47.0814 | test | 158 | 0 | 0 | 0 | 158 |
| 47.7229 | train | 158 | 139 | 139 | 19 | 0 |
| 48.3687 | train | 158 | 139 | 139 | 19 | 0 |
| 49.0224 | test | 158 | 0 | 0 | 0 | 158 |
| 49.6876 | train | 159 | 140 | 140 | 19 | 0 |
| 50.3681 | train | 158 | 139 | 139 | 19 | 0 |
| 51.0668 | train | 158 | 139 | 139 | 19 | 0 |
| 51.7864 | test | 158 | 0 | 0 | 0 | 158 |
| 52.5288 | train | 159 | 140 | 140 | 19 | 0 |
| 53.2942 | train | 158 | 139 | 139 | 19 | 0 |
| 54.0815 | train | 158 | 139 | 139 | 19 | 0 |
| 54.888 | train | 158 | 139 | 139 | 19 | 0 |
| 55.7096 | train | 158 | 139 | 139 | 19 | 0 |
| 56.5433 | train | 158 | 139 | 139 | 19 | 0 |
| 57.39 | train | 159 | 140 | 140 | 19 | 0 |
| 58.2626 | train | 158 | 139 | 139 | 19 | 0 |
| 59.2014 | train | 158 | 139 | 139 | 19 | 0 |
| 60.3077 | train | 159 | 140 | 140 | 19 | 0 |
| 61.756 | train | 158 | 139 | 139 | 19 | 0 |
| 63.4998 | train | 158 | 139 | 139 | 19 | 0 |
| 65.2598 | train | 158 | 139 | 139 | 19 | 0 |
| 66.9701 | train | 158 | 139 | 139 | 19 | 0 |
| 68.6497 | train | 158 | 139 | 139 | 19 | 0 |
| 70.3146 | test | 158 | 0 | 0 | 0 | 158 |
| 71.9729 | train | 158 | 139 | 139 | 19 | 0 |
| 73.6283 | train | 158 | 139 | 139 | 19 | 0 |
| 75.2823 | train | 158 | 139 | 139 | 19 | 0 |
| 76.9358 | train | 158 | 139 | 139 | 19 | 0 |
| 78.589 | train | 158 | 139 | 139 | 19 | 0 |
| 80.242 | train | 158 | 139 | 139 | 19 | 0 |
| 81.8947 | train | 158 | 139 | 139 | 19 | 0 |
| 83.5472 | train | 158 | 139 | 139 | 19 | 0 |
| 85.1991 | train | 158 | 139 | 139 | 19 | 0 |
| 86.8502 | train | 158 | 139 | 139 | 19 | 0 |
| 88.4998 | train | 158 | 139 | 139 | 19 | 0 |
| 90.1473 | train | 158 | 139 | 139 | 19 | 0 |
| 91.7922 | train | 158 | 139 | 139 | 19 | 0 |
| 93.4352 | train | 158 | 139 | 139 | 19 | 0 |
| 95.0817 | train | 158 | 139 | 139 | 19 | 0 |
| 96.7493 | train | 158 | 139 | 139 | 19 | 0 |
| 98.4804 | train | 158 | 139 | 139 | 19 | 0 |
| 100.352 | test | 158 | 0 | 0 | 0 | 158 |
| 102.44 | train | 158 | 139 | 139 | 19 | 0 |
| 104.711 | train | 158 | 139 | 139 | 19 | 0 |
| 107.051 | train | 158 | 139 | 139 | 19 | 0 |
| 109.396 | train | 158 | 139 | 139 | 19 | 0 |
| 111.734 | train | 158 | 139 | 139 | 19 | 0 |
| 114.066 | train | 158 | 139 | 139 | 19 | 0 |
| 116.395 | train | 158 | 139 | 139 | 19 | 0 |
| 118.723 | train | 158 | 139 | 139 | 19 | 0 |
| 121.05 | train | 158 | 139 | 139 | 19 | 0 |
| 123.377 | train | 158 | 139 | 139 | 19 | 0 |
| 125.703 | train | 159 | 140 | 140 | 19 | 0 |
| 128.029 | train | 158 | 139 | 139 | 19 | 0 |
| 130.355 | train | 158 | 139 | 139 | 19 | 0 |
| 132.68 | train | 158 | 139 | 139 | 19 | 0 |
| 135.004 | train | 158 | 139 | 139 | 19 | 0 |
| 137.325 | train | 159 | 140 | 140 | 19 | 0 |
| 139.642 | train | 159 | 140 | 140 | 19 | 0 |
| 141.956 | train | 158 | 139 | 139 | 19 | 0 |
| 144.273 | train | 158 | 139 | 139 | 19 | 0 |
| 146.62 | train | 158 | 139 | 139 | 19 | 0 |
| 149.059 | test | 158 | 0 | 0 | 0 | 158 |
| 151.686 | train | 158 | 139 | 139 | 19 | 0 |
| 154.521 | train | 158 | 139 | 139 | 19 | 0 |
| 157.46 | train | 158 | 139 | 139 | 19 | 0 |
| 160.415 | train | 158 | 139 | 139 | 19 | 0 |
| 163.364 | train | 158 | 139 | 139 | 19 | 0 |
| 166.306 | train | 158 | 139 | 139 | 19 | 0 |
| 169.245 | train | 158 | 139 | 139 | 19 | 0 |
| 172.182 | train | 158 | 139 | 139 | 19 | 0 |
| 175.118 | train | 158 | 139 | 139 | 19 | 0 |
| 178.054 | train | 158 | 139 | 139 | 19 | 0 |
| 180.992 | train | 158 | 139 | 139 | 19 | 0 |
| 183.933 | train | 158 | 139 | 139 | 19 | 0 |
| 186.885 | train | 159 | 140 | 140 | 19 | 0 |
| 189.862 | test | 158 | 0 | 0 | 0 | 158 |
| 192.912 | train | 159 | 140 | 140 | 19 | 0 |
| 196.161 | train | 158 | 139 | 139 | 19 | 0 |
| 200 | train | 159 | 140 | 140 | 19 | 0 |

## Aggregate Held-out Metrics

| Metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| rhs_l2 | 1.47951 | 1.54871 | 0.151112 | 4.52314 |
| pressure_head_l2 | 1.57158 | 2.2501 | 0.0191893 | 7.33336 |
| one_step_a_l2 | 0.233452 | 0.44965 | 0.0250194 | 1.62144 |
| one_step_b_l2 | 1.64231 | 2.36149 | 0.0405696 | 7.89356 |
| rollout_a_l2 | 0.746176 | 1.23233 | 0.0363813 | 4.52307 |
| rollout_b_l2 | 3.55772 | 6.94527 | 0.0549449 | 25.0669 |
| one_step_pressure_energy_error | 8.7661 | 18.3952 | 0.001524 | 63.7101 |
| rollout_pressure_energy_error | 52.9082 | 156.044 | 0.0116411 | 546.117 |

Error curve CSV: `/root/moe/V15_PhysicsGeneralizable/test_results_v15/results/V15_Base/V15_Base_physics_generalizable_ru16_rp16/V15_Base_physics_generalizable_ru16_rp16_error_vs_re.csv`

Error curve SVG: `/root/moe/V15_PhysicsGeneralizable/test_results_v15/results/V15_Base/V15_Base_physics_generalizable_ru16_rp16/V15_Base_physics_generalizable_ru16_rp16_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.374731 | 54.866 | - | 0.0219616 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 1.72141 | 54.866 | 4.50702 | 0.0961902 | 0.0946618 | 4.45699 | 0.65142 | 0.626432 | 4.67731 | 4 | 2.70267 | 1.03268 | 18 |
| 32.74006652832031 | Galerkin only | 0.329263 | 68.9859 | - | 0.0191824 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 0.966439 | 68.9859 | 2.59274 | 0.0548743 | 0.0538144 | 2.55917 | 0.445326 | 0.431016 | 3.25097 | 4 | 2.70398 | 1.02974 | 18 |
| 39.68547821044922 | Galerkin only | 0.299669 | 56.0379 | - | 0.0174179 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 0.674509 | 56.0379 | 0.963365 | 0.0382894 | 0.0379301 | 0.972075 | 0.31828 | 0.311955 | 1.16969 | 4 | 2.50933 | 1.02669 | 15 |
| 45.142704010009766 | Galerkin only | 0.279522 | 50.0437 | - | 0.0162395 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 0.561786 | 50.0437 | 0.698095 | 0.0313537 | 0.0312654 | 0.713164 | 0.23698 | 0.224681 | 0.737755 | 4 | 1.96483 | 1.02148 | 15 |
| 47.081356048583984 | Galerkin only | 2.17816 | 66.4526 | - | 0.269108 | - | - | - | - | - | - | - | - | - |
| 47.081356048583984 | HPRS-MoE | 4.52314 | 66.4526 | 0.382092 | 0.336466 | 0.336248 | 0.438042 | 0.899142 | 1.02541 | 1.99812 | 4 | 2.38027 | 1.01039 | 15 |
| 49.02235794067383 | Galerkin only | 1.76499 | 53.8859 | - | 0.211056 | - | - | - | - | - | - | - | - | - |
| 49.02235794067383 | HPRS-MoE | 3.23849 | 53.8859 | 0.694358 | 0.248205 | 0.247476 | 0.792387 | 0.62434 | 0.743642 | 1.74407 | 4.01 | 2.50403 | 1.00963 | 15 |
| 51.78644943237305 | Galerkin only | 2.70785 | 591.267 | - | 1.57782 | - | - | - | - | - | - | - | - | - |
| 51.78644943237305 | HPRS-MoE | 3.83443 | 591.267 | 7.33336 | 1.622 | 1.62144 | 7.89356 | 4.14613 | 4.52307 | 25.0669 | 4.03 | 1.87775 | 1.01489 | 15 |
| 70.31463623046875 | Galerkin only | 0.194353 | 1.4395 | - | 0.313395 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.151112 | 1.4395 | 0.0429467 | 0.137325 | 0.0556469 | 0.0970683 | 0.490043 | 0.185477 | 0.27592 | 4.48 | 2.10242 | 1.1545 | 11 |
| 100.35224914550781 | Galerkin only | 0.255587 | 0.936488 | - | 0.402269 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.217605 | 0.936488 | 0.0191893 | 0.157942 | 0.0384288 | 0.0609739 | 0.583075 | 0.0435468 | 0.0873192 | 4.66 | 2.52792 | 1.13088 | 16 |
| 149.05923461914062 | Galerkin only | 0.278351 | 0.800668 | - | 0.435167 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.205254 | 0.800668 | 0.0227992 | 0.169914 | 0.0260438 | 0.0405696 | 0.552554 | 0.0363813 | 0.0549449 | 4.45 | 2.21977 | 1.08641 | 12 |
| 189.86227416992188 | Galerkin only | 0.285972 | 0.715973 | - | 0.44897 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.180477 | 0.715973 | 0.0313885 | 0.168926 | 0.0250194 | 0.0414588 | 0.650717 | 0.0563221 | 0.0718861 | 4.65 | 1.79201 | 1.05549 | 14 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 32.74006652832031 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 39.68547821044922 | True | [0.066, 0.000, 0.934] | [0.066, 0.000, 0.934] | 0 |
| 45.142704010009766 | True | [0.295, 0.000, 0.705] | [0.295, 0.000, 0.705] | 0 |
| 47.081356048583984 | True | [0.886, 0.000, 0.114] | [0.886, 0.000, 0.114] | 0 |
| 49.02235794067383 | True | [0.930, 0.000, 0.070] | [0.930, 0.000, 0.070] | 0 |
| 51.78644943237305 | True | [0.646, 0.000, 0.354] | [0.646, 0.000, 0.354] | 0 |
| 70.31463623046875 | True | [0.854, 0.146, 0.000] | [0.854, 0.146, 0.000] | 0 |
| 100.35224914550781 | True | [0.025, 0.975, 0.000] | [0.025, 0.975, 0.000] | 0 |
| 149.05923461914062 | True | [0.025, 0.854, 0.120] | [0.025, 0.854, 0.120] | 0 |
| 189.86227416992188 | True | [0.000, 0.589, 0.411] | [0.000, 0.589, 0.411] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.836443 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 32.74006652832031 | 0.833411 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 39.68547821044922 | 0.830011 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 45.142704010009766 | 0.825244 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 47.081356048583984 | 0.812689 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 49.02235794067383 | 0.811568 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 51.78644943237305 | 0.802929 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 70.31463623046875 | 0.746739 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 100.35224914550781 | 0.740223 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 149.05923461914062 | 0.89777 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 189.86227416992188 | 0.91362 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |

Runtime: 36962.38 s.
