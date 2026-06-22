# V10 Shared-Expert Physical MoE-ROM Technical Report

## 1. Objective

V9/V9(2) already improved low-Re pressure prediction with amplitude-aware relative losses, but two architecture-level problems remained:

- Dead experts: low-Re test still had 2 dead routed experts.
- High-energy generalization: Re=300 velocity rollout degraded in V9(2), suggesting routed experts were competing between global residual dynamics and local high-frequency corrections.

V10 tests whether an explicit shared expert mechanism can separate global physical residuals from Re/phase-specific residuals, eliminate dead experts, and recover high-Re rollout stability.

## 2. Data and Tensor Files

The experiment used the Re=50-300 weighted-L2 dataset introduced in V8.

| Item | Cluster path |
|---|---|
| POD data root | `/root/moe/V8/data/Global_POD_Weighted_L2` |
| Velocity POD | `/root/moe/V8/data/Global_POD_Weighted_L2/global_velocity_pod_weighted_l2.npz` |
| Pressure POD | `/root/moe/V8/data/Global_POD_Weighted_L2/global_pressure_pod_weighted_l2.npz` |
| Mesh L2 weights | `/root/moe/V8/data/Global_POD_Weighted_L2/mesh_l2_point_weights.npz` |
| Semi-intrusive Galerkin tensor | `/root/moe/V8/data/semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_slim.npz` |
| Pressure surrogate tensor | `/root/moe/V8/data/pressure_poisson_surrogate_tensors_allRe100_weightedL2_ru80_rp80.npz` |
| Data manifest docs | `/root/moe/V8/data/*.md` |

Main truncation:

| Quantity | Value |
|---|---:|
| `r_u` | 16 |
| `r_p` | 16 |
| velocity POD energy | 0.943109 |
| pressure POD energy | 0.944973 |
| total snapshots | 12869 |
| valid samples | 12569 |
| held-out Re indices | 10, 59, 99 |
| held-out Re values | 56.3745, 120.0, 300.0 |

## 3. V10 Architecture

Code: `test_results_v10/deep_moe_rom_v10.py`

V10 keeps the V7/V8/V9 pressure-surrogate residual RK4 framework:

```text
PhysicalContextEncoder
  -> 2 Shared-Routed MoE blocks
  -> alpha_next_head
  -> rhs_correction_head
  -> pressure_next residual head
```

The autonomous rollout step is:

1. Compute corrected RHS from the semi-intrusive Galerkin RHS plus learned correction.
2. Integrate velocity coefficient `a_next` with RK4.
3. Compute pressure baseline from the rigid pressure surrogate:

```python
b_base = c_tilde + A_tilde @ a_next + einsum(H_tilde, a_next, a_next)
```

4. Predict only pressure residual `delta_b` from the network.
5. Compose autonomous pressure:

```python
b_next = b_base + delta_b
```

The V10 block differs from V9(2) in three places:

```python
shared_stack = torch.stack([expert(z) for expert in self.shared_experts], dim=1)
shared_gate = torch.softmax(self.shared_mixer(z), dim=-1)
shared = torch.einsum("bs,bsh->bh", shared_gate, shared_stack)

gate = topk_softmax_router(z)
gate = (1.0 - gate_floor) * gate + gate_floor / num_experts

h = h + shared_scale * shared + routed_scale * routed
```

This makes the always-on branch an explicit small shared expert pool, while the routed experts specialize in Re/phase/time-local residuals. The gate floor is intentionally used as a hard anti-collapse mechanism.

## 4. Main Hyperparameters

Main experiment: `v10_r16_rk4_b2_shared_floor`

| Hyperparameter | Value |
|---|---:|
| blocks | 2 |
| hidden dim | 144 |
| routed experts | 8 |
| shared experts | 2 |
| top-k | 2 |
| expert hidden | 224 |
| dropout | 0.035 |
| router temperature | 1.05 |
| gate floor | 0.12 |
| shared scale | 1.0 |
| routed scale | 0.55 |
| batch size | 768 |
| recon sampled columns | 2048 |
| train rollout steps | 8 |
| eval rollout steps | 16 |
| rollout curriculum | 1, 2, 4, 8 |
| max epochs | 200 |
| min epochs | 100 |
| patience | 40 |
| learning rate | 7.5e-4 |
| weight decay | 1.2e-4 |
| runtime | 1079.49 s |

Loss weights:

| Loss | Weight |
|---|---:|
| coefficient | 1.0 |
| dynamic residual | 1.0 |
| pressure residual | 0.65 |
| reconstruction | 0.08 |
| rollout | 0.16 |
| pressure rollout | 0.35 |
| alpha/RHS consistency | 0.15 |
| router balance | 0.08 |
| router entropy | -0.0015 |
| router smoothness | 0.06 |
| alpha relative | 0.08 |
| RHS relative | 0.08 |
| pressure relative | 0.35 |
| rollout relative mix | 0.30 |
| relative floor fraction | 0.05 |

## 5. Main V10 Results

| Test Re | Best epoch | Pressure L2 | RHS L2 | Auto a one-step | Auto b one-step | Auto a rollout | Auto b rollout | Load CV | Entropy | Dead experts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745 | 150 | 0.283692 | 0.078466 | 0.066361 | 0.289178 | 0.403320 | 0.543825 | 0.932987 | 1.373952 | 0 |
| 120.0 | 140 | 0.020780 | 0.028455 | 0.054604 | 0.063671 | 0.327383 | 0.311903 | 0.543456 | 1.332339 | 0 |
| 300.0 | 125 | 0.064957 | 0.016931 | 0.079498 | 0.087281 | 0.496450 | 0.587781 | 0.321574 | 1.383845 | 0 |

Compared with V9(2), V10 reduced mean pressure error from 0.1520 to 0.1231 and eliminated all dead experts.

| Metric | V9(2) mean | V10 mean | Change |
|---|---:|---:|---:|
| Pressure relative L2 | 0.152039 | 0.123143 | -19.0% |
| RHS relative L2 | 0.043316 | 0.041284 | -4.7% |
| Auto velocity rollout | 0.426001 | 0.409051 | -4.0% |
| Auto pressure rollout | 0.496979 | 0.481170 | -3.2% |
| Dead experts total | 2 | 0 | eliminated |

Per-Re comparison with V9(2):

| Test Re | Pressure change | RHS change | Auto velocity rollout change | Auto pressure rollout change |
|---:|---:|---:|---:|---:|
| 56.3745 | -21.7% | -10.5% | +21.7% | +6.5% |
| 120.0 | -5.8% | +13.6% | -8.3% | -3.6% |
| 300.0 | -9.5% | -2.0% | -15.8% | -10.5% |

High-Re recovery is the strongest success case: Re=300 pressure L2 improved from V9(2) 0.071738 to 0.064957, and autonomous velocity rollout improved from 0.589806 to 0.496450.

## 6. Expert Routing Analysis

The explicit gate floor eliminated dead experts in all tested Re cases. The routing is not uniformly flat; it still shows physical specialization by Re and phase.

| Test Re | Mean routed load | Top-1 pattern | Interpretation |
|---:|---|---|---|
| 56.3745 | `[0.015, 0.100, 0.154, 0.278, 0.028, 0.344, 0.047, 0.033]` | experts 3 and 5 dominate | low-Re remains concentrated, but no routed expert is dead |
| 120.0 | `[0.185, 0.071, 0.174, 0.040, 0.189, 0.024, 0.204, 0.113]` | experts 0, 2, 4, 6 active | mid-Re has multi-expert sharing |
| 300.0 | `[0.080, 0.168, 0.131, 0.093, 0.107, 0.111, 0.102, 0.208]` | no expert exceeds 31% top-1 | high-Re has the healthiest routed distribution |

The top-k count in V10 is computed from the largest `k` gate weights, not from nonzero gates, because the gate floor makes every expert nonzero. This keeps the routing diagnostics meaningful.

## 7. Low-Re Pressure-Focused Ablation

A pressure-focused ablation was run only on Re=56:

Experiment: `v10_r16_lowRe_pressure_focus`

Changes:

- `lambda_pressure`: 0.65 -> 0.75
- `lambda_pressure_rel`: 0.35 -> 0.70
- `relative_floor_frac`: 0.05 -> 0.02
- `lambda_rollout`: 0.16 -> 0.12
- `rollout_relative_mix`: 0.30 -> 0.45

| Experiment | Pressure L2 | RHS L2 | Auto a one-step | Auto b one-step | Auto a rollout | Auto b rollout | Load CV | Entropy | Dead experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V10 main Re=56 | 0.283692 | 0.078466 | 0.066361 | 0.289178 | 0.403320 | 0.543825 | 0.932987 | 1.373952 | 0 |
| Low-Re pressure focus | 0.287155 | 0.079134 | 0.085798 | 0.301697 | 0.464596 | 0.591496 | 0.702608 | 1.456979 | 0 |

The pressure-focused loss did not improve low-Re generalization. It slightly worsened pressure and rollout, so the main shared-floor configuration is kept as the V10 result.

## 8. Stability Notes

V10 improves autonomous rollout on Re=120 and Re=300, but low-Re teacher-forced rollout still has extreme outlier windows. In the main run, Re=56 teacher-forced rollout mean became very large due to a numerical overflow window; the autonomous pressure rollout remained finite at 0.543825. In the low-Re pressure-focused ablation, teacher-forced rollout produced no valid windows after overflow filtering.

This suggests that the low-Re issue is not solved by stronger pressure loss alone. The likely bottleneck is the interaction between low-amplitude pressure coefficients, corrected RHS sensitivity, and teacher-forced pressure inputs. Future work should test bounded RHS corrections, Re-aware residual scaling, low-Re oversampling, and trust-region clipping during rollout evaluation.

## 9. Conclusion

V10 achieves the two main architecture goals:

- Dead experts were eliminated: V9(2) had 2 dead experts at Re=56; V10 has 0 at every tested Re.
- High-Re rollout recovered: Re=300 autonomous velocity rollout improved by 15.8% and autonomous pressure rollout improved by 10.5% versus V9(2).

Accuracy improved overall, especially pressure:

- Mean pressure relative L2 improved by 19.0% versus V9(2).
- Re=120 and Re=300 pressure errors are below 10%.
- Re=56 pressure error improved from 0.362332 to 0.283692 but remains above 10%.

Therefore V10 is a better architecture baseline than V9(2), but it does not fully meet the all-Re `<10%` pressure target. The next precision-focused step should target the low-Re pressure/rollout coupling rather than simply increasing pressure relative-loss weight.
