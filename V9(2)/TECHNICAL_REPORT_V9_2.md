# V9(2) 技术报告：长 Epoch 与 Loss 稳定早停

日期：2026-06-21

代码：`test_results_v9_2/deep_moe_rom_v9_2.py`

结果目录：`test_results_v9_2/results/`

## 1. 目标

V9 已经通过 relative loss 明显提升了 V8 的压力精度，但用户希望进一步延长 epoch，并在 loss 稳定后停止训练。V9(2) 因此不改变 V9 的核心物理结构，而是改进训练日程：

- 延长最大 epoch。
- 提高验证频率。
- 引入 `early_stop_min_delta`，只有验证分数出现足够改善才刷新 best。
- 在 `min_epochs` 后，如果验证分数长期没有足够改善，则停止训练。

## 2. 继承的模型结构

V9(2) 继续使用 V7/V8/V9 的 pressure-surrogate residual RK4 架构：

```text
a_next = RK4(a_t, Galerkin velocity tensors + learned RHS correction)
b_base = c_tilde + A_tilde @ a_next
       + torch.einsum("pij,bi,bj->bp", H_tilde, a_next, a_next)
delta_b = pressure_next_head(x_t)
b_next = b_base + delta_b
```

训练损失继承 V9：

- coefficient loss
- RHS dynamic residual loss
- sampled reconstruction loss
- pressure surrogate residual loss
- alpha/RHS consistency loss
- short rollout loss
- alpha/RHS/pressure relative loss
- router load-balance、entropy、temporal smoothness

## 3. V9(2) 训练日程

V9(2) 使用 V9 中表现最好的 `r16-b2` 配置，并延长训练：

| Item | Value |
|---|---:|
| `r_u`, `r_p` | 16, 16 |
| MoE blocks | 2 |
| Experts | 8 |
| Hidden dim | 144 |
| Expert hidden | 224 |
| Max epochs | 180 |
| Min epochs | 100 |
| Patience | 35 |
| Eval every | 5 |
| Early-stop min delta | 8e-4 |
| Train rollout steps | 8 |
| Eval rollout steps | 16 |
| Rollout curriculum | 1, 2, 4, 8 |
| Reconstruction sampled columns | 2048 |

早停规则：

```text
if val_score < best_val - early_stop_min_delta:
    update best checkpoint
elif epoch >= min_epochs and epoch - best_epoch >= patience:
    stop
```

实际运行中没有跑满 180 epoch，而是在各 test split 的验证分数稳定后停止。三个 test split 的 best epoch 分别为：

| Test Re | Best epoch | Last recorded eval epoch |
|---:|---:|---:|
| 56.37 | 125 | 160 |
| 120.00 | 120 | 155 |
| 300.00 | 80 | 115 |

总运行时间：809.36 s。训练设备：NVIDIA GeForce RTX 3090。

## 4. V9(2) 指标

| Test Re | Base pressure L2 | Final pressure L2 | RHS L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | Auto a rollout mean | Auto b rollout mean | Entropy | Load CV | Dead experts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.37 | 2.7265 | 0.3623 | 0.0876 | 0.0854 | 0.0603 | 0.3691 | 0.3314 | 0.5106 | 1.0020 | 1.2767 | 2 |
| 120.00 | 0.4817 | 0.0220 | 0.0250 | 0.1029 | 0.0567 | 0.0640 | 0.3568 | 0.3235 | 1.1510 | 0.3964 | 0 |
| 300.00 | 0.3379 | 0.0717 | 0.0173 | 0.0999 | 0.0849 | 0.0935 | 0.5898 | 0.6568 | 0.9868 | 0.4271 | 0 |

三点平均：

| Metric | V8 r16-b2 | V9 r16-b2 | V9(2) r16-b2 |
|---|---:|---:|---:|
| Final pressure L2 | 0.3633 | 0.1698 | 0.1520 |
| Auto pressure one-step L2 | 0.3911 | 0.1925 | 0.1755 |
| Auto velocity rollout mean | 0.4250 | 0.4206 | 0.4260 |
| Auto pressure rollout mean | 0.7035 | 0.4910 | 0.4970 |
| RHS L2 | 0.0624 | 0.0512 | 0.0433 |
| TF one-step L2 | 0.0875 | 0.0868 | 0.0960 |

## 5. 与 V9 的对比

负值表示误差下降。

| Test Re | Final pressure | Auto pressure one-step | Auto velocity rollout | Auto pressure rollout | RHS | TF one-step |
|---:|---:|---:|---:|---:|---:|---:|
| 56.37 | -5.9% | -6.6% | -9.8% | -6.4% | -14.7% | +23.3% |
| 120.00 | -32.2% | -12.6% | +1.2% | +3.4% | -14.3% | +12.5% |
| 300.00 | -21.9% | -14.4% | +8.9% | +6.9% | -19.9% | +0.2% |

三点平均变化：

| Metric | V9 | V9(2) | Change |
|---|---:|---:|---:|
| Final pressure L2 | 0.1698 | 0.1520 | -10.4% |
| Auto pressure one-step L2 | 0.1925 | 0.1755 | -8.8% |
| Auto velocity rollout mean | 0.4206 | 0.4260 | +1.3% |
| Auto pressure rollout mean | 0.4910 | 0.4970 | +1.2% |
| RHS L2 | 0.0512 | 0.0433 | -15.4% |
| TF one-step L2 | 0.0868 | 0.0960 | +10.6% |

## 6. 与 V8 的对比

V9(2) 相比 V8 baseline 仍有明显优势：

| Test Re | Final pressure | Auto pressure one-step | Auto velocity rollout | Auto pressure rollout | RHS |
|---:|---:|---:|---:|---:|---:|
| 56.37 | -61.1% | -61.5% | -29.6% | -56.9% | -31.3% |
| 120.00 | -65.3% | -33.0% | +1.4% | -1.8% | -22.3% |
| 300.00 | -25.1% | -21.3% | +30.4% | +10.1% | -37.6% |

## 7. 结论

V9(2) 达到了“延长 epoch 并在 loss 稳定后停止”的目标。训练并未跑满 180 epoch，而是在验证分数稳定后停止，说明 early stopping 正常工作。

精度收益：

- 相比 V9，三点平均 final pressure L2 进一步下降 10.4%。
- 相比 V9，三点平均 auto pressure one-step L2 下降 8.8%。
- 相比 V9，三点平均 RHS L2 下降 15.4%。
- 低 Re 与中 Re 的压力精度继续提升，`Re=120` 的 final pressure L2 从 0.0325 降到 0.0220。

代价：

- rollout 均值没有继续改善，平均 auto pressure rollout 相比 V9 小幅上升 1.2%。
- `Re=300` 的 rollout 仍然是短板，velocity rollout 与 pressure rollout 均比 V9 略差。
- 低 Re router 仍有 2 个 dead experts，说明单纯延长 epoch 不能完全解决路由均衡问题。

综合判断：

- 如果目标是 one-step pressure / RHS 精度，V9(2) 优于 V9。
- 如果目标是 long rollout 稳定性，V9 和 V9(2) 基本持平，V9 略优。
- 下一步应考虑对 rollout loss 做后期 annealing 或单独针对 `Re=300` 加强 velocity-rollout 权重，而不是继续单纯增加 epoch。
