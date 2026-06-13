# 半侵入式物理一致 MoE-ROM 测试报告

日期：2026-06-13

## 1. 阅读后的方案选择

GitHub 方案文档推荐的主线是用 reduced RHS `R_r` 作为物理骨架，再用 Shared-Routed MoE 学习 closure correction。结合你指定的半侵入式文档，本次选择其中的 **方案 B：预测 RHS correction**，并把已有半侵入式 Galerkin 张量作为 `R_r`。

半侵入式张量采用本地文档 `/home/ray/Desktop/Cylinder_Results_Re500_1000_VTU/SEMI_INTRUSIVE_GALERKIN_TENSORS_allRe_ru80_rp80.md` 中的 Gram 修正形式：

```text
da/dt = c(Re) + A(Re) a + H(a,a) + P(Re) b
```

其中 `a` 是速度 POD 系数，`b` 是压力 POD 系数。本次测试没有重新生成张量，而是把本地已生成的 `semi_intrusive_galerkin_tensors_allRe_ru80_rp80.npz` 复制到集群数据目录后直接使用。

## 2. 最终测试实现

最终采用一个轻量、可解释、纯 `numpy` 的 Shared-Routed ridge-MoE closure：

```text
adot_hat = R_galerkin(a,b;Re) + C_shared(x) + sum_e gate_e(Re, phase) C_e(x)
```

输入特征 `x` 包含：`Re`、`1/Re`、phase Fourier features `sin(k theta), cos(k theta), k=1..4`、当前 `a`、当前 `b`、Galerkin RHS `R_galerkin`、低/中/高阶模态能量范数、压力范数和 RHS 范数。

专家结构：1 个 shared ridge expert + 8 个 routed ridge experts。router 是物理先验路由：2 个 Re band center `0.25/0.75` 乘 4 个 phase center `0/0.25/0.5/0.75`，top-2 soft routing。

训练目标：用中心差分得到 `adot_true`，训练 correction `adot_true - R_galerkin`。

最终选择阶数：`r_u = 16, r_p = 16`。原因是这个小阶数版本在 leave-one-Re 测试中最稳定，且能清晰展示“Galerkin 物理骨架 + MoE closure”带来的误差下降。

## 3. 集群运行信息

数据目录：

```text
/root/Cylinder_Results_Re500_1000_POD_data
```

远端 Python 环境：

```text
/root/miniconda3/envs/romtest/bin/python
numpy 2.4.6
```

最终运行命令：

```bash
/root/miniconda3/envs/romtest/bin/python /root/moe_rom_test.py --r-u 16 --r-p 16 --output-dir /root/moe_rom_test_results_r16
```

本地保存的脚本：`/home/ray/Desktop/vpn/semi_intrusive_moe_rom_test.py`

远端脚本：`/root/moe_rom_test.py`

远端结果：`/root/moe_rom_test_results_r16/semi_intrusive_moe_rom_metrics.json` 和 `semi_intrusive_moe_rom_summary.md`

## 4. 主实验结果

POD 能量覆盖：velocity first 16 = 0.912204，pressure first 16 = 0.951839。有效中心差分样本数：1194。

测试划分：

- `Re=700`：插值测试，训练使用其他 Re。
- `Re=1000`：外推测试，训练使用 `Re=500,600,700,800,900`。

| Test Re | Model | RHS relative L2 | RHS RMSE | centered R2 | one-step relative L2 | improvement vs Galerkin |
|---:|---|---:|---:|---:|---:|---:|
| 700 | Galerkin only | 0.203423 | 1.29909 | 0.958575 | 0.0649938 | 0% |
| 700 | Galerkin + shared | 0.142434 | 0.909605 | 0.979691 | 0.0518297 | 29.9812% |
| 700 | Galerkin + shared-routed | 0.11752 | 0.750502 | 0.986174 | 0.0490176 | 42.2285% |
| 1000 | Galerkin only | 0.17964 | 2.6541 | 0.967693 | 0.057334 | 0% |
| 1000 | Galerkin + shared | 0.10774 | 1.5918 | 0.988379 | 0.0474805 | 40.0248% |
| 1000 | Galerkin + shared-routed | 0.0985543 | 1.45609 | 0.990276 | 0.045681 | 45.138% |

20 步 Euler rollout 是一个轻量检查，使用真实压力 POD 系数和已知 phase 作为上下文，因此还不是完全 autonomous pressure-coupled ROM：

| Test Re | steps | windows | mean rollout relative L2 | median rollout relative L2 |
|---:|---:|---:|---:|---:|
| 700 | 20 | 9 | 0.426427 | 0.33093 |
| 1000 | 20 | 9 | 0.400615 | 0.385423 |

结论：在 `r=16` 下，shared-routed correction 相比 Galerkin-only 将 RHS relative L2 误差在 `Re=700` 上降低约 42.23%，在 `Re=1000` 上降低约 45.14%。这说明半侵入式 Galerkin RHS 已经提供了有效物理骨架，而路由专家能进一步学习 Re/phase 相关的 closure residual。

## 5. 阶数对照

| Run | r_u/r_p | POD energy velocity/pressure | shared-routed RHS improvement | 20-step rollout mean relative L2 |
|---|---:|---:|---:|---:|
| selected | 16/16 | 0.912204 / 0.951839 | Re=700: 42.2285%; Re=1000: 45.138% | Re=700: 0.426427; Re=1000: 0.400615 |
| r=24 check | 24/24 | 0.951398 / 0.979258 | Re=700: 16.3957%; Re=1000: 19.1617% | Re=700: 0.637304; Re=1000: 0.502369 |
| r=40 check | 40/40 | 0.980555 / 0.992929 | Re=700: -102.739%; Re=1000: -143.102% | Re=700: 3.21608e+54; Re=1000: 7.68308e+27 |

解释：`r=24` 仍有稳定正收益，但 rollout 误差略大；`r=40` 的 Galerkin-only 本身更强，但简单线性 ridge closure 在 leave-one-Re 上明显过拟合，导致校正后误差变差。因此这次技术验证选择 `r=16`，后续若要提升阶数，建议改用更强正则、非线性小网络或 rollout loss。

## 6. 局限与下一步

- 这次是 teacher-forced RHS/短 rollout 测试，pressure coefficient `b(t)` 使用真实 POD 数据；完全在线 ROM 需要同时预测压力分支或设计 pressure closure。
- router 采用物理先验软路由，不是端到端学习 router；这适合作为 MVP 和 sanity check，后续可替换为可训练 router。
- 当前 POD 是 unweighted global POD，半侵入式张量通过 Gram 修正适配；更严格版本可以补 mass-weighted POD。
- 下一步建议在 `r=16/24` 上加入 rollout loss，并做 `without R_galerkin`、`routed only`、`shared+routed` 的完整 ablation。

## 7. Python 代码

```python
#!/usr/bin/env python3
"""Semi-intrusive Galerkin + shared-routed ridge MoE closure test.

This script is intentionally dependency-light: only numpy is required.
It evaluates whether a small, physically routed correction improves the
semi-intrusive Galerkin reduced RHS for POD coefficients.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


EPS = 1.0e-12


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1.0e-10] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


@dataclass
class RoutedRidgeModel:
    standardizer: Standardizer
    shared_w: np.ndarray
    expert_w: np.ndarray
    settings: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/Cylinder_Results_Re500_1000_POD_data"),
    )
    parser.add_argument(
        "--tensor-path",
        type=Path,
        default=Path(
            "/root/Cylinder_Results_Re500_1000_POD_data/"
            "semi_intrusive_galerkin_tensors_allRe_ru80_rp80.npz"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/root/moe_rom_test"))
    parser.add_argument("--r-u", type=int, default=24)
    parser.add_argument("--r-p", type=int, default=24)
    parser.add_argument("--phase-harmonics", type=int, default=4)
    parser.add_argument("--test-res", type=int, nargs="+", default=[700, 1000])
    parser.add_argument("--ridge-shared", type=float, default=1.0e-2)
    parser.add_argument("--ridge-expert", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--sigma-phase", type=float, default=0.20)
    parser.add_argument("--sigma-re", type=float, default=0.35)
    parser.add_argument("--rollout-steps", type=int, default=20)
    return parser.parse_args()


def load_snapshot_index(path: Path) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    rows.sort(key=lambda row: int(row["snapshot_id"]))
    return {
        "snapshot_id": np.asarray([int(r["snapshot_id"]) for r in rows], dtype=int),
        "Re": np.asarray([int(float(r["Re"])) for r in rows], dtype=int),
        "time": np.asarray([float(r["time"]) for r in rows], dtype=float),
        "period": np.asarray([float(r["period"]) for r in rows], dtype=float),
        "cycle": np.asarray([float(r["cycle"]) for r in rows], dtype=float),
        "phase": np.asarray([float(r["phase"]) % 1.0 for r in rows], dtype=float),
        "local_snapshot_index": np.asarray(
            [int(float(r["local_snapshot_index"])) for r in rows], dtype=int
        ),
    }


def centered_time_derivative(
    coeff: np.ndarray, re_values: np.ndarray, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return derivative and valid interior indices."""
    deriv = np.full_like(coeff, np.nan, dtype=float)
    valid: List[int] = []
    for re in sorted(np.unique(re_values).tolist()):
        idx = np.where(re_values == re)[0]
        idx = idx[np.argsort(times[idx])]
        if len(idx) < 3:
            continue
        dt = times[idx[2:]] - times[idx[:-2]]
        deriv[idx[1:-1]] = (coeff[idx[2:]] - coeff[idx[:-2]]) / dt[:, None]
        valid.extend(idx[1:-1].tolist())
    return deriv, np.asarray(sorted(valid), dtype=int)


def galerkin_rhs_by_re(
    tensors: np.lib.npyio.NpzFile,
    a: np.ndarray,
    b: np.ndarray,
    re_values: np.ndarray,
    r_u: int,
    r_p: int,
) -> np.ndarray:
    rhs = np.empty((a.shape[0], r_u), dtype=float)
    for re in sorted(np.unique(re_values).tolist()):
        row = np.where(re_values == re)[0]
        prefix = f"Re_{int(re)}"
        c = tensors[f"{prefix}_c"][:r_u]
        A = tensors[f"{prefix}_A"][:r_u, :r_u]
        H = tensors[f"{prefix}_H"][:r_u, :r_u, :r_u]
        P = tensors[f"{prefix}_P"][:r_u, :r_p]
        ar = a[row]
        br = b[row]
        rhs[row] = (
            c[None, :]
            + ar @ A.T
            + np.einsum("ijk,nj,nk->ni", H, ar, ar, optimize=True)
            + br @ P.T
        )
    return rhs


def galerkin_rhs_single(
    tensors: np.lib.npyio.NpzFile,
    a: np.ndarray,
    b: np.ndarray,
    re: int,
    r_u: int,
    r_p: int,
) -> np.ndarray:
    prefix = f"Re_{int(re)}"
    c = tensors[f"{prefix}_c"][:r_u]
    A = tensors[f"{prefix}_A"][:r_u, :r_u]
    H = tensors[f"{prefix}_H"][:r_u, :r_u, :r_u]
    P = tensors[f"{prefix}_P"][:r_u, :r_p]
    return c + A @ a + np.einsum("ijk,j,k->i", H, a, a, optimize=True) + P @ b


def make_features(
    a: np.ndarray,
    b: np.ndarray,
    rhs: np.ndarray,
    re_values: np.ndarray,
    phase: np.ndarray,
    harmonics: int,
    include_rhs: bool = True,
) -> np.ndarray:
    re = re_values.astype(float)
    re_norm = ((re - 750.0) / 250.0)[:, None]
    inv_re = (1000.0 / re)[:, None]
    theta = 2.0 * np.pi * phase

    cols = [re_norm, inv_re]
    for k in range(1, harmonics + 1):
        cols.append(np.sin(k * theta)[:, None])
        cols.append(np.cos(k * theta)[:, None])

    r_u = a.shape[1]
    low = min(4, r_u)
    split = min(12, r_u)
    e_low = np.linalg.norm(a[:, :low], axis=1, keepdims=True)
    e_mid = np.linalg.norm(a[:, low:split], axis=1, keepdims=True)
    e_high = np.linalg.norm(a[:, split:], axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    rhs_norm = np.linalg.norm(rhs, axis=1, keepdims=True)

    cols.extend([a, b])
    if include_rhs:
        cols.append(rhs)
    cols.extend([e_low, e_mid, e_high, b_norm, rhs_norm])
    return np.hstack(cols)


def phase_distance(phase: np.ndarray, centers: np.ndarray) -> np.ndarray:
    raw = np.abs(phase[:, None] - centers[None, :])
    return np.minimum(raw, 1.0 - raw)


def router_weights(
    re_values: np.ndarray,
    phase: np.ndarray,
    top_k: int,
    sigma_phase: float,
    sigma_re: float,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    phase_centers = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float)
    re_centers = np.asarray([0.25, 0.75], dtype=float)
    centers: List[Tuple[float, float]] = []
    scores = []
    re_unit = ((re_values.astype(float) - 500.0) / 500.0)[:, None]

    for re_c in re_centers:
        pd = phase_distance(phase, phase_centers)
        rd = re_unit - re_c
        block = -0.5 * (pd / sigma_phase) ** 2 - 0.5 * (rd / sigma_re) ** 2
        scores.append(block)
        centers.extend([(float(re_c), float(pc)) for pc in phase_centers])

    score = np.hstack(scores)
    score -= score.max(axis=1, keepdims=True)
    weights = np.exp(score)

    if 0 < top_k < weights.shape[1]:
        keep = np.argpartition(weights, -top_k, axis=1)[:, -top_k:]
        mask = np.zeros_like(weights, dtype=bool)
        row = np.arange(weights.shape[0])[:, None]
        mask[row, keep] = True
        weights = np.where(mask, weights, 0.0)

    weights /= weights.sum(axis=1, keepdims=True) + EPS
    return weights, centers


def add_bias(x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])


def fit_ridge(
    x: np.ndarray, y: np.ndarray, ridge: float, weights: np.ndarray | None = None
) -> np.ndarray:
    xb = add_bias(x)
    if weights is not None:
        safe_w = np.maximum(weights, 0.0)
        sw = np.sqrt(safe_w)[:, None]
        xb = xb * sw
        y = y * sw

    gram = xb.T @ xb
    reg = ridge * np.eye(gram.shape[0], dtype=gram.dtype)
    reg[-1, -1] = 0.0
    rhs = xb.T @ y
    try:
        return np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram + reg) @ rhs


def predict_ridge(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return add_bias(x) @ w


def train_routed_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    re_train: np.ndarray,
    phase_train: np.ndarray,
    ridge_shared: float,
    ridge_expert: float,
    top_k: int,
    sigma_phase: float,
    sigma_re: float,
) -> Tuple[RoutedRidgeModel, Dict[str, object]]:
    standardizer = Standardizer.fit(x_train)
    xs = standardizer.transform(x_train)

    shared_w = fit_ridge(xs, y_train, ridge_shared)
    shared_pred = predict_ridge(xs, shared_w)
    remainder = y_train - shared_pred

    route_w, centers = router_weights(
        re_train, phase_train, top_k=top_k, sigma_phase=sigma_phase, sigma_re=sigma_re
    )
    expert_mats = []
    for expert_id in range(route_w.shape[1]):
        expert_mats.append(fit_ridge(xs, remainder, ridge_expert, route_w[:, expert_id]))
    expert_w = np.stack(expert_mats, axis=0)

    settings = {
        "ridge_shared": float(ridge_shared),
        "ridge_expert": float(ridge_expert),
        "top_k": int(top_k),
        "sigma_phase": float(sigma_phase),
        "sigma_re": float(sigma_re),
    }
    model = RoutedRidgeModel(
        standardizer=standardizer,
        shared_w=shared_w,
        expert_w=expert_w,
        settings=settings,
    )
    info = {
        "expert_centers": [
            {"re_unit_center": c[0], "phase_center": c[1]} for c in centers
        ],
        "train_expert_utilization": route_w.mean(axis=0).tolist(),
    }
    return model, info


def predict_routed_ridge(
    model: RoutedRidgeModel,
    x: np.ndarray,
    re_values: np.ndarray,
    phase: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = model.standardizer.transform(x)
    shared = predict_ridge(xs, model.shared_w)
    route_w, _ = router_weights(
        re_values,
        phase,
        top_k=int(model.settings["top_k"]),
        sigma_phase=float(model.settings["sigma_phase"]),
        sigma_re=float(model.settings["sigma_re"]),
    )
    expert_preds = np.stack([predict_ridge(xs, ew) for ew in model.expert_w], axis=1)
    routed = np.einsum("ne,neo->no", route_w, expert_preds, optimize=True)
    return shared + routed, route_w


def predict_shared_only(model: RoutedRidgeModel, x: np.ndarray) -> np.ndarray:
    xs = model.standardizer.transform(x)
    return predict_ridge(xs, model.shared_w)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    err = y_pred - y_true
    sse = float(np.sum(err * err))
    denom = float(np.sum(y_true * y_true)) + EPS
    centered = y_true - y_true.mean(axis=0, keepdims=True)
    centered_denom = float(np.sum(centered * centered)) + EPS
    mode_rel = np.sqrt(np.sum(err * err, axis=0) / (np.sum(y_true * y_true, axis=0) + EPS))
    return {
        "relative_l2": float(math.sqrt(sse / denom)),
        "rmse": float(math.sqrt(float(np.mean(err * err)))),
        "mae": float(np.mean(np.abs(err))),
        "r2_centered": float(1.0 - sse / centered_denom),
        "mode_relative_l2_first8": [float(v) for v in mode_rel[:8]],
    }


def one_step_metrics(
    a_true: np.ndarray,
    rhs_true: np.ndarray,
    rhs_pred: np.ndarray,
    re_values: np.ndarray,
    times: np.ndarray,
    valid_idx: np.ndarray,
    test_re: int,
) -> Dict[str, float]:
    idx = valid_idx[re_values[valid_idx] == test_re]
    idx = idx[np.argsort(times[idx])]
    if len(idx) < 2:
        return {"relative_l2": float("nan"), "rmse": float("nan")}
    current = idx[:-1]
    nxt = idx[1:]
    dt = (times[nxt] - times[current])[:, None]
    pred_next = a_true[current] + dt * rhs_pred[current]
    true_next = a_true[nxt]
    err = pred_next - true_next
    return {
        "relative_l2": float(np.linalg.norm(err) / (np.linalg.norm(true_next) + EPS)),
        "rmse": float(math.sqrt(float(np.mean(err * err)))),
    }


def rollout_error(
    tensors: np.lib.npyio.NpzFile,
    model: RoutedRidgeModel,
    a_all: np.ndarray,
    b_all: np.ndarray,
    re_values: np.ndarray,
    phase: np.ndarray,
    times: np.ndarray,
    seq_idx: np.ndarray,
    test_re: int,
    r_u: int,
    r_p: int,
    harmonics: int,
    steps: int,
) -> Dict[str, float]:
    idx = seq_idx[re_values[seq_idx] == test_re]
    idx = idx[np.argsort(times[idx])]
    if len(idx) <= steps + 1:
        return {"relative_l2_mean": float("nan"), "relative_l2_median": float("nan")}

    rel_errors: List[float] = []
    stride = max(1, steps)
    for start in range(0, len(idx) - steps - 1, stride):
        a_cur = a_all[idx[start]].copy()
        pred = []
        for h in range(steps):
            current_id = idx[start + h]
            next_id = idx[start + h + 1]
            re = int(re_values[current_id])
            rhs_g = galerkin_rhs_single(tensors, a_cur, b_all[current_id], re, r_u, r_p)
            x = make_features(
                a_cur[None, :],
                b_all[current_id][None, :],
                rhs_g[None, :],
                re_values[current_id : current_id + 1],
                phase[current_id : current_id + 1],
                harmonics,
                include_rhs=True,
            )
            corr, _ = predict_routed_ridge(
                model,
                x,
                re_values[current_id : current_id + 1],
                phase[current_id : current_id + 1],
            )
            rhs = rhs_g + corr[0]
            dt = float(times[next_id] - times[current_id])
            a_cur = a_cur + dt * rhs
            if not np.all(np.isfinite(a_cur)):
                break
            pred.append(a_cur.copy())

        if len(pred) != steps:
            continue
        true = a_all[idx[start + 1 : start + steps + 1]]
        pred_arr = np.asarray(pred)
        rel_errors.append(float(np.linalg.norm(pred_arr - true) / (np.linalg.norm(true) + EPS)))

    if not rel_errors:
        return {"relative_l2_mean": float("nan"), "relative_l2_median": float("nan")}
    return {
        "relative_l2_mean": float(np.mean(rel_errors)),
        "relative_l2_median": float(np.median(rel_errors)),
        "num_windows": int(len(rel_errors)),
    }


def summarize_improvement(base: Dict[str, object], candidate: Dict[str, object]) -> float:
    b = float(base["relative_l2"])
    c = float(candidate["relative_l2"])
    return float(100.0 * (1.0 - c / (b + EPS)))


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    started = time.time()
    pod_dir = args.data_root / "Global_POD_Unweighted"
    index = load_snapshot_index(pod_dir / "pod_snapshot_index.csv")

    vel = np.load(pod_dir / "global_velocity_pod.npz")
    pre = np.load(pod_dir / "global_pressure_pod.npz")
    tensors = np.load(args.tensor_path)

    a_full = vel["coeff_uv"][:, : args.r_u].astype(float)
    b_full = pre["coeff_p"][:, : args.r_p].astype(float)
    re_values = index["Re"]
    phase = index["phase"]
    times = index["time"]

    adot_full, valid_idx = centered_time_derivative(a_full, re_values, times)
    rhs_full = galerkin_rhs_by_re(
        tensors, a_full, b_full, re_values, args.r_u, args.r_p
    )

    x_full = make_features(
        a_full,
        b_full,
        rhs_full,
        re_values,
        phase,
        args.phase_harmonics,
        include_rhs=True,
    )
    residual_full = adot_full - rhs_full

    results = []
    saved_models: Dict[int, RoutedRidgeModel] = {}

    for test_re in args.test_res:
        train = valid_idx[re_values[valid_idx] != test_re]
        test = valid_idx[re_values[valid_idx] == test_re]
        y_train = residual_full[train]
        x_train = x_full[train]
        model, model_info = train_routed_ridge(
            x_train,
            y_train,
            re_values[train],
            phase[train],
            ridge_shared=args.ridge_shared,
            ridge_expert=args.ridge_expert,
            top_k=args.top_k,
            sigma_phase=args.sigma_phase,
            sigma_re=args.sigma_re,
        )
        saved_models[int(test_re)] = model

        x_test = x_full[test]
        corr_shared = predict_shared_only(model, x_test)
        corr_routed, test_route = predict_routed_ridge(
            model, x_test, re_values[test], phase[test]
        )

        y_true = adot_full[test]
        pred_galerkin = rhs_full[test]
        pred_shared = rhs_full[test] + corr_shared
        pred_routed = rhs_full[test] + corr_routed

        m_g = metrics(y_true, pred_galerkin)
        m_s = metrics(y_true, pred_shared)
        m_r = metrics(y_true, pred_routed)

        one_g = one_step_metrics(a_full, adot_full, rhs_full, re_values, times, valid_idx, test_re)
        rhs_pred_shared_full = rhs_full.copy()
        rhs_pred_routed_full = rhs_full.copy()
        rhs_pred_shared_full[test] = pred_shared
        rhs_pred_routed_full[test] = pred_routed
        one_s = one_step_metrics(
            a_full, adot_full, rhs_pred_shared_full, re_values, times, valid_idx, test_re
        )
        one_r = one_step_metrics(
            a_full, adot_full, rhs_pred_routed_full, re_values, times, valid_idx, test_re
        )
        roll_r = rollout_error(
            tensors,
            model,
            a_full,
            b_full,
            re_values,
            phase,
            times,
            valid_idx,
            test_re,
            args.r_u,
            args.r_p,
            args.phase_harmonics,
            args.rollout_steps,
        )

        results.append(
            {
                "test_Re": int(test_re),
                "train_Re": sorted(set(map(int, re_values[train].tolist()))),
                "num_train": int(len(train)),
                "num_test": int(len(test)),
                "metrics": {
                    "galerkin_only": m_g,
                    "galerkin_plus_shared": m_s,
                    "galerkin_plus_shared_routed": m_r,
                },
                "one_step_euler": {
                    "galerkin_only": one_g,
                    "galerkin_plus_shared": one_s,
                    "galerkin_plus_shared_routed": one_r,
                },
                "rollout_teacher_forced_pressure": {
                    "galerkin_plus_shared_routed": roll_r,
                    "steps": int(args.rollout_steps),
                    "note": "Uses true pressure POD coefficients and known phase as context.",
                },
                "improvement_percent_vs_galerkin_relative_l2": {
                    "shared": summarize_improvement(m_g, m_s),
                    "shared_routed": summarize_improvement(m_g, m_r),
                },
                "router": {
                    **model_info,
                    "test_expert_utilization": test_route.mean(axis=0).tolist(),
                },
            }
        )

    cumulative_uv = vel["cumulative_energy_uv"]
    cumulative_p = pre["cumulative_energy_p"]
    out = {
        "scheme": "semi_intrusive_galerkin_plus_shared_routed_ridge_moe",
        "data_root": str(args.data_root),
        "tensor_path": str(args.tensor_path),
        "settings": {
            "r_u": int(args.r_u),
            "r_p": int(args.r_p),
            "phase_harmonics": int(args.phase_harmonics),
            "ridge_shared": float(args.ridge_shared),
            "ridge_expert": float(args.ridge_expert),
            "top_k": int(args.top_k),
            "sigma_phase": float(args.sigma_phase),
            "sigma_re": float(args.sigma_re),
            "rollout_steps": int(args.rollout_steps),
        },
        "pod_energy": {
            f"velocity_first_{args.r_u}": float(cumulative_uv[args.r_u - 1]),
            f"pressure_first_{args.r_p}": float(cumulative_p[args.r_p - 1]),
        },
        "samples": {
            "total_snapshots": int(a_full.shape[0]),
            "valid_derivative_samples": int(len(valid_idx)),
        },
        "results": results,
        "runtime_seconds": float(time.time() - started),
    }
    return out


def format_float(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.6g}"


def write_markdown_summary(path: Path, result: Dict[str, object]) -> None:
    lines = []
    settings = result["settings"]
    lines.append("# Semi-intrusive Galerkin + Shared-Routed MoE Test Summary")
    lines.append("")
    lines.append("## Scheme")
    lines.append("")
    lines.append(
        "Reduced RHS: `adot = R_galerkin(a,b;Re) + C_shared(x) + "
        "sum_e gate_e(Re,phase) C_e(x)`."
    )
    lines.append("")
    lines.append(
        f"Ranks: velocity r_u={settings['r_u']}, pressure r_p={settings['r_p']}; "
        f"phase Fourier harmonics K={settings['phase_harmonics']}; "
        f"top-k router={settings['top_k']}."
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "| Test Re | Model | RHS relative L2 | RHS RMSE | centered R2 | "
        "one-step relative L2 | improvement vs Galerkin |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for item in result["results"]:
        test_re = item["test_Re"]
        base = item["metrics"]["galerkin_only"]
        one_base = item["one_step_euler"]["galerkin_only"]
        rows = [
            ("Galerkin only", "galerkin_only", base, one_base, 0.0),
            (
                "Galerkin + shared",
                "galerkin_plus_shared",
                item["metrics"]["galerkin_plus_shared"],
                item["one_step_euler"]["galerkin_plus_shared"],
                item["improvement_percent_vs_galerkin_relative_l2"]["shared"],
            ),
            (
                "Galerkin + shared-routed",
                "galerkin_plus_shared_routed",
                item["metrics"]["galerkin_plus_shared_routed"],
                item["one_step_euler"]["galerkin_plus_shared_routed"],
                item["improvement_percent_vs_galerkin_relative_l2"]["shared_routed"],
            ),
        ]
        for name, _, metric, one, imp in rows:
            lines.append(
                f"| {test_re} | {name} | {format_float(metric['relative_l2'])} | "
                f"{format_float(metric['rmse'])} | {format_float(metric['r2_centered'])} | "
                f"{format_float(one['relative_l2'])} | {format_float(imp)}% |"
            )
    lines.append("")
    lines.append("## Rollout")
    lines.append("")
    lines.append(
        "Rollout is a lightweight Euler check using true pressure coefficients and known phase "
        "as context; it is not yet a fully autonomous pressure-coupled ROM."
    )
    lines.append("")
    lines.append("| Test Re | steps | windows | mean relative L2 | median relative L2 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for item in result["results"]:
        roll = item["rollout_teacher_forced_pressure"]["galerkin_plus_shared_routed"]
        lines.append(
            f"| {item['test_Re']} | {item['rollout_teacher_forced_pressure']['steps']} | "
            f"{roll.get('num_windows', 0)} | {format_float(roll['relative_l2_mean'])} | "
            f"{format_float(roll['relative_l2_median'])} |"
        )
    lines.append("")
    lines.append(f"Runtime: {format_float(result['runtime_seconds'])} s.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_experiment(args)

    json_path = args.output_dir / "semi_intrusive_moe_rom_metrics.json"
    md_path = args.output_dir / "semi_intrusive_moe_rom_summary.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown_summary(md_path, result)

    print(json.dumps(result, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
```
