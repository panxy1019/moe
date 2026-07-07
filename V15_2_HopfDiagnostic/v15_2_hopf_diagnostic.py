#!/usr/bin/env python3
"""Diagnostic-only Hopf analysis for V15_1_AdaptiveGate.

This script loads an existing V15_1_AdaptiveGate checkpoint and evaluates
autonomous 24-step RK4 windows. It does not retrain or alter the network.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


EPS = 1.0e-12
TWO_PI = 2.0 * math.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v15-module",
        type=Path,
        default=Path(
            "/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/"
            "train_v15_1_pressure_base_evolution.py"
        ),
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path(
            "/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/results/"
            "V15_1_AdaptiveGate/V15_1_AdaptiveGate_physics_generalizable_ru16_rp16/"
            "V15_1_AdaptiveGate_physics_generalizable_ru16_rp16_metrics.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/results/"
            "V15_1_AdaptiveGate/V15_1_AdaptiveGate_physics_generalizable_ru16_rp16/"
            "V15_1_AdaptiveGate_physics_generalizable_ru16_rp16_Re_24p630436_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/moe/V15_2_HopfDiagnostic/results"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--window-stride", type=int, default=24)
    parser.add_argument("--max-pair-mode", type=int, default=16)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v15_1_train_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import V15 module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_corr(x: Iterable[float], y: Iterable[float]) -> float:
    xv = np.asarray(list(x), dtype=np.float64)
    yv = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if int(np.sum(mask)) < 3:
        return float("nan")
    xv = xv[mask]
    yv = yv[mask]
    if float(np.std(xv)) < 1.0e-12 or float(np.std(yv)) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * x))


def stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def relative_l2(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + EPS))


def regime_group(regime: str) -> str:
    if regime in {"steady_wake", "pre_hopf_steady"}:
        return "Steady"
    if regime == "hopf_transition":
        return "Hopf"
    return "Periodic"


def namespace_from_settings(settings: Dict[str, object]) -> SimpleNamespace:
    defaults: Dict[str, object] = {
        "data_root": Path(
            "/root/moe/ROM_PhysicsGeneralizable/data/Global_POD_AreaWeighted_L2"
        ),
        "tensor_path": Path(
            "/root/moe/ROM_PhysicsGeneralizable/data/"
            "semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz"
        ),
        "pressure_surrogate_path": Path(
            "/root/moe/ROM_PhysicsGeneralizable/data/"
            "pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz"
        ),
        "regime_rom_root": Path("/root/moe/ROM_PhysicsGeneralizable/Regime_ROM_Library"),
        "r_u": 16,
        "r_p": 16,
        "phase_harmonics": 4,
        "history_len": 3,
        "recon_dim": 2048,
        "seed": 1234,
        "test_re_selection": "regime_default",
        "test_re_indices": [10, 59, 99],
        "test_re_values": None,
        "num_uniform_test_re": 10,
        "rhs_target": "residual",
        "pressure_target": "closure",
        "pressure_input_mode": "pressure_only",
        "closure_mode": "adaptive_gate",
        "pressure_base_mode": "static",
        "film_base_hidden": 64,
        "film_base_scale": 0.2,
        "integrator": "rk4",
        "rollout_steps": 24,
        "train_time_stride": 1,
        "train_time_offset": 0,
        "train_re_stride": 1,
        "train_re_offset": 0,
        "regime_balanced_sampling": False,
    }
    for key, value in settings.items():
        cli_key = key.replace("-", "_")
        if cli_key in {"data_root", "tensor_path", "pressure_surrogate_path", "regime_rom_root"}:
            defaults[cli_key] = Path(str(value))
        else:
            defaults[cli_key] = value
    return SimpleNamespace(**defaults)


def checkpoint_scalers(v15, ckpt: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for key, item in ckpt["scalers"].items():
        out[key] = v15.Standardizer(
            mean=np.asarray(item["mean"], dtype=np.float32),
            scale=np.asarray(item["scale"], dtype=np.float32),
        )
    return out


def build_model(v15, arrays: Dict[str, np.ndarray], settings: Dict[str, object], args_ns: SimpleNamespace, ckpt: Dict[str, object], device: torch.device):
    model = v15.OperatorSpaceMoEROM(
        in_dim=arrays["x"].shape[1],
        out_dim=int(args_ns.r_u),
        pressure_dim=int(args_ns.r_p),
        hidden_dim=int(settings.get("hidden_dim", 224)),
        expert_hidden=int(settings.get("expert_hidden", 768)),
        num_blocks=int(settings.get("num_blocks", 3)),
        num_experts=int(settings.get("legacy_num_experts_arg", 6)),
        num_operator_spaces=int(settings.get("num_operator_spaces", 3)),
        num_regime_groups=int(settings.get("num_regime_groups", 3)),
        experts_per_group=int(settings.get("experts_per_group", 6)),
        top_k=int(settings.get("top_k", 2)),
        group_top_k=int(settings.get("group_top_k", 1)),
        dropout=float(settings.get("dropout", 0.04)),
        temperature=float(settings.get("temperature", 0.95)),
        gate_floor=float(settings.get("gate_floor", 0.0)),
        group_temperature=float(settings.get("group_temperature", 0.9)),
        group_gate_floor=float(settings.get("group_gate_floor", 0.0)),
        shared_scale=float(settings.get("shared_scale", 1.0)),
        routed_scale=float(settings.get("routed_scale", 0.85)),
        expert_blocks=int(settings.get("expert_blocks", 3)),
        quadratic_rank=int(settings.get("quadratic_rank", 4)),
        quadratic_scale=float(settings.get("quadratic_scale", 0.05)),
        phase_harmonics=int(settings.get("phase_harmonics", 4)),
        closure_mode=str(settings.get("closure_mode", "adaptive_gate")),
        pressure_base_mode=str(settings.get("pressure_base_mode", "static")),
        film_base_hidden=int(settings.get("film_base_hidden", 64)),
        film_base_scale=float(settings.get("film_base_scale", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def integrate_step(v15, model, a_state, b_state, cur, dt, a_hist, b_hist, rhs_hist, arrays, scalers, tensors, pressure_tensors, args_ns, device):
    k1, pressure_op, rhs_g, closure_params = v15.model_outputs_from_states_np(
        model, a_state, b_state, cur, a_hist, b_hist, rhs_hist, arrays, scalers, tensors, args_ns, device
    )
    if args_ns.integrator == "rk4":
        k2, _, _, _ = v15.model_outputs_from_states_np(
            model, a_state + 0.5 * dt * k1, b_state, cur, a_hist, b_hist, rhs_hist,
            arrays, scalers, tensors, args_ns, device
        )
        k3, _, _, _ = v15.model_outputs_from_states_np(
            model, a_state + 0.5 * dt * k2, b_state, cur, a_hist, b_hist, rhs_hist,
            arrays, scalers, tensors, args_ns, device
        )
        k4, _, _, _ = v15.model_outputs_from_states_np(
            model, a_state + dt * k3, b_state, cur, a_hist, b_hist, rhs_hist,
            arrays, scalers, tensors, args_ns, device
        )
        a_next = a_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        a_next = a_state + dt * k1

    b_base = v15.pressure_surrogate_by_label(
        pressure_tensors,
        a_next[None, :],
        arrays["label_id"][cur : cur + 1],
        arrays["labels"],
        args_ns.r_u,
        args_ns.r_p,
    )[0]
    if args_ns.pressure_target == "state":
        b_next = pressure_op
    else:
        b_next, _, _, _, _ = v15.closure_components_np(
            args_ns.closure_mode,
            b_base[None, :],
            pressure_op[None, :],
            closure_params["alpha"],
            closure_params["beta"],
        )
        b_next = b_next[0]
    return a_next.astype(np.float32), b_next.astype(np.float32), rhs_g.astype(np.float32)


def trace_window(v15, model, arrays, scalers, tensors, pressure_tensors, args_ns, device, start: int, steps: int):
    a_cur = arrays["a"][start].copy()
    b_cur = arrays["b"][start].copy()
    a_hist, b_hist, rhs_hist = v15.init_history_states_np(start, arrays)
    cur = int(start)
    pred_a: List[np.ndarray] = []
    pred_b: List[np.ndarray] = []
    true_a: List[np.ndarray] = []
    true_b: List[np.ndarray] = []
    times: List[float] = []
    ok = True
    for _ in range(steps):
        nxt = int(arrays["next_idx"][cur])
        if nxt < 0 or int(arrays["label_id"][nxt]) != int(arrays["label_id"][cur]):
            ok = False
            break
        dt = float(arrays["time"][nxt] - arrays["time"][cur])
        a_next, b_next, rhs_g = integrate_step(
            v15, model, a_cur, b_cur, cur, dt, a_hist, b_hist, rhs_hist,
            arrays, scalers, tensors, pressure_tensors, args_ns, device
        )
        if not (np.all(np.isfinite(a_next)) and np.all(np.isfinite(b_next))):
            ok = False
            break
        pred_a.append(a_next.copy())
        pred_b.append(b_next.copy())
        true_a.append(arrays["a"][nxt].copy())
        true_b.append(arrays["b"][nxt].copy())
        times.append(float(arrays["time"][nxt]))
        a_hist = np.concatenate([a_next[None, None, :], a_hist[:, :-1, :]], axis=1)
        b_hist = np.concatenate([b_next[None, None, :], b_hist[:, :-1, :]], axis=1)
        rhs_hist = np.concatenate([rhs_g[None, None, :], rhs_hist[:, :-1, :]], axis=1)
        a_cur = a_next
        b_cur = b_next
        cur = nxt
    return ok and len(pred_a) == steps, np.asarray(times), np.asarray(true_a), np.asarray(pred_a), np.asarray(true_b), np.asarray(pred_b)


def dominant_pair(arrays: Dict[str, np.ndarray], heldout_label_ids: List[int], max_mode: int) -> Tuple[int, int, Dict[str, float]]:
    candidates = [(i, i + 1) for i in range(0, max(2, max_mode - 1), 2)]
    mask = np.isin(arrays["label_id"], np.asarray(heldout_label_ids, dtype=np.int64))
    groups = np.asarray([regime_group(r) for r in arrays["regime"]])
    osc_mask = mask & (groups != "Steady")
    if not np.any(osc_mask):
        osc_mask = mask
    scores: Dict[str, float] = {}
    a = arrays["a"][osc_mask]
    for i, j in candidates:
        if j >= a.shape[1]:
            continue
        pair = a[:, [i, j]]
        centered = pair - pair.mean(axis=0, keepdims=True)
        scores[f"{i},{j}"] = float(np.mean(np.sum(centered * centered, axis=1)))
    if not scores:
        return 0, 1, {}
    best = max(scores, key=scores.get)
    i, j = [int(part) for part in best.split(",")]
    return i, j, scores


def modal_metrics(times: np.ndarray, true_a: np.ndarray, pred_a: np.ndarray, pair: Tuple[int, int]) -> Dict[str, object]:
    i, j = pair
    true_i = true_a[:, i]
    true_j = true_a[:, j]
    pred_i = pred_a[:, i]
    pred_j = pred_a[:, j]
    r_true = np.sqrt(true_i * true_i + true_j * true_j)
    r_pred = np.sqrt(pred_i * pred_i + pred_j * pred_j)
    theta_true = np.unwrap(np.arctan2(true_j, true_i))
    theta_pred = np.unwrap(np.arctan2(pred_j, pred_i))
    if len(theta_true):
        offset = round(float((theta_true[0] - theta_pred[0]) / TWO_PI)) * TWO_PI
        theta_pred = theta_pred + offset
    phase_error = wrap_angle(theta_pred - theta_true)
    amp_error = np.abs(r_pred - r_true) / (np.abs(r_true) + 1.0e-8)
    if len(times) >= 3:
        omega_true = np.gradient(theta_true, times)
        omega_pred = np.gradient(theta_pred, times)
        slope_true = float(np.polyfit(times, theta_true, 1)[0])
        slope_pred = float(np.polyfit(times, theta_pred, 1)[0])
    else:
        omega_true = np.full_like(theta_true, np.nan)
        omega_pred = np.full_like(theta_pred, np.nan)
        slope_true = float("nan")
        slope_pred = float("nan")
    f_true = abs(slope_true) / TWO_PI if math.isfinite(slope_true) else float("nan")
    f_pred = abs(slope_pred) / TWO_PI if math.isfinite(slope_pred) else float("nan")
    amp_scale = float(np.median(r_true)) if len(r_true) else float("nan")
    if math.isfinite(f_true) and f_true > 1.0e-8 and math.isfinite(amp_scale) and amp_scale > 1.0e-8:
        freq_rel_error = abs(f_pred - f_true) / (abs(f_true) + EPS)
    else:
        freq_rel_error = float("nan")
    return {
        "r_true": r_true,
        "r_pred": r_pred,
        "theta_true": theta_true,
        "theta_pred": theta_pred,
        "omega_true": omega_true,
        "omega_pred": omega_pred,
        "phase_error": phase_error,
        "amp_error": amp_error,
        "amp_error_mean": float(np.mean(amp_error)) if len(amp_error) else float("nan"),
        "amp_error_median": float(np.median(amp_error)) if len(amp_error) else float("nan"),
        "phase_abs_mean": float(np.mean(np.abs(phase_error))) if len(phase_error) else float("nan"),
        "phase_abs_final": float(abs(phase_error[-1])) if len(phase_error) else float("nan"),
        "freq_true": f_true,
        "freq_pred": f_pred,
        "freq_rel_error": float(freq_rel_error),
        "r_true_mean": float(np.mean(r_true)) if len(r_true) else float("nan"),
        "r_pred_mean": float(np.mean(r_pred)) if len(r_pred) else float("nan"),
    }


def pod_tail_summary(data_root: Path, arrays: Dict[str, np.ndarray], heldout_label_ids: List[int], r_u: int, r_p: int) -> List[Dict[str, object]]:
    vel = np.load(data_root / "global_velocity_pod_area_weighted_l2.npz")
    pre = np.load(data_root / "global_pressure_pod_area_weighted_l2.npz")
    coeff_u = vel["coeff_uv"].astype(np.float64)
    coeff_p = pre["coeff_p"].astype(np.float64)
    rows = []
    for label_id in heldout_label_ids:
        ids = np.where(arrays["label_id"] == int(label_id))[0]
        u_head = np.sum(coeff_u[ids, :r_u] ** 2, axis=1)
        u_tail = np.sum(coeff_u[ids, r_u:] ** 2, axis=1)
        p_head = np.sum(coeff_p[ids, :r_p] ** 2, axis=1)
        p_tail = np.sum(coeff_p[ids, r_p:] ** 2, axis=1)
        rows.append(
            {
                "label_id": int(label_id),
                "label": str(arrays["labels"][int(label_id)]),
                "Re": float(np.mean(arrays["re"][ids])),
                "regime": str(arrays["regime"][ids[0]]) if len(ids) else "unknown",
                "regime_group": regime_group(str(arrays["regime"][ids[0]])) if len(ids) else "unknown",
                "velocity_tail_rel_l2": float(np.sqrt(np.sum(u_tail) / (np.sum(u_head + u_tail) + EPS))),
                "pressure_tail_rel_l2": float(np.sqrt(np.sum(p_tail) / (np.sum(p_head + p_tail) + EPS))),
                "velocity_head_energy_fraction": float(np.sum(u_head) / (np.sum(u_head + u_tail) + EPS)),
                "pressure_head_energy_fraction": float(np.sum(p_head) / (np.sum(p_head + p_tail) + EPS)),
            }
        )
    return rows


def polyline(points: List[Tuple[float, float]], sx, sy, color: str, width: float = 2.0) -> str:
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points if math.isfinite(x) and math.isfinite(y))
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def panel_svg(x: float, y: float, w: float, h: float, title: str, series: List[Tuple[str, np.ndarray, np.ndarray, str]]) -> List[str]:
    vals_x = np.concatenate([sx for _, sx, sy, _ in series if len(sx) and len(sy)])
    vals_y = np.concatenate([sy for _, sx, sy, _ in series if len(sx) and len(sy)])
    vals_x = vals_x[np.isfinite(vals_x)]
    vals_y = vals_y[np.isfinite(vals_y)]
    if len(vals_x) == 0 or len(vals_y) == 0:
        return []
    xmin, xmax = float(vals_x.min()), float(vals_x.max())
    ymin, ymax = float(vals_y.min()), float(vals_y.max())
    if abs(xmax - xmin) < 1.0e-10:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1.0e-10:
        ymax = ymin + 1.0
    ypad = 0.08 * (ymax - ymin)
    xpad = 0.05 * (xmax - xmin)
    ymin -= ypad
    ymax += ypad
    xmin -= xpad
    xmax += xpad

    def sx(v: float) -> float:
        return x + (v - xmin) / (xmax - xmin) * w

    def sy(v: float) -> float:
        return y + h - (v - ymin) / (ymax - ymin) * h

    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#d8e0e8"/>',
        f'<text x="{x + 8}" y="{y + 18}" font-size="13" font-weight="700">{title}</text>',
        f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#536575"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#536575"/>',
    ]
    for label, xs, ys, color in series:
        parts.append(polyline(list(zip(xs.tolist(), ys.tolist())), sx, sy, color))
    for n, (label, _xs, _ys, color) in enumerate(series[:3]):
        lx = x + 8 + n * 88
        ly = y + h - 8
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{lx + 23}" y="{ly + 4}" font-size="11">{label}</text>')
    return parts


def write_re_svg(path: Path, label: str, pair: Tuple[int, int], times: np.ndarray, true_a: np.ndarray, pred_a: np.ndarray, metrics: Dict[str, object]) -> None:
    i, j = pair
    rel_t = times - times[0] if len(times) else times
    width, height = 1180, 920
    panel_w, panel_h = 520, 245
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2d3a}</style>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="42" y="36" font-size="22" font-weight="700">{label}: POD pair ({i},{j}) Hopf diagnostic</text>',
    ]
    panels = [
        (
            42,
            64,
            "phase portrait",
            [
                ("true", true_a[:, i], true_a[:, j], "#1f77b4"),
                ("pred", pred_a[:, i], pred_a[:, j], "#d95f02"),
            ],
        ),
        (
            610,
            64,
            "amplitude r(t)",
            [
                ("true", rel_t, metrics["r_true"], "#1f77b4"),
                ("pred", rel_t, metrics["r_pred"], "#d95f02"),
            ],
        ),
        (
            42,
            354,
            "phase theta(t)",
            [
                ("true", rel_t, metrics["theta_true"], "#1f77b4"),
                ("pred", rel_t, metrics["theta_pred"], "#d95f02"),
            ],
        ),
        (
            610,
            354,
            "frequency omega(t)",
            [
                ("true", rel_t, metrics["omega_true"], "#1f77b4"),
                ("pred", rel_t, metrics["omega_pred"], "#d95f02"),
            ],
        ),
        (
            42,
            644,
            "relative amplitude error",
            [("|r_pred-r_true|/|r_true|", rel_t, metrics["amp_error"], "#756bb1")],
        ),
        (
            610,
            644,
            "wrapped phase error",
            [("wrap(theta_pred-theta_true)", rel_t, metrics["phase_error"], "#31a354")],
        ),
    ]
    for x, y, title, series in panels:
        parts.extend(panel_svg(x, y, panel_w, panel_h, title, series))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_re_timeseries(path: Path, times: np.ndarray, true_a: np.ndarray, pred_a: np.ndarray, true_b: np.ndarray, pred_b: np.ndarray, pair: Tuple[int, int], metrics: Dict[str, object]) -> None:
    i, j = pair
    fields = [
        "time",
        f"a{i}_true",
        f"a{j}_true",
        f"a{i}_pred",
        f"a{j}_pred",
        "r_true",
        "r_pred",
        "theta_true",
        "theta_pred",
        "omega_true",
        "omega_pred",
        "amp_error",
        "phase_error",
        "velocity_step_l2",
        "pressure_step_l2",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for k in range(len(times)):
            writer.writerow(
                {
                    "time": float(times[k]),
                    f"a{i}_true": float(true_a[k, i]),
                    f"a{j}_true": float(true_a[k, j]),
                    f"a{i}_pred": float(pred_a[k, i]),
                    f"a{j}_pred": float(pred_a[k, j]),
                    "r_true": float(metrics["r_true"][k]),
                    "r_pred": float(metrics["r_pred"][k]),
                    "theta_true": float(metrics["theta_true"][k]),
                    "theta_pred": float(metrics["theta_pred"][k]),
                    "omega_true": float(metrics["omega_true"][k]),
                    "omega_pred": float(metrics["omega_pred"][k]),
                    "amp_error": float(metrics["amp_error"][k]),
                    "phase_error": float(metrics["phase_error"][k]),
                    "velocity_step_l2": relative_l2(true_a[k], pred_a[k]),
                    "pressure_step_l2": relative_l2(true_b[k], pred_b[k]),
                }
            )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(v):
        return "nan"
    if abs(v) >= 100:
        return f"{v:.3g}"
    if abs(v) >= 1:
        return f"{v:.4g}"
    return f"{v:.5f}"


def write_report(path: Path, pair: Tuple[int, int], pair_scores: Dict[str, float], per_re: List[Dict[str, object]], pod_rows: List[Dict[str, object]], output_dir: Path) -> None:
    by_label = {row["label"]: row for row in per_re}
    hopf = [row for row in per_re if row["regime_group"] == "Hopf"]
    target = by_label.get("Re_51p786450")
    lines = [
        "# V15_2 Hopf Diagnostic Report",
        "",
        "## Scope",
        "",
        "This diagnostic uses the frozen `V15_1_AdaptiveGate` checkpoint. No network, router, expert, loss, or training data changes were made.",
        "",
        f"Dominant oscillatory POD pair selected from held-out Hopf/Periodic true trajectories: `{pair}`.",
        "",
        "True Lift/Drag coefficient time series are not present in the retained ROM_PhysicsGeneralizable POD database. Therefore true CL/CD amplitude, phase, and frequency errors are marked unavailable rather than replaced by a POD proxy.",
        "",
        "## Main Conclusion",
        "",
    ]
    if target:
        pod_target = next((row for row in pod_rows if row["label"] == "Re_51p786450"), {})
        lines.extend(
            [
                f"For `Re=51.786`, the median representative 24-step window has velocity relative L2 `{fmt(target['representative_velocity_l2'])}` and pressure relative L2 `{fmt(target['representative_pressure_l2'])}`.",
                f"The modal amplitude error is `{fmt(target['amp_error_mean'])}`, mean absolute wrapped phase error is `{fmt(target['phase_abs_mean_rad'])}` rad, and frequency/Strouhal relative error is `{fmt(target['freq_rel_error'])}`.",
                f"The velocity POD tail beyond ru=16 is `{fmt(pod_target.get('velocity_tail_rel_l2', float('nan')))}` and pressure POD tail beyond rp=16 is `{fmt(pod_target.get('pressure_tail_rel_l2', float('nan')))}`.",
                "",
            ]
        )
        amp = float(target["amp_error_mean"])
        phase = float(target["phase_abs_mean_rad"])
        freq = float(target["freq_rel_error"]) if math.isfinite(float(target["freq_rel_error"])) else float("nan")
        vel_tail = float(pod_target.get("velocity_tail_rel_l2", float("nan")))
        pressure_tail = float(pod_target.get("pressure_tail_rel_l2", float("nan")))
        verdict = []
        if math.isfinite(amp) and amp > 0.5:
            verdict.append("amplitude is a major error source")
        if math.isfinite(phase) and phase > 0.8:
            verdict.append("phase drift is a major error source")
        if math.isfinite(freq) and freq > 0.2:
            verdict.append("frequency/Strouhal mismatch is significant")
        if math.isfinite(max(vel_tail, pressure_tail)) and max(vel_tail, pressure_tail) > 0.25:
            verdict.append("POD truncation/projection is non-negligible")
        if not verdict:
            verdict.append("errors are moderate and coupled rather than dominated by a single scalar diagnostic")
        lines.append("Diagnosis: " + "; ".join(verdict) + ".")
    lines.extend(
        [
            "",
            "## Per-Re Summary",
            "",
            "| Re | Regime | windows | 24-step u L2 mean | 24-step p L2 mean | amp err | phase err(rad) | freq/Strouhal err | L2-phase corr | POD u tail | POD p tail |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    pod_by_label = {row["label"]: row for row in pod_rows}
    for row in sorted(per_re, key=lambda item: float(item["Re"])):
        pod = pod_by_label.get(row["label"], {})
        lines.append(
            f"| {float(row['Re']):.3f} | {row['regime_group']} | {int(row['num_windows'])} | "
            f"{fmt(row['velocity_l2_mean'])} | {fmt(row['pressure_l2_mean'])} | "
            f"{fmt(row['amp_error_mean'])} | {fmt(row['phase_abs_mean_rad'])} | "
            f"{fmt(row['freq_rel_error'])} | {fmt(row['l2_phase_corr'])} | "
            f"{fmt(pod.get('velocity_tail_rel_l2', float('nan')))} | "
            f"{fmt(pod.get('pressure_tail_rel_l2', float('nan')))} |"
        )
    lines.extend(
        [
            "",
            "## Hopf-Specific Reading",
            "",
        ]
    )
    for row in sorted(hopf, key=lambda item: float(item["Re"])):
        lines.append(
            f"- Re={float(row['Re']):.3f}: amp error={fmt(row['amp_error_mean'])}, "
            f"phase error={fmt(row['phase_abs_mean_rad'])} rad, "
            f"freq error={fmt(row['freq_rel_error'])}, "
            f"L2-phase corr={fmt(row['l2_phase_corr'])}."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `v15_2_per_re_summary.csv`: per-Re aggregate diagnostics.",
            "- `v15_2_per_window_metrics.csv`: every 24-step window with relative L2 and phase/amplitude/frequency metrics.",
            "- `v15_2_pod_projection_summary.csv`: ru/rp truncation tail energy diagnostics.",
            "- `figures/*_hopf_diagnostic.svg`: phase portrait, amplitude, phase, omega, amplitude error, and wrapped phase error for a representative median-error window.",
            "- `timeseries/*_representative.csv`: raw representative-window traces used by the SVG figures.",
            "",
            "## Dominant Pair Scores",
            "",
        ]
    )
    for key, value in sorted(pair_scores.items()):
        lines.append(f"- pair `{key}`: oscillatory energy score {fmt(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = cli.output_dir / "figures"
    ts_dir = cli.output_dir / "timeseries"
    fig_dir.mkdir(exist_ok=True)
    ts_dir.mkdir(exist_ok=True)

    v15 = load_module(cli.v15_module)
    metrics = json.loads(cli.metrics_json.read_text(encoding="utf-8"))
    settings = metrics["settings"]
    args_ns = namespace_from_settings(settings)
    args_ns.rollout_steps = int(cli.rollout_steps)

    device = torch.device(cli.device if torch.cuda.is_available() and cli.device == "cuda" else "cpu")
    ckpt = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    scalers = checkpoint_scalers(v15, ckpt)

    arrays, meta = v15.build_arrays(args_ns)
    heldout_label_ids = [int(v) for v in ckpt["holdout_label_ids"]]
    tensors = np.load(args_ns.tensor_path)
    pressure_tensors = np.load(args_ns.pressure_surrogate_path)
    model = build_model(v15, arrays, settings, args_ns, ckpt, device)

    pair = dominant_pair(arrays, heldout_label_ids, min(int(cli.max_pair_mode), int(args_ns.r_u)))
    pair_ij = (pair[0], pair[1])
    pair_scores = pair[2]
    pod_rows = pod_tail_summary(args_ns.data_root, arrays, heldout_label_ids, int(args_ns.r_u), int(args_ns.r_p))

    per_re_rows: List[Dict[str, object]] = []
    per_window_rows: List[Dict[str, object]] = []
    valid_sample_set = set(arrays["sample_ids"].tolist())

    for label_id in heldout_label_ids:
        idx = np.where(arrays["label_id"] == int(label_id))[0]
        idx = idx[np.argsort(arrays["time"][idx])]
        label = str(arrays["labels"][int(label_id)])
        re_value = float(np.mean(arrays["re"][idx]))
        regime = str(arrays["regime"][idx[0]]) if len(idx) else "unknown"
        starts = []
        for start_pos in range(1, len(idx) - int(cli.rollout_steps) - 1, max(1, int(cli.window_stride))):
            start = int(idx[start_pos])
            if start in valid_sample_set:
                starts.append(start)
        windows = []
        for start in starts:
            ok, times, true_a, pred_a, true_b, pred_b = trace_window(
                v15, model, arrays, scalers, tensors, pressure_tensors, args_ns, device, start, int(cli.rollout_steps)
            )
            if not ok:
                continue
            mm = modal_metrics(times, true_a, pred_a, pair_ij)
            row = {
                "label_id": int(label_id),
                "label": label,
                "Re": re_value,
                "regime": regime,
                "regime_group": regime_group(regime),
                "start_index": int(start),
                "start_time": float(arrays["time"][start]),
                "end_time": float(times[-1]) if len(times) else float("nan"),
                "velocity_l2": relative_l2(true_a, pred_a),
                "pressure_l2": relative_l2(true_b, pred_b),
                "amp_error_mean": mm["amp_error_mean"],
                "amp_error_median": mm["amp_error_median"],
                "phase_abs_mean_rad": mm["phase_abs_mean"],
                "phase_abs_final_rad": mm["phase_abs_final"],
                "freq_true": mm["freq_true"],
                "freq_pred": mm["freq_pred"],
                "freq_rel_error": mm["freq_rel_error"],
                "strouhal_rel_error": mm["freq_rel_error"],
                "r_true_mean": mm["r_true_mean"],
                "r_pred_mean": mm["r_pred_mean"],
                "lift_drag_available": False,
                "lift_phase_error": float("nan"),
                "lift_amp_error": float("nan"),
                "lift_freq_error": float("nan"),
                "drag_phase_error": float("nan"),
                "drag_amp_error": float("nan"),
                "drag_freq_error": float("nan"),
            }
            windows.append((row, times, true_a, pred_a, true_b, pred_b, mm))
            per_window_rows.append(row)
        if not windows:
            continue
        velocity_l2_values = [float(item[0]["velocity_l2"]) for item in windows]
        phase_values = [float(item[0]["phase_abs_mean_rad"]) for item in windows]
        median_l2 = float(np.median(velocity_l2_values))
        rep_idx = int(np.argmin(np.abs(np.asarray(velocity_l2_values) - median_l2)))
        rep_row, times, true_a, pred_a, true_b, pred_b, mm = windows[rep_idx]
        safe_label = label.replace(".", "p")
        write_re_svg(fig_dir / f"{safe_label}_hopf_diagnostic.svg", label, pair_ij, times, true_a, pred_a, mm)
        write_re_timeseries(
            ts_dir / f"{safe_label}_representative.csv",
            times,
            true_a,
            pred_a,
            true_b,
            pred_b,
            pair_ij,
            mm,
        )
        per_re_rows.append(
            {
                "label_id": int(label_id),
                "label": label,
                "Re": re_value,
                "regime": regime,
                "regime_group": regime_group(regime),
                "dominant_pair_i": pair_ij[0],
                "dominant_pair_j": pair_ij[1],
                "num_windows": int(len(windows)),
                "representative_start_time": rep_row["start_time"],
                "representative_velocity_l2": rep_row["velocity_l2"],
                "representative_pressure_l2": rep_row["pressure_l2"],
                "velocity_l2_mean": stats(velocity_l2_values)["mean"],
                "velocity_l2_std": stats(velocity_l2_values)["std"],
                "pressure_l2_mean": stats([float(item[0]["pressure_l2"]) for item in windows])["mean"],
                "pressure_l2_std": stats([float(item[0]["pressure_l2"]) for item in windows])["std"],
                "amp_error_mean": stats([float(item[0]["amp_error_mean"]) for item in windows])["mean"],
                "amp_error_std": stats([float(item[0]["amp_error_mean"]) for item in windows])["std"],
                "phase_abs_mean_rad": stats(phase_values)["mean"],
                "phase_abs_std_rad": stats(phase_values)["std"],
                "phase_abs_final_rad_mean": stats([float(item[0]["phase_abs_final_rad"]) for item in windows])["mean"],
                "freq_true": stats([float(item[0]["freq_true"]) for item in windows])["mean"],
                "freq_pred": stats([float(item[0]["freq_pred"]) for item in windows])["mean"],
                "freq_rel_error": stats([float(item[0]["freq_rel_error"]) for item in windows])["mean"],
                "strouhal_rel_error": stats([float(item[0]["strouhal_rel_error"]) for item in windows])["mean"],
                "l2_phase_corr": finite_corr(velocity_l2_values, phase_values),
                "l2_amp_corr": finite_corr(velocity_l2_values, [float(item[0]["amp_error_mean"]) for item in windows]),
                "lift_drag_available": False,
            }
        )

    write_csv(cli.output_dir / "v15_2_per_re_summary.csv", per_re_rows)
    write_csv(cli.output_dir / "v15_2_per_window_metrics.csv", per_window_rows)
    write_csv(cli.output_dir / "v15_2_pod_projection_summary.csv", pod_rows)
    summary = {
        "checkpoint": str(cli.checkpoint),
        "metrics_json": str(cli.metrics_json),
        "dominant_pair": list(pair_ij),
        "dominant_pair_scores": pair_scores,
        "per_re": per_re_rows,
        "pod_projection": pod_rows,
        "lift_drag_available": False,
        "data_meta": {
            "num_re": meta.get("num_re"),
            "valid_samples": meta.get("valid_samples"),
            "history_len": meta.get("history_len"),
        },
    }
    (cli.output_dir / "v15_2_hopf_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path = cli.output_dir / "V15_2_HOPF_DIAGNOSTIC_REPORT.md"
    write_report(report_path, pair_ij, pair_scores, per_re_rows, pod_rows, cli.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(cli.output_dir),
                "report": str(report_path),
                "dominant_pair": list(pair_ij),
                "num_re": len(per_re_rows),
                "num_windows": len(per_window_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
