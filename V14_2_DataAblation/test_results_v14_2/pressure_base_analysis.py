#!/usr/bin/env python3
"""Pressure base/residual/closure diagnostics for a V14/V14_2 closure run.

This script does not alter or retrain the model. It reloads the checkpoint from
an existing dense closure run and evaluates three pressure readout modes:

* BaseOnly: b_pred = b_base
* ResidualOnlyState: b_pred = pressure_head
* ClosureCurrent: b_pred = b_base + pressure_head

Velocity integration, Galerkin RHS, RK4, encoder, routers, and expert weights
are kept identical to the source training run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from train_v14_2 import (
    EPS,
    OperatorSpaceMoEROM,
    Standardizer,
    build_arrays,
    init_history_states_np,
    model_outputs_from_states_np,
    pressure_surrogate_by_label,
    relative_l2_np,
    rmse_np,
)


PRESSURE_MODES = {
    "BaseOnly": "base",
    "ResidualOnlyState": "residual",
    "ClosureCurrent": "closure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--tensor-path", type=Path)
    parser.add_argument("--pressure-surrogate-path", type=Path)
    parser.add_argument(
        "--poisson-doc",
        type=Path,
        default=Path(
            "/root/moe/V8/data/"
            "PRESSURE_POISSON_SURROGATE_TENSORS_allRe100_weightedL2_ru80_rp80.md"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-tf32", action="store_true", default=True)
    return parser.parse_args()


def as_path(value: object, fallback: Path | None = None) -> Path:
    if value is None:
        if fallback is None:
            raise ValueError("Missing required path.")
        return fallback
    return Path(str(value))


def namespace_from_metrics(metrics: Dict[str, object], cli: argparse.Namespace) -> SimpleNamespace:
    settings = metrics["settings"]
    args = SimpleNamespace()
    args.data_root = cli.data_root or as_path(metrics.get("data_root"))
    args.tensor_path = cli.tensor_path or as_path(metrics.get("tensor_path"))
    args.pressure_surrogate_path = cli.pressure_surrogate_path or as_path(
        metrics.get("pressure_surrogate_path")
    )
    args.r_u = int(settings["r_u"])
    args.r_p = int(settings["r_p"])
    args.phase_harmonics = int(settings["phase_harmonics"])
    args.history_len = int(settings["history_len"])
    args.recon_dim = int(settings.get("recon_dim", 0))
    args.seed = int(settings.get("seed", 1234))
    args.rhs_target = str(settings.get("rhs_target", "residual"))
    args.pressure_target = str(settings.get("pressure_target", "closure"))
    args.integrator = str(settings.get("integrator", "rk4"))
    args.test_re_selection = str(settings.get("test_re_selection", "explicit"))
    args.num_uniform_test_re = int(settings.get("num_uniform_test_re", 10))
    args.test_re_indices = [int(v) for v in settings.get("test_re_indices", [])]
    args.train_time_stride = int(settings.get("train_time_stride", 1))
    args.train_time_offset = int(settings.get("train_time_offset", 0))
    args.train_re_stride = int(settings.get("train_re_stride", 1))
    args.train_re_offset = int(settings.get("train_re_offset", 0))
    args.rollout_steps = int(settings.get("rollout_steps", 24))
    args.device = cli.device
    return args


def load_scalers(raw: Dict[str, object]) -> Dict[str, Standardizer]:
    out: Dict[str, Standardizer] = {}
    for key, value in raw.items():
        item = value
        out[key] = Standardizer(
            mean=np.asarray(item["mean"], dtype=np.float32),
            scale=np.asarray(item["scale"], dtype=np.float32),
        )
    return out


def instantiate_model(
    settings: Dict[str, object],
    in_dim: int,
    device: torch.device,
) -> OperatorSpaceMoEROM:
    return OperatorSpaceMoEROM(
        in_dim=in_dim,
        out_dim=int(settings["r_u"]),
        pressure_dim=int(settings["r_p"]),
        hidden_dim=int(settings["hidden_dim"]),
        expert_hidden=int(settings["expert_hidden"]),
        num_blocks=int(settings["num_blocks"]),
        num_experts=int(settings.get("legacy_num_experts_arg", settings.get("num_experts", 6))),
        num_operator_spaces=int(settings.get("num_operator_spaces", 1)),
        num_regime_groups=int(settings["num_regime_groups"]),
        experts_per_group=int(settings["experts_per_group"]),
        top_k=int(settings["top_k"]),
        group_top_k=int(settings["group_top_k"]),
        dropout=float(settings["dropout"]),
        temperature=float(settings["temperature"]),
        gate_floor=float(settings["gate_floor"]),
        group_temperature=float(settings["group_temperature"]),
        group_gate_floor=float(settings["group_gate_floor"]),
        shared_scale=float(settings["shared_scale"]),
        routed_scale=float(settings["routed_scale"]),
        expert_blocks=int(settings["expert_blocks"]),
        quadratic_rank=int(settings["quadratic_rank"]),
        quadratic_scale=float(settings["quadratic_scale"]),
        phase_harmonics=int(settings["phase_harmonics"]),
    ).to(device)


def l2_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def row_relative_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    err = np.linalg.norm(y_pred - y_true, axis=1)
    denom = np.linalg.norm(y_true, axis=1) + EPS
    return (err / denom).astype(np.float64)


def pressure_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    true_energy = np.sum(y_true * y_true, axis=1)
    pred_energy = np.sum(y_pred * y_pred, axis=1)
    err_energy = np.sum((y_pred - y_true) ** 2, axis=1)
    return {
        "relative_l2": relative_l2_np(y_true, y_pred),
        "rmse": rmse_np(y_true, y_pred),
        "sample_relative_l2_mean": float(np.mean(row_relative_errors(y_true, y_pred))),
        "sample_relative_l2_median": float(np.median(row_relative_errors(y_true, y_pred))),
        "pressure_true_energy_mean": float(np.mean(true_energy)),
        "pressure_pred_energy_mean": float(np.mean(pred_energy)),
        "pressure_error_energy_mean": float(np.mean(err_energy)),
        "pressure_energy_relative_error": float(
            abs(np.sum(pred_energy) - np.sum(true_energy)) / (np.sum(true_energy) + EPS)
        ),
    }


def component_diagnostics(
    true_b: np.ndarray,
    base_b: np.ndarray,
    residual_b: np.ndarray,
    closure_b: np.ndarray,
) -> Dict[str, float | bool]:
    base_error = relative_l2_np(true_b, base_b)
    residual_error = relative_l2_np(true_b, residual_b)
    closure_error = relative_l2_np(true_b, closure_b)
    residual_norm = l2_norm(residual_b)
    base_norm = l2_norm(base_b)
    true_norm = l2_norm(true_b)
    closure_norm = l2_norm(closure_b)
    contribution_ratio = residual_norm / (base_norm + residual_norm + EPS)
    final_contribution_ratio = residual_norm / (closure_norm + EPS)
    return {
        "base_pressure_error_relative_l2": base_error,
        "residual_only_pressure_error_relative_l2": residual_error,
        "closure_pressure_error_relative_l2": closure_error,
        "closure_absolute_improvement_vs_base": float(base_error - closure_error),
        "closure_relative_improvement_percent_vs_base": float(
            100.0 * (base_error - closure_error) / (base_error + EPS)
        ),
        "residual_energy_ratio": float(residual_norm / (true_norm + EPS)),
        "residual_to_base_ratio": float(residual_norm / (base_norm + EPS)),
        "pressure_head_contribution_ratio": float(contribution_ratio),
        "pressure_head_final_contribution_ratio": float(final_contribution_ratio),
        "pressure_head_dominant": bool(contribution_ratio >= 0.5 or residual_norm > base_norm),
    }


def select_pressure_prediction(
    mode: str,
    base_b: np.ndarray,
    residual_b: np.ndarray,
) -> np.ndarray:
    if mode == "base":
        return base_b
    if mode == "residual":
        return residual_b
    if mode == "closure":
        return base_b + residual_b
    raise ValueError(f"Unknown pressure mode: {mode}")


def integrate_pressure_mode_np(
    model: OperatorSpaceMoEROM,
    a_state: np.ndarray,
    b_state: np.ndarray,
    cur: int,
    dt: float,
    a_hist: np.ndarray,
    b_hist: np.ndarray,
    rhs_hist: np.ndarray,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    args: SimpleNamespace,
    mode: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    k1, residual_b, rhs_g = model_outputs_from_states_np(
        model, a_state, b_state, cur, a_hist, b_hist, rhs_hist, arrays, scalers, tensors, args, device
    )
    if args.integrator == "rk4":
        k2, _, _ = model_outputs_from_states_np(
            model,
            a_state + 0.5 * dt * k1,
            b_state,
            cur,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            args,
            device,
        )
        k3, _, _ = model_outputs_from_states_np(
            model,
            a_state + 0.5 * dt * k2,
            b_state,
            cur,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            args,
            device,
        )
        k4, _, _ = model_outputs_from_states_np(
            model,
            a_state + dt * k3,
            b_state,
            cur,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            args,
            device,
        )
        a_next = a_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        a_next = a_state + dt * k1
    base_b = pressure_surrogate_by_label(
        pressure_tensors,
        a_next[None, :],
        arrays["label_id"][cur : cur + 1],
        arrays["labels"],
        args.r_u,
        args.r_p,
    )[0]
    b_next = select_pressure_prediction(mode, base_b, residual_b)
    components = {
        "base": base_b.astype(np.float32),
        "residual": residual_b.astype(np.float32),
        "closure": (base_b + residual_b).astype(np.float32),
    }
    return a_next.astype(np.float32), b_next.astype(np.float32), rhs_g.astype(np.float32), components


def one_step_pressure_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    sample_ids: np.ndarray,
    args: SimpleNamespace,
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    true_a: List[np.ndarray] = []
    pred_a: List[np.ndarray] = []
    true_b: List[np.ndarray] = []
    base_b: List[np.ndarray] = []
    residual_b: List[np.ndarray] = []
    closure_b: List[np.ndarray] = []
    for cur in sample_ids.tolist():
        nxt = int(arrays["next_idx"][cur])
        if nxt < 0 or int(arrays["label_id"][nxt]) != int(arrays["label_id"][cur]):
            continue
        dt = float(arrays["time"][nxt] - arrays["time"][cur])
        a_hist, b_hist, rhs_hist = init_history_states_np(int(cur), arrays)
        a_next, _, _, components = integrate_pressure_mode_np(
            model,
            arrays["a"][cur].copy(),
            arrays["b"][cur].copy(),
            int(cur),
            dt,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            pressure_tensors,
            args,
            "closure",
            device,
        )
        true_a.append(arrays["a"][nxt])
        pred_a.append(a_next)
        true_b.append(arrays["b"][nxt])
        base_b.append(components["base"])
        residual_b.append(components["residual"])
        closure_b.append(components["closure"])

    if not true_b:
        empty = np.empty((0, args.r_p), dtype=np.float32)
        return {"num_samples": 0, "modes": {}, "diagnostics": {}}, {
            "true_b": empty,
            "base": empty,
            "residual": empty,
            "closure": empty,
        }

    true_a_arr = np.asarray(true_a, dtype=np.float32)
    pred_a_arr = np.asarray(pred_a, dtype=np.float32)
    true_b_arr = np.asarray(true_b, dtype=np.float32)
    base_arr = np.asarray(base_b, dtype=np.float32)
    residual_arr = np.asarray(residual_b, dtype=np.float32)
    closure_arr = np.asarray(closure_b, dtype=np.float32)
    modes: Dict[str, object] = {}
    for public_name, mode_name in PRESSURE_MODES.items():
        pred_b = select_pressure_prediction(mode_name, base_arr, residual_arr)
        modes[public_name] = {
            "a_relative_l2": relative_l2_np(true_a_arr, pred_a_arr),
            "a_rmse": rmse_np(true_a_arr, pred_a_arr),
            **pressure_metrics(true_b_arr, pred_b),
            "num_samples": int(len(true_b_arr)),
        }
    return {
        "num_samples": int(len(true_b_arr)),
        "modes": modes,
        "diagnostics": component_diagnostics(true_b_arr, base_arr, residual_arr, closure_arr),
    }, {
        "true_b": true_b_arr,
        "base": base_arr,
        "residual": residual_arr,
        "closure": closure_arr,
    }


def rollout_pressure_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    test_label_id: int,
    args: SimpleNamespace,
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    idx = np.where(arrays["label_id"] == test_label_id)[0]
    idx = idx[np.argsort(arrays["time"][idx])]
    valid_sample_set = set(arrays["sample_ids"].tolist())
    mode_outputs: Dict[str, Dict[str, List[np.ndarray] | List[float]]] = {
        name: {
            "true_a": [],
            "pred_a": [],
            "true_b": [],
            "pred_b": [],
            "base": [],
            "residual": [],
            "closure": [],
            "a_window_errors": [],
            "b_window_errors": [],
        }
        for name in PRESSURE_MODES
    }
    stride = max(1, int(args.rollout_steps))
    model.eval()
    for public_name, mode_name in PRESSURE_MODES.items():
        store = mode_outputs[public_name]
        for start_pos in range(1, len(idx) - int(args.rollout_steps) - 1, stride):
            start = int(idx[start_pos])
            if start not in valid_sample_set:
                continue
            a_cur = arrays["a"][start].copy()
            b_cur = arrays["b"][start].copy()
            a_hist, b_hist, rhs_hist = init_history_states_np(start, arrays)
            cur = start
            pred_a_window: List[np.ndarray] = []
            pred_b_window: List[np.ndarray] = []
            base_window: List[np.ndarray] = []
            residual_window: List[np.ndarray] = []
            closure_window: List[np.ndarray] = []
            ok = True
            for _ in range(int(args.rollout_steps)):
                nxt = int(arrays["next_idx"][cur])
                if nxt < 0 or int(arrays["label_id"][nxt]) != int(test_label_id):
                    ok = False
                    break
                dt = float(arrays["time"][nxt] - arrays["time"][cur])
                a_next, b_next, rhs_g, components = integrate_pressure_mode_np(
                    model,
                    a_cur,
                    b_cur,
                    cur,
                    dt,
                    a_hist,
                    b_hist,
                    rhs_hist,
                    arrays,
                    scalers,
                    tensors,
                    pressure_tensors,
                    args,
                    mode_name,
                    device,
                )
                if not (np.all(np.isfinite(a_next)) and np.all(np.isfinite(b_next))):
                    ok = False
                    break
                pred_a_window.append(a_next.copy())
                pred_b_window.append(b_next.copy())
                base_window.append(components["base"].copy())
                residual_window.append(components["residual"].copy())
                closure_window.append(components["closure"].copy())
                a_hist = np.concatenate([a_next[None, None, :], a_hist[:, :-1, :]], axis=1)
                b_hist = np.concatenate([b_next[None, None, :], b_hist[:, :-1, :]], axis=1)
                rhs_hist = np.concatenate([rhs_g[None, None, :], rhs_hist[:, :-1, :]], axis=1)
                a_cur = a_next
                b_cur = b_next
                cur = nxt
            if ok and len(pred_a_window) == int(args.rollout_steps):
                true_a_window = arrays["a"][idx[start_pos + 1 : start_pos + int(args.rollout_steps) + 1]]
                true_b_window = arrays["b"][idx[start_pos + 1 : start_pos + int(args.rollout_steps) + 1]]
                pred_a_arr = np.asarray(pred_a_window, dtype=np.float32)
                pred_b_arr = np.asarray(pred_b_window, dtype=np.float32)
                store["true_a"].append(true_a_window)
                store["pred_a"].append(pred_a_arr)
                store["true_b"].append(true_b_window)
                store["pred_b"].append(pred_b_arr)
                store["base"].append(np.asarray(base_window, dtype=np.float32))
                store["residual"].append(np.asarray(residual_window, dtype=np.float32))
                store["closure"].append(np.asarray(closure_window, dtype=np.float32))
                store["a_window_errors"].append(relative_l2_np(true_a_window, pred_a_arr))
                store["b_window_errors"].append(relative_l2_np(true_b_window, pred_b_arr))

    modes: Dict[str, object] = {}
    diagnostic_reference: Dict[str, np.ndarray] = {}
    distributions: Dict[str, np.ndarray] = {}
    for public_name, store in mode_outputs.items():
        if not store["true_b"]:
            modes[public_name] = {
                "a_relative_l2_mean": float("nan"),
                "b_relative_l2_mean": float("nan"),
                "num_windows": 0,
            }
            continue
        true_a_arr = np.concatenate(store["true_a"], axis=0)
        pred_a_arr = np.concatenate(store["pred_a"], axis=0)
        true_b_arr = np.concatenate(store["true_b"], axis=0)
        pred_b_arr = np.concatenate(store["pred_b"], axis=0)
        b_window_errors = np.asarray(store["b_window_errors"], dtype=np.float64)
        a_window_errors = np.asarray(store["a_window_errors"], dtype=np.float64)
        modes[public_name] = {
            "a_relative_l2_mean": float(np.mean(a_window_errors)),
            "a_relative_l2_median": float(np.median(a_window_errors)),
            "a_relative_l2_p90": float(np.percentile(a_window_errors, 90)),
            "a_relative_l2_max": float(np.max(a_window_errors)),
            "b_relative_l2_mean": float(np.mean(b_window_errors)),
            "b_relative_l2_median": float(np.median(b_window_errors)),
            "b_relative_l2_p90": float(np.percentile(b_window_errors, 90)),
            "b_relative_l2_max": float(np.max(b_window_errors)),
            **pressure_metrics(true_b_arr, pred_b_arr),
            "num_windows": int(len(b_window_errors)),
            "num_points": int(len(true_b_arr)),
        }
        distributions[f"{public_name}_window_b_relative_l2"] = b_window_errors
        distributions[f"{public_name}_point_b_relative_l2"] = row_relative_errors(true_b_arr, pred_b_arr)
        if public_name == "ClosureCurrent":
            diagnostic_reference = {
                "true_b": true_b_arr,
                "base": np.concatenate(store["base"], axis=0),
                "residual": np.concatenate(store["residual"], axis=0),
                "closure": np.concatenate(store["closure"], axis=0),
            }

    diagnostics = {}
    if diagnostic_reference:
        diagnostics = component_diagnostics(
            diagnostic_reference["true_b"],
            diagnostic_reference["base"],
            diagnostic_reference["residual"],
            diagnostic_reference["closure"],
        )
        # For rollout, BaseOnly and ClosureCurrent are independent closed-loop
        # trajectories because the chosen pressure state feeds back into the
        # next Galerkin RHS. Keep component ratios from the closure trajectory,
        # but report improvement using the mode-level rollout errors.
        base_mode_error = float(modes.get("BaseOnly", {}).get("b_relative_l2_mean", float("nan")))
        residual_mode_error = float(
            modes.get("ResidualOnlyState", {}).get("b_relative_l2_mean", float("nan"))
        )
        closure_mode_error = float(
            modes.get("ClosureCurrent", {}).get("b_relative_l2_mean", float("nan"))
        )
        if all(np.isfinite(v) for v in [base_mode_error, residual_mode_error, closure_mode_error]):
            diagnostics["component_base_pressure_error_relative_l2"] = diagnostics[
                "base_pressure_error_relative_l2"
            ]
            diagnostics["component_closure_pressure_error_relative_l2"] = diagnostics[
                "closure_pressure_error_relative_l2"
            ]
            diagnostics["base_pressure_error_relative_l2"] = base_mode_error
            diagnostics["residual_only_pressure_error_relative_l2"] = residual_mode_error
            diagnostics["closure_pressure_error_relative_l2"] = closure_mode_error
            diagnostics["closure_absolute_improvement_vs_base"] = float(
                base_mode_error - closure_mode_error
            )
            diagnostics["closure_relative_improvement_percent_vs_base"] = float(
                100.0 * (base_mode_error - closure_mode_error) / (base_mode_error + EPS)
            )
    return {"modes": modes, "diagnostics": diagnostics}, diagnostic_reference | distributions


def stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate(results: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"num_test_re": int(len(results)), "one_step": {}, "rollout": {}}
    for phase in ["one_step", "rollout"]:
        phase_out: Dict[str, object] = {}
        for public_name in PRESSURE_MODES:
            if phase == "one_step":
                rels = [
                    float(item[phase]["modes"][public_name]["relative_l2"])
                    for item in results
                ]
                energy = [
                    float(item[phase]["modes"][public_name]["pressure_energy_relative_error"])
                    for item in results
                ]
            else:
                rels = [
                    float(item[phase]["modes"][public_name]["b_relative_l2_mean"])
                    for item in results
                ]
                energy = [
                    float(item[phase]["modes"][public_name]["pressure_energy_relative_error"])
                    for item in results
                ]
            phase_out[public_name] = {
                "pressure_relative_l2": stats(rels),
                "pressure_energy_relative_error": stats(energy),
            }
        phase_out["diagnostics"] = {
            "base_error": stats(
                [float(item[phase]["diagnostics"]["base_pressure_error_relative_l2"]) for item in results]
            ),
            "closure_error": stats(
                [float(item[phase]["diagnostics"]["closure_pressure_error_relative_l2"]) for item in results]
            ),
            "closure_improvement_percent": stats(
                [
                    float(item[phase]["diagnostics"]["closure_relative_improvement_percent_vs_base"])
                    for item in results
                ]
            ),
            "residual_energy_ratio": stats(
                [float(item[phase]["diagnostics"]["residual_energy_ratio"]) for item in results]
            ),
            "residual_to_base_ratio": stats(
                [float(item[phase]["diagnostics"]["residual_to_base_ratio"]) for item in results]
            ),
            "contribution_ratio": stats(
                [float(item[phase]["diagnostics"]["pressure_head_contribution_ratio"]) for item in results]
            ),
        }
        out[phase] = phase_out
    return out


def write_comparison_csv(path: Path, results: List[Dict[str, object]]) -> None:
    fields = [
        "Re",
        "label",
        "num_test",
        "one_base_l2",
        "one_residual_only_l2",
        "one_closure_l2",
        "one_abs_improvement",
        "one_rel_improvement_pct",
        "one_residual_energy_ratio",
        "one_residual_to_base_ratio",
        "one_contribution_ratio",
        "one_head_dominant",
        "roll_base_l2",
        "roll_residual_only_l2",
        "roll_closure_l2",
        "roll_abs_improvement",
        "roll_rel_improvement_pct",
        "roll_residual_energy_ratio",
        "roll_residual_to_base_ratio",
        "roll_contribution_ratio",
        "roll_head_dominant",
        "rollout_windows",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in sorted(results, key=lambda row: float(row["test_Re"])):
            one = item["one_step"]
            roll = item["rollout"]
            writer.writerow(
                {
                    "Re": item["test_Re"],
                    "label": item["test_Re_label"],
                    "num_test": item["num_test"],
                    "one_base_l2": one["modes"]["BaseOnly"]["relative_l2"],
                    "one_residual_only_l2": one["modes"]["ResidualOnlyState"]["relative_l2"],
                    "one_closure_l2": one["modes"]["ClosureCurrent"]["relative_l2"],
                    "one_abs_improvement": one["diagnostics"]["closure_absolute_improvement_vs_base"],
                    "one_rel_improvement_pct": one["diagnostics"][
                        "closure_relative_improvement_percent_vs_base"
                    ],
                    "one_residual_energy_ratio": one["diagnostics"]["residual_energy_ratio"],
                    "one_residual_to_base_ratio": one["diagnostics"]["residual_to_base_ratio"],
                    "one_contribution_ratio": one["diagnostics"]["pressure_head_contribution_ratio"],
                    "one_head_dominant": one["diagnostics"]["pressure_head_dominant"],
                    "roll_base_l2": roll["modes"]["BaseOnly"]["b_relative_l2_mean"],
                    "roll_residual_only_l2": roll["modes"]["ResidualOnlyState"]["b_relative_l2_mean"],
                    "roll_closure_l2": roll["modes"]["ClosureCurrent"]["b_relative_l2_mean"],
                    "roll_abs_improvement": roll["diagnostics"]["closure_absolute_improvement_vs_base"],
                    "roll_rel_improvement_pct": roll["diagnostics"][
                        "closure_relative_improvement_percent_vs_base"
                    ],
                    "roll_residual_energy_ratio": roll["diagnostics"]["residual_energy_ratio"],
                    "roll_residual_to_base_ratio": roll["diagnostics"]["residual_to_base_ratio"],
                    "roll_contribution_ratio": roll["diagnostics"]["pressure_head_contribution_ratio"],
                    "roll_head_dominant": roll["diagnostics"]["pressure_head_dominant"],
                    "rollout_windows": roll["modes"]["ClosureCurrent"].get("num_windows", 0),
                }
            )


def write_distribution_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = ["Re", "phase", "mode", "kind", "relative_l2"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def svg_polyline(points: List[Tuple[float, float]], color: str) -> str:
    if len(points) < 2:
        return ""
    return (
        f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" '
        f'fill="none" stroke="{color}" stroke-width="2.5"/>'
    )


def write_error_vs_re_svg(path: Path, results: List[Dict[str, object]], phase: str) -> None:
    ordered = sorted(results, key=lambda row: float(row["test_Re"]))
    width, height = 1180, 700
    left, right, top, bottom = 85, 35, 55, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    re_vals = np.asarray([float(item["test_Re"]) for item in ordered], dtype=np.float64)
    colors = {
        "BaseOnly": "#1f77b4",
        "ResidualOnlyState": "#ff7f0e",
        "ClosureCurrent": "#2ca02c",
    }
    series: Dict[str, List[float]] = {}
    for name in PRESSURE_MODES:
        if phase == "one_step":
            series[name] = [float(item[phase]["modes"][name]["relative_l2"]) for item in ordered]
        else:
            series[name] = [float(item[phase]["modes"][name]["b_relative_l2_mean"]) for item in ordered]
    all_vals = np.asarray([v for vals in series.values() for v in vals if np.isfinite(v)])
    y_max = float(max(0.1, np.max(all_vals) * 1.08)) if len(all_vals) else 1.0
    x_min, x_max = float(np.min(re_vals)), float(np.max(re_vals))
    if abs(x_max - x_min) < EPS:
        x_max = x_min + 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - min(max(y, 0.0), y_max) / y_max * plot_h

    title = f"Pressure {phase.replace('_', '-')} relative L2 vs Re"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#1f2d3a}.axis{stroke:#4d5d6c;stroke-width:1.4}.grid{stroke:#d8e0e8;stroke-width:1}</style>",
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="{left}" y="34" font-size="24" font-weight="700">{title}</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
    ]
    for frac in np.linspace(0.0, 1.0, 6):
        y = top + plot_h - frac * plot_h
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text x="18" y="{y + 4:.1f}" font-size="12">{frac * y_max:.2f}</text>')
    for x in np.linspace(x_min, x_max, 6):
        parts.append(f'<text x="{sx(float(x)) - 18:.1f}" y="{top + plot_h + 28}" font-size="12">{x:.0f}</text>')
    parts.append(f'<text x="{left + plot_w / 2 - 35:.1f}" y="{height - 25}" font-size="14">Reynolds number</text>')
    legend_x, legend_y = left + 18, top + 20
    for i, (name, vals) in enumerate(series.items()):
        points = [(sx(float(r)), sy(float(v))) for r, v in zip(re_vals, vals) if np.isfinite(v)]
        parts.append(svg_polyline(points, colors[name]))
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[name]}" stroke="#fff"/>')
        lx = legend_x + i * 245
        parts.append(f'<rect x="{lx}" y="{legend_y - 10}" width="18" height="5" fill="{colors[name]}"/>')
        parts.append(f'<text x="{lx + 26}" y="{legend_y - 4}" font-size="13">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_base_residual_svg(path: Path, results: List[Dict[str, object]]) -> None:
    ordered = sorted(results, key=lambda row: float(row["test_Re"]))
    width, height = 1180, 700
    left, right, top, bottom = 85, 35, 55, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    re_vals = np.asarray([float(item["test_Re"]) for item in ordered], dtype=np.float64)
    series = {
        "Base error one-step": [
            float(item["one_step"]["diagnostics"]["base_pressure_error_relative_l2"])
            for item in ordered
        ],
        "Closure error one-step": [
            float(item["one_step"]["diagnostics"]["closure_pressure_error_relative_l2"])
            for item in ordered
        ],
        "Residual magnitude / true": [
            float(item["one_step"]["diagnostics"]["residual_energy_ratio"])
            for item in ordered
        ],
        "Residual / base": [
            float(item["one_step"]["diagnostics"]["residual_to_base_ratio"])
            for item in ordered
        ],
    }
    colors = {
        "Base error one-step": "#1f77b4",
        "Closure error one-step": "#2ca02c",
        "Residual magnitude / true": "#d62728",
        "Residual / base": "#9467bd",
    }
    vals = np.asarray([v for row in series.values() for v in row if np.isfinite(v)])
    y_max = float(max(0.1, np.max(vals) * 1.08)) if len(vals) else 1.0
    x_min, x_max = float(np.min(re_vals)), float(np.max(re_vals))

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - min(max(y, 0.0), y_max) / y_max * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#1f2d3a}.axis{stroke:#4d5d6c;stroke-width:1.4}.grid{stroke:#d8e0e8;stroke-width:1}</style>",
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="{left}" y="34" font-size="24" font-weight="700">Base error and residual magnitude vs Re</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
    ]
    for frac in np.linspace(0.0, 1.0, 6):
        y = top + plot_h - frac * plot_h
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text x="18" y="{y + 4:.1f}" font-size="12">{frac * y_max:.2f}</text>')
    for x in np.linspace(x_min, x_max, 6):
        parts.append(f'<text x="{sx(float(x)) - 18:.1f}" y="{top + plot_h + 28}" font-size="12">{x:.0f}</text>')
    for i, (name, values) in enumerate(series.items()):
        points = [(sx(float(r)), sy(float(v))) for r, v in zip(re_vals, values) if np.isfinite(v)]
        parts.append(svg_polyline(points, colors[name]))
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[name]}" stroke="#fff"/>')
        lx = left + 18 + (i % 2) * 360
        ly = top + 20 + (i // 2) * 24
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="18" height="5" fill="{colors[name]}"/>')
        parts.append(f'<text x="{lx + 26}" y="{ly - 4}" font-size="13">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_distribution_svg(path: Path, rows: List[Dict[str, object]], phase: str, kind: str) -> None:
    selected = [row for row in rows if row["phase"] == phase and row["kind"] == kind]
    width, height = 1180, 690
    left, right, top, bottom = 85, 35, 55, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = {
        "BaseOnly": "#1f77b4",
        "ResidualOnlyState": "#ff7f0e",
        "ClosureCurrent": "#2ca02c",
    }
    values_by_mode = {
        mode: np.asarray([float(row["relative_l2"]) for row in selected if row["mode"] == mode])
        for mode in PRESSURE_MODES
    }
    all_vals = np.asarray([v for vals in values_by_mode.values() for v in vals if np.isfinite(v)])
    if len(all_vals) == 0:
        path.write_text("", encoding="utf-8")
        return
    x_max = float(max(0.1, np.percentile(all_vals, 98) * 1.15))
    bins = np.linspace(0.0, x_max, 28)
    hist_by_mode = {
        mode: np.histogram(np.clip(vals, 0.0, x_max), bins=bins)[0].astype(np.float64)
        for mode, vals in values_by_mode.items()
    }
    y_max = float(max(1.0, max(np.max(v) for v in hist_by_mode.values())))
    bin_w = plot_w / (len(bins) - 1)

    def sx(x: float) -> float:
        return left + x / x_max * plot_w

    def sy(y: float) -> float:
        return top + plot_h - y / y_max * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#1f2d3a}.axis{stroke:#4d5d6c;stroke-width:1.4}.grid{stroke:#d8e0e8;stroke-width:1}</style>",
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="{left}" y="34" font-size="24" font-weight="700">Pressure error distribution: {phase} {kind}</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
    ]
    offsets = [-0.25, 0.0, 0.25]
    bar_w = bin_w / 4.2
    for mode_i, (mode, counts) in enumerate(hist_by_mode.items()):
        for i, count in enumerate(counts):
            x0 = sx(float(bins[i])) + offsets[mode_i] * bin_w
            y0 = sy(float(count))
            parts.append(
                f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" '
                f'height="{top + plot_h - y0:.1f}" fill="{colors[mode]}" opacity="0.68"/>'
            )
    for x in np.linspace(0.0, x_max, 6):
        parts.append(f'<text x="{sx(float(x)) - 14:.1f}" y="{top + plot_h + 28}" font-size="12">{x:.2f}</text>')
    for frac in np.linspace(0.0, 1.0, 5):
        y = top + plot_h - frac * plot_h
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text x="24" y="{y + 4:.1f}" font-size="12">{frac * y_max:.0f}</text>')
    for i, mode in enumerate(PRESSURE_MODES):
        lx = left + 18 + i * 245
        ly = top + 20
        parts.append(f'<rect x="{lx}" y="{ly - 12}" width="16" height="10" fill="{colors[mode]}" opacity="0.75"/>')
        parts.append(f'<text x="{lx + 24}" y="{ly - 3}" font-size="13">{mode}</text>')
    parts.append(f'<text x="{left + plot_w / 2 - 35:.1f}" y="{height - 25}" font-size="14">relative L2</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def read_poisson_doc_summary(path: Path) -> str:
    if not path.exists():
        return f"Poisson surrogate markdown was not found at `{path}` during report generation."
    text = path.read_text(encoding="utf-8", errors="replace")
    keep = []
    for line in text.splitlines():
        if any(
            marker in line
            for marker in [
                "L.shape",
                "H_tilde.shape",
                "L condition",
                "b(t) =",
                "计算 Re 数量",
                "压力泊松",
            ]
        ):
            keep.append(line.strip())
    return "\n".join(f"- {line}" for line in keep[:16]) or f"Read `{path}`."


def subset_results(results: List[Dict[str, object]], lo: float, hi: float) -> List[Dict[str, object]]:
    return [item for item in results if lo <= float(item["test_Re"]) <= hi]


def mean_diag(results: List[Dict[str, object]], phase: str, key: str) -> float:
    vals = [float(item[phase]["diagnostics"][key]) for item in results]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def write_report(
    path: Path,
    result: Dict[str, object],
    comparison_csv: Path,
    figures: Dict[str, str],
    poisson_doc_summary: str,
) -> None:
    results = sorted(result["results"], key=lambda row: float(row["test_Re"]))
    agg = result["aggregate_metrics"]
    low = subset_results(results, 0.0, 80.0)
    high = subset_results(results, 240.0, 1.0e9)
    one_base_mean = agg["one_step"]["BaseOnly"]["pressure_relative_l2"]["mean"]
    one_closure_mean = agg["one_step"]["ClosureCurrent"]["pressure_relative_l2"]["mean"]
    roll_base_mean = agg["rollout"]["BaseOnly"]["pressure_relative_l2"]["mean"]
    roll_closure_mean = agg["rollout"]["ClosureCurrent"]["pressure_relative_l2"]["mean"]
    residual_to_base = agg["one_step"]["diagnostics"]["residual_to_base_ratio"]["mean"]
    contribution = agg["one_step"]["diagnostics"]["contribution_ratio"]["mean"]
    low_base = mean_diag(low, "one_step", "base_pressure_error_relative_l2")
    low_closure = mean_diag(low, "one_step", "closure_pressure_error_relative_l2")
    high_one_closure = mean_diag(high, "one_step", "closure_pressure_error_relative_l2")
    high_roll_closure = mean_diag(high, "rollout", "closure_pressure_error_relative_l2")

    if one_base_mean < 0.1:
        base_judgement = "BaseOnly is already accurate at one-step scale."
    elif one_base_mean < 0.3:
        base_judgement = "BaseOnly is usable but not sufficiently accurate by the 10 percent target."
    else:
        base_judgement = "BaseOnly is not accurate enough; the pressure head is compensating for a large base error."

    head_role = (
        "dominant pressure term"
        if contribution >= 0.5 or residual_to_base >= 1.0
        else "small-to-moderate correction"
    )
    low_reason = (
        "Base error is already large at low Re; residual learning cannot fully repair it."
        if low_base >= 0.3 and low_closure >= 0.2
        else "The base is not the only low-Re issue; residual coupling/training also contributes."
    )
    high_reason = (
        "High-Re pressure is more rollout-drift limited than one-step limited."
        if high_roll_closure > max(0.15, 1.5 * high_one_closure)
        else "High-Re pressure is mainly limited by the base/residual readout rather than rollout drift alone."
    )

    lines = [
        "# Pressure Base Analysis Report",
        "",
        "## Scope",
        "",
        "This diagnostic keeps the V14/V14_2 HPRS-MoE structure and training setup fixed. "
        "Only the pressure readout is switched at evaluation time:",
        "",
        "- BaseOnly: `b_pred = b_base`",
        "- ResidualOnly(State): `b_pred = pressure_head`",
        "- Closure(Current): `b_pred = b_base + pressure_head`",
        "",
        f"Source metrics JSON: `{result['source_metrics_json']}`",
        "",
        f"Checkpoint: `{result['checkpoint_path']}`",
        "",
        "## Poisson Surrogate Notes",
        "",
        poisson_doc_summary,
        "",
        "## Aggregate Findings",
        "",
        "| Phase | BaseOnly pressure L2 | ResidualOnly pressure L2 | Closure pressure L2 | Closure improvement vs Base | Residual/Base | Contribution |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in ["one_step", "rollout"]:
        phase_agg = agg[phase]
        diag = phase_agg["diagnostics"]
        lines.append(
            f"| {phase} | "
            f"{phase_agg['BaseOnly']['pressure_relative_l2']['mean']:.6g} | "
            f"{phase_agg['ResidualOnlyState']['pressure_relative_l2']['mean']:.6g} | "
            f"{phase_agg['ClosureCurrent']['pressure_relative_l2']['mean']:.6g} | "
            f"{diag['closure_improvement_percent']['mean']:.3g}% | "
            f"{diag['residual_to_base_ratio']['mean']:.6g} | "
            f"{diag['contribution_ratio']['mean']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"- Poisson base judgement: {base_judgement}",
            f"- Pressure head role: mean residual/base = {residual_to_base:.4g}, "
            f"mean contribution ratio = {contribution:.4g}; this behaves as a {head_role}.",
            f"- Low-Re diagnosis: {low_reason}",
            f"- High-Re diagnosis: {high_reason}",
            "",
            "## Per-Re Comparison",
            "",
            "| Re | Base one-step | ResidualOnly one-step | Closure one-step | Improve % | Base rollout | ResidualOnly rollout | Closure rollout | Roll improve % | Residual/Base | Dominant? |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in results:
        one = item["one_step"]
        roll = item["rollout"]
        lines.append(
            f"| {float(item['test_Re']):.6g} | "
            f"{one['modes']['BaseOnly']['relative_l2']:.6g} | "
            f"{one['modes']['ResidualOnlyState']['relative_l2']:.6g} | "
            f"{one['modes']['ClosureCurrent']['relative_l2']:.6g} | "
            f"{one['diagnostics']['closure_relative_improvement_percent_vs_base']:.3g}% | "
            f"{roll['modes']['BaseOnly']['b_relative_l2_mean']:.6g} | "
            f"{roll['modes']['ResidualOnlyState']['b_relative_l2_mean']:.6g} | "
            f"{roll['modes']['ClosureCurrent']['b_relative_l2_mean']:.6g} | "
            f"{roll['diagnostics']['closure_relative_improvement_percent_vs_base']:.3g}% | "
            f"{one['diagnostics']['residual_to_base_ratio']:.6g} | "
            f"{one['diagnostics']['pressure_head_dominant']} |"
        )
    lines.extend(
        [
            "",
            "## Direct Answers",
            "",
            f"1. Poisson surrogate enough precision? {base_judgement} "
            f"Mean one-step BaseOnly pressure L2 is {one_base_mean:.4g}; "
            f"mean rollout BaseOnly pressure L2 is {roll_base_mean:.4g}.",
            "",
            f"2. Is the head a small residual? It is a {head_role}. "
            f"Mean one-step residual/base is {residual_to_base:.4g}, and contribution ratio is {contribution:.4g}.",
            "",
            f"3. Low-Re failure source: {low_reason} "
            f"Low-Re mean base/closure one-step errors are {low_base:.4g}/{low_closure:.4g}.",
            "",
            f"4. High-Re pressure source: {high_reason} "
            f"High-Re mean closure one-step/rollout errors are {high_one_closure:.4g}/{high_roll_closure:.4g}.",
            "",
            "5. Current bottleneck: judge from the table above. If Closure only marginally improves Base "
            "while residual/base is large, the bottleneck is the Poisson surrogate plus base-residual "
            "coupling. If Base is good but Closure degrades, residual learning is the main issue. "
            "If one-step is good but rollout is poor, pressure is mainly limited by autonomous trajectory drift.",
            "",
            "## Artifacts",
            "",
            f"- Comparison CSV: `{comparison_csv}`",
        ]
    )
    for name, fig_path in figures.items():
        lines.append(f"- {name}: `{fig_path}`")
    lines.extend(
        [
            "",
            f"Runtime: {float(result['runtime_seconds']):.2f} s.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cli = parse_args()
    started = time.time()
    source = json.loads(cli.metrics_json.read_text(encoding="utf-8"))
    settings = source["settings"]
    args = namespace_from_metrics(source, cli)
    if args.pressure_target != "closure":
        raise ValueError("Pressure Base Analysis requires a closure-trained checkpoint.")
    output_dir = cli.output_dir or (cli.metrics_json.parent / "pressure_base_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = cli.experiment_name or f"{settings['experiment_name']}_pressure_base_analysis"
    device = torch.device(cli.device if torch.cuda.is_available() and cli.device == "cuda" else "cpu")
    if cli.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    arrays, meta = build_arrays(args)
    tensors = np.load(args.tensor_path)
    pressure_tensors = np.load(args.pressure_surrogate_path)
    checkpoint_path = Path(str(source["results"][0]["checkpoint_path"]))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scalers = load_scalers(ckpt["scalers"])
    model = instantiate_model(settings, arrays["x"].shape[1], device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    holdout_ids = [int(v) for v in ckpt.get("holdout_label_ids", settings["test_re_indices"])]
    sample_ids = arrays["sample_ids"]
    results: List[Dict[str, object]] = []
    distribution_rows: List[Dict[str, object]] = []
    for label_id in holdout_ids:
        test_ids = sample_ids[arrays["label_id"][sample_ids] == int(label_id)]
        if len(test_ids) == 0:
            continue
        one, one_arrays = one_step_pressure_analysis(
            model, arrays, scalers, tensors, pressure_tensors, test_ids, args, device
        )
        roll, roll_arrays = rollout_pressure_analysis(
            model, arrays, scalers, tensors, pressure_tensors, int(label_id), args, device
        )
        for mode_name, mode_key in PRESSURE_MODES.items():
            one_pred = select_pressure_prediction(
                mode_key, one_arrays["base"], one_arrays["residual"]
            )
            for value in row_relative_errors(one_arrays["true_b"], one_pred).tolist():
                distribution_rows.append(
                    {
                        "Re": float(arrays["re"][test_ids[0]]),
                        "phase": "one_step",
                        "mode": mode_name,
                        "kind": "sample",
                        "relative_l2": float(value),
                    }
                )
            roll_key = f"{mode_name}_window_b_relative_l2"
            for value in np.asarray(roll_arrays.get(roll_key, []), dtype=np.float64).tolist():
                distribution_rows.append(
                    {
                        "Re": float(arrays["re"][test_ids[0]]),
                        "phase": "rollout",
                        "mode": mode_name,
                        "kind": "window",
                        "relative_l2": float(value),
                    }
                )
        results.append(
            {
                "test_Re": float(arrays["re"][test_ids[0]]),
                "test_Re_label": str(arrays["labels"][int(label_id)]),
                "num_test": int(len(test_ids)),
                "one_step": one,
                "rollout": roll,
            }
        )
        print(
            json.dumps(
                {
                    "event": "pressure_base_re_done",
                    "Re": float(arrays["re"][test_ids[0]]),
                    "one_base": one["modes"]["BaseOnly"]["relative_l2"],
                    "one_closure": one["modes"]["ClosureCurrent"]["relative_l2"],
                    "roll_base": roll["modes"]["BaseOnly"]["b_relative_l2_mean"],
                    "roll_closure": roll["modes"]["ClosureCurrent"]["b_relative_l2_mean"],
                }
            ),
            flush=True,
        )

    comparison_csv = output_dir / f"{experiment_name}_pressure_mode_comparison.csv"
    distribution_csv = output_dir / f"{experiment_name}_pressure_error_distribution.csv"
    one_svg = output_dir / f"{experiment_name}_one_step_pressure_error_vs_re.svg"
    roll_svg = output_dir / f"{experiment_name}_rollout_pressure_error_vs_re.svg"
    base_residual_svg = output_dir / f"{experiment_name}_base_error_residual_magnitude_vs_re.svg"
    one_dist_svg = output_dir / f"{experiment_name}_one_step_pressure_error_distribution.svg"
    roll_dist_svg = output_dir / f"{experiment_name}_rollout_pressure_error_distribution.svg"
    write_comparison_csv(comparison_csv, results)
    write_distribution_csv(distribution_csv, distribution_rows)
    write_error_vs_re_svg(one_svg, results, "one_step")
    write_error_vs_re_svg(roll_svg, results, "rollout")
    write_base_residual_svg(base_residual_svg, results)
    write_distribution_svg(one_dist_svg, distribution_rows, "one_step", "sample")
    write_distribution_svg(roll_dist_svg, distribution_rows, "rollout", "window")
    figures = {
        "one_step_error_vs_re": str(one_svg),
        "rollout_error_vs_re": str(roll_svg),
        "base_error_residual_magnitude_vs_re": str(base_residual_svg),
        "one_step_error_distribution": str(one_dist_svg),
        "rollout_error_distribution": str(roll_dist_svg),
    }
    out = {
        "scheme": "pressure_base_analysis_v14_closure_dense_uniform10",
        "source_metrics_json": str(cli.metrics_json),
        "checkpoint_path": str(checkpoint_path),
        "experiment_name": experiment_name,
        "settings": settings,
        "data_meta": meta,
        "results": results,
        "aggregate_metrics": aggregate(results),
        "comparison_csv": str(comparison_csv),
        "distribution_csv": str(distribution_csv),
        "figures": figures,
        "runtime_seconds": float(time.time() - started),
    }
    json_path = output_dir / f"{experiment_name}_pressure_base_analysis.json"
    report_path = output_dir / f"{experiment_name}_pressure_base_analysis_report.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_report(
        report_path,
        out,
        comparison_csv,
        figures,
        read_poisson_doc_summary(cli.poisson_doc),
    )
    print(
        json.dumps(
            {
                "event": "pressure_base_analysis_done",
                "json": str(json_path),
                "report": str(report_path),
                "comparison_csv": str(comparison_csv),
                "runtime_seconds": out["runtime_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
