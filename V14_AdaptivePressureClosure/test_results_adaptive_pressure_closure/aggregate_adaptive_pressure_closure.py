#!/usr/bin/env python3
"""Aggregate V14 adaptive pressure closure experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


METRIC_PATHS = {
    "rhs_l2": ("deep_moe", "rhs_relative_l2"),
    "one_step_velocity_l2": ("one_step_autonomous_pressure", "a_relative_l2"),
    "one_step_pressure_l2": ("one_step_autonomous_pressure", "b_relative_l2"),
    "one_step_pressure_energy_error": (
        "one_step_autonomous_pressure",
        "b_energy_relative_error",
    ),
    "rollout_velocity_l2": ("rollout_autonomous_pressure", "a_relative_l2_mean"),
    "rollout_pressure_l2": ("rollout_autonomous_pressure", "b_relative_l2_mean"),
    "rollout_pressure_energy_error": (
        "rollout_autonomous_pressure",
        "b_energy_relative_error",
    ),
    "pressure_base_l2": ("deep_moe", "pressure_surrogate_base_relative_l2"),
    "pressure_head_l2": ("deep_moe", "pressure_head_relative_l2"),
    "pressure_effective_residual_l2": (
        "deep_moe",
        "pressure_effective_residual_relative_l2",
    ),
    "alpha_mean": ("deep_moe", "closure_alpha_mean"),
    "alpha_std": ("deep_moe", "closure_alpha_std"),
    "beta_mean": ("deep_moe", "closure_beta_mean"),
    "beta_std": ("deep_moe", "closure_beta_std"),
    "base_scale_mean": ("deep_moe", "closure_base_scale_mean"),
    "residual_scale_mean": ("deep_moe", "closure_residual_scale_mean"),
    "base_error_mean": ("deep_moe", "closure_base_error_mean"),
    "residual_magnitude_mean": ("deep_moe", "closure_residual_magnitude_mean"),
    "effective_residual_magnitude_mean": (
        "deep_moe",
        "closure_effective_residual_magnitude_mean",
    ),
    "base_contribution_ratio_mean": (
        "deep_moe",
        "closure_base_contribution_ratio_mean",
    ),
    "residual_contribution_ratio_mean": (
        "deep_moe",
        "closure_residual_contribution_ratio_mean",
    ),
    "alpha_base_error_corr": ("deep_moe", "closure_alpha_base_error_corr"),
    "alpha_residual_magnitude_corr": (
        "deep_moe",
        "closure_alpha_residual_magnitude_corr",
    ),
    "beta_base_error_corr": ("deep_moe", "closure_beta_base_error_corr"),
    "beta_residual_magnitude_corr": (
        "deep_moe",
        "closure_beta_residual_magnitude_corr",
    ),
}

MODE_LABELS = {
    "baseline": "Baseline",
    "residual_scaling": "AdaptiveResidualScaling",
    "base_scaling": "AdaptiveBaseScaling",
    "dual_adaptive": "DualAdaptiveClosure",
}

COLORS = {
    "Baseline": "#1f77b4",
    "AdaptiveResidualScaling": "#ff7f0e",
    "AdaptiveBaseScaling": "#2ca02c",
    "DualAdaptiveClosure": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def nested_metric(item: Dict[str, object], path: Iterable[str]) -> float:
    cur: object = item
    for key in path:
        if not isinstance(cur, dict):
            return float("nan")
        cur = cur.get(key)
    try:
        return float(cur)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def mode_label(data: Dict[str, object], fallback: str) -> str:
    settings = data.get("settings", {})
    if isinstance(settings, dict):
        raw = str(settings.get("closure_mode", ""))
        if raw in MODE_LABELS:
            return MODE_LABELS[raw]
        if settings.get("pressure_input_mode") == "pressure_only":
            return MODE_LABELS["baseline"]
    return fallback


def load_experiment(path: Path, fallback: str) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "mode": mode_label(data, fallback),
        "path": str(path),
        "settings": data.get("settings", {}),
        "results": data.get("results", []),
        "aggregate_metrics": data.get("aggregate_metrics", {}),
        "runtime_seconds": data.get("runtime_seconds"),
    }


def rows_for_experiment(exp: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for item in exp["results"]:  # type: ignore[index]
        if not isinstance(item, dict):
            continue
        row: Dict[str, object] = {
            "mode": exp["mode"],
            "Re": float(item["test_Re"]),
            "label": str(item["test_Re_label"]),
            "num_test": int(item.get("num_test", 0)),
            "best_epoch": int(item.get("best_epoch", -1)),
        }
        for name, path in METRIC_PATHS.items():
            row[name] = nested_metric(item, path)
        rows.append(row)
    return sorted(rows, key=lambda item: float(item["Re"]))


def sample_rows_for_experiment(exp: Dict[str, object]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for item in exp["results"]:  # type: ignore[index]
        if not isinstance(item, dict):
            continue
        re_value = float(item["test_Re"])
        deep = item.get("deep_moe", {})
        if not isinstance(deep, dict):
            continue
        diag = deep.get("closure_diagnostics", {})
        if not isinstance(diag, dict):
            continue
        for sample in diag.get("time_series", []):
            if not isinstance(sample, dict):
                continue
            row = {"mode": exp["mode"], "Re": re_value}
            for key in [
                "time",
                "alpha",
                "beta",
                "base_scale",
                "residual_scale",
                "base_error",
                "closure_error",
                "residual_magnitude",
                "effective_residual_magnitude",
                "base_contribution_ratio",
                "residual_contribution_ratio",
            ]:
                row[key] = float(sample.get(key, float("nan")))
            out.append(row)
    return out


def finite_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate_rows(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    return {
        name: finite_stats([float(row.get(name, float("nan"))) for row in rows])
        for name in METRIC_PATHS
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = ["mode", "Re", "label", "num_test", "best_epoch", *METRIC_PATHS.keys()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_sample_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "mode",
        "Re",
        "time",
        "alpha",
        "beta",
        "base_scale",
        "residual_scale",
        "base_error",
        "closure_error",
        "residual_magnitude",
        "effective_residual_magnitude",
        "base_contribution_ratio",
        "residual_contribution_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def svg_line_chart(
    path: Path,
    title: str,
    y_label: str,
    series: Dict[str, List[tuple[float, float]]],
    x_label: str = "Reynolds number",
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    width, height = 1180, 720
    left, right, top, bottom = 88, 35, 58, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_points = [(x, y) for points in series.values() for x, y in points if np.isfinite(y)]
    if not all_points:
        path.write_text("", encoding="utf-8")
        return
    xs = np.asarray([p[0] for p in all_points], dtype=np.float64)
    ys = np.asarray([p[1] for p in all_points], dtype=np.float64)
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    if abs(x_max - x_min) < 1.0e-12:
        x_max = x_min + 1.0
    if y_min is None:
        y_min = min(0.0, float(np.min(ys)) * 0.96)
    if y_max is None:
        y_max = max(0.1, float(np.max(ys)) * 1.08)
    if abs(y_max - y_min) < 1.0e-12:
        y_max = y_min + 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>text{font-family:Arial,sans-serif;fill:#1f2d3a}.axis{stroke:#536575;stroke-width:1.4}.grid{stroke:#d8e0e8;stroke-width:1}.line{fill:none;stroke-width:2.4}.dot{stroke:#fff;stroke-width:1.2}</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="{left}" y="34" font-size="23" font-weight="700">{title}</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
    ]
    for frac in np.linspace(0, 1, 6):
        y = top + plot_h - frac * plot_h
        val = y_min + frac * (y_max - y_min)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text x="18" y="{y + 4:.1f}" font-size="12">{val:.3g}</text>')
    for x in np.linspace(x_min, x_max, 6):
        px = sx(float(x))
        parts.append(f'<text x="{px - 22:.1f}" y="{top + plot_h + 28}" font-size="12">{x:.0f}</text>')
    parts.append(f'<text x="{left + plot_w / 2 - 38:.1f}" y="{height - 28}" font-size="14">{x_label}</text>')
    parts.append(f'<text x="18" y="{top - 18}" font-size="14">{y_label}</text>')
    legend_x, legend_y = left + 12, top + 20
    for idx, (name, points) in enumerate(series.items()):
        color = COLORS.get(name.split(" ")[0], "#444")
        points = [(x, y) for x, y in sorted(points) if np.isfinite(y)]
        if not points:
            continue
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f'<polyline class="line" points="{d}" stroke="{color}"/>')
        for x, y in points:
            parts.append(f'<circle class="dot" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.2" fill="{color}"/>')
        ly = legend_y + 22 * idx
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 24}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 32}" y="{ly + 4}" font-size="13">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_mode_curves(output_dir: Path, rows: List[Dict[str, object]]) -> Dict[str, str]:
    modes = sorted({str(row["mode"]) for row in rows})
    figures: Dict[str, str] = {}

    def series_for(metric: str) -> Dict[str, List[tuple[float, float]]]:
        return {
            mode: [
                (float(row["Re"]), float(row.get(metric, float("nan"))))
                for row in rows
                if row["mode"] == mode
            ]
            for mode in modes
        }

    specs = {
        "one_step_pressure": ("one_step_pressure_l2", "One-step Pressure vs Re", "relative L2"),
        "rollout_pressure": ("rollout_pressure_l2", "24-step Rollout Pressure vs Re", "relative L2"),
        "pressure_energy": (
            "rollout_pressure_energy_error",
            "Rollout Pressure Energy Error vs Re",
            "relative error",
        ),
        "alpha_mean": ("alpha_mean", "Adaptive Alpha Mean vs Re", "alpha"),
        "beta_mean": ("beta_mean", "Adaptive Beta Mean vs Re", "beta"),
        "base_contribution": (
            "base_contribution_ratio_mean",
            "Base Contribution Ratio vs Re",
            "ratio",
        ),
        "residual_contribution": (
            "residual_contribution_ratio_mean",
            "Residual Contribution Ratio vs Re",
            "ratio",
        ),
        "base_error": ("base_error_mean", "Base Error vs Re", "relative L2"),
        "residual_magnitude": (
            "residual_magnitude_mean",
            "Residual Magnitude vs Re",
            "relative norm",
        ),
    }
    for key, (metric, title, y_label) in specs.items():
        path = output_dir / f"v14_adaptive_{key}.svg"
        svg_line_chart(path, title, y_label, series_for(metric))
        figures[key] = str(path)
    return figures


def write_focus_timeseries(output_dir: Path, sample_rows: List[Dict[str, object]]) -> Dict[str, str]:
    figures: Dict[str, str] = {}
    for target in [50.0, 78.0906, 300.0]:
        selected = [
            row for row in sample_rows if abs(float(row["Re"]) - target) < 0.6
        ]
        if not selected:
            continue
        for metric in ["alpha", "beta"]:
            modes = sorted({str(row["mode"]) for row in selected})
            series = {
                mode: [
                    (float(row["time"]), float(row.get(metric, float("nan"))))
                    for row in selected
                    if row["mode"] == mode
                ]
                for mode in modes
            }
            path = output_dir / f"v14_adaptive_{metric}_timeseries_Re{int(round(target))}.svg"
            svg_line_chart(
                path,
                f"{metric} time evolution at Re={target:g}",
                metric,
                series,
                x_label="time",
                y_min=-0.55 if metric == "beta" else 0.0,
                y_max=1.05 if metric == "alpha" else 0.55,
            )
            figures[f"{metric}_timeseries_Re{int(round(target))}"] = str(path)
    return figures


def group_stats(rows: List[Dict[str, object]], predicate) -> Dict[str, Dict[str, float]]:
    selected = [row for row in rows if predicate(float(row["Re"]))]
    return aggregate_rows(selected)


def rel_change(candidate: float, baseline: float) -> float:
    return float(100.0 * (baseline - candidate) / (abs(baseline) + 1.0e-12))


def write_report(
    path: Path,
    rows: List[Dict[str, object]],
    aggregate: Dict[str, Dict[str, Dict[str, float]]],
    figures: Dict[str, str],
    sample_csv: Path,
) -> None:
    modes = ["Baseline", "AdaptiveResidualScaling", "AdaptiveBaseScaling", "DualAdaptiveClosure"]
    modes = [mode for mode in modes if mode in aggregate]
    lines = [
        "# V14 Adaptive Pressure Closure Report",
        "",
        "## Setup",
        "",
        "This experiment keeps the V14/V14_3 HPRS-MoE backbone fixed: shared encoder, routers, velocity head, pressure head input `[a_t,b_t]`, physics-aware pressure experts, Galerkin RHS, RK4, dense V14 training split, losses, optimizer settings, and Poisson surrogate are unchanged. Only the pressure closure fusion is changed.",
        "",
        "- Baseline: `b_pred = b_base + r`.",
        "- AdaptiveResidualScaling: `b_pred = b_base + alpha(x) r`, with `alpha in [0,1]`.",
        "- AdaptiveBaseScaling: `b_pred = alpha(x) b_base + r`, with `alpha in [0,1]`.",
        "- DualAdaptiveClosure: `b_pred = (1+beta(x)) b_base + alpha(x) r`, with `alpha in [0,1]`, `beta in [-0.5,0.5]`.",
        "",
        "The confidence head is a two-layer MLP on the existing encoder feature `h`; it does not re-encode raw input and does not affect the router.",
        "",
        "## Aggregate Metrics",
        "",
        "| Mode | 1-step u | 1-step p | 24-step u | 24-step p | RHS | p energy rollout | alpha mean | beta mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        stats = aggregate[mode]
        lines.append(
            f"| {mode} | "
            f"{stats['one_step_velocity_l2']['mean']:.6g} | "
            f"{stats['one_step_pressure_l2']['mean']:.6g} | "
            f"{stats['rollout_velocity_l2']['mean']:.6g} | "
            f"{stats['rollout_pressure_l2']['mean']:.6g} | "
            f"{stats['rhs_l2']['mean']:.6g} | "
            f"{stats['rollout_pressure_energy_error']['mean']:.6g} | "
            f"{stats['alpha_mean']['mean']:.6g} | "
            f"{stats['beta_mean']['mean']:.6g} |"
        )
    lines.extend(["", "## Low-Re And High-Re", ""])
    lines.append(
        "| Mode | Low-Re 1-step p | Low-Re 24-step p | High-Re 1-step p | High-Re 24-step p | base contrib | residual contrib |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        low = group_stats(mode_rows, lambda re: re <= 80.0)
        high = group_stats(mode_rows, lambda re: re >= 240.0)
        lines.append(
            f"| {mode} | "
            f"{low['one_step_pressure_l2']['mean']:.6g} | "
            f"{low['rollout_pressure_l2']['mean']:.6g} | "
            f"{high['one_step_pressure_l2']['mean']:.6g} | "
            f"{high['rollout_pressure_l2']['mean']:.6g} | "
            f"{aggregate[mode]['base_contribution_ratio_mean']['mean']:.6g} | "
            f"{aggregate[mode]['residual_contribution_ratio_mean']['mean']:.6g} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if "Baseline" in aggregate:
        base = aggregate["Baseline"]["rollout_pressure_l2"]["mean"]
        best_mode = min(modes, key=lambda mode: aggregate[mode]["rollout_pressure_l2"]["mean"])
        best = aggregate[best_mode]["rollout_pressure_l2"]["mean"]
        lines.append(
            f"Best mean 24-step pressure error is `{best_mode}` at `{best:.6g}`. "
            f"Relative improvement vs Baseline is `{rel_change(best, base):.2f}%`."
        )
        for mode in modes:
            if mode == "Baseline":
                continue
            one = aggregate[mode]["one_step_pressure_l2"]["mean"]
            roll = aggregate[mode]["rollout_pressure_l2"]["mean"]
            base_one = aggregate["Baseline"]["one_step_pressure_l2"]["mean"]
            lines.append(
                f"- {mode}: one-step pressure change vs Baseline "
                f"{rel_change(one, base_one):.2f}%, rollout pressure change "
                f"{rel_change(roll, base):.2f}%."
            )
    lines.extend(
        [
            "",
            "The expected adaptive pattern is: high Re should keep high base contribution and low residual correction, while low Re should reduce fixed trust in the Poisson base and rely more on residual correction. The alpha/beta curves and contribution ratios below test whether the confidence head learned that behavior.",
            "",
            "If adaptive gating does not beat Baseline with statistical consistency across the 10 held-out Reynolds numbers, the likely bottleneck is not the fixed closure weight itself. The remaining suspects are the Poisson surrogate quality at low Re, pressure residual target conditioning, and the coupling between pressure closure and autonomous velocity rollout.",
            "",
            "## Artifacts",
            "",
            f"- Combined CSV: `{path.parent / 'v14_adaptive_pressure_closure_combined.csv'}`",
            f"- Closure sample CSV: `{sample_csv}`",
        ]
    )
    for name, fig in sorted(figures.items()):
        lines.append(f"- {name}: `{fig}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = []
    if args.baseline_metrics and args.baseline_metrics.exists():
        experiments.append(load_experiment(args.baseline_metrics, "Baseline"))
    for path in args.metrics:
        experiments.append(load_experiment(path, path.stem))

    rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    for exp in experiments:
        rows.extend(rows_for_experiment(exp))
        sample_rows.extend(sample_rows_for_experiment(exp))
    rows.sort(key=lambda row: (str(row["mode"]), float(row["Re"])))
    sample_rows.sort(key=lambda row: (str(row["mode"]), float(row["Re"]), float(row["time"])))

    aggregate = {}
    for mode in sorted({str(row["mode"]) for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        stats = aggregate_rows(mode_rows)
        stats["low_Re_le_80"] = group_stats(mode_rows, lambda re: re <= 80.0)  # type: ignore[assignment]
        stats["high_Re_ge_240"] = group_stats(mode_rows, lambda re: re >= 240.0)  # type: ignore[assignment]
        aggregate[mode] = stats

    combined_csv = args.output_dir / "v14_adaptive_pressure_closure_combined.csv"
    sample_csv = args.output_dir / "v14_adaptive_pressure_closure_samples.csv"
    write_csv(combined_csv, rows)
    write_sample_csv(sample_csv, sample_rows)
    figures = write_mode_curves(args.output_dir, rows)
    figures.update(write_focus_timeseries(args.output_dir, sample_rows))

    report = args.output_dir / "TECHNICAL_REPORT_V14_ADAPTIVE_PRESSURE_CLOSURE.md"
    write_report(report, rows, aggregate, figures, sample_csv)
    aggregate_json = args.output_dir / "v14_adaptive_pressure_closure_aggregate.json"
    aggregate_json.write_text(
        json.dumps(
            {
                "experiments": experiments,
                "aggregate": aggregate,
                "combined_csv": str(combined_csv),
                "sample_csv": str(sample_csv),
                "figures": figures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report),
                "aggregate": str(aggregate_json),
                "combined_csv": str(combined_csv),
                "sample_csv": str(sample_csv),
            }
        )
    )


if __name__ == "__main__":
    main()
