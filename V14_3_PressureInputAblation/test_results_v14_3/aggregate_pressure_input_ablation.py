#!/usr/bin/env python3
"""Aggregate V14_3 pressure input ablation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List


MODE_LABELS = {
    "pressure_only": "PressureOnly",
    "velocity_only": "VelocityOnly",
    "hybrid": "Hybrid",
}

MODE_COLORS = {
    "PressureOnly": "#1f77b4",
    "VelocityOnly": "#ff7f0e",
    "Hybrid": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("metrics", nargs="+", type=Path)
    return parser.parse_args()


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def stats(values: Iterable[float]) -> Dict[str, float]:
    vals = finite(values)
    if not vals:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def metric(item: Dict[str, object], group: str, key: str) -> float:
    value = item.get(group, {})
    if not isinstance(value, dict):
        return math.nan
    return float(value.get(key, math.nan))


def load_rows(paths: List[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    experiments: list[dict[str, object]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        settings = data["settings"]
        mode = MODE_LABELS.get(settings.get("pressure_input_mode", "pressure_only"), "PressureOnly")
        experiments.append(
            {
                "mode": mode,
                "path": str(path),
                "settings": settings,
                "aggregate_metrics": data.get("aggregate_metrics", {}),
                "runtime_seconds": data.get("runtime_seconds"),
            }
        )
        for item in data["results"]:
            one = item["one_step_autonomous_pressure"]
            roll = item["rollout_autonomous_pressure"]
            deep = item["deep_moe"]
            rows.append(
                {
                    "mode": mode,
                    "Re": float(item["test_Re"]),
                    "label": str(item["test_Re_label"]),
                    "rhs_l2": metric(item, "deep_moe", "rhs_relative_l2"),
                    "one_step_velocity_l2": metric(item, "one_step_autonomous_pressure", "a_relative_l2"),
                    "one_step_pressure_l2": metric(item, "one_step_autonomous_pressure", "b_relative_l2"),
                    "one_step_pressure_energy_error": float(
                        one.get("b_energy_relative_error", math.nan)
                    ),
                    "rollout_velocity_l2": metric(
                        item, "rollout_autonomous_pressure", "a_relative_l2_mean"
                    ),
                    "rollout_pressure_l2": metric(
                        item, "rollout_autonomous_pressure", "b_relative_l2_mean"
                    ),
                    "rollout_pressure_energy_error": float(
                        roll.get("b_energy_relative_error", math.nan)
                    ),
                    "pressure_head_l2": float(deep.get("pressure_head_relative_l2", math.nan)),
                    "pressure_head_energy_error": float(
                        deep.get("pressure_energy_relative_error", math.nan)
                    ),
                    "num_test": int(item.get("num_test", 0)),
                    "rollout_windows": int(roll.get("num_windows", 0)),
                    "best_epoch": int(item.get("best_epoch", -1)),
                }
            )
    return rows, experiments


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "mode",
        "Re",
        "label",
        "rhs_l2",
        "one_step_velocity_l2",
        "one_step_pressure_l2",
        "one_step_pressure_energy_error",
        "rollout_velocity_l2",
        "rollout_pressure_l2",
        "rollout_pressure_energy_error",
        "pressure_head_l2",
        "pressure_head_energy_error",
        "num_test",
        "rollout_windows",
        "best_epoch",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (str(x["mode"]), float(x["Re"]))):
            writer.writerow(row)


def aggregate_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    metrics = [
        "rhs_l2",
        "one_step_velocity_l2",
        "one_step_pressure_l2",
        "one_step_pressure_energy_error",
        "rollout_velocity_l2",
        "rollout_pressure_l2",
        "rollout_pressure_energy_error",
        "pressure_head_l2",
        "pressure_head_energy_error",
    ]
    for mode in sorted({str(row["mode"]) for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        out[mode] = {name: stats(float(row[name]) for row in mode_rows) for name in metrics}
        low_rows = [row for row in mode_rows if float(row["Re"]) <= 80.0]
        high_rows = [row for row in mode_rows if float(row["Re"]) >= 240.0]
        out[mode]["low_Re_le_80"] = {
            name: stats(float(row[name]) for row in low_rows) for name in metrics
        }
        out[mode]["high_Re_ge_240"] = {
            name: stats(float(row[name]) for row in high_rows) for name in metrics
        }
    return out


def improvement(new: float, base: float) -> float:
    return 100.0 * (base - new) / (base + 1.0e-12)


def svg_curve(path: Path, rows: List[Dict[str, object]], metric_name: str, title: str) -> None:
    width, height = 1180, 700
    left, right, top, bottom = 85, 35, 55, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    re_values = sorted({float(row["Re"]) for row in rows})
    modes = ["PressureOnly", "VelocityOnly", "Hybrid"]
    values = [
        float(row[metric_name])
        for row in rows
        if math.isfinite(float(row.get(metric_name, math.nan)))
    ]
    if not values:
        path.write_text("", encoding="utf-8")
        return
    x_min, x_max = min(re_values), max(re_values)
    y_max = max(0.1, max(values) * 1.08)

    def sx(x: float) -> float:
        return left + (x - x_min) / max(x_max - x_min, 1.0e-12) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - min(max(y, 0.0), y_max) / y_max * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#1f2d3a}.axis{stroke:#4d5d6c;stroke-width:1.4}.grid{stroke:#d8e0e8;stroke-width:1}</style>",
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="{left}" y="34" font-size="24" font-weight="700">{title}</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
    ]
    for frac in [i / 5 for i in range(6)]:
        y = top + plot_h - frac * plot_h
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text x="18" y="{y + 4:.1f}" font-size="12">{frac * y_max:.2f}</text>')
    for x in [x_min + (x_max - x_min) * i / 5 for i in range(6)]:
        parts.append(f'<text x="{sx(x) - 18:.1f}" y="{top + plot_h + 28}" font-size="12">{x:.0f}</text>')
    for i, mode in enumerate(modes):
        points = []
        mode_rows = sorted([row for row in rows if row["mode"] == mode], key=lambda r: float(r["Re"]))
        for row in mode_rows:
            val = float(row.get(metric_name, math.nan))
            if math.isfinite(val):
                points.append((sx(float(row["Re"])), sy(val)))
        if len(points) >= 2:
            parts.append(
                f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" '
                f'fill="none" stroke="{MODE_COLORS[mode]}" stroke-width="2.5"/>'
            )
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{MODE_COLORS[mode]}" stroke="#fff"/>')
        lx = left + 18 + i * 230
        ly = top + 20
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="18" height="5" fill="{MODE_COLORS[mode]}"/>')
        parts.append(f'<text x="{lx + 26}" y="{ly - 4}" font-size="13">{mode}</text>')
    parts.append(f'<text x="{left + plot_w / 2 - 35:.1f}" y="{height - 25}" font-size="14">Reynolds number</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(
    path: Path,
    rows: List[Dict[str, object]],
    aggregate: Dict[str, object],
    csv_path: Path,
    figures: Dict[str, Path],
    experiments: List[Dict[str, object]],
) -> None:
    pressure = aggregate.get("PressureOnly", {})
    velocity = aggregate.get("VelocityOnly", {})
    hybrid = aggregate.get("Hybrid", {})
    base_one = pressure.get("one_step_pressure_l2", {}).get("mean", math.nan)
    vel_one = velocity.get("one_step_pressure_l2", {}).get("mean", math.nan)
    hyb_one = hybrid.get("one_step_pressure_l2", {}).get("mean", math.nan)
    base_roll = pressure.get("rollout_pressure_l2", {}).get("mean", math.nan)
    vel_roll = velocity.get("rollout_pressure_l2", {}).get("mean", math.nan)
    hyb_roll = hybrid.get("rollout_pressure_l2", {}).get("mean", math.nan)
    low_base = pressure.get("low_Re_le_80", {}).get("one_step_pressure_l2", {}).get("mean", math.nan)
    low_vel = velocity.get("low_Re_le_80", {}).get("one_step_pressure_l2", {}).get("mean", math.nan)
    low_hyb = hybrid.get("low_Re_le_80", {}).get("one_step_pressure_l2", {}).get("mean", math.nan)

    lines = [
        "# V14_3 Pressure Input Ablation Report",
        "",
        "## Experiment Design",
        "",
        "All runs keep HPRS-MoE, routers, Galerkin RHS, RK4, losses, optimizer settings, "
        "training schedule, dense V14 data organization, and `--pressure-target=closure` fixed. "
        "Only the pressure expert state input changes.",
        "",
        "- PressureOnly: unchanged V14 baseline pressure state from current code path.",
        "- VelocityOnly: pressure state is `[a_next, 0]`, so the head sees next-step velocity only.",
        "- Hybrid: pressure state is `[a_next, b_base]`, so the head sees next-step velocity and the Poisson prior.",
        "",
        "The pressure expert state dimension is held at `r_u+r_p` for all modes, so the Linear, "
        "low-rank quadratic, and FFN parameterization remains identical.",
        "",
        "## Aggregate Metrics",
        "",
        "| Mode | one-step pressure | rollout pressure | one-step velocity | rollout velocity | RHS | pressure energy one-step | pressure energy rollout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ["PressureOnly", "VelocityOnly", "Hybrid"]:
        item = aggregate.get(mode, {})
        lines.append(
            f"| {mode} | "
            f"{item.get('one_step_pressure_l2', {}).get('mean', math.nan):.6g} | "
            f"{item.get('rollout_pressure_l2', {}).get('mean', math.nan):.6g} | "
            f"{item.get('one_step_velocity_l2', {}).get('mean', math.nan):.6g} | "
            f"{item.get('rollout_velocity_l2', {}).get('mean', math.nan):.6g} | "
            f"{item.get('rhs_l2', {}).get('mean', math.nan):.6g} | "
            f"{item.get('one_step_pressure_energy_error', {}).get('mean', math.nan):.6g} | "
            f"{item.get('rollout_pressure_energy_error', {}).get('mean', math.nan):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Low-Re Focus",
            "",
            "| Mode | Re<=80 one-step pressure | Re<=80 rollout pressure | Re<=80 pressure energy one-step |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode in ["PressureOnly", "VelocityOnly", "Hybrid"]:
        item = aggregate.get(mode, {}).get("low_Re_le_80", {})
        lines.append(
            f"| {mode} | "
            f"{item.get('one_step_pressure_l2', {}).get('mean', math.nan):.6g} | "
            f"{item.get('rollout_pressure_l2', {}).get('mean', math.nan):.6g} | "
            f"{item.get('one_step_pressure_energy_error', {}).get('mean', math.nan):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Per-Re Pressure Metrics",
            "",
            "| Re | PressureOnly one-step | VelocityOnly one-step | Hybrid one-step | PressureOnly rollout | VelocityOnly rollout | Hybrid rollout |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    re_values = sorted({float(row["Re"]) for row in rows})
    for re_value in re_values:
        by_mode = {row["mode"]: row for row in rows if abs(float(row["Re"]) - re_value) < 1.0e-9}
        lines.append(
            f"| {re_value:.6g} | "
            f"{float(by_mode.get('PressureOnly', {}).get('one_step_pressure_l2', math.nan)):.6g} | "
            f"{float(by_mode.get('VelocityOnly', {}).get('one_step_pressure_l2', math.nan)):.6g} | "
            f"{float(by_mode.get('Hybrid', {}).get('one_step_pressure_l2', math.nan)):.6g} | "
            f"{float(by_mode.get('PressureOnly', {}).get('rollout_pressure_l2', math.nan)):.6g} | "
            f"{float(by_mode.get('VelocityOnly', {}).get('rollout_pressure_l2', math.nan)):.6g} | "
            f"{float(by_mode.get('Hybrid', {}).get('rollout_pressure_l2', math.nan)):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Direct Answers",
            "",
            f"- VelocityOnly vs PressureOnly: mean one-step pressure change is "
            f"{improvement(vel_one, base_one):.3g}% and rollout pressure change is "
            f"{improvement(vel_roll, base_roll):.3g}% relative improvement.",
            f"- Hybrid vs PressureOnly: mean one-step pressure change is "
            f"{improvement(hyb_one, base_one):.3g}% and rollout pressure change is "
            f"{improvement(hyb_roll, base_roll):.3g}% relative improvement.",
            f"- Low-Re one-step pressure means: PressureOnly={low_base:.6g}, "
            f"VelocityOnly={low_vel:.6g}, Hybrid={low_hyb:.6g}.",
            "",
            "Interpretation rule: if VelocityOnly clearly beats PressureOnly at low Re, "
            "the current pressure-state input is likely a major bottleneck. If Hybrid beats both, "
            "the pressure head should use velocity modes plus the Poisson prior. If neither improves, "
            "the bottleneck is more likely pressure residual learning, the Poisson base, or rollout coupling.",
            "",
            "## Artifacts",
            "",
            f"- Combined CSV: `{csv_path}`",
        ]
    )
    for name, figure_path in figures.items():
        lines.append(f"- {name}: `{figure_path}`")
    lines.append("")
    lines.append("## Source Metrics")
    lines.append("")
    for exp in experiments:
        lines.append(f"- {exp['mode']}: `{exp['path']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, experiments = load_rows(args.metrics)
    combined_csv = args.output_dir / "v14_3_pressure_input_ablation_combined.csv"
    aggregate_json = args.output_dir / "v14_3_pressure_input_ablation_aggregate.json"
    report_md = args.output_dir / "TECHNICAL_REPORT_V14_3_PRESSURE_INPUT_ABLATION.md"
    figures = {
        "one_step_pressure_vs_re": args.output_dir / "v14_3_one_step_pressure_vs_re.svg",
        "rollout_pressure_vs_re": args.output_dir / "v14_3_rollout_pressure_vs_re.svg",
        "pressure_energy_vs_re": args.output_dir / "v14_3_pressure_energy_vs_re.svg",
    }
    write_csv(combined_csv, rows)
    agg = aggregate_rows(rows)
    aggregate_json.write_text(
        json.dumps(
            {
                "experiments": experiments,
                "aggregate": agg,
                "combined_csv": str(combined_csv),
                "figures": {key: str(value) for key, value in figures.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    svg_curve(figures["one_step_pressure_vs_re"], rows, "one_step_pressure_l2", "V14_3 one-step pressure error vs Re")
    svg_curve(figures["rollout_pressure_vs_re"], rows, "rollout_pressure_l2", "V14_3 24-step rollout pressure error vs Re")
    svg_curve(figures["pressure_energy_vs_re"], rows, "one_step_pressure_energy_error", "V14_3 pressure energy error vs Re")
    write_report(report_md, rows, agg, combined_csv, figures, experiments)
    print(
        json.dumps(
            {
                "report": str(report_md),
                "aggregate": str(aggregate_json),
                "combined_csv": str(combined_csv),
            }
        )
    )


if __name__ == "__main__":
    main()
