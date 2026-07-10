#!/usr/bin/env python3
"""Aggregate V16_2 AttractorStabilityMoE metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


CASE_ORDER = [
    "V16_1_SteadyPressureAnchor32",
    "V16_2_SteadyContractivePressureROM32",
    "V16_2_HopfLogRadiusNormalForm32",
    "V16_2_RegimeGroupedMoE32",
]

METRIC_PATHS = {
    "rhs_l2": ("deep_moe", "rhs_relative_l2"),
    "direct_pressure_closure_l2": ("deep_moe", "pressure_head_relative_l2"),
    "pressure_base_l2": ("deep_moe", "pressure_surrogate_base_relative_l2"),
    "pressure_residual_only_l2": ("deep_moe", "pressure_residual_only_relative_l2"),
    "one_step_velocity_l2": ("one_step_autonomous_pressure", "a_relative_l2"),
    "one_step_pressure_l2": ("one_step_autonomous_pressure", "b_relative_l2"),
    "one_step_pressure_energy_error": ("one_step_autonomous_pressure", "b_energy_relative_error"),
    "rollout_velocity_l2": ("rollout_autonomous_pressure", "a_relative_l2_mean"),
    "rollout_pressure_l2": ("rollout_autonomous_pressure", "b_relative_l2_mean"),
    "rollout_pressure_energy_error": ("rollout_autonomous_pressure", "b_energy_relative_error"),
    "alpha_mean": ("deep_moe", "closure_alpha_mean"),
    "alpha_std": ("deep_moe", "closure_alpha_std"),
    "base_contribution_ratio": ("deep_moe", "closure_base_contribution_ratio_mean"),
    "residual_contribution_ratio": ("deep_moe", "closure_residual_contribution_ratio_mean"),
    "base_error_mean": ("deep_moe", "closure_base_error_mean"),
    "residual_magnitude_mean": ("deep_moe", "closure_residual_magnitude_mean"),
    "router_active_experts": ("routing_analysis_test", "active_experts_mean"),
    "dead_experts_1pct": ("routing_analysis_test", "dead_experts_threshold_1pct"),
    "router_entropy": ("routing_analysis_test", "entropy_mean"),
    "group_entropy": ("group_routing_analysis_test", "entropy_mean"),
    "group_active": ("group_routing_analysis_test", "active_groups_mean"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
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


def case_name(data: Dict[str, object], path: Path) -> str:
    settings = data.get("settings", {})
    if isinstance(settings, dict) and settings.get("experiment_tag"):
        return str(settings["experiment_tag"])
    return path.parent.parent.name


def case_sort_key(case: str) -> tuple[int, str]:
    return (CASE_ORDER.index(case) if case in CASE_ORDER else 99, case)


def regime_group(regime: str) -> str:
    if regime in {"steady_wake", "pre_hopf_steady"}:
        return "Steady"
    if regime in {"hopf_transition", "near_onset_hopf"}:
        return "Hopf"
    return "Periodic"


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


def top_summary(values: List[float], limit: int = 3) -> str:
    if not values:
        return ""
    order = np.argsort(-np.asarray(values, dtype=np.float64))[:limit]
    return "; ".join(f"e{int(i)}:{float(values[int(i)]):.3f}" for i in order if values[int(i)] > 1.0e-6)


def load_rows(path: Path) -> tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    case = case_name(data, path)
    rows: List[Dict[str, object]] = []
    router_rows: List[Dict[str, object]] = []
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        regime = str(item.get("test_regime", "unknown"))
        row: Dict[str, object] = {
            "case": case,
            "Re": float(item["test_Re"]),
            "label": str(item["test_Re_label"]),
            "regime": regime,
            "regime_group": regime_group(regime),
            "attractor": str(item.get("test_attractor", "unknown")),
            "best_epoch": int(item.get("best_epoch", -1)),
            "best_val_score": float(item.get("best_val_score", float("nan"))),
            "num_test": int(item.get("num_test", 0)),
        }
        for name, metric_path in METRIC_PATHS.items():
            row[name] = nested_metric(item, metric_path)
        hopf = item.get("rollout_autonomous_pressure", {}).get("hopf_amplitude", {})
        if isinstance(hopf, dict):
            for key in [
                "r_true_mean",
                "r_pred_mean",
                "amplitude_relative_l2",
                "log_amplitude_mae",
                "energy_log_mae",
                "overshoot_ratio_mean",
                "overshoot_gt2_fraction",
                "phase_abs_error_mean",
                "frequency_increment_mae",
            ]:
                try:
                    row[f"hopf_{key}"] = float(hopf.get(key, float("nan")))
                except (TypeError, ValueError):
                    row[f"hopf_{key}"] = float("nan")
        route = item.get("routing_analysis_test", {})
        if isinstance(route, dict):
            row["expert_top1_top3"] = top_summary(route.get("top1_fraction", []))
            row["expert_mean_load_top3"] = top_summary(route.get("mean_load", []))
            row["topk_set_counts"] = json.dumps(route.get("topk_set_counts", {}), ensure_ascii=False)
        group_route = item.get("group_routing_analysis_test", {})
        if isinstance(group_route, dict):
            row["group_top1_top3"] = top_summary(group_route.get("top1_fraction", []))
            row["group_mean_load_top3"] = top_summary(group_route.get("mean_load", []))
        rows.append(row)

        by_regime = item.get("routing_by_regime_test", {})
        if isinstance(by_regime, dict):
            for reg_name, diag in by_regime.items():
                if not isinstance(diag, dict) or int(diag.get("num_samples", 0)) <= 0:
                    continue
                router_rows.append(
                    {
                        "case": case,
                        "Re": row["Re"],
                        "test_regime": regime,
                        "diagnostic_regime": reg_name,
                        "num_samples": int(diag.get("num_samples", 0)),
                        "expert_top1_top3": top_summary(diag.get("expert_top1_fraction", [])),
                        "expert_top2_top3": top_summary(diag.get("expert_top2_fraction", [])),
                        "expert_mean_load_top3": top_summary(diag.get("expert_mean_load", [])),
                        "group_top1_top3": top_summary(diag.get("group_top1_fraction", [])),
                        "group_mean_load_top3": top_summary(diag.get("group_mean_load", [])),
                        "dead_experts_1pct": int(diag.get("dead_experts_threshold_1pct", -1)),
                        "active_experts_mean": float(diag.get("active_experts_mean", float("nan"))),
                        "expert_entropy": float(diag.get("expert_entropy_mean", float("nan"))),
                        "group_entropy": float(diag.get("group_entropy_mean", float("nan"))),
                        "topk_set_counts": json.dumps(diag.get("expert_topk_set_counts", {}), ensure_ascii=False),
                    }
                )
    return data, sorted(rows, key=lambda row: float(row["Re"])), router_rows


def aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"overall": {}, "by_regime_group": {}, "by_regime": {}}
    cases = sorted({str(row["case"]) for row in rows}, key=case_sort_key)
    for case in cases:
        sub = [row for row in rows if row["case"] == case]
        out["overall"][case] = {
            metric: finite_stats([float(row.get(metric, float("nan"))) for row in sub])
            for metric in METRIC_PATHS
        }
    for key, target_name in [("regime_group", "by_regime_group"), ("regime", "by_regime")]:
        groups = sorted({str(row[key]) for row in rows})
        for group in groups:
            out[target_name][group] = {}
            for case in cases:
                sub = [row for row in rows if row["case"] == case and row[key] == group]
                out[target_name][group][case] = {
                    metric: finite_stats([float(row.get(metric, float("nan"))) for row in sub])
                    for metric in METRIC_PATHS
                }
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(path: Path, stats: Dict[str, object], rows: List[Dict[str, object]]) -> None:
    cases = sorted({str(row["case"]) for row in rows}, key=case_sort_key)
    lines = [
        "# V16_2 AttractorStabilityMoE Aggregate Report",
        "",
        "Baseline is `V16_1_SteadyPressureAnchor32`. V16_2 cases are independent: steady contraction, Hopf log-radius normal-form, and regime-grouped routing.",
        "",
        "## Overall Means",
        "",
        "| Case | 1-step u | 1-step p | 24-step u | 24-step p | RHS | p energy | alpha | active experts | dead experts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overall = stats.get("overall", {})
    for case in cases:
        item = overall.get(case, {})
        def m(key: str) -> float:
            val = item.get(key, {})
            return float(val.get("mean", float("nan"))) if isinstance(val, dict) else float("nan")
        lines.append(
            f"| `{case}` | {m('one_step_velocity_l2'):.4g} | {m('one_step_pressure_l2'):.4g} | "
            f"{m('rollout_velocity_l2'):.4g} | {m('rollout_pressure_l2'):.4g} | {m('rhs_l2'):.4g} | "
            f"{m('rollout_pressure_energy_error'):.4g} | {m('alpha_mean'):.4g} | "
            f"{m('router_active_experts'):.4g} | {m('dead_experts_1pct'):.4g} |"
        )
    lines.extend(["", "## Regime Means", ""])
    for group in ["Steady", "Hopf", "Periodic"]:
        lines.extend(
            [
                f"### {group}",
                "",
                "| Case | 1-step u | 1-step p | 24-step u | 24-step p | p energy | active experts |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for case in cases:
            item = stats.get("by_regime_group", {}).get(group, {}).get(case, {})
            def gm(key: str) -> float:
                val = item.get(key, {})
                return float(val.get("mean", float("nan"))) if isinstance(val, dict) else float("nan")
            lines.append(
                f"| `{case}` | {gm('one_step_velocity_l2'):.4g} | {gm('one_step_pressure_l2'):.4g} | "
                f"{gm('rollout_velocity_l2'):.4g} | {gm('rollout_pressure_l2'):.4g} | "
                f"{gm('rollout_pressure_energy_error'):.4g} | {gm('router_active_experts'):.4g} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Output Files",
            "",
            "- `v16_2_summary_metrics.json`",
            "- `v16_2_per_re_metrics.csv`",
            "- `v16_2_hopf_near_onset_diagnostics.csv`",
            "- `v16_2_steady_pressure_drift.csv`",
            "- `v16_2_periodic_degradation.csv`",
            "- `v16_2_router_diagnostics.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []
    all_router_rows: List[Dict[str, object]] = []
    generated_from = []
    for metric_path in args.metrics:
        data, rows, router_rows = load_rows(metric_path)
        generated_from.append(str(metric_path))
        all_rows.extend(rows)
        all_router_rows.extend(router_rows)
    all_rows.sort(key=lambda row: (case_sort_key(str(row["case"])), float(row["Re"])))
    stats = aggregate(all_rows)
    summary = {"generated_from": generated_from, "stats": stats}
    (args.output_dir / "v16_2_summary_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(args.output_dir / "v16_2_per_re_metrics.csv", all_rows)
    write_csv(
        args.output_dir / "v16_2_hopf_near_onset_diagnostics.csv",
        [row for row in all_rows if row.get("regime_group") == "Hopf"],
    )
    write_csv(
        args.output_dir / "v16_2_steady_pressure_drift.csv",
        [row for row in all_rows if row.get("regime_group") == "Steady"],
    )
    write_csv(
        args.output_dir / "v16_2_periodic_degradation.csv",
        [row for row in all_rows if row.get("regime_group") == "Periodic"],
    )
    write_csv(args.output_dir / "v16_2_router_diagnostics.csv", all_router_rows)
    write_report(args.output_dir / "V16_2_AttractorStabilityMoE_TECHNICAL_REPORT.md", stats, all_rows)
    print(json.dumps({"output_dir": str(args.output_dir), "num_rows": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
