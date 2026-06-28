#!/usr/bin/env python3
"""Print routed and shared expert utilization from V13 metrics JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n# {data['settings']['experiment_name']}")
        for item in data["results"]:
            route = item.get("routing_analysis_test", {})
            shared = item.get("shared_operator_analysis_test", {})
            print(f"Re={item['test_Re']:.6g} combined_load={fmt(route.get('mean_load', []))}")
            for name, info in route.get("by_router", {}).items():
                print(
                    f"  {name}: cv={info.get('load_cv', float('nan')):.3f} "
                    f"dead={info.get('dead_experts_threshold_1pct', 0)} "
                    f"top1={fmt(info.get('top1_fraction', []))}"
                )
            if shared:
                print(
                    "  shared velocity="
                    f"{fmt(shared.get('velocity_mean_mixer_weight', []))} "
                    "pressure="
                    f"{fmt(shared.get('pressure_mean_mixer_weight', []))}"
                )


if __name__ == "__main__":
    main()
