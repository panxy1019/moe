#!/usr/bin/env python3
"""Print HPRS group and expert utilization from V14 metrics JSON files."""

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
            group = item.get("group_routing_analysis_test", {})
            shared = item.get("shared_operator_analysis_test", {})
            print(
                f"Re={item['test_Re']:.6g} "
                f"active_experts={route.get('active_experts_mean', float('nan')):.2f} "
                f"combined_load={fmt(route.get('mean_load', []))}"
            )
            if group:
                print(
                    f"  group: load={fmt(group.get('mean_load', []))} "
                    f"top1={fmt(group.get('top1_fraction', []))} "
                    f"active_groups={group.get('active_groups_mean', float('nan')):.2f}"
                )
            for name, info in route.get("by_router", {}).items():
                print(
                    f"  {name}: cv={info.get('load_cv', float('nan')):.3f} "
                    f"dead={info.get('dead_experts_threshold_1pct', 0)} "
                    f"active={info.get('active_experts_mean', float('nan')):.2f} "
                    f"top1={fmt(info.get('top1_fraction', []))}"
                )
            if shared:
                print(
                    "  shared-in-group: "
                    f"load={fmt(shared.get('group_mean_load', []))} "
                    f"entropy={shared.get('mixer_entropy_mean', float('nan')):.3f}"
                )


if __name__ == "__main__":
    main()
