#!/usr/bin/env python3
"""构建 Physics-Generalizable 数据集的 Regime-specific area-weighted POD。

输出文件沿用当前 V15/V16 ROM 脚本读取的键名：

    Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz
    Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz
    Global_POD_AreaWeighted_L2/mesh_l2_point_area_weights.npz
    Global_POD_AreaWeighted_L2/pod_snapshot_index.csv
    Global_POD_AreaWeighted_L2/pod_area_weighted_l2_metadata.json

POD 对每个 Re 的快照先减去该 Re 的均值场，再做 lumped point-area 加权。
由于 periodic 子库的快照矩阵过大，本脚本采用可复现的 randomized block SVD，
只在原始快照上流式做矩阵乘法，不复用统一全局 ROM 张量。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import scipy.linalg


Array = np.ndarray

DEFAULT_DATA_DIR = Path("/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re")
POD_DIR_NAME = "Global_POD_AreaWeighted_L2"
OUT_ROOT_NAME = "Regime_ROM_Library"

REGIME_GROUPS: dict[str, tuple[str, ...]] = {
    "steady": ("steady_wake", "pre_hopf_steady"),
    "hopf": ("hopf_transition",),
    "periodic": (
        "developing_periodic_shedding",
        "mature_periodic_shedding",
        "high_re_2d_periodic_near_modeA",
    ),
}


@dataclass(frozen=True)
class CaseInfo:
    label: str
    re_value: float
    source_regime: str
    target_regime: str
    path: Path
    n_times: int


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def stamp(self) -> str:
        return f"{time.perf_counter() - self.t0:8.1f}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建三类 Regime-specific area-weighted POD 数据库",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="原始 Physics-Generalizable 数据目录")
    parser.add_argument("--out-root", type=Path, default=None, help="输出根目录；默认 data-dir/Regime_ROM_Library")
    parser.add_argument("--rank", type=int, default=80, help="保留 POD 模态数")
    parser.add_argument("--oversampling", type=int, default=32, help="randomized SVD 过采样数")
    parser.add_argument("--power-iter", type=int, default=2, help="randomized SVD power iterations")
    parser.add_argument("--seed", type=int, default=20260705, help="随机种子")
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=tuple(REGIME_GROUPS),
        default=list(REGIME_GROUPS),
        help="要构建的目标 Regime",
    )
    return parser.parse_args()


def read_snapshot_index(pod_dir: Path) -> list[dict[str, str]]:
    with (pod_dir / "pod_snapshot_index.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_cases(data_dir: Path, rows: list[dict[str, str]], target_regime: str) -> list[CaseInfo]:
    allowed = set(REGIME_GROUPS[target_regime])
    by_label: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["regime"] not in allowed:
            continue
        label = row["Re_label"]
        item = by_label.setdefault(
            label,
            {
                "re_value": float(row["Re"]),
                "source_regime": row["regime"],
                "n_times": 0,
            },
        )
        item["n_times"] = int(item["n_times"]) + 1

    cases: list[CaseInfo] = []
    for label, item in sorted(by_label.items(), key=lambda kv: float(kv[1]["re_value"])):
        path = data_dir / f"{label}_uvp_pointData.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            n_times = int(z["u"].shape[0])
        expected = int(item["n_times"])
        if n_times != expected:
            raise ValueError(f"{label} snapshot count mismatch: file={n_times}, csv={expected}")
        cases.append(
            CaseInfo(
                label=label,
                re_value=float(item["re_value"]),
                source_regime=str(item["source_regime"]),
                target_regime=target_regime,
                path=path,
                n_times=n_times,
            )
        )
    if not cases:
        raise ValueError(f"No cases found for target regime {target_regime}")
    return cases


def copy_weights(global_pod_dir: Path, out_pod_dir: Path) -> tuple[Array, Array, Array]:
    src = global_pod_dir / "mesh_l2_point_area_weights.npz"
    with np.load(src, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    out_pod_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_pod_dir / "mesh_l2_point_area_weights.npz", **arrays)
    points = np.asarray(arrays["points"], dtype=np.float64)
    point_areas = np.asarray(arrays["point_areas"], dtype=np.float64)
    sqrt_point_areas = np.asarray(arrays["sqrt_point_areas"], dtype=np.float64)
    return points, point_areas, sqrt_point_areas


def compute_case_means(cases: list[CaseInfo], n_points: int, timer: Timer) -> tuple[Array, Array]:
    mean_uv = np.empty((len(cases), 2 * n_points), dtype=np.float32)
    mean_p = np.empty((len(cases), n_points), dtype=np.float32)
    for i, case in enumerate(cases):
        with np.load(case.path, allow_pickle=False) as z:
            mean_u = np.asarray(z["u"], dtype=np.float64).mean(axis=0)
            mean_v = np.asarray(z["v"], dtype=np.float64).mean(axis=0)
            mean_press = np.asarray(z["p"], dtype=np.float64).mean(axis=0)
            points = np.asarray(z["points"], dtype=np.float64)
        mean_uv[i, :n_points] = mean_u.astype(np.float32)
        mean_uv[i, n_points:] = mean_v.astype(np.float32)
        mean_p[i] = mean_press.astype(np.float32)
        if i == 0:
            print(f"[{timer.stamp()}]   first case points shape={points.shape}", flush=True)
        print(f"[{timer.stamp()}]   means {case.label} ({i + 1}/{len(cases)})", flush=True)
    return mean_uv, mean_p


def snapshot_offsets(cases: list[CaseInfo]) -> Array:
    offsets = np.zeros(len(cases) + 1, dtype=np.int64)
    for i, case in enumerate(cases):
        offsets[i + 1] = offsets[i] + case.n_times
    return offsets


def load_weighted_velocity_block(case: CaseInfo, mean_uv_row: Array, sqrt_area: Array) -> tuple[Array, Array]:
    n_points = sqrt_area.shape[0]
    with np.load(case.path, allow_pickle=False) as z:
        u = np.asarray(z["u"], dtype=np.float64) - np.asarray(mean_uv_row[:n_points], dtype=np.float64)[None, :]
        v = np.asarray(z["v"], dtype=np.float64) - np.asarray(mean_uv_row[n_points:], dtype=np.float64)[None, :]
    u *= sqrt_area[None, :]
    v *= sqrt_area[None, :]
    return np.ascontiguousarray(u), np.ascontiguousarray(v)


def load_weighted_pressure_block(case: CaseInfo, mean_p_row: Array, sqrt_area: Array) -> Array:
    with np.load(case.path, allow_pickle=False) as z:
        p = np.asarray(z["p"], dtype=np.float64) - np.asarray(mean_p_row, dtype=np.float64)[None, :]
    p *= sqrt_area[None, :]
    return np.ascontiguousarray(p)


def randomized_velocity_pod(
    cases: list[CaseInfo],
    mean_uv: Array,
    sqrt_area: Array,
    rank: int,
    oversampling: int,
    power_iter: int,
    seed: int,
    timer: Timer,
) -> tuple[Array, Array, Array, Array, Array, float]:
    n_points = sqrt_area.shape[0]
    offsets = snapshot_offsets(cases)
    n_snap = int(offsets[-1])
    n_features = 2 * n_points
    work_rank = min(n_snap, rank + oversampling)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n_snap, work_rank), dtype=np.float64)

    def x_times(mat: Array, with_energy: bool = False) -> tuple[Array, float]:
        y = np.zeros((n_features, mat.shape[1]), dtype=np.float64)
        energy = 0.0
        for i, case in enumerate(cases):
            start, stop = int(offsets[i]), int(offsets[i + 1])
            u, v = load_weighted_velocity_block(case, mean_uv[i], sqrt_area)
            block = mat[start:stop]
            y[:n_points] += u.T @ block
            y[n_points:] += v.T @ block
            if with_energy:
                energy += float(np.sum(u * u) + np.sum(v * v))
        return y, energy

    def xt_times(q: Array) -> Array:
        z_out = np.zeros((n_snap, q.shape[1]), dtype=np.float64)
        q_u = np.ascontiguousarray(q[:n_points])
        q_v = np.ascontiguousarray(q[n_points:])
        for i, case in enumerate(cases):
            start, stop = int(offsets[i]), int(offsets[i + 1])
            u, v = load_weighted_velocity_block(case, mean_uv[i], sqrt_area)
            z_out[start:stop] = u @ q_u + v @ q_v
        return z_out

    print(f"[{timer.stamp()}] velocity randomized range: n_snap={n_snap}, work_rank={work_rank}", flush=True)
    y, total_energy = x_times(omega, with_energy=True)
    q, _ = np.linalg.qr(y, mode="reduced")
    for it in range(power_iter):
        print(f"[{timer.stamp()}] velocity power iteration {it + 1}/{power_iter}", flush=True)
        z = xt_times(q)
        y, _ = x_times(z, with_energy=False)
        q, _ = np.linalg.qr(y, mode="reduced")
    print(f"[{timer.stamp()}] velocity small SVD", flush=True)
    b_mat = xt_times(q).T
    u_hat, singular_values, vt = scipy.linalg.svd(b_mat, full_matrices=False, lapack_driver="gesdd")
    r = min(rank, singular_values.size)
    weighted_modes = q @ u_hat[:, :r]
    raw_modes = weighted_modes.copy()
    raw_modes[:n_points] /= sqrt_area[:, None]
    raw_modes[n_points:] /= sqrt_area[:, None]
    coeff = vt[:r].T * singular_values[:r][None, :]
    energy = singular_values[:r] ** 2 / max(total_energy, np.finfo(np.float64).tiny)
    cumulative = np.cumsum(singular_values[:r] ** 2) / max(total_energy, np.finfo(np.float64).tiny)
    return (
        raw_modes.T.astype(np.float32),
        weighted_modes.T.astype(np.float32),
        coeff.astype(np.float32),
        singular_values[:r].astype(np.float64),
        energy.astype(np.float64),
        total_energy,
    )


def randomized_pressure_pod(
    cases: list[CaseInfo],
    mean_p: Array,
    sqrt_area: Array,
    rank: int,
    oversampling: int,
    power_iter: int,
    seed: int,
    timer: Timer,
) -> tuple[Array, Array, Array, Array, Array, float]:
    n_points = sqrt_area.shape[0]
    offsets = snapshot_offsets(cases)
    n_snap = int(offsets[-1])
    work_rank = min(n_snap, rank + oversampling)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n_snap, work_rank), dtype=np.float64)

    def x_times(mat: Array, with_energy: bool = False) -> tuple[Array, float]:
        y = np.zeros((n_points, mat.shape[1]), dtype=np.float64)
        energy = 0.0
        for i, case in enumerate(cases):
            start, stop = int(offsets[i]), int(offsets[i + 1])
            p = load_weighted_pressure_block(case, mean_p[i], sqrt_area)
            y += p.T @ mat[start:stop]
            if with_energy:
                energy += float(np.sum(p * p))
        return y, energy

    def xt_times(q: Array) -> Array:
        z_out = np.zeros((n_snap, q.shape[1]), dtype=np.float64)
        q_c = np.ascontiguousarray(q)
        for i, case in enumerate(cases):
            start, stop = int(offsets[i]), int(offsets[i + 1])
            p = load_weighted_pressure_block(case, mean_p[i], sqrt_area)
            z_out[start:stop] = p @ q_c
        return z_out

    print(f"[{timer.stamp()}] pressure randomized range: n_snap={n_snap}, work_rank={work_rank}", flush=True)
    y, total_energy = x_times(omega, with_energy=True)
    q, _ = np.linalg.qr(y, mode="reduced")
    for it in range(power_iter):
        print(f"[{timer.stamp()}] pressure power iteration {it + 1}/{power_iter}", flush=True)
        z = xt_times(q)
        y, _ = x_times(z, with_energy=False)
        q, _ = np.linalg.qr(y, mode="reduced")
    print(f"[{timer.stamp()}] pressure small SVD", flush=True)
    b_mat = xt_times(q).T
    u_hat, singular_values, vt = scipy.linalg.svd(b_mat, full_matrices=False, lapack_driver="gesdd")
    r = min(rank, singular_values.size)
    weighted_modes = q @ u_hat[:, :r]
    raw_modes = weighted_modes / sqrt_area[:, None]
    coeff = vt[:r].T * singular_values[:r][None, :]
    energy = singular_values[:r] ** 2 / max(total_energy, np.finfo(np.float64).tiny)
    cumulative = np.cumsum(singular_values[:r] ** 2) / max(total_energy, np.finfo(np.float64).tiny)
    return (
        raw_modes.T.astype(np.float32),
        weighted_modes.T.astype(np.float32),
        coeff.astype(np.float32),
        singular_values[:r].astype(np.float64),
        energy.astype(np.float64),
        total_energy,
    )


def write_snapshot_index(out_pod_dir: Path, source_rows: list[dict[str, str]], cases: list[CaseInfo]) -> None:
    labels = {case.label for case in cases}
    rows = [row for row in source_rows if row["Re_label"] in labels]
    with (out_pod_dir / "pod_snapshot_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["snapshot_id", "Re", "Re_label", "regime", "time", "estimated_period", "local_snapshot_index"])
        writer.writeheader()
        for new_id, row in enumerate(rows):
            out = dict(row)
            out["snapshot_id"] = str(new_id)
            writer.writerow(out)


def save_pod_files(
    out_pod_dir: Path,
    cases: list[CaseInfo],
    points: Array,
    point_areas: Array,
    sqrt_point_areas: Array,
    mean_uv: Array,
    mean_p: Array,
    vel_pod: tuple[Array, Array, Array, Array, Array, float],
    prs_pod: tuple[Array, Array, Array, Array, Array, float],
    rank: int,
    oversampling: int,
    power_iter: int,
    seed: int,
) -> None:
    phi_uv, phi_uv_weighted, coeff_uv, sv_uv, energy_uv, total_uv = vel_pod
    phi_p, phi_p_weighted, coeff_p, sv_p, energy_p, total_p = prs_pod
    re_values = np.asarray([case.re_value for case in cases], dtype=np.float64)
    re_labels = np.asarray([case.label for case in cases])
    regimes = np.asarray([case.source_regime for case in cases])
    np.savez_compressed(
        out_pod_dir / "global_velocity_pod_area_weighted_l2.npz",
        phi_uv=phi_uv,
        phi_uv_weighted=phi_uv_weighted,
        coeff_uv=coeff_uv,
        mean_uv_by_Re=mean_uv.astype(np.float32),
        Re_values=re_values,
        Re_labels=re_labels,
        regimes=regimes,
        points=points.astype(np.float32),
        point_areas=point_areas.astype(np.float32),
        sqrt_point_areas=sqrt_point_areas.astype(np.float32),
        singular_values_uv=sv_uv,
        energy_uv=energy_uv,
        cumulative_energy_uv=np.cumsum(sv_uv**2) / max(total_uv, np.finfo(np.float64).tiny),
        total_weighted_energy_uv=np.asarray(total_uv, dtype=np.float64),
    )
    np.savez_compressed(
        out_pod_dir / "global_pressure_pod_area_weighted_l2.npz",
        phi_p=phi_p,
        phi_p_weighted=phi_p_weighted,
        coeff_p=coeff_p,
        mean_p_by_Re=mean_p.astype(np.float32),
        Re_values=re_values,
        Re_labels=re_labels,
        regimes=regimes,
        points=points.astype(np.float32),
        point_areas=point_areas.astype(np.float32),
        sqrt_point_areas=sqrt_point_areas.astype(np.float32),
        singular_values_p=sv_p,
        energy_p=energy_p,
        cumulative_energy_p=np.cumsum(sv_p**2) / max(total_p, np.finfo(np.float64).tiny),
        total_weighted_energy_p=np.asarray(total_p, dtype=np.float64),
    )
    metadata = {
        "method": "regime-specific randomized block POD with lumped nodal 2D area weights",
        "rank": rank,
        "oversampling": oversampling,
        "power_iter": power_iter,
        "seed": seed,
        "n_cases": len(cases),
        "total_snapshots": int(sum(case.n_times for case in cases)),
        "n_points": int(points.shape[0]),
        "velocity": {
            "phi_shape": list(phi_uv.shape),
            "coeff_shape": list(coeff_uv.shape),
            "captured_energy_rank": float(np.sum(sv_uv**2) / max(total_uv, np.finfo(np.float64).tiny)),
            "total_weighted_energy": float(total_uv),
        },
        "pressure": {
            "phi_shape": list(phi_p.shape),
            "coeff_shape": list(coeff_p.shape),
            "captured_energy_rank": float(np.sum(sv_p**2) / max(total_p, np.finfo(np.float64).tiny)),
            "total_weighted_energy": float(total_p),
        },
        "regime_groups": REGIME_GROUPS,
    }
    (out_pod_dir / "pod_area_weighted_l2_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_report(
    report_path: Path,
    target_regime: str,
    cases: list[CaseInfo],
    vel_pod: tuple[Array, Array, Array, Array, Array, float],
    prs_pod: tuple[Array, Array, Array, Array, Array, float],
    elapsed: float,
) -> None:
    phi_uv, _, coeff_uv, sv_uv, _, total_uv = vel_pod
    phi_p, _, coeff_p, sv_p, _, total_p = prs_pod
    lines = [
        f"# Regime-specific Area-weighted POD: {target_regime}",
        "",
        "## Regime Mapping",
        "",
        f"- Target regime: `{target_regime}`",
        f"- Source labels: `{list(REGIME_GROUPS[target_regime])}`",
        f"- Number of Re cases: `{len(cases)}`",
        f"- Total snapshots: `{sum(case.n_times for case in cases)}`",
        f"- Re range: `{min(case.re_value for case in cases):.12g}` to `{max(case.re_value for case in cases):.12g}`",
        "",
        "## POD Method",
        "",
        "- Per-Re mean subtraction is used before POD.",
        "- Inner product uses lumped `point_areas` from `mesh_l2_point_area_weights.npz`.",
        "- Randomized block SVD is applied directly to weighted raw snapshots; no global ROM tensors are reused.",
        "",
        "## Outputs",
        "",
        f"- `phi_uv.shape = {phi_uv.shape}`",
        f"- `coeff_uv.shape = {coeff_uv.shape}`",
        f"- `velocity captured energy = {float(np.sum(sv_uv**2) / max(total_uv, np.finfo(np.float64).tiny)):.12e}`",
        f"- `phi_p.shape = {phi_p.shape}`",
        f"- `coeff_p.shape = {coeff_p.shape}`",
        f"- `pressure captured energy = {float(np.sum(sv_p**2) / max(total_p, np.finfo(np.float64).tiny)):.12e}`",
        f"- elapsed: `{elapsed:.1f} s`",
        "",
        "## Cases",
        "",
    ]
    for case in cases:
        lines.append(f"- `{case.label}`: Re={case.re_value:.12g}, source=`{case.source_regime}`, snapshots={case.n_times}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_one_regime(
    data_dir: Path,
    out_root: Path,
    source_rows: list[dict[str, str]],
    target_regime: str,
    rank: int,
    oversampling: int,
    power_iter: int,
    seed: int,
) -> None:
    timer = Timer()
    global_pod_dir = data_dir / POD_DIR_NAME
    out_dir = out_root / target_regime
    out_pod_dir = out_dir / POD_DIR_NAME
    cases = load_cases(data_dir, source_rows, target_regime)
    print(f"[{timer.stamp()}] Build {target_regime}: cases={len(cases)}, snapshots={sum(c.n_times for c in cases)}", flush=True)
    points, point_areas, sqrt_point_areas = copy_weights(global_pod_dir, out_pod_dir)
    mean_uv, mean_p = compute_case_means(cases, points.shape[0], timer)
    vel_pod = randomized_velocity_pod(
        cases=cases,
        mean_uv=mean_uv,
        sqrt_area=sqrt_point_areas,
        rank=rank,
        oversampling=oversampling,
        power_iter=power_iter,
        seed=seed + 17,
        timer=timer,
    )
    prs_pod = randomized_pressure_pod(
        cases=cases,
        mean_p=mean_p,
        sqrt_area=sqrt_point_areas,
        rank=rank,
        oversampling=oversampling,
        power_iter=power_iter,
        seed=seed + 31,
        timer=timer,
    )
    write_snapshot_index(out_pod_dir, source_rows, cases)
    save_pod_files(
        out_pod_dir=out_pod_dir,
        cases=cases,
        points=points,
        point_areas=point_areas,
        sqrt_point_areas=sqrt_point_areas,
        mean_uv=mean_uv,
        mean_p=mean_p,
        vel_pod=vel_pod,
        prs_pod=prs_pod,
        rank=rank,
        oversampling=oversampling,
        power_iter=power_iter,
        seed=seed,
    )
    write_report(
        report_path=out_dir / f"REGIME_SPECIFIC_POD_{target_regime}.md",
        target_regime=target_regime,
        cases=cases,
        vel_pod=vel_pod,
        prs_pod=prs_pod,
        elapsed=time.perf_counter() - timer.t0,
    )
    print(f"[{timer.stamp()}] Done {target_regime}: {out_dir}", flush=True)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve() if args.out_root else data_dir / OUT_ROOT_NAME
    out_root.mkdir(parents=True, exist_ok=True)
    source_rows = read_snapshot_index(data_dir / POD_DIR_NAME)
    for regime in args.regimes:
        build_one_regime(
            data_dir=data_dir,
            out_root=out_root,
            source_rows=source_rows,
            target_regime=regime,
            rank=int(args.rank),
            oversampling=int(args.oversampling),
            power_iter=int(args.power_iter),
            seed=int(args.seed),
        )


if __name__ == "__main__":
    main()
