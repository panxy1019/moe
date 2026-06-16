#!/usr/bin/env python3
"""计算压力泊松代数代理系统张量。

目标方程：

    L b(t) = c^p + A^p a(t) + H^p(a(t), a(t))

其中速度/压力 POD 均来自 Global_POD_Weighted_L2，内积使用 mesh_l2_point_weights.npz
中的 point_volumes。脚本同时输出伪逆后的等效张量：

    b(t) = c_tilde + A_tilde a(t) + H_tilde(a(t), a(t)).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from compute_weighted_l2_semi_intrusive_galerkin_tensors import (
    DEFAULT_DATA_DIR,
    DEFAULT_MESH_VTU,
    POD_DIR_NAME,
    Array,
    PyVistaDerivativeOperator,
    Timer,
    load_pod_data,
    read_mesh,
    validate_point_alignment,
)


@dataclass(frozen=True)
class PressureSurrogateResult:
    """某个 Re 均值场对应的压力泊松 RHS 张量。"""

    c_p: Array
    a_p: Array
    c_tilde: Array
    a_tilde: Array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算 pressure Poisson surrogate 张量",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="30Re weighted L2 POD 数据目录")
    parser.add_argument("--mesh-vtu", type=Path, default=DEFAULT_MESH_VTU, help="同网格 VTU 模板")
    parser.add_argument("--re", type=float, default=1000.0, help="单个 Re；未设置 --all-re 时使用")
    parser.add_argument("--all-re", action="store_true", help="对所有 Re 均值场计算 c^p/A^p")
    parser.add_argument("--ru", type=int, default=0, help="速度模态数；0 表示全部保留模态")
    parser.add_argument("--rp", type=int, default=0, help="压力模态数；0 表示全部保留模态")
    parser.add_argument("--chunk-size", type=int, default=2048, help="H^p 节点分块大小")
    parser.add_argument("--pinv-rcond", type=float, default=1e-10, help="L 的 SVD 伪逆截断阈值")
    parser.add_argument("--output", type=Path, default=None, help="输出 .npz 文件")
    parser.add_argument("--report", type=Path, default=None, help="输出 Markdown 技术报告")
    return parser.parse_args()


def compute_mode_gradients(
    deriv: PyVistaDerivativeOperator,
    phi_u: Array,
    phi_p: Array,
    timer: Timer,
) -> tuple[Array, Array]:
    """计算速度模态梯度和压力模态梯度。"""

    n_points, _, ru = phi_u.shape
    rp = phi_p.shape[1]
    grad_phi_u = np.empty((n_points, 3, 3, ru), dtype=np.float64)
    grad_phi_p = np.empty((n_points, 3, rp), dtype=np.float64)

    print(f"[{timer.stamp()}] 计算速度模态梯度: r_u={ru}", flush=True)
    for j in range(ru):
        grad_phi_u[:, :, :, j] = deriv.vector_gradient(phi_u[:, :, j], f"phi_u_{j:04d}")
        if (j + 1) % max(1, ru // 10) == 0 or j + 1 == ru:
            print(f"[{timer.stamp()}]   velocity mode gradients {j + 1}/{ru}", flush=True)

    print(f"[{timer.stamp()}] 计算压力模态梯度: r_p={rp}", flush=True)
    for m in range(rp):
        grad_phi_p[:, :, m] = deriv.scalar_gradient(phi_p[:, m], f"phi_p_{m:04d}")
        if (m + 1) % max(1, rp // 10) == 0 or m + 1 == rp:
            print(f"[{timer.stamp()}]   pressure mode gradients {m + 1}/{rp}", flush=True)

    return grad_phi_u, grad_phi_p


def compute_shared_tensors(
    grad_phi_p: Array,
    phi_u: Array,
    grad_phi_u: Array,
    weights: Array,
    chunk_size: int,
    timer: Timer,
) -> tuple[Array, Array]:
    """计算共享的 L 和 H^p。"""

    print(f"[{timer.stamp()}] 计算压力 Laplacian 矩阵 L", flush=True)
    l_mat = -np.einsum("ncm,nck,n->mk", grad_phi_p, grad_phi_p, weights, optimize=True)
    l_mat = 0.5 * (l_mat + l_mat.T)

    n_points, _, ru = phi_u.shape
    rp = grad_phi_p.shape[2]
    h_p = np.zeros((rp, ru, ru), dtype=np.float64)
    if chunk_size <= 0:
        chunk_size = n_points
    n_chunks = (n_points + chunk_size - 1) // chunk_size
    print(f"[{timer.stamp()}] 计算非线性压力源张量 H^p: shape=({rp},{ru},{ru}), chunks={n_chunks}", flush=True)
    for chunk_id, start in enumerate(range(0, n_points, chunk_size), start=1):
        stop = min(start + chunk_size, n_points)
        h_p += np.einsum(
            "ncm,naj,ncak,n->mjk",
            grad_phi_p[start:stop],
            phi_u[start:stop],
            grad_phi_u[start:stop],
            weights[start:stop],
            optimize=True,
        )
        if chunk_id % max(1, n_chunks // 20) == 0 or chunk_id == n_chunks:
            print(f"[{timer.stamp()}]   H^p chunks {chunk_id}/{n_chunks}", flush=True)
    return l_mat, h_p


def compute_mean_rhs_tensors(
    deriv: PyVistaDerivativeOperator,
    grad_phi_p: Array,
    phi_u: Array,
    grad_phi_u: Array,
    u_bar: Array,
    p_bar: Array,
    weights: Array,
    timer: Timer,
    label: str,
) -> tuple[Array, Array]:
    """计算某个 Re 均值下的 c^p 和 A^p。

    符号来自：
        Δp = -div((u·grad)u)
        -∫ grad(psi_m)·grad(p) = ∫ grad(psi_m)·(u·grad)u

    对 p = p_bar + Psi b，得到：
        L b = ∫ grad(psi_m)·((u·grad)u + grad(p_bar)).
    """

    print(f"[{timer.stamp()}] 计算均值梯度: {label}", flush=True)
    grad_u_bar = deriv.vector_gradient(u_bar, f"u_bar_{label}")
    grad_p_bar = deriv.scalar_gradient(p_bar, f"p_bar_{label}")

    print(f"[{timer.stamp()}] 装配 c^p/A^p: {label}", flush=True)
    conv_bar = np.einsum("na,nca->nc", u_bar, grad_u_bar, optimize=True)
    c_p = np.einsum("ncm,nc,n->m", grad_phi_p, conv_bar + grad_p_bar, weights, optimize=True)

    cross_1 = np.einsum("na,ncaj->ncj", u_bar, grad_phi_u, optimize=True)
    cross_2 = np.einsum("naj,nca->ncj", phi_u, grad_u_bar, optimize=True)
    a_p = np.einsum("ncm,ncj,n->mj", grad_phi_p, cross_1 + cross_2, weights, optimize=True)
    return c_p, a_p


def pinv_solve_tensors(
    l_mat: Array,
    c_p: Array,
    a_p: Array,
    h_p: Array,
    rcond: float,
) -> tuple[Array, Array, Array, Array, Array, int]:
    """使用 L 的 Moore-Penrose 伪逆计算等效张量。"""

    u_svd, singular_values, vt_svd = np.linalg.svd(l_mat, full_matrices=False)
    if singular_values.size == 0:
        raise ValueError("L 矩阵为空")
    cutoff = float(rcond) * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > cutoff))
    inv_s = np.zeros_like(singular_values)
    inv_s[singular_values > cutoff] = 1.0 / singular_values[singular_values > cutoff]
    l_pinv = (vt_svd.T * inv_s) @ u_svd.T

    rp = l_mat.shape[0]
    c_tilde = l_pinv @ c_p
    a_tilde = l_pinv @ a_p
    h_tilde = (l_pinv @ h_p.reshape(rp, -1)).reshape(h_p.shape)
    return l_pinv, singular_values, c_tilde, a_tilde, h_tilde, rank


def select_re_indices(re_values: Array, re_value: float, all_re: bool) -> list[int]:
    if all_re:
        return list(range(len(re_values)))
    nearest = int(np.argmin(np.abs(re_values - float(re_value))))
    if abs(float(re_values[nearest]) - float(re_value)) > 1e-6:
        raise ValueError(f"Re={re_value} 不在 POD Re_values 中；最近值为 {re_values[nearest]}")
    return [nearest]


def save_results(
    output: Path,
    data_dir: Path,
    mesh_vtu: Path,
    re_indices: list[int],
    re_values: Array,
    re_labels: Array,
    weights: Array,
    l_mat: Array,
    l_pinv: Array,
    singular_values: Array,
    rank: int,
    h_p: Array,
    h_tilde: Array,
    results: dict[str, PressureSurrogateResult],
    metadata_extra: dict[str, object],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(re_indices, dtype=np.int32)
    arrays: dict[str, Array] = {
        "Re_values_computed": np.asarray(re_values[idx], dtype=np.float64),
        "Re_labels_computed": np.asarray(re_labels[idx]),
        "pod_Re_values": np.asarray(re_values, dtype=np.float64),
        "pod_Re_labels": np.asarray(re_labels),
        "mass_weights": weights,
        "L": l_mat,
        "L_pinv": l_pinv,
        "L_singular_values": singular_values,
        "L_pinv_rank": np.asarray(rank, dtype=np.int32),
        "H_p": h_p,
        "H_tilde": h_tilde,
    }
    metadata = {
        "data_dir": str(data_dir),
        "mesh_vtu": str(mesh_vtu),
        "pod_dir": str(data_dir / POD_DIR_NAME),
        "equation": "L b = c_p + A_p a + H_p(a,a)",
        "effective_equation": "b = c_tilde + A_tilde a + H_tilde(a,a)",
        "L_definition": "L_mk = -int grad(psi_m) dot grad(psi_k) dOmega",
        "rhs_sign_convention": "Delta p = -div((u dot grad)u); boundary terms are neglected in the projected weak form",
        **metadata_extra,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, indent=2))
    for label, result in results.items():
        arrays[f"{label}_c_p"] = result.c_p
        arrays[f"{label}_A_p"] = result.a_p
        arrays[f"{label}_c_tilde"] = result.c_tilde
        arrays[f"{label}_A_tilde"] = result.a_tilde
    np.savez_compressed(output, **arrays)


def tensor_norms(result: PressureSurrogateResult) -> dict[str, float]:
    return {
        "norm_c_p": float(np.linalg.norm(result.c_p)),
        "norm_A_p": float(np.linalg.norm(result.a_p)),
        "norm_c_tilde": float(np.linalg.norm(result.c_tilde)),
        "norm_A_tilde": float(np.linalg.norm(result.a_tilde)),
    }


def write_report(
    report: Path,
    data_dir: Path,
    mesh_vtu: Path,
    output: Path,
    re_indices: list[int],
    re_values: Array,
    re_labels: Array,
    weights: Array,
    l_mat: Array,
    singular_values: Array,
    rank: int,
    rcond: float,
    h_p: Array,
    h_tilde: Array,
    results: dict[str, PressureSurrogateResult],
    elapsed: float,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(re_indices, dtype=np.int32)
    cond_l = float(np.inf) if singular_values[-1] == 0.0 else float(singular_values[0] / singular_values[-1])
    cutoff = float(rcond) * float(singular_values[0])

    lines: list[str] = []
    lines.append("# Pressure Poisson Surrogate Galerkin Tensors\n")
    lines.append("## 数据来源\n")
    lines.append(f"- 数据目录：`{data_dir}`")
    lines.append(f"- POD 目录：`{data_dir / POD_DIR_NAME}`")
    lines.append(f"- 网格模板：`{mesh_vtu}`")
    lines.append(f"- 输出文件：`{output}`")
    lines.append(f"- 计算 Re 数量：`{len(re_indices)}`")
    lines.append(f"- 计算 Re 标签：`{re_labels[idx].tolist()}`\n")

    lines.append("## 弱形式与符号约定\n")
    lines.append("不可压缩动量方程取散度后采用压力泊松形式：\n")
    lines.append("```text")
    lines.append("Delta p = - div((u dot grad) u)")
    lines.append("```\n")
    lines.append("用压力基函数 `psi_m` 测试并忽略边界项：\n")
    lines.append("```text")
    lines.append("- int grad(psi_m) dot grad(p) dOmega = int grad(psi_m) dot ((u dot grad)u) dOmega")
    lines.append("```\n")
    lines.append("令 `p = p_bar + sum_k psi_k b_k`，`u = u_bar + sum_j phi_j a_j`，得到：\n")
    lines.append("```text")
    lines.append("L b = c^p + A^p a + H^p(a,a)")
    lines.append("L_mk = - int grad(psi_m) dot grad(psi_k) dOmega")
    lines.append("c^p_m = int grad(psi_m) dot ((u_bar dot grad)u_bar + grad(p_bar)) dOmega")
    lines.append("A^p_mj = int grad(psi_m) dot ((u_bar dot grad)phi_j + (phi_j dot grad)u_bar) dOmega")
    lines.append("H^p_mjk = int grad(psi_m) dot ((phi_j dot grad)phi_k) dOmega")
    lines.append("```\n")

    lines.append("## 数值实现\n")
    lines.append("- 导数由 `pyvista.UnstructuredGrid.compute_derivative()` 在非结构 VTU 网格上计算。")
    lines.append("- 体积分权重使用 weighted L2 POD 的 `point_volumes`。")
    lines.append("- 向量梯度 reshape 为 `(N, 3, 3)`，轴含义是 `[速度分量, 空间导数方向]`。")
    lines.append("- `H^p` 使用节点分块和 `np.einsum('ncm,naj,ncak,n->mjk', ...)` 装配。\n")

    lines.append("## 输出张量\n")
    lines.append(f"- `L.shape = {l_mat.shape}`")
    lines.append(f"- `H_p.shape = {h_p.shape}`")
    lines.append(f"- `H_tilde.shape = {h_tilde.shape}`")
    lines.append(f"- `mass_weights.shape = {weights.shape}`")
    lines.append(f"- `sum(mass_weights) = {float(np.sum(weights)):.12e}`")
    lines.append(f"- `L` SVD rank = `{rank}` / `{l_mat.shape[0]}` with `rcond={rcond:g}`")
    lines.append(f"- `L` singular cutoff = `{cutoff:.6e}`")
    lines.append(f"- `L` condition estimate = `{cond_l:.6e}`")
    lines.append(f"- `||H_p||_F = {float(np.linalg.norm(h_p)):.6e}`")
    lines.append(f"- `||H_tilde||_F = {float(np.linalg.norm(h_tilde)):.6e}`\n")

    lines.append("## 等效代数代理系统\n")
    lines.append("脚本保存 `L_pinv`，并已左乘得到最终等效张量：\n")
    lines.append("```text")
    lines.append("c_tilde = L_pinv c^p")
    lines.append("A_tilde = L_pinv A^p")
    lines.append("H_tilde[:,j,k] = L_pinv H^p[:,j,k]")
    lines.append("b(t) = c_tilde + A_tilde a(t) + H_tilde(a(t),a(t))")
    lines.append("```\n")

    lines.append("## 本次运行结果\n")
    for label, result in results.items():
        re_idx = int(np.flatnonzero(re_labels == label)[0])
        norms = tensor_norms(result)
        lines.append(f"### {label} (`Re = {float(re_values[re_idx]):.12g}`)\n")
        lines.append(f"- `c_p.shape = {result.c_p.shape}`")
        lines.append(f"- `A_p.shape = {result.a_p.shape}`")
        lines.append(f"- `c_tilde.shape = {result.c_tilde.shape}`")
        lines.append(f"- `A_tilde.shape = {result.a_tilde.shape}`")
        lines.append(f"- `||c^p||_2 = {norms['norm_c_p']:.6e}`")
        lines.append(f"- `||A^p||_F = {norms['norm_A_p']:.6e}`")
        lines.append(f"- `||c_tilde||_2 = {norms['norm_c_tilde']:.6e}`")
        lines.append(f"- `||A_tilde||_F = {norms['norm_A_tilde']:.6e}`\n")

    lines.append(f"总运行时间：`{elapsed:.1f} s`。\n")
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    timer = Timer()

    data_dir = args.data_dir.expanduser().resolve()
    mesh_vtu = args.mesh_vtu.expanduser().resolve()
    pod = load_pod_data(data_dir, args.ru, args.rp)
    re_indices = select_re_indices(pod.re_values, float(args.re), bool(args.all_re))
    selected_labels = pod.re_labels[np.asarray(re_indices, dtype=np.int32)]

    output = args.output
    if output is None:
        suffix = "allRe30_weightedL2" if args.all_re else str(selected_labels[0])
        output = data_dir / f"pressure_poisson_surrogate_tensors_{suffix}_ru{pod.phi_u.shape[2]}_rp{pod.phi_p.shape[1]}.npz"
    output = output.expanduser().resolve()

    report = args.report
    if report is None:
        suffix = "allRe30_weightedL2" if args.all_re else str(selected_labels[0])
        report = data_dir / f"PRESSURE_POISSON_SURROGATE_TENSORS_{suffix}_ru{pod.phi_u.shape[2]}_rp{pod.phi_p.shape[1]}.md"
    report = report.expanduser().resolve()

    print(f"[{timer.stamp()}] 数据目录: {data_dir}")
    print(f"[{timer.stamp()}] 网格文件: {mesh_vtu}")
    print(f"[{timer.stamp()}] Re labels: {selected_labels.tolist()}")
    print(f"[{timer.stamp()}] r_u={pod.phi_u.shape[2]}, r_p={pod.phi_p.shape[1]}", flush=True)

    mesh = read_mesh(mesh_vtu)
    validate_point_alignment(mesh, pod.points)
    weights = pod.point_volumes
    print(
        f"[{timer.stamp()}] mass sum={np.sum(weights):.12e}, "
        f"min={np.min(weights):.3e}, max={np.max(weights):.3e}",
        flush=True,
    )

    deriv = PyVistaDerivativeOperator(mesh)
    grad_phi_u, grad_phi_p = compute_mode_gradients(deriv, pod.phi_u, pod.phi_p, timer)
    l_mat, h_p = compute_shared_tensors(
        grad_phi_p=grad_phi_p,
        phi_u=pod.phi_u,
        grad_phi_u=grad_phi_u,
        weights=weights,
        chunk_size=int(args.chunk_size),
        timer=timer,
    )

    l_pinv, singular_values, _, _, h_tilde, rank = pinv_solve_tensors(
        l_mat=l_mat,
        c_p=np.zeros(pod.phi_p.shape[1], dtype=np.float64),
        a_p=np.zeros((pod.phi_p.shape[1], pod.phi_u.shape[2]), dtype=np.float64),
        h_p=h_p,
        rcond=float(args.pinv_rcond),
    )

    results: dict[str, PressureSurrogateResult] = {}
    for idx in re_indices:
        label = str(pod.re_labels[idx])
        c_p, a_p = compute_mean_rhs_tensors(
            deriv=deriv,
            grad_phi_p=grad_phi_p,
            phi_u=pod.phi_u,
            grad_phi_u=grad_phi_u,
            u_bar=pod.u_bar_by_re[idx],
            p_bar=pod.p_bar_by_re[idx],
            weights=weights,
            timer=timer,
            label=label,
        )
        c_tilde = l_pinv @ c_p
        a_tilde = l_pinv @ a_p
        results[label] = PressureSurrogateResult(
            c_p=c_p,
            a_p=a_p,
            c_tilde=c_tilde,
            a_tilde=a_tilde,
        )

    print(f"[{timer.stamp()}] 保存张量: {output}", flush=True)
    save_results(
        output=output,
        data_dir=data_dir,
        mesh_vtu=mesh_vtu,
        re_indices=re_indices,
        re_values=pod.re_values,
        re_labels=pod.re_labels,
        weights=weights,
        l_mat=l_mat,
        l_pinv=l_pinv,
        singular_values=singular_values,
        rank=rank,
        h_p=h_p,
        h_tilde=h_tilde,
        results=results,
        metadata_extra={
            "pinv_rcond": float(args.pinv_rcond),
            "r_u": int(pod.phi_u.shape[2]),
            "r_p": int(pod.phi_p.shape[1]),
        },
    )

    elapsed = timer.elapsed()
    print(f"[{timer.stamp()}] 写技术报告: {report}", flush=True)
    write_report(
        report=report,
        data_dir=data_dir,
        mesh_vtu=mesh_vtu,
        output=output,
        re_indices=re_indices,
        re_values=pod.re_values,
        re_labels=pod.re_labels,
        weights=weights,
        l_mat=l_mat,
        singular_values=singular_values,
        rank=rank,
        rcond=float(args.pinv_rcond),
        h_p=h_p,
        h_tilde=h_tilde,
        results=results,
        elapsed=elapsed,
    )
    print(f"[{timer.stamp()}] 完成", flush=True)


if __name__ == "__main__":
    main()
