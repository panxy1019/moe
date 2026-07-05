#!/usr/bin/env python3
"""从 area-weighted L2 POD 数据计算半侵入式 Galerkin 投影张量。

本脚本面向 /home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re 中的
Global_POD_AreaWeighted_L2 输出。
速度 POD 文件中的模态布局为 [u(所有点), v(所有点)]，这里会补齐 w=0，形成 (N, 3, r_u)。
压力 POD 是标量模态，形成 (N, r_p)。该数据集的 phi_uv/phi_p 已经是 raw 物理空间模态，
并满足 point_areas 对应的 lumped mass 正交。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyvista as pv


Array = np.ndarray


DEFAULT_DATA_DIR = Path("/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re")
POD_DIR_NAME = "Global_POD_AreaWeighted_L2"
DEFAULT_MESH_VTU = Path("/home/ray/Desktop/Cylinder_Results/Re_100_VTK/run_Re_100_7107.vtk")
NU_DEFAULT = 1.0e-3


@dataclass(frozen=True)
class PodData:
    """POD 与均值场数据。"""

    re_values: Array
    re_labels: Array
    points: Array
    point_volumes: Array
    u_bar_by_re: Array  # (n_re, N, 3)
    p_bar_by_re: Array  # (n_re, N)
    phi_u: Array  # (N, 3, r_u)
    phi_p: Array  # (N, r_p)
    retained_rank_u: int
    retained_rank_p: int
    cumulative_energy_u: Array
    cumulative_energy_p: Array


@dataclass(frozen=True)
class TensorResult:
    """原始投影张量和经过 Gram 矩阵左乘修正后的 ODE 张量。"""

    gram_u: Array
    c_raw: Array
    a_raw: Array
    h_raw: Array
    p_raw: Array
    c: Array
    a: Array
    h: Array
    p: Array


class Timer:
    """简单计时器，用于生成可读日志。"""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def stamp(self) -> str:
        return f"{self.elapsed():8.1f}s"


class PyVistaDerivativeOperator:
    """将 pyvista.compute_derivative() 包装为 NumPy 输入/输出接口。"""

    def __init__(self, mesh: pv.UnstructuredGrid) -> None:
        # 只保留几何和拓扑；每次求导时写入临时 point_data。
        self.mesh = mesh.copy(deep=True)
        self.mesh.clear_data()

    def scalar_gradient(self, values: Array, name: str = "q") -> Array:
        """计算标量场梯度 grad(q)，返回形状 (N, 3)。"""

        q = as_point_scalar(values, self.mesh.n_points)
        self.mesh.clear_data()
        self.mesh.point_data[name] = np.ascontiguousarray(q, dtype=np.float64)
        out = self.mesh.compute_derivative(
            scalars=name,
            gradient=f"grad_{name}",
            preference="point",
        )
        return np.asarray(out.point_data[f"grad_{name}"], dtype=np.float64).copy()

    def vector_gradient(self, values: Array, name: str = "u") -> Array:
        """计算向量场梯度 grad(u)，返回形状 (N, 3, 3)。

        PyVista/VTK 返回的 9 分量顺序是：
        [du/dx, du/dy, du/dz, dv/dx, ..., dw/dz]。
        因此 reshape 后的轴含义为 [速度分量, 空间导数方向]。
        """

        vec = as_point_vector(values, self.mesh.n_points)
        self.mesh.clear_data()
        self.mesh.point_data[name] = np.ascontiguousarray(vec, dtype=np.float64)
        out = self.mesh.compute_derivative(
            scalars=name,
            gradient=f"grad_{name}",
            preference="point",
        )
        grad_flat = np.asarray(out.point_data[f"grad_{name}"], dtype=np.float64)
        return grad_flat.reshape(self.mesh.n_points, 3, 3).copy()

    def scalar_laplacian(self, values: Array, name: str = "q") -> Array:
        """计算标量场 Laplacian(q)=div(grad(q))，返回形状 (N,)。"""

        q = as_point_scalar(values, self.mesh.n_points)
        self.mesh.clear_data()
        self.mesh.point_data[name] = np.ascontiguousarray(q, dtype=np.float64)
        grad_name = f"grad_{name}"
        lap_name = f"lap_{name}"
        with_grad = self.mesh.compute_derivative(
            scalars=name,
            gradient=grad_name,
            preference="point",
        )
        with_lap = with_grad.compute_derivative(
            scalars=grad_name,
            divergence=lap_name,
            preference="point",
        )
        return np.asarray(with_lap.point_data[lap_name], dtype=np.float64).copy()

    def vector_laplacian(self, values: Array, name: str = "u") -> Array:
        """逐分量计算向量场 Laplacian(u)，返回形状 (N, 3)。"""

        vec = as_point_vector(values, self.mesh.n_points)
        lap = np.empty_like(vec, dtype=np.float64)
        for comp, label in enumerate(("x", "y", "z")):
            lap[:, comp] = self.scalar_laplacian(vec[:, comp], f"{name}_{label}")
        return lap


def as_point_scalar(values: Array, n_points: int) -> Array:
    """校验并压平点标量场。"""

    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == (n_points, 1):
        arr = arr[:, 0]
    if arr.shape != (n_points,):
        raise ValueError(f"标量场形状应为 ({n_points},) 或 ({n_points}, 1)，实际为 {arr.shape}")
    return arr


def as_point_vector(values: Array, n_points: int) -> Array:
    """校验点向量场形状。"""

    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n_points, 3):
        raise ValueError(f"向量场形状应为 ({n_points}, 3)，实际为 {arr.shape}")
    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算不可压 Navier-Stokes 半侵入式 Galerkin 投影张量",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="数据主目录")
    parser.add_argument("--re", type=float, default=1000.0, help="用于均值场 c/A 的 Reynolds 数")
    parser.add_argument("--all-re", action="store_true", help="对所有 Re 分别计算 c/A；H/P 共用")
    parser.add_argument("--ru", type=int, default=20, help="速度模态数；0 表示使用文件中全部模态")
    parser.add_argument("--rp", type=int, default=20, help="压力模态数；0 表示使用文件中全部模态")
    parser.add_argument("--nu", type=float, default=NU_DEFAULT, help="运动粘度")
    parser.add_argument(
        "--mesh-vtu",
        type=Path,
        default=DEFAULT_MESH_VTU,
        help="用于拓扑/求导的 VTK/VTU 文件；新数据集未保存完整拓扑，默认使用同网格旧数据的一帧",
    )
    parser.add_argument("--chunk-size", type=int, default=2048, help="H 张量节点分块大小")
    parser.add_argument("--output", type=Path, default=None, help="输出 .npz 文件路径")
    parser.add_argument("--report", type=Path, default=None, help="输出 Markdown 报告路径")
    parser.add_argument("--skip-raw", action="store_true", help="不在 .npz 中保存 raw 张量，可减小文件")
    parser.add_argument(
        "--reuse-shared-npz",
        type=Path,
        default=None,
        help="从已有 .npz 复用 G_u/H_raw/P_raw，适合补算多个 Re 的 c/A",
    )
    return parser.parse_args()


def load_pod_data(data_dir: Path, ru: int, rp: int) -> PodData:
    """读取 area-weighted L2 POD 文件，并整理为节点优先的 NumPy 数组。"""

    pod_dir = data_dir / POD_DIR_NAME
    vel_path = pod_dir / "global_velocity_pod_area_weighted_l2.npz"
    prs_path = pod_dir / "global_pressure_pod_area_weighted_l2.npz"
    weights_path = pod_dir / "mesh_l2_point_area_weights.npz"
    if not vel_path.exists() or not prs_path.exists():
        raise FileNotFoundError(f"未找到 POD 文件：{vel_path} 或 {prs_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"未找到 area-weighted L2 权重文件：{weights_path}")

    vel = np.load(vel_path)
    prs = np.load(prs_path)
    weights_npz = np.load(weights_path)

    n_points = int(np.asarray(vel["points"]).shape[0])
    re_values = np.asarray(vel["Re_values"], dtype=np.float64)
    re_labels = np.asarray(vel["Re_labels"]).astype(str)
    if not np.allclose(re_values, np.asarray(prs["Re_values"], dtype=np.float64), rtol=0.0, atol=1e-10):
        raise ValueError("速度 POD 和压力 POD 的 Re_values 不一致")
    if not np.array_equal(re_labels, np.asarray(prs["Re_labels"]).astype(str)):
        raise ValueError("速度 POD 和压力 POD 的 Re_labels 不一致")

    retained_u = int(np.asarray(vel["phi_uv"]).shape[0])
    retained_p = int(np.asarray(prs["phi_p"]).shape[0])
    ru_use = retained_u if ru == 0 else min(ru, retained_u)
    rp_use = retained_p if rp == 0 else min(rp, retained_p)

    points = np.asarray(vel["points"], dtype=np.float64)
    if points.shape != (n_points, 3):
        raise ValueError(f"POD points 形状异常：{points.shape}")
    if not np.allclose(points, np.asarray(prs["points"], dtype=np.float64), rtol=0.0, atol=1e-7):
        raise ValueError("速度 POD 和压力 POD 的 points 不一致")
    if not np.allclose(points, np.asarray(weights_npz["points"], dtype=np.float64), rtol=0.0, atol=1e-7):
        raise ValueError("POD points 和 mesh_l2_point_area_weights points 不一致")

    weight_key = "point_areas" if "point_areas" in weights_npz.files else "point_volumes"
    point_volumes = np.asarray(weights_npz[weight_key], dtype=np.float64)
    if point_volumes.shape != (n_points,):
        raise ValueError(f"{weight_key} 形状异常：{point_volumes.shape}")
    if np.any(point_volumes <= 0.0) or not np.all(np.isfinite(point_volumes)):
        raise ValueError(f"{weight_key} 含非正值或非有限值")

    mean_uv = np.asarray(vel["mean_uv_by_Re"], dtype=np.float64)
    mean_p = np.asarray(prs["mean_p_by_Re"], dtype=np.float64)
    u_bar_by_re = np.zeros((len(re_values), n_points, 3), dtype=np.float64)
    u_bar_by_re[:, :, 0] = mean_uv[:, :n_points]
    u_bar_by_re[:, :, 1] = mean_uv[:, n_points:]
    p_bar_by_re = mean_p

    phi_uv = np.asarray(vel["phi_uv"][:ru_use], dtype=np.float64)
    phi_u = np.zeros((n_points, 3, ru_use), dtype=np.float64)
    phi_u[:, 0, :] = phi_uv[:, :n_points].T
    phi_u[:, 1, :] = phi_uv[:, n_points:].T

    phi_p = np.asarray(prs["phi_p"][:rp_use], dtype=np.float64).T

    return PodData(
        re_values=re_values,
        re_labels=re_labels,
        points=points,
        point_volumes=point_volumes,
        u_bar_by_re=u_bar_by_re,
        p_bar_by_re=p_bar_by_re,
        phi_u=phi_u,
        phi_p=phi_p,
        retained_rank_u=retained_u,
        retained_rank_p=retained_p,
        cumulative_energy_u=np.asarray(vel["cumulative_energy_uv"], dtype=np.float64),
        cumulative_energy_p=np.asarray(prs["cumulative_energy_p"], dtype=np.float64),
    )


def default_vtu_path(data_dir: Path, re_value: float) -> Path:
    """返回同网格 VTK/VTU 模板文件。

    area-weighted POD 数据集只保存了 uvp npz 和 POD 结果，没有随目录保存完整拓扑。
    本脚本默认使用已校验与 POD points 完全对齐的旧 VTK 数据帧。
    """

    del data_dir, re_value
    return DEFAULT_MESH_VTU


def parse_flow_time(path: Path) -> float:
    """从 flow_<time>.vtu 文件名提取时间。"""

    stem = path.stem
    try:
        return float(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return float("inf")


def read_mesh(mesh_vtu: Path) -> pv.UnstructuredGrid:
    """读取 VTK/VTU 网格。"""

    mesh = pv.read(mesh_vtu)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError(f"期望 UnstructuredGrid，实际为 {type(mesh)}")
    return mesh


def validate_point_alignment(mesh: pv.UnstructuredGrid, pod_points: Array) -> None:
    """确认 POD 点序与 VTK/VTU 点序一致。"""

    mesh_points = np.asarray(mesh.points, dtype=np.float64)
    if mesh_points.shape != pod_points.shape:
        raise ValueError(f"mesh points {mesh_points.shape} 与 POD points {pod_points.shape} 不一致")
    max_abs = float(np.max(np.abs(mesh_points - pod_points)))
    if max_abs > 1e-6:
        raise ValueError(f"mesh 与 POD 点坐标未对齐，max_abs_diff={max_abs:.3e}")


def nodal_volume_weights(mesh: pv.UnstructuredGrid) -> Array:
    """由 VTU 单元体积构造节点 lumped mass 权重。

    对每个单元，把单元体积均分到该单元的所有顶点：
        M_ii = sum_{cell contains i} volume(cell) / n_points(cell)
    """

    sized = mesh.compute_cell_sizes(length=False, area=False, volume=True)
    volumes = np.asarray(sized.cell_data["Volume"], dtype=np.float64)
    weights = np.zeros(mesh.n_points, dtype=np.float64)

    cells = np.asarray(mesh.cells, dtype=np.int64)
    cursor = 0
    for cell_id in range(mesh.n_cells):
        n_cell_points = int(cells[cursor])
        point_ids = cells[cursor + 1 : cursor + 1 + n_cell_points]
        weights[point_ids] += volumes[cell_id] / float(n_cell_points)
        cursor += n_cell_points + 1

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("构造出的 mass_weights 含非有限值或非正值")
    return weights


def compute_mode_derivatives(
    deriv: PyVistaDerivativeOperator,
    phi_u: Array,
    phi_p: Array,
    timer: Timer,
    compute_pressure: bool = True,
) -> tuple[Array, Array, Array | None]:
    """计算速度模态梯度/拉普拉斯和压力模态梯度。"""

    n_points, _, ru = phi_u.shape
    rp = phi_p.shape[1]
    grad_phi_u = np.empty((n_points, 3, 3, ru), dtype=np.float64)
    lap_phi_u = np.empty((n_points, 3, ru), dtype=np.float64)
    grad_phi_p: Array | None = None

    print(f"[{timer.stamp()}] 计算速度模态导数：r_u={ru}")
    for j in range(ru):
        grad_phi_u[:, :, :, j] = deriv.vector_gradient(phi_u[:, :, j], f"phi_u_{j:04d}")
        lap_phi_u[:, :, j] = deriv.vector_laplacian(phi_u[:, :, j], f"phi_u_{j:04d}")
        if (j + 1) % max(1, ru // 10) == 0 or j + 1 == ru:
            print(f"[{timer.stamp()}]   velocity mode derivatives {j + 1}/{ru}", flush=True)

    if compute_pressure:
        grad_phi_p = np.empty((n_points, 3, rp), dtype=np.float64)
        print(f"[{timer.stamp()}] 计算压力模态梯度：r_p={rp}")
        for m in range(rp):
            grad_phi_p[:, :, m] = deriv.scalar_gradient(phi_p[:, m], f"phi_p_{m:04d}")
            if (m + 1) % max(1, rp // 10) == 0 or m + 1 == rp:
                print(f"[{timer.stamp()}]   pressure mode gradients {m + 1}/{rp}", flush=True)
    else:
        print(f"[{timer.stamp()}] 跳过压力模态梯度：将复用已有 P_raw", flush=True)

    return grad_phi_u, lap_phi_u, grad_phi_p


def compute_mean_derivatives(
    deriv: PyVistaDerivativeOperator,
    u_bar: Array,
    p_bar: Array,
    timer: Timer,
    label: str,
) -> tuple[Array, Array, Array]:
    """计算均值场梯度、速度拉普拉斯和压力梯度。"""

    print(f"[{timer.stamp()}] 计算均值场导数：{label}", flush=True)
    grad_u_bar = deriv.vector_gradient(u_bar, f"u_bar_{label}")
    lap_u_bar = deriv.vector_laplacian(u_bar, f"u_bar_{label}")
    grad_p_bar = deriv.scalar_gradient(p_bar, f"p_bar_{label}")
    return grad_u_bar, lap_u_bar, grad_p_bar


def inner_vector_modes(phi_u: Array, vector_field: Array, weights: Array) -> Array:
    """<phi_i, f>_M。"""

    return np.einsum("nci,nc,n->i", phi_u, vector_field, weights, optimize=True)


def compute_shared_raw_tensors(
    phi_u: Array,
    grad_phi_u: Array,
    grad_phi_p: Array,
    weights: Array,
    chunk_size: int,
    timer: Timer,
) -> tuple[Array, Array, Array]:
    """计算与均值场无关的 Gram、H、P 原始投影张量。"""

    print(f"[{timer.stamp()}] 计算共享 Gram/P/H", flush=True)
    gram_u = np.einsum("nci,ncj,n->ij", phi_u, phi_u, weights, optimize=True)

    # P_im = <phi_i, -grad(psi_m)>_M
    p_raw = -np.einsum("nci,ncm,n->im", phi_u, grad_phi_p, weights, optimize=True)

    h_raw = compute_quadratic_tensor(phi_u, grad_phi_u, weights, chunk_size, timer)
    return gram_u, h_raw, p_raw


def compute_raw_mean_terms(
    phi_u: Array,
    grad_phi_u: Array,
    lap_phi_u: Array,
    u_bar: Array,
    grad_u_bar: Array,
    lap_u_bar: Array,
    grad_p_bar: Array,
    weights: Array,
    nu: float,
    timer: Timer,
    label: str,
) -> tuple[Array, Array]:
    """按给定均值场计算 c、A 原始投影张量。"""

    print(f"[{timer.stamp()}] 计算均值相关 c/A：{label}", flush=True)

    # c_i = <phi_i, -(ubar·grad)ubar + nu Lap(ubar) - grad(pbar)>_M
    mean_convection = -np.einsum("na,nca->nc", u_bar, grad_u_bar, optimize=True)
    mean_residual = mean_convection + nu * lap_u_bar - grad_p_bar
    c_raw = inner_vector_modes(phi_u, mean_residual, weights)

    # A_ij = <phi_i, -(ubar·grad)phi_j -(phi_j·grad)ubar + nu Lap(phi_j)>_M
    cross_1 = -np.einsum("na,ncaj->ncj", u_bar, grad_phi_u, optimize=True)
    cross_2 = -np.einsum("naj,nca->ncj", phi_u, grad_u_bar, optimize=True)
    linear_field = cross_1 + cross_2 + nu * lap_phi_u
    a_raw = np.einsum("nci,ncj,n->ij", phi_u, linear_field, weights, optimize=True)

    return c_raw, a_raw


def compute_quadratic_tensor(
    phi_u: Array,
    grad_phi_u: Array,
    weights: Array,
    chunk_size: int,
    timer: Timer,
) -> Array:
    """计算 H_ijk = <phi_i, -(phi_j·grad)phi_k>_M。

    使用节点分块降低内存峰值；每个块内部使用 np.einsum 完成 i,j,k 的高维收缩，
    不在模态维度 r_u 上写 Python 原生 for 循环。
    """

    n_points, _, ru = phi_u.shape
    h_raw = np.zeros((ru, ru, ru), dtype=np.float64)
    if chunk_size <= 0:
        chunk_size = n_points
    n_chunks = (n_points + chunk_size - 1) // chunk_size
    print(f"[{timer.stamp()}] 计算二次对流张量 H：r_u={ru}, chunks={n_chunks}", flush=True)

    for chunk_id, start in enumerate(range(0, n_points, chunk_size), start=1):
        stop = min(start + chunk_size, n_points)
        h_raw -= np.einsum(
            "nci,naj,ncak,n->ijk",
            phi_u[start:stop],
            phi_u[start:stop],
            grad_phi_u[start:stop],
            weights[start:stop],
            optimize=True,
        )
        if chunk_id % max(1, n_chunks // 20) == 0 or chunk_id == n_chunks:
            print(f"[{timer.stamp()}]   H chunks {chunk_id}/{n_chunks}", flush=True)

    return h_raw


def left_multiply_gram_inverse(
    gram_u: Array,
    c_raw: Array,
    a_raw: Array,
    h_raw: Array,
    p_raw: Array,
) -> tuple[Array, Array, Array, Array]:
    """将 G da/dt = raw_rhs 转换为 da/dt = G^{-1} raw_rhs。"""

    ru = gram_u.shape[0]
    c = np.linalg.solve(gram_u, c_raw)
    a = np.linalg.solve(gram_u, a_raw)
    h = np.linalg.solve(gram_u, h_raw.reshape(ru, -1)).reshape(h_raw.shape)
    p = np.linalg.solve(gram_u, p_raw)
    return c, a, h, p


def compute_tensors(
    pod: PodData,
    mesh: pv.UnstructuredGrid,
    re_indices: Iterable[int],
    weights: Array,
    nu: float,
    chunk_size: int,
    timer: Timer,
    shared_raw: tuple[Array, Array, Array] | None = None,
) -> dict[str, object]:
    """计算一个或多个 Re 均值下的投影张量。"""

    deriv = PyVistaDerivativeOperator(mesh)
    grad_phi_u, lap_phi_u, grad_phi_p = compute_mode_derivatives(
        deriv,
        pod.phi_u,
        pod.phi_p,
        timer,
        compute_pressure=shared_raw is None,
    )
    if shared_raw is None:
        if grad_phi_p is None:
            raise RuntimeError("内部错误：未计算 grad_phi_p，无法装配 P_raw")
        gram_u, h_raw, p_raw = compute_shared_raw_tensors(
            phi_u=pod.phi_u,
            grad_phi_u=grad_phi_u,
            grad_phi_p=grad_phi_p,
            weights=weights,
            chunk_size=chunk_size,
            timer=timer,
        )
    else:
        gram_u, h_raw, p_raw = shared_raw
        print(f"[{timer.stamp()}] 已复用共享 Gram/H_raw/P_raw", flush=True)

    re_indices_list = [int(idx) for idx in re_indices]
    results: dict[str, TensorResult] = {}
    for idx in re_indices_list:
        re_value = float(pod.re_values[idx])
        label = str(pod.re_labels[idx])
        u_bar = pod.u_bar_by_re[idx]
        p_bar = pod.p_bar_by_re[idx]
        grad_u_bar, lap_u_bar, grad_p_bar = compute_mean_derivatives(
            deriv,
            u_bar,
            p_bar,
            timer,
            label,
        )
        c_raw, a_raw = compute_raw_mean_terms(
            phi_u=pod.phi_u,
            grad_phi_u=grad_phi_u,
            lap_phi_u=lap_phi_u,
            u_bar=u_bar,
            grad_u_bar=grad_u_bar,
            lap_u_bar=lap_u_bar,
            grad_p_bar=grad_p_bar,
            weights=weights,
            nu=nu,
            timer=timer,
            label=label,
        )
        c, a, h, p = left_multiply_gram_inverse(gram_u, c_raw, a_raw, h_raw, p_raw)
        results[label] = TensorResult(
            gram_u=gram_u,
            c_raw=c_raw,
            a_raw=a_raw,
            h_raw=h_raw,
            p_raw=p_raw,
            c=c,
            a=a,
            h=h,
            p=p,
        )

    return {
        "mode_derivative_shapes": {
            "grad_phi_u": grad_phi_u.shape,
            "lap_phi_u": lap_phi_u.shape,
            "grad_phi_p": None if grad_phi_p is None else grad_phi_p.shape,
        },
        "results": results,
        "computed_indices": np.asarray(re_indices_list, dtype=np.int32),
    }


def load_shared_raw_tensors(path: Path, ru: int, rp: int) -> tuple[Array, Array, Array]:
    """从已有输出文件复用 G_u、H_raw、P_raw。"""

    z = np.load(path, allow_pickle=False)
    g_key = next((key for key in z.files if key.endswith("_G_u")), None)
    h_key = next((key for key in z.files if key.endswith("_H_raw")), None)
    p_key = next((key for key in z.files if key.endswith("_P_raw")), None)
    if g_key is None or h_key is None or p_key is None:
        raise KeyError(f"{path} 中未找到 *_G_u、*_H_raw、*_P_raw")

    gram_u = np.asarray(z[g_key], dtype=np.float64)
    h_raw = np.asarray(z[h_key], dtype=np.float64)
    p_raw = np.asarray(z[p_key], dtype=np.float64)
    if gram_u.shape != (ru, ru):
        raise ValueError(f"复用 G_u 形状 {gram_u.shape} 与 r_u={ru} 不符")
    if h_raw.shape != (ru, ru, ru):
        raise ValueError(f"复用 H_raw 形状 {h_raw.shape} 与 r_u={ru} 不符")
    if p_raw.shape != (ru, rp):
        raise ValueError(f"复用 P_raw 形状 {p_raw.shape} 与 r_u={ru}, r_p={rp} 不符")
    if not (np.isfinite(gram_u).all() and np.isfinite(h_raw).all() and np.isfinite(p_raw).all()):
        raise ValueError("复用的共享张量含非有限值")
    return gram_u, h_raw, p_raw


def save_npz(
    path: Path,
    pod: PodData,
    mesh_vtu: Path,
    weights: Array,
    re_indices: list[int],
    nu: float,
    chunk_size: int,
    tensor_data: dict[str, object],
    skip_raw: bool,
) -> None:
    """保存张量和元数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    re_indices_arr = np.asarray(re_indices, dtype=np.int32)
    arrays: dict[str, Array] = {
        "Re_values_computed": np.asarray(pod.re_values[re_indices_arr], dtype=np.float64),
        "Re_labels_computed": np.asarray(pod.re_labels[re_indices_arr]),
        "pod_Re_values": np.asarray(pod.re_values, dtype=np.float64),
        "pod_Re_labels": np.asarray(pod.re_labels),
        "mass_weights": weights,
        "points": pod.points,
        "nu": np.asarray(nu, dtype=np.float64),
        "chunk_size": np.asarray(chunk_size, dtype=np.int32),
        "r_u": np.asarray(pod.phi_u.shape[2], dtype=np.int32),
        "r_p": np.asarray(pod.phi_p.shape[1], dtype=np.int32),
    }

    metadata = {
        "mesh_vtu": str(mesh_vtu),
        "data_layout": {
            "phi_u": "(N, 3, r_u), velocity components are x/y/z; z modes are zero for this 2D case",
            "phi_p": "(N, r_p)",
            "mass_weights": "(N,), area-weighted L2 point_areas from mesh_l2_point_area_weights.npz",
        },
        "equations": {
            "raw": "G da/dt = c_raw + A_raw a + H_raw(a,a) + P_raw b",
            "ode": "da/dt = c + A a + H(a,a) + P b, with c/A/H/P = G^{-1} raw tensors",
        },
        "compact_layout": bool(skip_raw),
        "compact_layout_note": "When skip_raw=True, shared G_u/H/P are written once and per-Re arrays contain c/A only.",
        "weighted_l2_note": "For this dataset phi_uv/phi_p are raw-space area-weighted POD modes and are M-orthonormal up to storage precision.",
        "derivative": "pyvista.UnstructuredGrid.compute_derivative(), point preference",
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, indent=2))

    results = tensor_data["results"]
    assert isinstance(results, dict)
    result_items = list(results.items())
    if not result_items:
        raise ValueError("没有可保存的 Re 结果")

    if skip_raw:
        first_result = result_items[0][1]
        assert isinstance(first_result, TensorResult)
        arrays["G_u"] = first_result.gram_u
        arrays["H"] = first_result.h
        arrays["P"] = first_result.p
        arrays["c_all"] = np.stack([result.c for _, result in result_items], axis=0)
        arrays["A_all"] = np.stack([result.a for _, result in result_items], axis=0)

    for label, result in result_items:
        assert isinstance(result, TensorResult)
        prefix = label
        arrays[f"{prefix}_c"] = result.c
        arrays[f"{prefix}_A"] = result.a
        if not skip_raw:
            arrays[f"{prefix}_G_u"] = result.gram_u
            arrays[f"{prefix}_H"] = result.h
            arrays[f"{prefix}_P"] = result.p
            arrays[f"{prefix}_c_raw"] = result.c_raw
            arrays[f"{prefix}_A_raw"] = result.a_raw
            arrays[f"{prefix}_H_raw"] = result.h_raw
            arrays[f"{prefix}_P_raw"] = result.p_raw

    np.savez(path, **arrays)


def tensor_norms(result: TensorResult) -> dict[str, float]:
    """报告用范数和 Gram 条件数。"""

    return {
        "cond_G_u": float(np.linalg.cond(result.gram_u)),
        "norm_c": float(np.linalg.norm(result.c)),
        "norm_A": float(np.linalg.norm(result.a)),
        "norm_H": float(np.linalg.norm(result.h)),
        "norm_P": float(np.linalg.norm(result.p)),
        "norm_c_raw": float(np.linalg.norm(result.c_raw)),
        "norm_A_raw": float(np.linalg.norm(result.a_raw)),
        "norm_H_raw": float(np.linalg.norm(result.h_raw)),
        "norm_P_raw": float(np.linalg.norm(result.p_raw)),
    }


def write_report(
    path: Path,
    data_dir: Path,
    mesh_vtu: Path,
    pod: PodData,
    weights: Array,
    re_indices: list[int],
    nu: float,
    output_npz: Path,
    tensor_data: dict[str, object],
    elapsed: float,
) -> None:
    """写出 Markdown 过程说明和最终方程。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    results = tensor_data["results"]
    assert isinstance(results, dict)
    re_indices_arr = np.asarray(re_indices, dtype=np.int32)
    computed_re_values = pod.re_values[re_indices_arr]
    computed_re_labels = pod.re_labels[re_indices_arr]

    lines: list[str] = []
    lines.append("# Area-weighted L2 Semi-intrusive Galerkin Projection Tensors\n")
    lines.append("## 数据来源\n")
    lines.append(f"- 数据目录：`{data_dir}`")
    lines.append(f"- 网格模板：`{mesh_vtu}`")
    lines.append(f"- POD 目录：`{data_dir / POD_DIR_NAME}`")
    lines.append(f"- 计算 Re 数量：`{len(re_indices)}`")
    lines.append(f"- 计算 Re 标签：`{computed_re_labels.tolist()}`")
    lines.append(f"- 运动粘度：`nu = {nu:.8g}`")
    lines.append(f"- 输出张量文件：`{output_npz}`\n")

    ru = pod.phi_u.shape[2]
    rp = pod.phi_p.shape[1]
    n_points = pod.points.shape[0]
    lines.append("## 数组形状\n")
    lines.append(f"- `N = {n_points}`")
    lines.append(f"- `r_u = {ru}`，`phi_u.shape = {pod.phi_u.shape}`")
    lines.append(f"- `r_p = {rp}`，`phi_p.shape = {pod.phi_p.shape}`")
    lines.append(f"- `mass_weights.shape = {weights.shape}`")
    lines.append(f"- `sum(mass_weights) = {float(np.sum(weights)):.12e}`")
    lines.append(f"- `min/max(mass_weights) = {float(np.min(weights)):.12e} / {float(np.max(weights)):.12e}`\n")

    lines.append("## 离散内积与质量权重\n")
    lines.append("离散内积采用 lumped mass 形式：\n")
    lines.append("```text")
    lines.append("<f, g>_M = sum_n mass_weights[n] * sum_c f[n,c] * g[n,c]")
    lines.append("```\n")
    lines.append("本数据集已经提供 `Global_POD_AreaWeighted_L2/mesh_l2_point_area_weights.npz`，脚本直接读取其中的 `point_areas` 作为 `mass_weights`。这些权重是 2D 计算域单元面积按节点 lumping 后的 nodal control areas，与 area-weighted L2 POD 报告中的内积一致。\n")
    lines.append("与上一版 unweighted 数据集不同，本次 `phi_uv/phi_p` 是 raw 物理空间 area-weighted POD 模态，并满足质量矩阵正交：`Phi^T M Phi ≈ I`。脚本仍保存 `G_u` 和 Gram 修正后的 `c/A/H/P`，用于消除 float32 存储和导出误差带来的小偏差。\n")

    lines.append("## PyVista 导数计算\n")
    lines.append("脚本通过 `pyvista.UnstructuredGrid.compute_derivative()` 在点数据上计算导数，并封装成 NumPy 接口：\n")
    lines.append("```python")
    lines.append("grad_q = mesh.compute_derivative(scalars='q', gradient='grad_q', preference='point')")
    lines.append("lap_q = grad_q.compute_derivative(scalars='grad_q', divergence='lap_q', preference='point')")
    lines.append("grad_u = mesh.compute_derivative(scalars='u', gradient='grad_u', preference='point')")
    lines.append("```")
    lines.append("向量梯度的 9 个分量按 `[du/dx, du/dy, du/dz, dv/dx, ..., dw/dz]` 排列，脚本 reshape 为 `(N, 3, 3)`，轴含义为 `[速度分量, 空间导数方向]`。\n")

    lines.append("## 投影张量定义\n")
    lines.append("原始投影张量为：\n")
    lines.append("```text")
    lines.append("G_ij     = <phi_i, phi_j>_M")
    lines.append("c_raw_i  = <phi_i, -(ubar · grad) ubar + nu Lap(ubar) - grad(pbar)>_M")
    lines.append("A_raw_ij = <phi_i, -(ubar · grad) phi_j - (phi_j · grad) ubar + nu Lap(phi_j)>_M")
    lines.append("H_raw_ijk= <phi_i, -(phi_j · grad) phi_k>_M")
    lines.append("P_raw_im = <phi_i, -grad(psi_m)>_M")
    lines.append("```\n")
    lines.append("对 area-weighted L2 raw-space 模态，理论上 `G≈I`。为保持数值严格性，脚本先装配：\n")
    lines.append("```text")
    lines.append("G da/dt = c_raw + A_raw a + H_raw(a,a) + P_raw b")
    lines.append("```\n")
    lines.append("脚本同时保存用于显式 ROM 的 Gram 修正张量：\n")
    lines.append("```text")
    lines.append("c = G^{-1} c_raw")
    lines.append("A = G^{-1} A_raw")
    lines.append("H[:,j,k] = G^{-1} H_raw[:,j,k]")
    lines.append("P = G^{-1} P_raw")
    lines.append("da/dt = c + A a + H(a,a) + P b")
    lines.append("```\n")

    lines.append("## `np.einsum` 收缩实现\n")
    lines.append("关键对流项没有在模态维度上使用 Python 原生循环。核心收缩为：\n")
    lines.append("```python")
    lines.append("cross_1 = -np.einsum('na,ncaj->ncj', u_bar, grad_phi_u)")
    lines.append("cross_2 = -np.einsum('naj,nca->ncj', phi_u, grad_u_bar)")
    lines.append("H_raw  -=  np.einsum('nci,naj,ncak,n->ijk', phi_u_blk, phi_u_blk, grad_phi_u_blk, w_blk)")
    lines.append("P_raw   = -np.einsum('nci,ncm,n->im', phi_u, grad_phi_p, weights)")
    lines.append("```\n")

    lines.append("## 本次运行结果\n")
    for re_value, result in results.items():
        assert isinstance(result, TensorResult)
        norms = tensor_norms(result)
        re_idx = int(np.flatnonzero(pod.re_labels == re_value)[0])
        re_float = float(pod.re_values[re_idx])
        max_g_err = float(np.max(np.abs(result.gram_u - np.eye(result.gram_u.shape[0]))))
        lines.append(f"### {re_value}  (`Re = {re_float:.12g}`)\n")
        lines.append(f"- `G_u.shape = {result.gram_u.shape}`")
        lines.append(f"- `c.shape = {result.c.shape}`")
        lines.append(f"- `A.shape = {result.a.shape}`")
        lines.append(f"- `H.shape = {result.h.shape}`")
        lines.append(f"- `P.shape = {result.p.shape}`")
        lines.append(f"- `cond(G_u) = {norms['cond_G_u']:.6e}`")
        lines.append(f"- `max_abs(G_u - I) = {max_g_err:.6e}`")
        lines.append(f"- `||c||_2 = {norms['norm_c']:.6e}`")
        lines.append(f"- `||A||_F = {norms['norm_A']:.6e}`")
        lines.append(f"- `||H||_F = {norms['norm_H']:.6e}`")
        lines.append(f"- `||P||_F = {norms['norm_P']:.6e}`\n")

    lines.append("## 最终侵入式 ROM 方程\n")
    lines.append("对本次输出文件中的每个 `Re_xxx`，使用 Gram 修正后的张量：\n")
    lines.append("```text")
    lines.append("da_i/dt = c_i + sum_j A_ij a_j + sum_j sum_k H_ijk a_j a_k + sum_m P_im b_m")
    lines.append("```\n")
    lines.append("其中 `a` 是速度模态系数，`b` 是压力模态系数；`c/A` 对应 `.npz` 中的 `Re_xxx_c`、`Re_xxx_A`。在全 Re compact 输出中，`G_u/H/P` 是共享张量；单 Re 或非 compact raw 输出中也可能包含 `Re_xxx_H`、`Re_xxx_P` 形式的兼容键。\n")
    lines.append(f"总运行时间：`{elapsed:.1f} s`。\n")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    timer = Timer()

    data_dir = args.data_dir.expanduser().resolve()
    pod = load_pod_data(data_dir, args.ru, args.rp)
    if args.all_re:
        re_indices = list(range(len(pod.re_values)))
    else:
        nearest = int(np.argmin(np.abs(pod.re_values - float(args.re))))
        if abs(float(pod.re_values[nearest]) - float(args.re)) > 1e-6:
            raise ValueError(f"Re={args.re} 不在 POD Re_values 中；最近值为 {pod.re_values[nearest]}")
        re_indices = [nearest]
    re_values_selected = pod.re_values[re_indices]
    re_labels_selected = pod.re_labels[re_indices]

    mesh_vtu = args.mesh_vtu.expanduser().resolve() if args.mesh_vtu else default_vtu_path(data_dir, float(re_values_selected[0]))
    output = args.output
    if output is None:
        suffix = "allRe100_areaWeightedL2" if args.all_re else str(re_labels_selected[0])
        output = data_dir / f"semi_intrusive_galerkin_tensors_{suffix}_ru{pod.phi_u.shape[2]}_rp{pod.phi_p.shape[1]}.npz"
    output = output.expanduser().resolve()

    report = args.report
    if report is None:
        suffix = "allRe100_areaWeightedL2" if args.all_re else str(re_labels_selected[0])
        report = data_dir / f"SEMI_INTRUSIVE_GALERKIN_TENSORS_{suffix}_ru{pod.phi_u.shape[2]}_rp{pod.phi_p.shape[1]}.md"
    report = report.expanduser().resolve()

    print(f"[{timer.stamp()}] 数据目录: {data_dir}")
    print(f"[{timer.stamp()}] 网格文件: {mesh_vtu}")
    print(f"[{timer.stamp()}] Re labels: {re_labels_selected.tolist()}")
    print(f"[{timer.stamp()}] r_u={pod.phi_u.shape[2]}, r_p={pod.phi_p.shape[1]}, nu={args.nu:g}")

    mesh = read_mesh(mesh_vtu)
    validate_point_alignment(mesh, pod.points)

    print(f"[{timer.stamp()}] 读取 area-weighted L2 point_areas 作为质量权重", flush=True)
    weights = pod.point_volumes
    print(
        f"[{timer.stamp()}] mass sum={np.sum(weights):.12e}, "
        f"min={np.min(weights):.3e}, max={np.max(weights):.3e}",
        flush=True,
    )

    shared_raw = None
    if args.reuse_shared_npz is not None:
        reuse_path = args.reuse_shared_npz.expanduser().resolve()
        print(f"[{timer.stamp()}] 复用共享张量: {reuse_path}", flush=True)
        shared_raw = load_shared_raw_tensors(
            reuse_path,
            ru=pod.phi_u.shape[2],
            rp=pod.phi_p.shape[1],
        )

    tensor_data = compute_tensors(
        pod=pod,
        mesh=mesh,
        re_indices=re_indices,
        weights=weights,
        nu=float(args.nu),
        chunk_size=int(args.chunk_size),
        timer=timer,
        shared_raw=shared_raw,
    )

    print(f"[{timer.stamp()}] 保存张量: {output}", flush=True)
    save_npz(
        path=output,
        pod=pod,
        mesh_vtu=mesh_vtu,
        weights=weights,
        re_indices=re_indices,
        nu=float(args.nu),
        chunk_size=int(args.chunk_size),
        tensor_data=tensor_data,
        skip_raw=bool(args.skip_raw),
    )

    elapsed = timer.elapsed()
    print(f"[{timer.stamp()}] 写 Markdown 报告: {report}", flush=True)
    write_report(
        path=report,
        data_dir=data_dir,
        mesh_vtu=mesh_vtu,
        pod=pod,
        weights=weights,
        re_indices=re_indices,
        nu=float(args.nu),
        output_npz=output,
        tensor_data=tensor_data,
        elapsed=elapsed,
    )
    print(f"[{timer.stamp()}] 完成", flush=True)


if __name__ == "__main__":
    main()
