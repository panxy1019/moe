#!/usr/bin/env python3
"""Deep semi-intrusive MoE-ROM v2 experiment.

The model upgrades the v1 ridge closure to a PyTorch network:

    PhysicalContextEncoder -> stacked Shared-Routed MoE blocks -> dual heads

The heads predict both the next velocity POD state and the RHS closure
correction on top of the semi-intrusive Galerkin RHS.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1.0e-12


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0).astype(np.float32)
        scale = x.std(axis=0).astype(np.float32)
        scale[scale < 1.0e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/Cylinder_Results_Re500_1000_POD_data"),
    )
    parser.add_argument(
        "--tensor-path",
        type=Path,
        default=Path(
            "/root/Cylinder_Results_Re500_1000_POD_data/"
            "semi_intrusive_galerkin_tensors_allRe_ru80_rp80.npz"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/root/moe_rom_v2"))
    parser.add_argument("--experiment-name", default="deep_moe_v2")
    parser.add_argument("--r-u", type=int, default=16)
    parser.add_argument("--r-p", type=int, default=16)
    parser.add_argument("--test-res", type=int, nargs="+", default=[700, 1000])
    parser.add_argument("--phase-harmonics", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=55)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--train-rollout-steps", type=int, default=4)
    parser.add_argument("--rollout-batch", type=int, default=16)
    parser.add_argument("--recon-dim", type=int, default=4096)
    parser.add_argument("--lambda-coeff", type=float, default=1.0)
    parser.add_argument("--lambda-dyn", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=0.08)
    parser.add_argument("--lambda-rollout", type=float, default=0.12)
    parser.add_argument("--lambda-consistency", type=float, default=0.15)
    parser.add_argument("--lambda-router-balance", type=float, default=0.02)
    parser.add_argument("--lambda-router-entropy", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_snapshot_index(path: Path) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda row: int(row["snapshot_id"]))
    return {
        "snapshot_id": np.asarray([int(r["snapshot_id"]) for r in rows], dtype=np.int64),
        "Re": np.asarray([int(float(r["Re"])) for r in rows], dtype=np.int64),
        "time": np.asarray([float(r["time"]) for r in rows], dtype=np.float32),
        "period": np.asarray([float(r["period"]) for r in rows], dtype=np.float32),
        "phase": np.asarray([float(r["phase"]) % 1.0 for r in rows], dtype=np.float32),
        "local_snapshot_index": np.asarray(
            [int(float(r["local_snapshot_index"])) for r in rows], dtype=np.int64
        ),
    }


def centered_time_derivative(
    coeff: np.ndarray, re_values: np.ndarray, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    deriv = np.full_like(coeff, np.nan, dtype=np.float32)
    valid: List[int] = []
    for re in sorted(np.unique(re_values).tolist()):
        idx = np.where(re_values == re)[0]
        idx = idx[np.argsort(times[idx])]
        if len(idx) < 3:
            continue
        dt = (times[idx[2:]] - times[idx[:-2]]).astype(np.float32)
        deriv[idx[1:-1]] = (coeff[idx[2:]] - coeff[idx[:-2]]) / dt[:, None]
        valid.extend(idx[1:-1].tolist())
    return deriv, np.asarray(sorted(valid), dtype=np.int64)


def next_index_by_sequence(re_values: np.ndarray, times: np.ndarray) -> np.ndarray:
    nxt = np.full(re_values.shape[0], -1, dtype=np.int64)
    for re in sorted(np.unique(re_values).tolist()):
        idx = np.where(re_values == re)[0]
        idx = idx[np.argsort(times[idx])]
        nxt[idx[:-1]] = idx[1:]
    return nxt


def galerkin_rhs_by_re(
    tensors: np.lib.npyio.NpzFile,
    a: np.ndarray,
    b: np.ndarray,
    re_values: np.ndarray,
    r_u: int,
    r_p: int,
) -> np.ndarray:
    rhs = np.empty((a.shape[0], r_u), dtype=np.float32)
    for re in sorted(np.unique(re_values).tolist()):
        row = np.where(re_values == re)[0]
        prefix = f"Re_{int(re)}"
        c = tensors[f"{prefix}_c"][:r_u].astype(np.float32)
        A = tensors[f"{prefix}_A"][:r_u, :r_u].astype(np.float32)
        H = tensors[f"{prefix}_H"][:r_u, :r_u, :r_u].astype(np.float32)
        P = tensors[f"{prefix}_P"][:r_u, :r_p].astype(np.float32)
        ar = a[row]
        br = b[row]
        rhs[row] = (
            c[None, :]
            + ar @ A.T
            + np.einsum("ijk,nj,nk->ni", H, ar, ar, optimize=True)
            + br @ P.T
        )
    return rhs.astype(np.float32)


def make_features_np(
    a: np.ndarray,
    b: np.ndarray,
    rhs: np.ndarray,
    re_values: np.ndarray,
    phase: np.ndarray,
    harmonics: int,
) -> np.ndarray:
    re = re_values.astype(np.float32)
    re_norm = ((re - 750.0) / 250.0)[:, None]
    inv_re = (1000.0 / re)[:, None]
    theta = (2.0 * np.pi * phase.astype(np.float32))[:, None]
    cols = [re_norm, inv_re]
    for k in range(1, harmonics + 1):
        cols.append(np.sin(k * theta))
        cols.append(np.cos(k * theta))

    r_u = a.shape[1]
    low = min(4, r_u)
    split = min(12, r_u)
    e_low = np.linalg.norm(a[:, :low], axis=1, keepdims=True)
    e_mid = np.linalg.norm(a[:, low:split], axis=1, keepdims=True)
    e_high = np.linalg.norm(a[:, split:], axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    rhs_norm = np.linalg.norm(rhs, axis=1, keepdims=True)
    cols.extend([a, b, rhs, e_low, e_mid, e_high, b_norm, rhs_norm])
    return np.hstack(cols).astype(np.float32)


def make_features_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    rhs: torch.Tensor,
    re_values: torch.Tensor,
    phase: torch.Tensor,
    harmonics: int,
) -> torch.Tensor:
    re = re_values.float()
    cols = [((re - 750.0) / 250.0).unsqueeze(1), (1000.0 / re).unsqueeze(1)]
    theta = 2.0 * math.pi * phase.float()
    for k in range(1, harmonics + 1):
        cols.append(torch.sin(k * theta).unsqueeze(1))
        cols.append(torch.cos(k * theta).unsqueeze(1))
    r_u = a.shape[1]
    low = min(4, r_u)
    split = min(12, r_u)
    cols.extend(
        [
            a,
            b,
            rhs,
            torch.linalg.norm(a[:, :low], dim=1, keepdim=True),
            torch.linalg.norm(a[:, low:split], dim=1, keepdim=True),
            torch.linalg.norm(a[:, split:], dim=1, keepdim=True),
            torch.linalg.norm(b, dim=1, keepdim=True),
            torch.linalg.norm(rhs, dim=1, keepdim=True),
        ]
    )
    return torch.cat(cols, dim=1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicalContextEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedRoutedMoEBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        expert_hidden: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        temperature: float,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.norm = nn.LayerNorm(hidden_dim)
        self.shared = MLP(hidden_dim, expert_hidden, hidden_dim, dropout)
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList(
            [MLP(hidden_dim, expert_hidden, hidden_dim, dropout) for _ in range(num_experts)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.norm(h)
        shared = self.shared(z)
        logits = self.router(z) / max(self.temperature, 1.0e-4)
        probs = torch.softmax(logits, dim=-1)
        if 0 < self.top_k < self.num_experts:
            top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)
            gate = torch.zeros_like(probs)
            gate.scatter_(1, top_idx, top_val)
            gate = gate / (gate.sum(dim=-1, keepdim=True) + EPS)
        else:
            gate = probs
        expert_out = torch.stack([expert(z) for expert in self.experts], dim=1)
        routed = torch.einsum("be,beh->bh", gate, expert_out)
        h = h + shared + routed
        h = h + self.post(self.out_norm(h))
        return h, gate


class DualHeadMoEROM(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        expert_hidden: int,
        num_blocks: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        temperature: float,
    ):
        super().__init__()
        self.encoder = PhysicalContextEncoder(in_dim, hidden_dim, dropout)
        self.blocks = nn.ModuleList(
            [
                SharedRoutedMoEBlock(
                    hidden_dim,
                    expert_hidden,
                    num_experts,
                    top_k,
                    dropout,
                    temperature,
                )
                for _ in range(num_blocks)
            ]
        )
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.corr_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        h = self.encoder(x)
        gates = []
        for block in self.blocks:
            h, gate = block(h)
            gates.append(gate)
        alpha_next_std = self.alpha_head(h)
        corr_std = self.corr_head(h)
        return alpha_next_std, corr_std, gates


def router_regularization(gates: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if not gates:
        zero = torch.tensor(0.0)
        return zero, zero, {"balance": 0.0, "entropy": 0.0, "utilization": [1.0]}

    balance = torch.tensor(0.0, device=gates[0].device)
    entropy = torch.tensor(0.0, device=gates[0].device)
    utilization = []
    for gate in gates:
        mean_gate = gate.mean(dim=0)
        target = torch.full_like(mean_gate, 1.0 / gate.shape[1])
        balance = balance + torch.mean((mean_gate - target) ** 2)
        ent = -torch.sum(gate * torch.log(gate + EPS), dim=1).mean()
        entropy = entropy + ent
        utilization.append(mean_gate.detach().cpu().numpy())
    balance = balance / max(len(gates), 1)
    entropy = entropy / max(len(gates), 1)
    util = np.mean(np.stack(utilization, axis=0), axis=0) if utilization else np.zeros(1)
    return balance, entropy, {
        "balance": float(balance.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "utilization": [float(v) for v in util.tolist()],
    }


def relative_l2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + EPS))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def centered_r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(np.sum((y_pred - y_true) ** 2))
    cen = y_true - y_true.mean(axis=0, keepdims=True)
    return float(1.0 - sse / (float(np.sum(cen * cen)) + EPS))


def build_galerkin_torch(
    tensors: np.lib.npyio.NpzFile,
    re_list: Iterable[int],
    r_u: int,
    r_p: int,
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for re in sorted(set(int(v) for v in re_list)):
        prefix = f"Re_{re}"
        out[re] = {
            "c": torch.tensor(tensors[f"{prefix}_c"][:r_u], dtype=torch.float32, device=device),
            "A": torch.tensor(
                tensors[f"{prefix}_A"][:r_u, :r_u], dtype=torch.float32, device=device
            ),
            "H": torch.tensor(
                tensors[f"{prefix}_H"][:r_u, :r_u, :r_u], dtype=torch.float32, device=device
            ),
            "P": torch.tensor(
                tensors[f"{prefix}_P"][:r_u, :r_p], dtype=torch.float32, device=device
            ),
        }
    return out


def galerkin_rhs_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    re_values: torch.Tensor,
    gal: Dict[int, Dict[str, torch.Tensor]],
) -> torch.Tensor:
    rhs = torch.empty_like(a)
    for re in torch.unique(re_values).detach().cpu().tolist():
        re_int = int(re)
        mask = re_values == re_int
        ar = a[mask]
        br = b[mask]
        item = gal[re_int]
        rhs[mask] = (
            item["c"].unsqueeze(0)
            + ar @ item["A"].T
            + torch.einsum("ijk,bj,bk->bi", item["H"], ar, ar)
            + br @ item["P"].T
        )
    return rhs


def evaluate_model(
    model: DualHeadMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        alpha_std, corr_std, gates = model(x)
    alpha_pred = (
        alpha_std.detach().cpu().numpy() * scalers["alpha_next"].scale
        + scalers["alpha_next"].mean
    )
    corr_pred = (
        corr_std.detach().cpu().numpy() * scalers["residual"].scale
        + scalers["residual"].mean
    )
    rhs_pred = arrays["rhs_g"][sample_ids] + corr_pred
    y_rhs = arrays["adot"][sample_ids]
    y_alpha = arrays["a_next"][sample_ids]
    a_curr = arrays["a"][sample_ids]
    dt = arrays["dt_next"][sample_ids][:, None]
    euler_pred = a_curr + dt * rhs_pred

    util = []
    ent = []
    for gate in gates:
        g = gate.detach().cpu().numpy()
        util.append(g.mean(axis=0))
        ent.append(float(np.mean(-np.sum(g * np.log(g + EPS), axis=1))))
    util_mean = np.mean(np.stack(util, axis=0), axis=0) if util else np.zeros(1)

    return {
        "rhs_relative_l2": relative_l2_np(y_rhs, rhs_pred),
        "rhs_rmse": rmse_np(y_rhs, rhs_pred),
        "rhs_centered_r2": centered_r2_np(y_rhs, rhs_pred),
        "alpha_head_relative_l2": relative_l2_np(y_alpha, alpha_pred),
        "alpha_head_rmse": rmse_np(y_alpha, alpha_pred),
        "one_step_euler_relative_l2": relative_l2_np(y_alpha, euler_pred),
        "one_step_euler_rmse": rmse_np(y_alpha, euler_pred),
        "router_entropy": float(np.mean(ent)) if ent else 0.0,
        "router_utilization": [float(v) for v in util_mean.tolist()],
    }


def sequence_start_ids(
    sample_ids: np.ndarray,
    re_values: np.ndarray,
    next_idx: np.ndarray,
    steps: int,
) -> np.ndarray:
    sample_set = set(int(i) for i in sample_ids.tolist())
    starts = []
    for idx in sample_ids.tolist():
        cur = int(idx)
        ok = True
        re = int(re_values[cur])
        for _ in range(steps):
            nxt = int(next_idx[cur])
            if nxt < 0 or nxt not in sample_set or int(re_values[nxt]) != re:
                ok = False
                break
            cur = nxt
        if ok:
            starts.append(int(idx))
    return np.asarray(starts, dtype=np.int64)


def rollout_loss_torch(
    model: DualHeadMoEROM,
    start_ids: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> torch.Tensor:
    if start_ids.numel() == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    a_cur = arrays_t["a"][start_ids]
    current = start_ids
    losses = []
    for _ in range(args.train_rollout_steps):
        b = arrays_t["b"][current]
        re = arrays_t["re"][current]
        ph = arrays_t["phase"][current]
        rhs_g = galerkin_rhs_torch(a_cur, b, re, gal)
        x = make_features_torch(a_cur, b, rhs_g, re, ph, args.phase_harmonics)
        x = (x - scalers_t["x_mean"]) / scalers_t["x_scale"]
        _, corr_std, _ = model(x)
        corr = corr_std * scalers_t["res_scale"] + scalers_t["res_mean"]
        rhs = rhs_g + corr
        nxt = arrays_t["next_idx"][current]
        dt = arrays_t["time"][nxt].unsqueeze(1) - arrays_t["time"][current].unsqueeze(1)
        a_cur = a_cur + dt * rhs
        target = arrays_t["a"][nxt]
        losses.append(F.mse_loss((a_cur - target) / scalers_t["alpha_scale"], torch.zeros_like(a_cur)))
        current = nxt
    return torch.stack(losses).mean()


def rollout_eval_np(
    model: DualHeadMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    test_re: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    idx = np.where(arrays["re"] == test_re)[0]
    idx = idx[np.argsort(arrays["time"][idx])]
    valid_sample_set = set(arrays["sample_ids"].tolist())
    rel_errors: List[float] = []
    stride = max(1, args.rollout_steps)
    model.eval()
    for start_pos in range(1, len(idx) - args.rollout_steps - 1, stride):
        start = int(idx[start_pos])
        if start not in valid_sample_set:
            continue
        a_cur = arrays["a"][start].copy()
        pred = []
        cur = start
        ok = True
        for _ in range(args.rollout_steps):
            nxt = int(arrays["next_idx"][cur])
            if nxt < 0 or int(arrays["re"][nxt]) != test_re:
                ok = False
                break
            rhs_g = galerkin_rhs_by_re(
                tensors,
                a_cur[None, :],
                arrays["b"][cur : cur + 1],
                arrays["re"][cur : cur + 1],
                args.r_u,
                args.r_p,
            )[0]
            x = make_features_np(
                a_cur[None, :],
                arrays["b"][cur : cur + 1],
                rhs_g[None, :],
                arrays["re"][cur : cur + 1],
                arrays["phase"][cur : cur + 1],
                args.phase_harmonics,
            )
            xt = torch.tensor(scalers["x"].transform(x), dtype=torch.float32, device=device)
            with torch.no_grad():
                _, corr_std, _ = model(xt)
            corr = (
                corr_std.detach().cpu().numpy()[0] * scalers["residual"].scale
                + scalers["residual"].mean
            )
            dt = float(arrays["time"][nxt] - arrays["time"][cur])
            a_cur = a_cur + dt * (rhs_g + corr)
            if not np.all(np.isfinite(a_cur)):
                ok = False
                break
            pred.append(a_cur.copy())
            cur = nxt
        if ok and len(pred) == args.rollout_steps:
            true = arrays["a"][idx[start_pos + 1 : start_pos + args.rollout_steps + 1]]
            rel_errors.append(relative_l2_np(true, np.asarray(pred)))
    if not rel_errors:
        return {"relative_l2_mean": float("nan"), "relative_l2_median": float("nan"), "num_windows": 0}
    return {
        "relative_l2_mean": float(np.mean(rel_errors)),
        "relative_l2_median": float(np.median(rel_errors)),
        "num_windows": int(len(rel_errors)),
    }


def build_arrays(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    pod_dir = args.data_root / "Global_POD_Unweighted"
    idx = load_snapshot_index(pod_dir / "pod_snapshot_index.csv")
    vel = np.load(pod_dir / "global_velocity_pod.npz")
    pre = np.load(pod_dir / "global_pressure_pod.npz")
    tensors = np.load(args.tensor_path)

    a = vel["coeff_uv"][:, : args.r_u].astype(np.float32)
    b = pre["coeff_p"][:, : args.r_p].astype(np.float32)
    re = idx["Re"]
    phase = idx["phase"]
    time_arr = idx["time"]
    rhs_g = galerkin_rhs_by_re(tensors, a, b, re, args.r_u, args.r_p)
    adot, valid_deriv = centered_time_derivative(a, re, time_arr)
    nxt = next_index_by_sequence(re, time_arr)

    sample_ids = valid_deriv[nxt[valid_deriv] >= 0]
    sample_ids = sample_ids[re[nxt[sample_ids]] == re[sample_ids]]
    a_next = a[nxt]
    dt_next = np.zeros_like(time_arr, dtype=np.float32)
    ok = nxt >= 0
    dt_next[ok] = time_arr[nxt[ok]] - time_arr[ok]
    x = make_features_np(a, b, rhs_g, re, phase, args.phase_harmonics)
    residual = (adot - rhs_g).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    phi = vel["phi_uv"][: args.r_u].astype(np.float32)
    if args.recon_dim > 0 and args.recon_dim < phi.shape[1]:
        recon_cols = np.sort(rng.choice(phi.shape[1], size=args.recon_dim, replace=False))
        phi_recon = phi[:, recon_cols].copy()
    else:
        recon_cols = np.arange(phi.shape[1])
        phi_recon = phi

    arrays = {
        "a": a,
        "b": b,
        "rhs_g": rhs_g,
        "adot": adot,
        "residual": residual,
        "x": x,
        "a_next": a_next.astype(np.float32),
        "re": re,
        "phase": phase,
        "time": time_arr,
        "next_idx": nxt,
        "dt_next": dt_next,
        "sample_ids": sample_ids,
        "phi_recon": phi_recon.astype(np.float32),
    }
    meta = {
        "pod_energy": {
            f"velocity_first_{args.r_u}": float(vel["cumulative_energy_uv"][args.r_u - 1]),
            f"pressure_first_{args.r_p}": float(pre["cumulative_energy_p"][args.r_p - 1]),
        },
        "recon_columns": int(len(recon_cols)),
        "total_snapshots": int(a.shape[0]),
        "valid_samples": int(len(sample_ids)),
    }
    return arrays, meta


def train_one_split(
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, object],
    tensors: np.lib.npyio.NpzFile,
    test_re: int,
    device: torch.device,
) -> Dict[str, object]:
    sample_ids = arrays["sample_ids"]
    train_all = sample_ids[arrays["re"][sample_ids] != test_re]
    test_ids = sample_ids[arrays["re"][sample_ids] == test_re]

    train_ids: List[int] = []
    val_ids: List[int] = []
    for re in sorted(np.unique(arrays["re"][train_all]).tolist()):
        ids = train_all[arrays["re"][train_all] == re]
        ids = ids[np.argsort(arrays["time"][ids])]
        val_count = max(10, int(round(0.12 * len(ids))))
        val_ids.extend(ids[-val_count:].tolist())
        train_ids.extend(ids[:-val_count].tolist())
    train_ids = np.asarray(train_ids, dtype=np.int64)
    val_ids = np.asarray(val_ids, dtype=np.int64)

    scalers = {
        "x": Standardizer.fit(arrays["x"][train_ids]),
        "residual": Standardizer.fit(arrays["residual"][train_ids]),
        "alpha_next": Standardizer.fit(arrays["a_next"][train_ids]),
    }

    x_train = torch.tensor(scalers["x"].transform(arrays["x"][train_ids]), device=device)
    res_train = torch.tensor(
        scalers["residual"].transform(arrays["residual"][train_ids]), device=device
    )
    alpha_train = torch.tensor(
        scalers["alpha_next"].transform(arrays["a_next"][train_ids]), device=device
    )
    current_a_train = torch.tensor(arrays["a"][train_ids], device=device)
    rhs_g_train = torch.tensor(arrays["rhs_g"][train_ids], device=device)
    dt_train = torch.tensor(arrays["dt_next"][train_ids][:, None], device=device)
    train_ids_t = torch.tensor(train_ids, dtype=torch.long, device=device)
    phi_recon = torch.tensor(arrays["phi_recon"], device=device)

    arrays_t = {
        "a": torch.tensor(arrays["a"], dtype=torch.float32, device=device),
        "b": torch.tensor(arrays["b"], dtype=torch.float32, device=device),
        "re": torch.tensor(arrays["re"], dtype=torch.long, device=device),
        "phase": torch.tensor(arrays["phase"], dtype=torch.float32, device=device),
        "time": torch.tensor(arrays["time"], dtype=torch.float32, device=device),
        "next_idx": torch.tensor(arrays["next_idx"], dtype=torch.long, device=device),
    }
    scalers_t = {
        "x_mean": torch.tensor(scalers["x"].mean, dtype=torch.float32, device=device),
        "x_scale": torch.tensor(scalers["x"].scale, dtype=torch.float32, device=device),
        "res_mean": torch.tensor(scalers["residual"].mean, dtype=torch.float32, device=device),
        "res_scale": torch.tensor(scalers["residual"].scale, dtype=torch.float32, device=device),
        "alpha_scale": torch.tensor(scalers["alpha_next"].scale, dtype=torch.float32, device=device),
    }
    gal = build_galerkin_torch(tensors, np.unique(arrays["re"]), args.r_u, args.r_p, device)

    model = DualHeadMoEROM(
        in_dim=arrays["x"].shape[1],
        out_dim=args.r_u,
        hidden_dim=args.hidden_dim,
        expert_hidden=args.expert_hidden,
        num_blocks=args.num_blocks,
        num_experts=args.num_experts,
        top_k=args.top_k,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(20, args.epochs))

    train_roll_starts = sequence_start_ids(train_ids, arrays["re"], arrays["next_idx"], args.train_rollout_steps)
    train_roll_starts_t = torch.tensor(train_roll_starts, dtype=torch.long, device=device)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    history = []
    num_train = len(train_ids)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(num_train, device=device)
        batch_losses = []
        for start in range(0, num_train, args.batch_size):
            bi = perm[start : start + args.batch_size]
            x = x_train[bi]
            alpha_target = alpha_train[bi]
            res_target = res_train[bi]
            alpha_pred_std, corr_std, gates = model(x)
            coeff_loss = F.mse_loss(alpha_pred_std, alpha_target)
            dyn_loss = F.mse_loss(corr_std, res_target)

            corr_phy = corr_std * scalers_t["res_scale"] + scalers_t["res_mean"]
            rhs_pred = rhs_g_train[bi] + corr_phy
            alpha_euler = current_a_train[bi] + dt_train[bi] * rhs_pred
            alpha_head_phy = (
                alpha_pred_std * scalers_t["alpha_scale"]
                + torch.tensor(scalers["alpha_next"].mean, dtype=torch.float32, device=device)
            )
            consistency_loss = F.mse_loss(
                (alpha_head_phy - alpha_euler) / scalers_t["alpha_scale"],
                torch.zeros_like(alpha_head_phy),
            )

            next_train_ids = arrays_t["next_idx"][train_ids_t[bi]]
            recon_delta = (alpha_head_phy - arrays_t["a"][next_train_ids]) @ phi_recon
            recon_true = arrays_t["a"][next_train_ids] @ phi_recon
            recon_loss = torch.mean(recon_delta**2) / (torch.mean(recon_true**2) + EPS)

            balance_loss, entropy_loss, router_info = router_regularization(gates)

            loss = (
                args.lambda_coeff * coeff_loss
                + args.lambda_dyn * dyn_loss
                + args.lambda_recon * recon_loss
                + args.lambda_consistency * consistency_loss
                + args.lambda_router_balance * balance_loss
                + args.lambda_router_entropy * entropy_loss
            )

            if args.lambda_rollout > 0 and train_roll_starts_t.numel() > 0:
                count = min(args.rollout_batch, int(train_roll_starts_t.numel()))
                rb = train_roll_starts_t[torch.randint(0, train_roll_starts_t.numel(), (count,), device=device)]
                roll_loss = rollout_loss_torch(model, rb, arrays_t, scalers_t, gal, args)
                loss = loss + args.lambda_rollout * roll_loss
            else:
                roll_loss = torch.tensor(0.0, device=device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "coeff": float(coeff_loss.detach().cpu()),
                    "dyn": float(dyn_loss.detach().cpu()),
                    "recon": float(recon_loss.detach().cpu()),
                    "rollout": float(roll_loss.detach().cpu()),
                }
            )
        sched.step()

        if epoch == 1 or epoch % 10 == 0:
            val = evaluate_model(model, arrays, scalers, val_ids, args, device)
            val_score = val["rhs_relative_l2"] + val["alpha_head_relative_l2"]
            mean_loss = float(np.mean([b["loss"] for b in batch_losses]))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "val_score": float(val_score),
                    "val_rhs_relative_l2": float(val["rhs_relative_l2"]),
                    "val_alpha_relative_l2": float(val["alpha_head_relative_l2"]),
                }
            )
            if val_score < best_val:
                best_val = float(val_score)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate_model(model, arrays, scalers, train_ids, args, device)
    val_metrics = evaluate_model(model, arrays, scalers, val_ids, args, device)
    test_metrics = evaluate_model(model, arrays, scalers, test_ids, args, device)

    base_rhs = arrays["rhs_g"][test_ids]
    true_rhs = arrays["adot"][test_ids]
    base_euler = arrays["a"][test_ids] + arrays["dt_next"][test_ids][:, None] * base_rhs
    base_metrics = {
        "rhs_relative_l2": relative_l2_np(true_rhs, base_rhs),
        "rhs_rmse": rmse_np(true_rhs, base_rhs),
        "rhs_centered_r2": centered_r2_np(true_rhs, base_rhs),
        "one_step_euler_relative_l2": relative_l2_np(arrays["a_next"][test_ids], base_euler),
        "one_step_euler_rmse": rmse_np(arrays["a_next"][test_ids], base_euler),
    }
    rollout = rollout_eval_np(model, arrays, scalers, tensors, test_re, args, device)
    return {
        "test_Re": int(test_re),
        "train_Re": sorted(set(int(v) for v in arrays["re"][train_ids].tolist())),
        "num_train": int(len(train_ids)),
        "num_val": int(len(val_ids)),
        "num_test": int(len(test_ids)),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val),
        "baseline_galerkin": base_metrics,
        "deep_moe": test_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "rollout_teacher_forced_pressure": rollout,
        "improvement_percent_vs_galerkin_rhs": float(
            100.0 * (1.0 - test_metrics["rhs_relative_l2"] / (base_metrics["rhs_relative_l2"] + EPS))
        ),
        "improvement_percent_vs_galerkin_one_step": float(
            100.0
            * (
                1.0
                - test_metrics["one_step_euler_relative_l2"]
                / (base_metrics["one_step_euler_relative_l2"] + EPS)
            )
        ),
        "history_tail": history[-12:],
    }


def write_summary_md(path: Path, result: Dict[str, object]) -> None:
    settings = result["settings"]
    lines = [
        "# Deep MoE-ROM v2 Summary",
        "",
        "## Architecture",
        "",
        (
            f"PhysicalContextEncoder + {settings['num_blocks']} Shared-Routed MoE blocks, "
            f"hidden_dim={settings['hidden_dim']}, experts={settings['num_experts']}, "
            f"top_k={settings['top_k']}, expert_hidden={settings['expert_hidden']}."
        ),
        "",
        "Dual heads: `alpha_next_head` and `rhs_correction_head`.",
        "",
        "Losses: coefficient, sampled reconstruction, dynamic residual, short rollout, "
        "alpha/RHS consistency, router load-balance and entropy.",
        "",
        "## Metrics",
        "",
        "| Test Re | Model | RHS relative L2 | one-step relative L2 | alpha-head relative L2 | rollout mean L2 | RHS improvement |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in result["results"]:
        base = item["baseline_galerkin"]
        deep = item["deep_moe"]
        roll = item["rollout_teacher_forced_pressure"]
        lines.append(
            f"| {item['test_Re']} | Galerkin only | {base['rhs_relative_l2']:.6g} | "
            f"{base['one_step_euler_relative_l2']:.6g} | - | - | 0% |"
        )
        lines.append(
            f"| {item['test_Re']} | Deep shared-routed MoE | {deep['rhs_relative_l2']:.6g} | "
            f"{deep['one_step_euler_relative_l2']:.6g} | {deep['alpha_head_relative_l2']:.6g} | "
            f"{roll['relative_l2_mean']:.6g} | {item['improvement_percent_vs_galerkin_rhs']:.6g}% |"
        )
    lines.append("")
    lines.append(f"Runtime: {result['runtime_seconds']:.2f} s.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    started = time.time()
    arrays, meta = build_arrays(args)
    tensors = np.load(args.tensor_path)

    results = []
    for test_re in args.test_res:
        results.append(train_one_split(args, arrays, meta, tensors, int(test_re), device))

    settings = {
        "experiment_name": args.experiment_name,
        "r_u": args.r_u,
        "r_p": args.r_p,
        "phase_harmonics": args.phase_harmonics,
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "expert_hidden": args.expert_hidden,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "train_rollout_steps": args.train_rollout_steps,
        "rollout_steps": args.rollout_steps,
        "recon_dim": args.recon_dim,
        "loss_weights": {
            "coeff": args.lambda_coeff,
            "dyn": args.lambda_dyn,
            "recon": args.lambda_recon,
            "rollout": args.lambda_rollout,
            "consistency": args.lambda_consistency,
            "router_balance": args.lambda_router_balance,
            "router_entropy": args.lambda_router_entropy,
        },
        "device": str(device),
        "torch_version": torch.__version__,
    }
    out = {
        "scheme": "deep_physical_context_shared_routed_moe_dual_head",
        "data_root": str(args.data_root),
        "tensor_path": str(args.tensor_path),
        "settings": settings,
        "data_meta": meta,
        "results": results,
        "runtime_seconds": float(time.time() - started),
    }
    json_path = args.output_dir / f"{args.experiment_name}_metrics.json"
    md_path = args.output_dir / f"{args.experiment_name}_summary.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_summary_md(md_path, out)
    print(json.dumps(out, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
