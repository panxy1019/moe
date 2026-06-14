#!/usr/bin/env python3
"""Deep semi-intrusive MoE-ROM v5 experiment with RK4 rollout and expert routing diagnostics.

The model extends V4 with RK4 integration, deeper MoE capacity tests, and expert diagnostics:

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
        default=Path("/root/moe/V4/data"),
    )
    parser.add_argument(
        "--tensor-path",
        type=Path,
        default=Path(
            "/root/moe/V4/data/"
            "semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/root/moe/V5/results"))
    parser.add_argument("--experiment-name", default="deep_moe_v5")
    parser.add_argument("--r-u", type=int, default=16)
    parser.add_argument("--r-p", type=int, default=16)
    parser.add_argument("--test-re-indices", type=int, nargs="+", default=[12, 29])
    parser.add_argument("--phase-harmonics", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=90)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--train-rollout-steps", type=int, default=16)
    parser.add_argument("--rollout-batch", type=int, default=24)
    parser.add_argument("--recon-dim", type=int, default=4096)
    parser.add_argument("--lambda-coeff", type=float, default=1.0)
    parser.add_argument("--lambda-dyn", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=0.08)
    parser.add_argument("--lambda-rollout", type=float, default=0.12)
    parser.add_argument("--lambda-consistency", type=float, default=0.15)
    parser.add_argument("--lambda-router-balance", type=float, default=0.02)
    parser.add_argument("--lambda-router-entropy", type=float, default=0.002)
    parser.add_argument("--lambda-router-smooth", type=float, default=0.05)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--curriculum-steps", default="2,4,8,16")
    parser.add_argument("--min-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--integrator", choices=["euler", "rk4"], default="euler")
    parser.add_argument("--analysis-bins", type=int, default=4)
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
        "Re": np.asarray([float(r["Re"]) for r in rows], dtype=np.float32),
        "Re_label": np.asarray([r["Re_label"] for r in rows]),
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


def prev_index_by_sequence(re_values: np.ndarray, times: np.ndarray) -> np.ndarray:
    prv = np.full(re_values.shape[0], -1, dtype=np.int64)
    for re in sorted(np.unique(re_values).tolist()):
        idx = np.where(re_values == re)[0]
        idx = idx[np.argsort(times[idx])]
        prv[idx[1:]] = idx[:-1]
    return prv


def galerkin_rhs_by_label(
    tensors: np.lib.npyio.NpzFile,
    a: np.ndarray,
    b: np.ndarray,
    label_ids: np.ndarray,
    labels: np.ndarray,
    r_u: int,
    r_p: int,
) -> np.ndarray:
    rhs = np.empty((a.shape[0], r_u), dtype=np.float32)
    H = tensors["H"][:r_u, :r_u, :r_u].astype(np.float32)
    P = tensors["P"][:r_u, :r_p].astype(np.float32)
    for label_id in sorted(np.unique(label_ids).tolist()):
        row = np.where(label_ids == label_id)[0]
        prefix = str(labels[int(label_id)])
        c = tensors[f"{prefix}_c"][:r_u].astype(np.float32)
        A = tensors[f"{prefix}_A"][:r_u, :r_u].astype(np.float32)
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


def history_index_matrix(
    sample_ids: np.ndarray,
    prev_idx: np.ndarray,
    history_len: int,
) -> np.ndarray:
    hist = np.full((len(sample_ids), max(history_len, 1)), -1, dtype=np.int64)
    for row, sid in enumerate(sample_ids.tolist()):
        cur = int(sid)
        for h in range(max(history_len, 1)):
            if cur < 0:
                break
            hist[row, h] = cur
            cur = int(prev_idx[cur])
    return hist


def make_history_features_np(
    base_x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    rhs: np.ndarray,
    hist_idx: np.ndarray,
) -> np.ndarray:
    current = hist_idx[:, 0]
    cols = [base_x[current]]
    for h in range(1, hist_idx.shape[1]):
        idx = hist_idx[:, h]
        valid = idx >= 0
        a_hist = np.zeros((hist_idx.shape[0], a.shape[1]), dtype=np.float32)
        b_hist = np.zeros((hist_idx.shape[0], b.shape[1]), dtype=np.float32)
        rhs_hist = np.zeros((hist_idx.shape[0], rhs.shape[1]), dtype=np.float32)
        da = np.zeros_like(a_hist)
        db = np.zeros_like(b_hist)
        drhs = np.zeros_like(rhs_hist)
        if np.any(valid):
            a_hist[valid] = a[idx[valid]]
            b_hist[valid] = b[idx[valid]]
            rhs_hist[valid] = rhs[idx[valid]]
            da[valid] = a[current[valid]] - a[idx[valid]]
            db[valid] = b[current[valid]] - b[idx[valid]]
            drhs[valid] = rhs[current[valid]] - rhs[idx[valid]]
        cols.extend([a_hist, b_hist, rhs_hist, da, db, drhs])
    return np.hstack(cols).astype(np.float32)


def make_history_features_from_current_np(
    base_current: np.ndarray,
    a_current: np.ndarray,
    b_current: np.ndarray,
    rhs_current: np.ndarray,
    hist_idx: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    cols = [base_current]
    for h in range(1, hist_idx.shape[1]):
        idx = hist_idx[:, h]
        valid = idx >= 0
        a_hist = np.zeros((hist_idx.shape[0], a.shape[1]), dtype=np.float32)
        b_hist = np.zeros((hist_idx.shape[0], b.shape[1]), dtype=np.float32)
        rhs_hist = np.zeros((hist_idx.shape[0], rhs.shape[1]), dtype=np.float32)
        da = np.zeros_like(a_hist)
        db = np.zeros_like(b_hist)
        drhs = np.zeros_like(rhs_hist)
        if np.any(valid):
            a_hist[valid] = a[idx[valid]]
            b_hist[valid] = b[idx[valid]]
            rhs_hist[valid] = rhs[idx[valid]]
            da[valid] = a_current[valid] - a_hist[valid]
            db[valid] = b_current[valid] - b_hist[valid]
            drhs[valid] = rhs_current[valid] - rhs_hist[valid]
        cols.extend([a_hist, b_hist, rhs_hist, da, db, drhs])
    return np.hstack(cols).astype(np.float32)


def shift_history_for_current_np(hist_row: np.ndarray, current_id: int) -> np.ndarray:
    shifted = hist_row.copy()
    if shifted.shape[0] > 1:
        shifted[1:] = shifted[:-1]
    shifted[0] = current_id
    return shifted


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


def make_history_features_torch(
    base_current: torch.Tensor,
    a_current: torch.Tensor,
    b_current: torch.Tensor,
    rhs_current: torch.Tensor,
    hist_ids: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
) -> torch.Tensor:
    cols = [base_current]
    for h in range(1, hist_ids.shape[1]):
        idx = hist_ids[:, h]
        valid = idx >= 0
        safe_idx = torch.clamp(idx, min=0)
        a_hist = torch.where(valid[:, None], arrays_t["a"][safe_idx], torch.zeros_like(a_current))
        b_hist = torch.where(valid[:, None], arrays_t["b"][safe_idx], torch.zeros_like(b_current))
        rhs_hist = torch.where(valid[:, None], arrays_t["rhs_g"][safe_idx], torch.zeros_like(rhs_current))
        da = torch.where(valid[:, None], a_current - a_hist, torch.zeros_like(a_current))
        db = torch.where(valid[:, None], b_current - b_hist, torch.zeros_like(b_current))
        drhs = torch.where(valid[:, None], rhs_current - rhs_hist, torch.zeros_like(rhs_current))
        cols.extend([a_hist, b_hist, rhs_hist, da, db, drhs])
    return torch.cat(cols, dim=1)


def shifted_history_ids_torch(current: torch.Tensor, arrays_t: Dict[str, torch.Tensor]) -> torch.Tensor:
    hist = arrays_t["hist_idx"][current].clone()
    if hist.shape[1] > 1:
        hist[:, 1:] = hist[:, :-1].clone()
    hist[:, 0] = current
    return hist


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


def gate_smoothness_loss(
    gates: List[torch.Tensor],
    prev_gates: List[torch.Tensor],
) -> torch.Tensor:
    if not gates or not prev_gates:
        return torch.tensor(0.0, device=gates[0].device if gates else "cpu")
    losses = []
    for gate, prev_gate in zip(gates, prev_gates):
        losses.append(F.mse_loss(gate, prev_gate))
    return torch.stack(losses).mean()


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
    labels: Iterable[str],
    r_u: int,
    r_p: int,
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    H = torch.tensor(tensors["H"][:r_u, :r_u, :r_u], dtype=torch.float32, device=device)
    P = torch.tensor(tensors["P"][:r_u, :r_p], dtype=torch.float32, device=device)
    for label_id, label in enumerate(labels):
        prefix = str(label)
        out[label_id] = {
            "c": torch.tensor(tensors[f"{prefix}_c"][:r_u], dtype=torch.float32, device=device),
            "A": torch.tensor(
                tensors[f"{prefix}_A"][:r_u, :r_u], dtype=torch.float32, device=device
            ),
            "H": H,
            "P": P,
        }
    return out


def galerkin_rhs_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    label_ids: torch.Tensor,
    gal: Dict[int, Dict[str, torch.Tensor]],
) -> torch.Tensor:
    rhs = torch.empty_like(a)
    for label_id in torch.unique(label_ids).detach().cpu().tolist():
        label_int = int(label_id)
        mask = label_ids == label_int
        ar = a[mask]
        br = b[mask]
        item = gal[label_int]
        rhs[mask] = (
            item["c"].unsqueeze(0)
            + ar @ item["A"].T
            + torch.einsum("ijk,bj,bk->bi", item["H"], ar, ar)
            + br @ item["P"].T
        )
    return rhs


def parse_curriculum_steps(text: str) -> List[int]:
    steps = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not steps:
        return [1]
    return steps


def curriculum_step_for_epoch(args: argparse.Namespace, epoch: int) -> int:
    steps = parse_curriculum_steps(args.curriculum_steps)
    stage_len = max(1, args.epochs // len(steps))
    stage = min(len(steps) - 1, (epoch - 1) // stage_len)
    return int(steps[stage])


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
    prev_ids = arrays["prev_idx"][sample_ids]
    smooth_mask = (prev_ids >= 0) & (
        arrays["label_id"][prev_ids] == arrays["label_id"][sample_ids]
    )
    gate_smooth = float("nan")
    if np.any(smooth_mask):
        px = torch.tensor(
            scalers["x"].transform(arrays["x"][prev_ids[smooth_mask]]),
            dtype=torch.float32,
            device=device,
        )
        cx = torch.tensor(
            scalers["x"].transform(arrays["x"][sample_ids[smooth_mask]]),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            _, _, cg = model(cx)
            _, _, pg = model(px)
            gate_smooth = float(gate_smoothness_loss(cg, pg).detach().cpu())

    return {
        "rhs_relative_l2": relative_l2_np(y_rhs, rhs_pred),
        "rhs_rmse": rmse_np(y_rhs, rhs_pred),
        "rhs_centered_r2": centered_r2_np(y_rhs, rhs_pred),
        "alpha_head_relative_l2": relative_l2_np(y_alpha, alpha_pred),
        "alpha_head_rmse": rmse_np(y_alpha, alpha_pred),
        "one_step_euler_relative_l2": relative_l2_np(y_alpha, euler_pred),
        "one_step_euler_rmse": rmse_np(y_alpha, euler_pred),
        "router_entropy": float(np.mean(ent)) if ent else 0.0,
        "router_temporal_smooth_mse": gate_smooth,
        "router_utilization": [float(v) for v in util_mean.tolist()],
    }


def routing_analysis(
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
        _, _, gates = model(x)
    if not gates:
        return {}
    gate_np = np.mean(np.stack([g.detach().cpu().numpy() for g in gates], axis=0), axis=0)
    top1 = np.argmax(gate_np, axis=1)
    top1_counts = np.bincount(top1, minlength=gate_np.shape[1]).astype(float)
    top1_frac = top1_counts / max(1, len(top1))
    topk_sets: Dict[str, int] = {}
    for row in gate_np:
        active = np.where(row > 1.0e-8)[0]
        key = ",".join(str(int(v)) for v in active.tolist())
        topk_sets[key] = topk_sets.get(key, 0) + 1

    phase = arrays["phase"][sample_ids]
    phase_bins = np.linspace(0.0, 1.0, args.analysis_bins + 1)
    by_phase = []
    for i in range(args.analysis_bins):
        if i == args.analysis_bins - 1:
            mask = (phase >= phase_bins[i]) & (phase <= phase_bins[i + 1])
        else:
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
        if np.any(mask):
            load = gate_np[mask].mean(axis=0)
            ent = -np.sum(gate_np[mask] * np.log(gate_np[mask] + EPS), axis=1).mean()
            top = np.bincount(top1[mask], minlength=gate_np.shape[1]).astype(float)
            top = top / max(1, int(np.sum(mask)))
        else:
            load = np.zeros(gate_np.shape[1], dtype=float)
            ent = float("nan")
            top = np.zeros(gate_np.shape[1], dtype=float)
        by_phase.append(
            {
                "phase_range": [float(phase_bins[i]), float(phase_bins[i + 1])],
                "num_samples": int(np.sum(mask)),
                "mean_load": [float(v) for v in load.tolist()],
                "top1_fraction": [float(v) for v in top.tolist()],
                "entropy": float(ent),
            }
        )

    load = gate_np.mean(axis=0)
    entropy = -np.sum(gate_np * np.log(gate_np + EPS), axis=1)
    return {
        "num_samples": int(len(sample_ids)),
        "num_experts": int(gate_np.shape[1]),
        "mean_load": [float(v) for v in load.tolist()],
        "load_cv": float(np.std(load) / (np.mean(load) + EPS)),
        "dead_experts_threshold_1pct": int(np.sum(load < 0.01)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_std": float(np.std(entropy)),
        "top1_fraction": [float(v) for v in top1_frac.tolist()],
        "topk_set_counts": {k: int(v) for k, v in sorted(topk_sets.items())},
        "by_phase_bin": by_phase,
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
    steps: int,
) -> torch.Tensor:
    if start_ids.numel() == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    a_cur = arrays_t["a"][start_ids]
    current = start_ids
    losses = []
    for _ in range(steps):
        nxt = arrays_t["next_idx"][current]
        dt = arrays_t["time"][nxt].unsqueeze(1) - arrays_t["time"][current].unsqueeze(1)
        a_cur = integrate_step_torch(model, a_cur, current, dt, arrays_t, scalers_t, gal, args)
        target = arrays_t["a"][nxt]
        losses.append(F.mse_loss((a_cur - target) / scalers_t["alpha_scale"], torch.zeros_like(a_cur)))
        current = nxt
    return torch.stack(losses).mean()


def model_rhs_torch(
    model: DualHeadMoEROM,
    a_state: torch.Tensor,
    current: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> torch.Tensor:
    b = arrays_t["b"][current]
    re = arrays_t["re"][current]
    label_id = arrays_t["label_id"][current]
    ph = arrays_t["phase"][current]
    rhs_g = galerkin_rhs_torch(a_state, b, label_id, gal)
    base_x = make_features_torch(a_state, b, rhs_g, re, ph, args.phase_harmonics)
    hist_ids = shifted_history_ids_torch(current, arrays_t)
    x = make_history_features_torch(base_x, a_state, b, rhs_g, hist_ids, arrays_t)
    x = (x - scalers_t["x_mean"]) / scalers_t["x_scale"]
    _, corr_std, _ = model(x)
    corr = corr_std * scalers_t["res_scale"] + scalers_t["res_mean"]
    return rhs_g + corr


def integrate_step_torch(
    model: DualHeadMoEROM,
    a_state: torch.Tensor,
    current: torch.Tensor,
    dt: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.integrator == "rk4":
        k1 = model_rhs_torch(model, a_state, current, arrays_t, scalers_t, gal, args)
        k2 = model_rhs_torch(model, a_state + 0.5 * dt * k1, current, arrays_t, scalers_t, gal, args)
        k3 = model_rhs_torch(model, a_state + 0.5 * dt * k2, current, arrays_t, scalers_t, gal, args)
        k4 = model_rhs_torch(model, a_state + dt * k3, current, arrays_t, scalers_t, gal, args)
        return a_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    rhs = model_rhs_torch(model, a_state, current, arrays_t, scalers_t, gal, args)
    return a_state + dt * rhs


def rollout_eval_np(
    model: DualHeadMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    test_label_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    idx = np.where(arrays["label_id"] == test_label_id)[0]
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
            if nxt < 0 or int(arrays["label_id"][nxt]) != int(test_label_id):
                ok = False
                break
            dt = float(arrays["time"][nxt] - arrays["time"][cur])
            a_cur = integrate_step_np(model, a_cur, cur, dt, arrays, scalers, tensors, args, device)
            if not np.all(np.isfinite(a_cur)):
                ok = False
                break
            pred.append(a_cur.copy())
            cur = nxt
        if ok and len(pred) == args.rollout_steps:
            true = arrays["a"][idx[start_pos + 1 : start_pos + args.rollout_steps + 1]]
            rel_errors.append(relative_l2_np(true, np.asarray(pred)))
    if not rel_errors:
        return {
            "relative_l2_mean": float("nan"),
            "relative_l2_median": float("nan"),
            "relative_l2_p90": float("nan"),
            "relative_l2_max": float("nan"),
            "num_windows": 0,
        }
    return {
        "relative_l2_mean": float(np.mean(rel_errors)),
        "relative_l2_median": float(np.median(rel_errors)),
        "relative_l2_p90": float(np.percentile(rel_errors, 90)),
        "relative_l2_max": float(np.max(rel_errors)),
        "num_windows": int(len(rel_errors)),
    }


def model_rhs_np(
    model: DualHeadMoEROM,
    a_state: np.ndarray,
    cur: int,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    rhs_g = galerkin_rhs_by_label(
        tensors,
        a_state[None, :],
        arrays["b"][cur : cur + 1],
        arrays["label_id"][cur : cur + 1],
        arrays["labels"],
        args.r_u,
        args.r_p,
    )[0]
    base_x = make_features_np(
        a_state[None, :],
        arrays["b"][cur : cur + 1],
        rhs_g[None, :],
        arrays["re"][cur : cur + 1],
        arrays["phase"][cur : cur + 1],
        args.phase_harmonics,
    )
    hist_row = shift_history_for_current_np(arrays["hist_idx"][cur], cur)[None, :]
    x = make_history_features_from_current_np(
        base_x,
        a_state[None, :],
        arrays["b"][cur : cur + 1],
        rhs_g[None, :],
        hist_row,
        arrays["a"],
        arrays["b"],
        arrays["rhs_g"],
    )
    xt = torch.tensor(scalers["x"].transform(x), dtype=torch.float32, device=device)
    with torch.no_grad():
        _, corr_std, _ = model(xt)
    corr = corr_std.detach().cpu().numpy()[0] * scalers["residual"].scale + scalers["residual"].mean
    return rhs_g + corr


def integrate_step_np(
    model: DualHeadMoEROM,
    a_state: np.ndarray,
    cur: int,
    dt: float,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    if args.integrator == "rk4":
        k1 = model_rhs_np(model, a_state, cur, arrays, scalers, tensors, args, device)
        k2 = model_rhs_np(model, a_state + 0.5 * dt * k1, cur, arrays, scalers, tensors, args, device)
        k3 = model_rhs_np(model, a_state + 0.5 * dt * k2, cur, arrays, scalers, tensors, args, device)
        k4 = model_rhs_np(model, a_state + dt * k3, cur, arrays, scalers, tensors, args, device)
        return a_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    rhs = model_rhs_np(model, a_state, cur, arrays, scalers, tensors, args, device)
    return a_state + dt * rhs


def one_step_integrator_eval_np(
    model: DualHeadMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    sample_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    preds = []
    targets = []
    for cur in sample_ids.tolist():
        nxt = int(arrays["next_idx"][cur])
        if nxt < 0 or int(arrays["label_id"][nxt]) != int(arrays["label_id"][cur]):
            continue
        dt = float(arrays["time"][nxt] - arrays["time"][cur])
        pred = integrate_step_np(
            model,
            arrays["a"][cur].copy(),
            int(cur),
            dt,
            arrays,
            scalers,
            tensors,
            args,
            device,
        )
        preds.append(pred)
        targets.append(arrays["a"][nxt])
    if not preds:
        return {"relative_l2": float("nan"), "rmse": float("nan"), "num_samples": 0}
    pred_arr = np.asarray(preds)
    target_arr = np.asarray(targets)
    return {
        "relative_l2": relative_l2_np(target_arr, pred_arr),
        "rmse": rmse_np(target_arr, pred_arr),
        "num_samples": int(len(preds)),
    }


def build_arrays(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    pod_dir = args.data_root
    idx = load_snapshot_index(pod_dir / "pod_snapshot_index.csv")
    vel = np.load(pod_dir / "global_velocity_pod_weighted_l2.npz")
    pre = np.load(pod_dir / "global_pressure_pod_weighted_l2.npz")
    tensors = np.load(args.tensor_path)

    a = vel["coeff_uv"][:, : args.r_u].astype(np.float32)
    b = pre["coeff_p"][:, : args.r_p].astype(np.float32)
    re = idx["Re"]
    labels = vel["Re_labels"].astype(str)
    label_to_id = {label: i for i, label in enumerate(labels.tolist())}
    label_id = np.asarray([label_to_id[str(label)] for label in idx["Re_label"]], dtype=np.int64)
    phase = idx["phase"]
    time_arr = idx["time"]
    rhs_g = galerkin_rhs_by_label(tensors, a, b, label_id, labels, args.r_u, args.r_p)
    adot, valid_deriv = centered_time_derivative(a, re, time_arr)
    nxt = next_index_by_sequence(re, time_arr)
    prv = prev_index_by_sequence(re, time_arr)

    sample_ids = valid_deriv[nxt[valid_deriv] >= 0]
    sample_ids = sample_ids[label_id[nxt[sample_ids]] == label_id[sample_ids]]
    hist_idx = history_index_matrix(np.arange(a.shape[0], dtype=np.int64), prv, args.history_len)
    enough_history = np.all(hist_idx[sample_ids] >= 0, axis=1)
    sample_ids = sample_ids[enough_history]
    a_next = a[nxt]
    dt_next = np.zeros_like(time_arr, dtype=np.float32)
    ok = nxt >= 0
    dt_next[ok] = time_arr[nxt[ok]] - time_arr[ok]
    base_x = make_features_np(a, b, rhs_g, re, phase, args.phase_harmonics)
    x = make_history_features_np(base_x, a, b, rhs_g, hist_idx)
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
        "base_x": base_x,
        "x": x,
        "a_next": a_next.astype(np.float32),
        "re": re,
        "label_id": label_id,
        "labels": labels,
        "phase": phase,
        "time": time_arr,
        "prev_idx": prv,
        "next_idx": nxt,
        "hist_idx": hist_idx,
        "dt_next": dt_next,
        "sample_ids": sample_ids,
        "phi_recon": phi_recon.astype(np.float32),
    }
    meta = {
        "pod_energy": {
            f"velocity_first_{args.r_u}": float(vel["cumulative_energy_uv"][args.r_u - 1]),
            f"pressure_first_{args.r_p}": float(pre["cumulative_energy_p"][args.r_p - 1]),
        },
        "num_re": int(len(labels)),
        "re_values": [float(v) for v in vel["Re_values"].tolist()],
        "re_labels": labels.tolist(),
        "recon_columns": int(len(recon_cols)),
        "total_snapshots": int(a.shape[0]),
        "valid_samples": int(len(sample_ids)),
        "history_len": int(args.history_len),
    }
    return arrays, meta


def train_one_split(
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, object],
    tensors: np.lib.npyio.NpzFile,
    test_label_id: int,
    device: torch.device,
) -> Dict[str, object]:
    sample_ids = arrays["sample_ids"]
    train_all = sample_ids[arrays["label_id"][sample_ids] != test_label_id]
    test_ids = sample_ids[arrays["label_id"][sample_ids] == test_label_id]

    train_ids: List[int] = []
    val_ids: List[int] = []
    for label_id_value in sorted(np.unique(arrays["label_id"][train_all]).tolist()):
        ids = train_all[arrays["label_id"][train_all] == label_id_value]
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
    x_all_t = torch.tensor(scalers["x"].transform(arrays["x"]), device=device)
    current_a_train = torch.tensor(arrays["a"][train_ids], device=device)
    rhs_g_train = torch.tensor(arrays["rhs_g"][train_ids], device=device)
    dt_train = torch.tensor(arrays["dt_next"][train_ids][:, None], device=device)
    train_ids_t = torch.tensor(train_ids, dtype=torch.long, device=device)
    phi_recon = torch.tensor(arrays["phi_recon"], device=device)

    arrays_t = {
        "a": torch.tensor(arrays["a"], dtype=torch.float32, device=device),
        "b": torch.tensor(arrays["b"], dtype=torch.float32, device=device),
        "rhs_g": torch.tensor(arrays["rhs_g"], dtype=torch.float32, device=device),
        "re": torch.tensor(arrays["re"], dtype=torch.float32, device=device),
        "label_id": torch.tensor(arrays["label_id"], dtype=torch.long, device=device),
        "phase": torch.tensor(arrays["phase"], dtype=torch.float32, device=device),
        "time": torch.tensor(arrays["time"], dtype=torch.float32, device=device),
        "prev_idx": torch.tensor(arrays["prev_idx"], dtype=torch.long, device=device),
        "next_idx": torch.tensor(arrays["next_idx"], dtype=torch.long, device=device),
        "hist_idx": torch.tensor(arrays["hist_idx"], dtype=torch.long, device=device),
    }
    scalers_t = {
        "x_mean": torch.tensor(scalers["x"].mean, dtype=torch.float32, device=device),
        "x_scale": torch.tensor(scalers["x"].scale, dtype=torch.float32, device=device),
        "res_mean": torch.tensor(scalers["residual"].mean, dtype=torch.float32, device=device),
        "res_scale": torch.tensor(scalers["residual"].scale, dtype=torch.float32, device=device),
        "alpha_mean": torch.tensor(scalers["alpha_next"].mean, dtype=torch.float32, device=device),
        "alpha_scale": torch.tensor(scalers["alpha_next"].scale, dtype=torch.float32, device=device),
    }
    gal = build_galerkin_torch(tensors, arrays["labels"], args.r_u, args.r_p, device)

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

    max_curriculum_steps = max(parse_curriculum_steps(args.curriculum_steps))
    train_roll_starts = sequence_start_ids(
        train_ids, arrays["label_id"], arrays["next_idx"], max_curriculum_steps
    )
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
                + scalers_t["alpha_mean"]
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
            batch_ids = train_ids_t[bi]
            prev_ids = arrays_t["prev_idx"][batch_ids]
            smooth_mask = (prev_ids >= 0) & (
                arrays_t["label_id"][prev_ids] == arrays_t["label_id"][batch_ids]
            )
            if bool(torch.any(smooth_mask)):
                _, _, prev_gates = model(x_all_t[prev_ids[smooth_mask]])
                smooth_loss = gate_smoothness_loss([g[smooth_mask] for g in gates], prev_gates)
            else:
                smooth_loss = torch.tensor(0.0, device=device)

            loss = (
                args.lambda_coeff * coeff_loss
                + args.lambda_dyn * dyn_loss
                + args.lambda_recon * recon_loss
                + args.lambda_consistency * consistency_loss
                + args.lambda_router_balance * balance_loss
                + args.lambda_router_entropy * entropy_loss
                + args.lambda_router_smooth * smooth_loss
            )

            if args.lambda_rollout > 0 and train_roll_starts_t.numel() > 0:
                current_rollout_steps = curriculum_step_for_epoch(args, epoch)
                count = min(args.rollout_batch, int(train_roll_starts_t.numel()))
                rb = train_roll_starts_t[torch.randint(0, train_roll_starts_t.numel(), (count,), device=device)]
                roll_loss = rollout_loss_torch(
                    model, rb, arrays_t, scalers_t, gal, args, current_rollout_steps
                )
                loss = loss + args.lambda_rollout * roll_loss
            else:
                current_rollout_steps = 0
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
                    "rollout_steps": int(current_rollout_steps),
                    "router_smooth": float(smooth_loss.detach().cpu()),
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
            elif epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
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
    one_step_integrator = one_step_integrator_eval_np(
        model, arrays, scalers, tensors, test_ids, args, device
    )
    rollout = rollout_eval_np(model, arrays, scalers, tensors, test_label_id, args, device)
    route_test = routing_analysis(model, arrays, scalers, test_ids, args, device)
    route_train = routing_analysis(model, arrays, scalers, train_ids, args, device)
    return {
        "test_Re": float(arrays["re"][test_ids[0]]),
        "test_Re_label": str(arrays["labels"][test_label_id]),
        "train_Re_labels": [
            str(arrays["labels"][i])
            for i in sorted(set(int(v) for v in arrays["label_id"][train_ids].tolist()))
        ],
        "num_train": int(len(train_ids)),
        "num_val": int(len(val_ids)),
        "num_test": int(len(test_ids)),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val),
        "baseline_galerkin": base_metrics,
        "deep_moe": test_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "one_step_integrator": one_step_integrator,
        "rollout_teacher_forced_pressure": rollout,
        "routing_analysis_test": route_test,
        "routing_analysis_train": route_train,
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
        "# Deep MoE-ROM v5 Summary",
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
        "alpha/RHS consistency, router load-balance, entropy and temporal smoothness.",
        "",
        "## Metrics",
        "",
        f"Integrator: `{settings['integrator']}`.",
        "",
        "| Test Re | Model | RHS relative L2 | Euler one-step L2 | Integrator one-step L2 | rollout mean L2 | load CV | entropy | dead experts | RHS improvement |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["results"]:
        base = item["baseline_galerkin"]
        deep = item["deep_moe"]
        roll = item["rollout_teacher_forced_pressure"]
        lines.append(
            f"| {item['test_Re']} | Galerkin only | {base['rhs_relative_l2']:.6g} | "
            f"{base['one_step_euler_relative_l2']:.6g} | - | - | - | - | - | 0% |"
        )
        routing = item["routing_analysis_test"]
        one = item["one_step_integrator"]
        lines.append(
            f"| {item['test_Re']} | Deep shared-routed MoE | {deep['rhs_relative_l2']:.6g} | "
            f"{deep['one_step_euler_relative_l2']:.6g} | {one['relative_l2']:.6g} | "
            f"{roll['relative_l2_mean']:.6g} | {routing['load_cv']:.6g} | "
            f"{routing['entropy_mean']:.6g} | {routing['dead_experts_threshold_1pct']} | "
            f"{item['improvement_percent_vs_galerkin_rhs']:.6g}% |"
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
    for test_label_id in args.test_re_indices:
        results.append(train_one_split(args, arrays, meta, tensors, int(test_label_id), device))

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
        "min_epochs": args.min_epochs,
        "curriculum_steps": parse_curriculum_steps(args.curriculum_steps),
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
            "router_smooth": args.lambda_router_smooth,
        },
        "history_len": args.history_len,
        "test_re_indices": args.test_re_indices,
        "integrator": args.integrator,
        "analysis_bins": args.analysis_bins,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    out = {
        "scheme": "weighted_l2_deep_moe_rom_v5_rk4_expert_diagnostics",
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
