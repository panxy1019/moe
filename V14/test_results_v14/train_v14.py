#!/usr/bin/env python3
"""Hierarchical Physics-Regime Sparse MoE-ROM v14.

V14 keeps the semi-intrusive Shared Encoder + Physics-aware Experts +
Galerkin + RK4 backbone, but replaces the flat expert pool with a physics
regime hierarchy. A shared group router first selects Low-Re / Transition /
High-Re style regimes; inside the active group, velocity and pressure branches
use Top-2 routed experts plus one group-shared expert. Training emphasizes
closed-loop multi-step dynamics rather than one-step head accuracy alone.

    [a_t, b_t, Re, phase, physical descriptors, history]
      -> shared encoder h_t
      -> group router pi^g and in-group velocity/pressure routers pi^u, pi^p
      -> group-shared expert + group-local Top-2 routed experts
      -> learned closure c_u, c_p with W a + low-rank a^T Q a + residual FFN
      -> f = Galerkin RHS + c_u
      -> RK4 velocity advance + pressure branch
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
RE_FEATURE_CENTER = 175.0
RE_FEATURE_SCALE = 125.0
RE_FEATURE_REF = 300.0


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
        default=Path("/root/moe/V8/data/Global_POD_Weighted_L2"),
    )
    parser.add_argument(
        "--tensor-path",
        type=Path,
        default=Path(
            "/root/moe/V8/data/"
            "semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_slim.npz"
        ),
    )
    parser.add_argument(
        "--pressure-surrogate-path",
        type=Path,
        default=Path(
            "/root/moe/V8/data/"
            "pressure_poisson_surrogate_tensors_allRe100_weightedL2_ru80_rp80.npz"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/root/moe/V14/test_results_v14/results")
    )
    parser.add_argument("--experiment-name", default="deep_moe_v14")
    parser.add_argument("--r-u", type=int, default=16)
    parser.add_argument("--r-p", type=int, default=16)
    parser.add_argument("--test-re-indices", type=int, nargs="+", default=[10, 59, 99])
    parser.add_argument("--phase-harmonics", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--num-experts", type=int, default=6)
    parser.add_argument("--num-shared-experts", type=int, default=1)
    parser.add_argument("--num-regime-groups", type=int, default=3)
    parser.add_argument("--experts-per-group", type=int, default=6)
    parser.add_argument("--group-top-k", type=int, default=1)
    parser.add_argument("--group-temperature", type=float, default=0.9)
    parser.add_argument("--group-gate-floor", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=1024)
    parser.add_argument("--expert-blocks", type=int, default=4)
    parser.add_argument("--quadratic-rank", type=int, default=4)
    parser.add_argument("--quadratic-scale", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gate-floor", type=float, default=0.0)
    parser.add_argument("--shared-scale", type=float, default=1.0)
    parser.add_argument("--routed-scale", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=90)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1.0e-3)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--train-rollout-steps", type=int, default=16)
    parser.add_argument("--rollout-batch", type=int, default=24)
    parser.add_argument("--rollout-every-batches", type=int, default=1)
    parser.add_argument("--recon-dim", type=int, default=4096)
    parser.add_argument("--lambda-coeff", type=float, default=1.0)
    parser.add_argument("--lambda-dyn", type=float, default=1.0)
    parser.add_argument("--lambda-pressure", type=float, default=0.55)
    parser.add_argument("--lambda-recon", type=float, default=0.08)
    parser.add_argument("--lambda-rollout", type=float, default=0.30)
    parser.add_argument("--lambda-pressure-rollout", type=float, default=0.25)
    parser.add_argument("--lambda-consistency", type=float, default=0.15)
    parser.add_argument("--lambda-router-balance", type=float, default=0.02)
    parser.add_argument("--lambda-router-entropy", type=float, default=0.002)
    parser.add_argument("--lambda-group-balance", type=float, default=0.04)
    parser.add_argument("--lambda-group-entropy", type=float, default=0.001)
    parser.add_argument("--lambda-group-supervision", type=float, default=0.03)
    parser.add_argument("--lambda-router-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-expert-diversity", type=float, default=0.01)
    parser.add_argument("--lambda-regime-router", type=float, default=0.005)
    parser.add_argument("--lambda-energy", type=float, default=0.04)
    parser.add_argument("--lambda-trajectory-consistency", type=float, default=0.10)
    parser.add_argument("--lambda-alpha-rel", type=float, default=0.20)
    parser.add_argument("--lambda-rhs-rel", type=float, default=0.20)
    parser.add_argument("--lambda-pressure-rel", type=float, default=0.60)
    parser.add_argument("--pressure-target", choices=["closure", "state"], default="closure")
    parser.add_argument("--rhs-target", choices=["full", "residual"], default="residual")
    parser.add_argument("--pressure-amplitude-weight-power", type=float, default=0.0)
    parser.add_argument("--pressure-amplitude-weight-max", type=float, default=6.0)
    parser.add_argument("--rollout-relative-mix", type=float, default=0.60)
    parser.add_argument("--relative-floor-frac", type=float, default=0.02)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--curriculum-steps", default="2,4,8,16")
    parser.add_argument("--scheduled-sampling-start", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling-end", type=float, default=0.75)
    parser.add_argument("--scheduled-sampling-warmup-frac", type=float, default=0.70)
    parser.add_argument("--min-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--integrator", choices=["euler", "rk4"], default="rk4")
    parser.add_argument("--analysis-bins", type=int, default=4)
    parser.add_argument("--eval-routing-every", type=int, default=0)
    parser.add_argument("--eval-routing-max-samples", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print-full-json", action="store_true")
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
    first_prefix = str(labels[int(label_ids[0])])
    H_key = "H" if "H" in tensors.files else f"{first_prefix}_H"
    P_key = "P" if "P" in tensors.files else f"{first_prefix}_P"
    H = tensors[H_key][:r_u, :r_u, :r_u].astype(np.float32)
    P = tensors[P_key][:r_u, :r_p].astype(np.float32)
    label_to_tensor_row = {}
    if "c_all" in tensors.files and "A_all" in tensors.files:
        computed_labels = tensors["Re_labels_computed"].astype(str)
        label_to_tensor_row = {
            str(label): i for i, label in enumerate(computed_labels.tolist())
        }
    for label_id in sorted(np.unique(label_ids).tolist()):
        row = np.where(label_ids == label_id)[0]
        prefix = str(labels[int(label_id)])
        if label_to_tensor_row:
            tensor_row = label_to_tensor_row[prefix]
            c = tensors["c_all"][tensor_row, :r_u].astype(np.float32)
            A = tensors["A_all"][tensor_row, :r_u, :r_u].astype(np.float32)
        else:
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


def pressure_surrogate_by_label(
    tensors: np.lib.npyio.NpzFile,
    a_next: np.ndarray,
    label_ids: np.ndarray,
    labels: np.ndarray,
    r_u: int,
    r_p: int,
) -> np.ndarray:
    out = np.empty((a_next.shape[0], r_p), dtype=np.float32)
    H_tilde = tensors["H_tilde"][:r_p, :r_u, :r_u].astype(np.float32)
    for label_id in sorted(np.unique(label_ids).tolist()):
        row = np.where(label_ids == label_id)[0]
        prefix = str(labels[int(label_id)])
        c_tilde = tensors[f"{prefix}_c_tilde"][:r_p].astype(np.float32)
        A_tilde = tensors[f"{prefix}_A_tilde"][:r_p, :r_u].astype(np.float32)
        ar = a_next[row]
        out[row] = (
            c_tilde[None, :]
            + ar @ A_tilde.T
            + np.einsum("pij,bi,bj->bp", H_tilde, ar, ar, optimize=True)
        )
    return out.astype(np.float32)


def make_features_np(
    a: np.ndarray,
    b: np.ndarray,
    rhs: np.ndarray,
    re_values: np.ndarray,
    phase: np.ndarray,
    harmonics: int,
) -> np.ndarray:
    re = re_values.astype(np.float32)
    re_norm = ((re - RE_FEATURE_CENTER) / RE_FEATURE_SCALE)[:, None]
    inv_re = (RE_FEATURE_REF / np.maximum(re, EPS))[:, None]
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
    a_energy = np.sum(a * a, axis=1, keepdims=True)
    b_energy = np.sum(b * b, axis=1, keepdims=True)
    total_energy = a_energy + b_energy
    low_fraction = (e_low * e_low) / (a_energy + EPS)
    high_fraction = (e_high * e_high) / (a_energy + EPS)
    pressure_velocity_ratio = b_norm / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)
    cols.extend(
        [
            a,
            b,
            rhs,
            e_low,
            e_mid,
            e_high,
            b_norm,
            rhs_norm,
            a_energy,
            b_energy,
            total_energy,
            low_fraction,
            high_fraction,
            pressure_velocity_ratio,
        ]
    )
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


def make_history_features_from_states_np(
    base_current: np.ndarray,
    a_current: np.ndarray,
    b_current: np.ndarray,
    rhs_current: np.ndarray,
    a_hist: np.ndarray,
    b_hist: np.ndarray,
    rhs_hist: np.ndarray,
) -> np.ndarray:
    cols = [base_current]
    for h in range(1, a_hist.shape[1]):
        ah = a_hist[:, h]
        bh = b_hist[:, h]
        rh = rhs_hist[:, h]
        cols.extend([ah, bh, rh, a_current - ah, b_current - bh, rhs_current - rh])
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
    cols = [
        ((re - RE_FEATURE_CENTER) / RE_FEATURE_SCALE).unsqueeze(1),
        (RE_FEATURE_REF / torch.clamp(re, min=EPS)).unsqueeze(1),
    ]
    theta = 2.0 * math.pi * phase.float()
    for k in range(1, harmonics + 1):
        cols.append(torch.sin(k * theta).unsqueeze(1))
        cols.append(torch.cos(k * theta).unsqueeze(1))
    r_u = a.shape[1]
    low = min(4, r_u)
    split = min(12, r_u)
    e_low = torch.linalg.norm(a[:, :low], dim=1, keepdim=True)
    e_mid = torch.linalg.norm(a[:, low:split], dim=1, keepdim=True)
    e_high = torch.linalg.norm(a[:, split:], dim=1, keepdim=True)
    a_norm = torch.linalg.norm(a, dim=1, keepdim=True)
    b_norm = torch.linalg.norm(b, dim=1, keepdim=True)
    rhs_norm = torch.linalg.norm(rhs, dim=1, keepdim=True)
    a_energy = torch.sum(a * a, dim=1, keepdim=True)
    b_energy = torch.sum(b * b, dim=1, keepdim=True)
    cols.extend(
        [
            a,
            b,
            rhs,
            e_low,
            e_mid,
            e_high,
            b_norm,
            rhs_norm,
            a_energy,
            b_energy,
            a_energy + b_energy,
            (e_low * e_low) / (a_energy + EPS),
            (e_high * e_high) / (a_energy + EPS),
            b_norm / (a_norm + EPS),
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


def make_history_features_from_states_torch(
    base_current: torch.Tensor,
    a_current: torch.Tensor,
    b_current: torch.Tensor,
    rhs_current: torch.Tensor,
    a_hist: torch.Tensor,
    b_hist: torch.Tensor,
    rhs_hist: torch.Tensor,
) -> torch.Tensor:
    cols = [base_current]
    for h in range(1, a_hist.shape[1]):
        ah = a_hist[:, h]
        bh = b_hist[:, h]
        rh = rhs_hist[:, h]
        cols.extend([ah, bh, rh, a_current - ah, b_current - bh, rhs_current - rh])
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def topk_router_from_logits(
    logits: torch.Tensor,
    top_k: int,
    gate_floor: float,
    temperature: float,
) -> torch.Tensor:
    probs = torch.softmax(logits / max(temperature, 1.0e-4), dim=-1)
    num_experts = probs.shape[1]
    if 0 < top_k < num_experts:
        top_val, top_idx = torch.topk(probs, top_k, dim=-1)
        gate = torch.zeros_like(probs)
        gate.scatter_(1, top_idx, top_val)
        gate = gate / (gate.sum(dim=-1, keepdim=True) + EPS)
        if top_k == 1:
            gate = gate - probs.detach() + probs
    else:
        gate = probs
    floor = max(0.0, min(gate_floor, 0.45))
    if floor > 0.0:
        gate = (1.0 - floor) * gate + floor / num_experts
        gate = gate / (gate.sum(dim=-1, keepdim=True) + EPS)
    return gate


class ExpandedFFNBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.ffn(self.norm(x))


class PhysicsAwareExpert(nn.Module):
    def __init__(
        self,
        h_dim: int,
        state_dim: int,
        out_dim: int,
        expert_hidden: int,
        num_blocks: int,
        quadratic_rank: int,
        quadratic_scale: float,
        dropout: float,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.out_dim = out_dim
        self.quadratic_rank = max(0, quadratic_rank)
        self.quadratic_scale = float(quadratic_scale)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(h_dim + state_dim),
            nn.Linear(h_dim + state_dim, h_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                ExpandedFFNBlock(h_dim, expert_hidden, dropout)
                for _ in range(max(1, num_blocks))
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, out_dim),
        )
        self.linear = nn.Linear(state_dim, out_dim, bias=False)
        if self.quadratic_rank > 0:
            self.quad_left = nn.Parameter(
                torch.empty(out_dim, self.quadratic_rank, state_dim)
            )
            self.quad_right = nn.Parameter(
                torch.empty(out_dim, self.quadratic_rank, state_dim)
            )
        else:
            self.register_parameter("quad_left", None)
            self.register_parameter("quad_right", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight, gain=0.25)
        head = self.mlp_head[-1]
        if isinstance(head, nn.Linear):
            nn.init.xavier_uniform_(head.weight, gain=0.25)
            nn.init.zeros_(head.bias)
        if self.quadratic_rank > 0:
            nn.init.normal_(self.quad_left, mean=0.0, std=0.015)
            nn.init.normal_(self.quad_right, mean=0.0, std=0.015)

    def forward(self, h: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(torch.cat([h, state], dim=1))
        for block in self.blocks:
            z = block(z)
        out = self.mlp_head(z) + self.linear(state)
        if self.quadratic_rank > 0:
            left = torch.einsum("bs,ors->bor", state, self.quad_left)
            right = torch.einsum("bs,ors->bor", state, self.quad_right)
            quad = torch.sum(left * right, dim=2)
            out = out + self.quadratic_scale * quad
        return out


class OperatorSpaceMoEROM(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        pressure_dim: int,
        hidden_dim: int,
        expert_hidden: int,
        num_blocks: int,
        num_experts: int,
        num_operator_spaces: int,
        num_regime_groups: int,
        experts_per_group: int,
        top_k: int,
        group_top_k: int,
        dropout: float,
        temperature: float,
        gate_floor: float,
        group_temperature: float,
        group_gate_floor: float,
        shared_scale: float,
        routed_scale: float,
        expert_blocks: int,
        quadratic_rank: int,
        quadratic_scale: float,
        phase_harmonics: int,
        diversity_samples: int = 32,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.pressure_dim = pressure_dim
        self.num_regime_groups = max(1, int(num_regime_groups))
        self.experts_per_group = max(1, int(experts_per_group or num_experts))
        self.shared_per_group = 1
        self.num_operator_spaces = self.num_regime_groups
        self.num_shared_experts = self.num_regime_groups * self.shared_per_group
        self.num_experts = self.num_regime_groups * (
            self.experts_per_group + self.shared_per_group
        )
        self.top_k = min(max(1, int(top_k)), self.experts_per_group)
        self.group_top_k = min(max(1, int(group_top_k)), self.num_regime_groups)
        self.temperature = temperature
        self.gate_floor = gate_floor
        self.group_temperature = group_temperature
        self.group_gate_floor = group_gate_floor
        self.shared_scale = shared_scale
        self.routed_scale = routed_scale
        self.expert_blocks = max(1, expert_blocks)
        self.quadratic_rank = max(0, quadratic_rank)
        self.quadratic_scale = quadratic_scale
        self.diversity_samples = max(0, diversity_samples)
        self.a_start = 2 + 2 * int(phase_harmonics)
        self.b_start = self.a_start + out_dim
        self.encoder = PhysicalContextEncoder(in_dim, hidden_dim, dropout)
        self.refine_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(max(0, num_blocks - 1))
            ]
        )
        router_dim = hidden_dim + in_dim
        self.group_router = nn.Sequential(
            nn.LayerNorm(router_dim),
            nn.Linear(router_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_regime_groups),
        )
        self.velocity_group_routers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(router_dim),
                    nn.Linear(router_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.experts_per_group),
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.pressure_group_routers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(router_dim),
                    nn.Linear(router_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.experts_per_group),
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.velocity_expert_groups = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PhysicsAwareExpert(
                            hidden_dim,
                            out_dim,
                            out_dim,
                            expert_hidden,
                            self.expert_blocks,
                            self.quadratic_rank,
                            self.quadratic_scale,
                            dropout,
                        )
                        for _ in range(self.experts_per_group)
                    ]
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.pressure_expert_groups = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PhysicsAwareExpert(
                            hidden_dim,
                            out_dim + pressure_dim,
                            pressure_dim,
                            expert_hidden,
                            self.expert_blocks,
                            self.quadratic_rank,
                            self.quadratic_scale,
                            dropout,
                        )
                        for _ in range(self.experts_per_group)
                    ]
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.velocity_shared_experts = nn.ModuleList(
            [
                PhysicsAwareExpert(
                    hidden_dim,
                    out_dim,
                    out_dim,
                    expert_hidden,
                    self.expert_blocks,
                    self.quadratic_rank,
                    self.quadratic_scale,
                    dropout,
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.pressure_shared_experts = nn.ModuleList(
            [
                PhysicsAwareExpert(
                    hidden_dim,
                    out_dim + pressure_dim,
                    pressure_dim,
                    expert_hidden,
                    self.expert_blocks,
                    self.quadratic_rank,
                    self.quadratic_scale,
                    dropout,
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.velocity_shared_mixer = None
        self.pressure_shared_mixer = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for block in self.refine_blocks:
            h = h + block(h)
        return h

    def _state_slices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_state = x[:, self.a_start : self.a_start + self.out_dim]
        b_state = x[:, self.b_start : self.b_start + self.pressure_dim]
        return a_state, torch.cat([a_state, b_state], dim=1)

    def _sparse_mix(
        self,
        experts: nn.ModuleList,
        h: torch.Tensor,
        state: torch.Tensor,
        gate: torch.Tensor,
        return_stack: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_dim = experts[0].out_dim if len(experts) else 0
        mixed = torch.zeros((h.shape[0], out_dim), dtype=h.dtype, device=h.device)
        for expert_id, expert in enumerate(experts):
            weights = gate[:, expert_id]
            active = weights > 0.0
            if bool(torch.any(active)):
                mixed[active] = mixed[active] + weights[active, None] * expert(
                    h[active], state[active]
                )
        if return_stack and self.diversity_samples > 0:
            n = min(int(self.diversity_samples), h.shape[0])
            stack = torch.stack([expert(h[:n], state[:n]) for expert in experts], dim=1)
        else:
            stack = torch.empty((0, len(experts), out_dim), dtype=h.dtype, device=h.device)
        return mixed, stack

    def _group_mix(
        self,
        expert_groups: nn.ModuleList,
        shared_experts: nn.ModuleList,
        local_routers: nn.ModuleList,
        h: torch.Tensor,
        state: torch.Tensor,
        router_in: torch.Tensor,
        group_gate: torch.Tensor,
        return_stack: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out_dim = shared_experts[0].out_dim
        mixed = torch.zeros((h.shape[0], out_dim), dtype=h.dtype, device=h.device)
        combined_gates = []
        stack_items = []
        denom = max(float(self.shared_scale + self.routed_scale), 1.0e-6)
        shared_part = float(self.shared_scale) / denom
        routed_part = float(self.routed_scale) / denom
        for group_id in range(self.num_regime_groups):
            logits = local_routers[group_id](router_in)
            local_gate = topk_router_from_logits(
                logits, self.top_k, self.gate_floor, self.temperature
            )
            group_weight = group_gate[:, group_id : group_id + 1]
            active = group_weight.squeeze(1) > 0.0
            if bool(torch.any(active)):
                routed, _ = self._sparse_mix(
                    expert_groups[group_id],
                    h[active],
                    state[active],
                    local_gate[active],
                    return_stack=False,
                )
                shared = shared_experts[group_id](h[active], state[active])
                group_out = shared_part * shared + routed_part * routed
                mixed[active] = mixed[active] + group_weight[active] * group_out
            combined_gates.append(
                torch.cat(
                    [
                        shared_part * group_weight,
                        routed_part * group_weight * local_gate,
                    ],
                    dim=1,
                )
            )
            if return_stack and self.diversity_samples > 0:
                n = min(int(self.diversity_samples), h.shape[0])
                stack_items.extend(
                    [expert(h[:n], state[:n]) for expert in expert_groups[group_id]]
                )
        combined_gate = torch.cat(combined_gates, dim=1)
        if return_stack and stack_items:
            stack = torch.stack(stack_items, dim=1)
        else:
            stack = torch.empty(
                (0, self.num_regime_groups * self.experts_per_group, out_dim),
                dtype=h.dtype,
                device=h.device,
            )
        return mixed, combined_gate, stack

    def group_router_outputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._encode(x)
        router_in = torch.cat([h, x], dim=1)
        logits = self.group_router(router_in)
        gate = topk_router_from_logits(
            logits, self.group_top_k, self.group_gate_floor, self.group_temperature
        )
        return gate, logits

    def forward(
        self, x: torch.Tensor, return_expert_stack: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        h = self._encode(x)
        a_state, pressure_state = self._state_slices(x)
        router_in = torch.cat([h, x], dim=1)
        group_logits = self.group_router(router_in)
        group_gate = topk_router_from_logits(
            group_logits,
            self.group_top_k,
            self.group_gate_floor,
            self.group_temperature,
        )
        rhs_op, vel_gate, velocity_stack = self._group_mix(
            self.velocity_expert_groups,
            self.velocity_shared_experts,
            self.velocity_group_routers,
            h,
            a_state,
            router_in,
            group_gate,
            return_stack=return_expert_stack,
        )
        pressure_op, pressure_gate, _ = self._group_mix(
            self.pressure_expert_groups,
            self.pressure_shared_experts,
            self.pressure_group_routers,
            h,
            pressure_state,
            router_in,
            group_gate,
            return_stack=False,
        )
        return rhs_op, pressure_op, [vel_gate, pressure_gate], velocity_stack


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


def expert_diversity_loss(expert_stack: torch.Tensor) -> torch.Tensor:
    if expert_stack.numel() == 0 or expert_stack.shape[1] <= 1:
        return torch.tensor(0.0, device=expert_stack.device)
    # Compare expert functions over the batch in operator-output space.
    expert_vectors = expert_stack.permute(1, 0, 2).reshape(expert_stack.shape[1], -1)
    expert_vectors = expert_vectors - expert_vectors.mean(dim=1, keepdim=True)
    expert_vectors = F.normalize(expert_vectors, dim=1, eps=1.0e-8)
    sim = expert_vectors @ expert_vectors.T
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return torch.mean(sim[~eye] ** 2)


def regime_router_loss(gates: List[torch.Tensor], re_values: torch.Tensor) -> torch.Tensor:
    if not gates:
        return torch.tensor(0.0, device=re_values.device)
    gate = torch.stack(gates, dim=0).mean(dim=0)
    masks = [re_values < 80.0, (re_values >= 80.0) & (re_values < 160.0), re_values >= 160.0]
    means = []
    for mask in masks:
        if bool(torch.sum(mask) >= 4):
            means.append(gate[mask].mean(dim=0))
    if len(means) <= 1:
        return torch.tensor(0.0, device=gate.device)
    sims = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            sims.append(
                F.cosine_similarity(
                    means[i].unsqueeze(0), means[j].unsqueeze(0), dim=1
                ).squeeze(0)
            )
    return torch.stack(sims).mean()


def re_regime_targets_torch(re_values: torch.Tensor, num_groups: int) -> torch.Tensor:
    targets = torch.zeros_like(re_values, dtype=torch.long)
    if num_groups <= 1:
        return targets
    if num_groups == 2:
        targets = torch.where(re_values >= 160.0, torch.ones_like(targets), targets)
        return targets
    targets = torch.where(re_values >= 80.0, torch.ones_like(targets), targets)
    high_value = torch.full_like(targets, min(num_groups - 1, 2))
    targets = torch.where(re_values >= 160.0, high_value, targets)
    return torch.clamp(targets, min=0, max=num_groups - 1)


def group_router_regularization(
    group_gate: torch.Tensor,
    group_logits: torch.Tensor,
    re_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean_gate = group_gate.mean(dim=0)
    target = torch.full_like(mean_gate, 1.0 / group_gate.shape[1])
    balance = torch.mean((mean_gate - target) ** 2)
    entropy = -torch.sum(group_gate * torch.log(group_gate + EPS), dim=1).mean()
    targets = re_regime_targets_torch(re_values, group_logits.shape[1])
    supervision = F.cross_entropy(group_logits, targets)
    return balance, entropy, supervision


def scheduled_sampling_probability(args: argparse.Namespace, epoch: int) -> float:
    start = float(np.clip(args.scheduled_sampling_start, 0.0, 1.0))
    end = float(np.clip(args.scheduled_sampling_end, 0.0, 1.0))
    warmup = float(np.clip(args.scheduled_sampling_warmup_frac, 1.0e-6, 1.0))
    denom = max(1.0, args.epochs * warmup)
    progress = float(np.clip((epoch - 1) / denom, 0.0, 1.0))
    return start + (end - start) * progress


def relative_l2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + EPS))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def relative_vector_loss_torch(
    diff: torch.Tensor, target: torch.Tensor, floor: torch.Tensor
) -> torch.Tensor:
    numerator = torch.sum(diff * diff, dim=1)
    denominator = torch.sum(target * target, dim=1) + torch.clamp(floor, min=EPS)
    return torch.mean(numerator / denominator)


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
    labels_list = [str(label) for label in labels]
    first_prefix = labels_list[0]
    H_key = "H" if "H" in tensors.files else f"{first_prefix}_H"
    P_key = "P" if "P" in tensors.files else f"{first_prefix}_P"
    H = torch.tensor(tensors[H_key][:r_u, :r_u, :r_u], dtype=torch.float32, device=device)
    P = torch.tensor(tensors[P_key][:r_u, :r_p], dtype=torch.float32, device=device)
    label_to_tensor_row = {}
    if "c_all" in tensors.files and "A_all" in tensors.files:
        computed_labels = tensors["Re_labels_computed"].astype(str)
        label_to_tensor_row = {
            str(label): i for i, label in enumerate(computed_labels.tolist())
        }
    for label_id, label in enumerate(labels_list):
        prefix = str(label)
        if label_to_tensor_row:
            tensor_row = label_to_tensor_row[prefix]
            c_np = tensors["c_all"][tensor_row, :r_u]
            A_np = tensors["A_all"][tensor_row, :r_u, :r_u]
        else:
            c_np = tensors[f"{prefix}_c"][:r_u]
            A_np = tensors[f"{prefix}_A"][:r_u, :r_u]
        out[label_id] = {
            "c": torch.tensor(c_np, dtype=torch.float32, device=device),
            "A": torch.tensor(A_np, dtype=torch.float32, device=device),
            "H": H,
            "P": P,
        }
    return out


def build_pressure_surrogate_torch(
    tensors: np.lib.npyio.NpzFile,
    labels: Iterable[str],
    r_u: int,
    r_p: int,
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    H_tilde = torch.tensor(
        tensors["H_tilde"][:r_p, :r_u, :r_u], dtype=torch.float32, device=device
    )
    for label_id, label in enumerate(labels):
        prefix = str(label)
        out[label_id] = {
            "c_tilde": torch.tensor(
                tensors[f"{prefix}_c_tilde"][:r_p], dtype=torch.float32, device=device
            ),
            "A_tilde": torch.tensor(
                tensors[f"{prefix}_A_tilde"][:r_p, :r_u],
                dtype=torch.float32,
                device=device,
            ),
            "H_tilde": H_tilde,
        }
    return out


def pressure_surrogate_torch(
    a_next: torch.Tensor,
    label_ids: torch.Tensor,
    sur: Dict[int, Dict[str, torch.Tensor]],
) -> torch.Tensor:
    r_p = next(iter(sur.values()))["c_tilde"].shape[0]
    out = torch.empty((a_next.shape[0], r_p), dtype=a_next.dtype, device=a_next.device)
    for label_id in torch.unique(label_ids).detach().cpu().tolist():
        label_int = int(label_id)
        mask = label_ids == label_int
        ar = a_next[mask]
        item = sur[label_int]
        out[mask] = (
            item["c_tilde"].unsqueeze(0)
            + ar @ item["A_tilde"].T
            + torch.einsum("pij,bi,bj->bp", item["H_tilde"], ar, ar)
        )
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
    model: OperatorSpaceMoEROM,
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
        rhs_std, pressure_std, gates, _ = model(x, return_expert_stack=False)
    rhs_pred = (
        rhs_std.detach().cpu().numpy() * scalers["rhs_operator"].scale
        + scalers["rhs_operator"].mean
    )
    if args.rhs_target == "residual":
        rhs_pred = arrays["rhs_g"][sample_ids] + rhs_pred
    pressure_op = (
        pressure_std.detach().cpu().numpy() * scalers["pressure_next"].scale
        + scalers["pressure_next"].mean
    )
    if args.pressure_target == "state":
        pressure_pred = pressure_op
        pressure_delta = pressure_pred - arrays["pressure_base_next"][sample_ids]
    else:
        pressure_delta = pressure_op
        pressure_pred = arrays["pressure_base_next"][sample_ids] + pressure_delta
    y_rhs = arrays["adot"][sample_ids]
    y_alpha = arrays["a_next"][sample_ids]
    y_pressure = arrays["b_next"][sample_ids]
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
            _, _, cg, _ = model(cx, return_expert_stack=False)
            _, _, pg, _ = model(px, return_expert_stack=False)
            gate_smooth = float(gate_smoothness_loss(cg, pg).detach().cpu())

    return {
        "rhs_relative_l2": relative_l2_np(y_rhs, rhs_pred),
        "rhs_rmse": rmse_np(y_rhs, rhs_pred),
        "rhs_centered_r2": centered_r2_np(y_rhs, rhs_pred),
        "alpha_head_relative_l2": relative_l2_np(y_alpha, euler_pred),
        "alpha_head_rmse": rmse_np(y_alpha, euler_pred),
        "operator_one_step_euler_relative_l2": relative_l2_np(y_alpha, euler_pred),
        "operator_one_step_euler_rmse": rmse_np(y_alpha, euler_pred),
        "pressure_head_relative_l2": relative_l2_np(y_pressure, pressure_pred),
        "pressure_head_rmse": rmse_np(y_pressure, pressure_pred),
        "pressure_surrogate_base_relative_l2": relative_l2_np(
            y_pressure, arrays["pressure_base_next"][sample_ids]
        ),
        "pressure_residual_relative_l2": relative_l2_np(
            arrays["pressure_residual"][sample_ids], pressure_delta
        ),
        "one_step_euler_relative_l2": relative_l2_np(y_alpha, euler_pred),
        "one_step_euler_rmse": rmse_np(y_alpha, euler_pred),
        "router_entropy": float(np.mean(ent)) if ent else 0.0,
        "router_temporal_smooth_mse": gate_smooth,
        "router_utilization": [float(v) for v in util_mean.tolist()],
    }


def routing_analysis(
    model: OperatorSpaceMoEROM,
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
        _, _, gates, _ = model(x, return_expert_stack=False)
    if not gates:
        return {}
    gate_arrays = [g.detach().cpu().numpy() for g in gates]
    gate_np = np.mean(np.stack(gate_arrays, axis=0), axis=0)
    top1 = np.argmax(gate_np, axis=1)
    top1_counts = np.bincount(top1, minlength=gate_np.shape[1]).astype(float)
    top1_frac = top1_counts / max(1, len(top1))
    topk_sets: Dict[str, int] = {}
    active_counts = np.sum(gate_np > 1.0e-6, axis=1)
    for row in gate_np:
        active = np.flatnonzero(row > 1.0e-6)
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
    by_router = {}
    for name, gate_arr in zip(["velocity", "pressure"], gate_arrays):
        r_load = gate_arr.mean(axis=0)
        r_entropy = -np.sum(gate_arr * np.log(gate_arr + EPS), axis=1)
        r_top1 = np.argmax(gate_arr, axis=1)
        r_top = np.bincount(r_top1, minlength=gate_arr.shape[1]).astype(float)
        r_top = r_top / max(1, len(r_top1))
        r_active = np.sum(gate_arr > 1.0e-6, axis=1)
        by_router[name] = {
            "mean_load": [float(v) for v in r_load.tolist()],
            "top1_fraction": [float(v) for v in r_top.tolist()],
            "load_cv": float(np.std(r_load) / (np.mean(r_load) + EPS)),
            "dead_experts_threshold_1pct": int(np.sum(r_load < 0.01)),
            "entropy_mean": float(np.mean(r_entropy)),
            "active_experts_mean": float(np.mean(r_active)),
            "active_experts_std": float(np.std(r_active)),
            "active_experts_min": int(np.min(r_active)),
            "active_experts_max": int(np.max(r_active)),
        }
    return {
        "num_samples": int(len(sample_ids)),
        "num_experts": int(gate_np.shape[1]),
        "num_regime_groups": int(getattr(model, "num_regime_groups", 1)),
        "experts_per_group": int(getattr(model, "experts_per_group", gate_np.shape[1])),
        "mean_load": [float(v) for v in load.tolist()],
        "load_cv": float(np.std(load) / (np.mean(load) + EPS)),
        "dead_experts_threshold_1pct": int(np.sum(load < 0.01)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_std": float(np.std(entropy)),
        "top1_fraction": [float(v) for v in top1_frac.tolist()],
        "active_experts_mean": float(np.mean(active_counts)),
        "active_experts_std": float(np.std(active_counts)),
        "active_experts_min": int(np.min(active_counts)),
        "active_experts_max": int(np.max(active_counts)),
        "effective_top_k_for_counts": int(np.max(active_counts)) if len(active_counts) else 0,
        "topk_set_counts": {k: int(v) for k, v in sorted(topk_sets.items())},
        "by_router": by_router,
        "by_phase_bin": by_phase,
    }


def routing_by_re_group_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        _, _, gates, _ = model(x, return_expert_stack=False)
    if not gates:
        return {}
    gate_np = np.mean(np.stack([g.detach().cpu().numpy() for g in gates], axis=0), axis=0)
    top1 = np.argmax(gate_np, axis=1)
    re = arrays["re"][sample_ids]
    groups = {
        "low_Re_lt_80": re < 80.0,
        "mid_Re_80_160": (re >= 80.0) & (re < 160.0),
        "high_Re_ge_160": re >= 160.0,
    }
    out: Dict[str, object] = {}
    for name, mask in groups.items():
        if not np.any(mask):
            out[name] = {"num_samples": 0}
            continue
        load = gate_np[mask].mean(axis=0)
        top = np.bincount(top1[mask], minlength=gate_np.shape[1]).astype(float)
        top = top / max(1, int(np.sum(mask)))
        entropy = -np.sum(gate_np[mask] * np.log(gate_np[mask] + EPS), axis=1)
        out[name] = {
            "num_samples": int(np.sum(mask)),
            "mean_load": [float(v) for v in load.tolist()],
            "top1_fraction": [float(v) for v in top.tolist()],
            "load_cv": float(np.std(load) / (np.mean(load) + EPS)),
            "entropy_mean": float(np.mean(entropy)),
        }
    return out


def group_routing_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        group_gate, _ = model.group_router_outputs(x)
    gate_np = group_gate.detach().cpu().numpy()
    top1 = np.argmax(gate_np, axis=1)
    entropy = -np.sum(gate_np * np.log(gate_np + EPS), axis=1)
    active = np.sum(gate_np > 1.0e-6, axis=1)
    re = arrays["re"][sample_ids]
    re_groups = {
        "low_Re_lt_80": re < 80.0,
        "transition_Re_80_160": (re >= 80.0) & (re < 160.0),
        "high_Re_ge_160": re >= 160.0,
    }
    by_re: Dict[str, object] = {}
    for name, mask in re_groups.items():
        if not np.any(mask):
            by_re[name] = {"num_samples": 0}
            continue
        top = np.bincount(top1[mask], minlength=gate_np.shape[1]).astype(float)
        top = top / max(1, int(np.sum(mask)))
        by_re[name] = {
            "num_samples": int(np.sum(mask)),
            "mean_load": [float(v) for v in gate_np[mask].mean(axis=0).tolist()],
            "top1_fraction": [float(v) for v in top.tolist()],
            "entropy_mean": float(np.mean(entropy[mask])),
            "active_groups_mean": float(np.mean(active[mask])),
        }
    top_all = np.bincount(top1, minlength=gate_np.shape[1]).astype(float)
    top_all = top_all / max(1, len(top1))
    return {
        "num_samples": int(len(sample_ids)),
        "num_groups": int(gate_np.shape[1]),
        "mean_load": [float(v) for v in gate_np.mean(axis=0).tolist()],
        "top1_fraction": [float(v) for v in top_all.tolist()],
        "entropy_mean": float(np.mean(entropy)),
        "entropy_std": float(np.std(entropy)),
        "active_groups_mean": float(np.mean(active)),
        "active_groups_std": float(np.std(active)),
        "by_re_group": by_re,
    }


def shared_operator_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    if model.num_shared_experts <= 0:
        return {"num_shared_experts": 0, "always_active": False}
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        group_gate, _ = model.group_router_outputs(x)
    gate_np = group_gate.detach().cpu().numpy()
    top1 = np.argmax(gate_np, axis=1)
    top = np.bincount(top1, minlength=gate_np.shape[1]).astype(float)
    top = top / max(1, len(top1))
    entropy = -np.sum(gate_np * np.log(gate_np + EPS), axis=1)
    return {
        "num_shared_experts": int(model.num_shared_experts),
        "shared_experts_per_group": int(model.shared_per_group),
        "always_active": True,
        "always_active_in_selected_group": True,
        "mean_active_shared_groups": float(np.mean(np.sum(gate_np > 1.0e-6, axis=1))),
        "mean_mixer_weight": [float(v) for v in gate_np.mean(axis=0).tolist()],
        "group_mean_load": [float(v) for v in gate_np.mean(axis=0).tolist()],
        "group_top1_fraction": [float(v) for v in top.tolist()],
        "mixer_weight_std": [float(v) for v in gate_np.std(axis=0).tolist()],
        "mixer_entropy_mean": float(np.mean(entropy)),
        "mixer_entropy_std": float(np.std(entropy)),
    }


def expert_operator_diversity_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
    max_samples: int = 1024,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    model.eval()
    chosen = sample_ids
    if len(chosen) > max_samples:
        rng = np.random.default_rng(12345)
        chosen = np.sort(rng.choice(chosen, size=max_samples, replace=False))
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][chosen]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        _, _, _, expert_stack = model(x)
    expert_vectors = expert_stack.detach().cpu().numpy().transpose(1, 0, 2)
    flat = expert_vectors.reshape(expert_vectors.shape[0], -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(flat, axis=1, keepdims=True) + EPS
    sim = (flat / norm) @ (flat / norm).T
    mask = ~np.eye(sim.shape[0], dtype=bool)
    off = sim[mask]
    return {
        "num_samples": int(len(chosen)),
        "pairwise_cosine_mean": float(np.mean(off)),
        "pairwise_cosine_abs_mean": float(np.mean(np.abs(off))),
        "pairwise_cosine_max_abs": float(np.max(np.abs(off))),
        "pairwise_cosine_matrix": sim.astype(float).round(6).tolist(),
        "collapse_flag_abs_cos_gt_0p95": bool(np.max(np.abs(off)) > 0.95),
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


def init_history_states_torch(
    start_ids: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hist_ids = arrays_t["hist_idx"][start_ids]
    safe = torch.clamp(hist_ids, min=0)
    valid = hist_ids >= 0
    a_hist = torch.where(valid[:, :, None], arrays_t["a"][safe], torch.zeros_like(arrays_t["a"][safe]))
    b_hist = torch.where(valid[:, :, None], arrays_t["b"][safe], torch.zeros_like(arrays_t["b"][safe]))
    rhs_hist = torch.where(
        valid[:, :, None], arrays_t["rhs_g"][safe], torch.zeros_like(arrays_t["rhs_g"][safe])
    )
    return a_hist, b_hist, rhs_hist


def model_outputs_from_states_torch(
    model: OperatorSpaceMoEROM,
    a_state: torch.Tensor,
    b_state: torch.Tensor,
    current: torch.Tensor,
    a_hist: torch.Tensor,
    b_hist: torch.Tensor,
    rhs_hist: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    re = arrays_t["re"][current]
    label_id = arrays_t["label_id"][current]
    ph = arrays_t["phase"][current]
    rhs_g = galerkin_rhs_torch(a_state, b_state, label_id, gal)
    base_x = make_features_torch(a_state, b_state, rhs_g, re, ph, args.phase_harmonics)
    x = make_history_features_from_states_torch(
        base_x, a_state, b_state, rhs_g, a_hist, b_hist, rhs_hist
    )
    x = (x - scalers_t["x_mean"]) / scalers_t["x_scale"]
    rhs_std, pressure_std, _, _ = model(x, return_expert_stack=False)
    rhs_op = rhs_std * scalers_t["rhs_op_scale"] + scalers_t["rhs_op_mean"]
    if args.rhs_target == "residual":
        rhs_op = rhs_g + rhs_op
    pressure_op = pressure_std * scalers_t["pressure_scale"] + scalers_t["pressure_mean"]
    return rhs_op, pressure_op, rhs_g


def integrate_autonomous_step_torch(
    model: OperatorSpaceMoEROM,
    a_state: torch.Tensor,
    b_state: torch.Tensor,
    current: torch.Tensor,
    dt: torch.Tensor,
    a_hist: torch.Tensor,
    b_hist: torch.Tensor,
    rhs_hist: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    sur: Dict[int, Dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k1, pressure_op, rhs_g = model_outputs_from_states_torch(
        model, a_state, b_state, current, a_hist, b_hist, rhs_hist, arrays_t, scalers_t, gal, args
    )
    if args.integrator == "rk4":
        k2, _, _ = model_outputs_from_states_torch(
            model,
            a_state + 0.5 * dt * k1,
            b_state,
            current,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            args,
        )
        k3, _, _ = model_outputs_from_states_torch(
            model,
            a_state + 0.5 * dt * k2,
            b_state,
            current,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            args,
        )
        k4, _, _ = model_outputs_from_states_torch(
            model,
            a_state + dt * k3,
            b_state,
            current,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            args,
        )
        a_next = a_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        a_next = a_state + dt * k1
    if args.pressure_target == "state":
        b_next = pressure_op
    else:
        b_base = pressure_surrogate_torch(a_next, arrays_t["label_id"][current], sur)
        b_next = b_base + pressure_op
    return a_next, b_next, rhs_g


def rollout_loss_torch(
    model: OperatorSpaceMoEROM,
    start_ids: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    sur: Dict[int, Dict[str, torch.Tensor]],
    args: argparse.Namespace,
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if start_ids.numel() == 0:
        zero = torch.tensor(0.0, device=next(model.parameters()).device)
        return zero, zero, zero
    a_cur = arrays_t["a"][start_ids]
    b_cur = arrays_t["b"][start_ids]
    a_hist, b_hist, rhs_hist = init_history_states_torch(start_ids, arrays_t)
    current = start_ids
    losses = []
    energy_losses = []
    last_pred_a = a_cur
    last_pred_b = b_cur
    last_target_a = a_cur
    last_target_b = b_cur
    sampling_prob = float(np.clip(getattr(args, "scheduled_sampling_probability", 1.0), 0.0, 1.0))
    for _ in range(steps):
        nxt = arrays_t["next_idx"][current]
        dt = arrays_t["time"][nxt].unsqueeze(1) - arrays_t["time"][current].unsqueeze(1)
        a_next, b_next, rhs_g = integrate_autonomous_step_torch(
            model, a_cur, b_cur, current, dt, a_hist, b_hist, rhs_hist, arrays_t, scalers_t, gal, sur, args
        )
        target_a = arrays_t["a"][nxt]
        target_b = arrays_t["b"][nxt]
        a_loss_std = F.mse_loss(
            (a_next - target_a) / scalers_t["alpha_scale"], torch.zeros_like(a_next)
        )
        b_loss_std = F.mse_loss(
            (b_next - target_b) / scalers_t["pressure_state_scale"], torch.zeros_like(b_next)
        )
        a_loss_rel = relative_vector_loss_torch(
            a_next - target_a, target_a, scalers_t["alpha_rel_floor"]
        )
        b_loss_rel = relative_vector_loss_torch(
            b_next - target_b, target_b, scalers_t["pressure_rel_floor"]
        )
        mix = float(np.clip(args.rollout_relative_mix, 0.0, 1.0))
        a_loss = (1.0 - mix) * a_loss_std + mix * a_loss_rel
        b_loss = (1.0 - mix) * b_loss_std + mix * b_loss_rel
        losses.append(a_loss + args.lambda_pressure_rollout * b_loss)

        pred_energy = torch.sum(a_next * a_next, dim=1)
        target_energy = torch.sum(target_a * target_a, dim=1)
        energy_losses.append(
            torch.mean(
                (pred_energy - target_energy) ** 2
                / (target_energy + torch.clamp(scalers_t["alpha_rel_floor"], min=EPS)) ** 2
            )
        )

        a_feed = sampling_prob * a_next + (1.0 - sampling_prob) * target_a
        b_feed = sampling_prob * b_next + (1.0 - sampling_prob) * target_b
        a_hist = torch.cat([a_feed[:, None, :], a_hist[:, :-1, :]], dim=1)
        b_hist = torch.cat([b_feed[:, None, :], b_hist[:, :-1, :]], dim=1)
        rhs_hist = torch.cat([rhs_g[:, None, :], rhs_hist[:, :-1, :]], dim=1)
        a_cur = a_feed
        b_cur = b_feed
        last_pred_a = a_next
        last_pred_b = b_next
        last_target_a = target_a
        last_target_b = target_b
        current = nxt
    rollout_loss = torch.stack(losses).mean()
    energy_loss = torch.stack(energy_losses).mean() if energy_losses else rollout_loss * 0.0
    traj_a = relative_vector_loss_torch(
        last_pred_a - last_target_a, last_target_a, scalers_t["alpha_rel_floor"]
    )
    traj_b = relative_vector_loss_torch(
        last_pred_b - last_target_b, last_target_b, scalers_t["pressure_rel_floor"]
    )
    trajectory_loss = traj_a + args.lambda_pressure_rollout * traj_b
    return rollout_loss, energy_loss, trajectory_loss


def model_rhs_torch(
    model: OperatorSpaceMoEROM,
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
    rhs_std, _, _, _ = model(x, return_expert_stack=False)
    rhs_op = rhs_std * scalers_t["rhs_op_scale"] + scalers_t["rhs_op_mean"]
    if args.rhs_target == "residual":
        rhs_op = rhs_g + rhs_op
    return rhs_op


def integrate_step_torch(
    model: OperatorSpaceMoEROM,
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
    model: OperatorSpaceMoEROM,
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
    model: OperatorSpaceMoEROM,
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
        rhs_std, _, _, _ = model(xt, return_expert_stack=False)
    rhs_op = (
        rhs_std.detach().cpu().numpy()[0] * scalers["rhs_operator"].scale
        + scalers["rhs_operator"].mean
    )
    if args.rhs_target == "residual":
        rhs_op = rhs_g + rhs_op
    return rhs_op


def integrate_step_np(
    model: OperatorSpaceMoEROM,
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


def init_history_states_np(
    start: int,
    arrays: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist = arrays["hist_idx"][start]
    a_hist = np.zeros((1, len(hist), arrays["a"].shape[1]), dtype=np.float32)
    b_hist = np.zeros((1, len(hist), arrays["b"].shape[1]), dtype=np.float32)
    rhs_hist = np.zeros((1, len(hist), arrays["rhs_g"].shape[1]), dtype=np.float32)
    valid = hist >= 0
    if np.any(valid):
        a_hist[0, valid] = arrays["a"][hist[valid]]
        b_hist[0, valid] = arrays["b"][hist[valid]]
        rhs_hist[0, valid] = arrays["rhs_g"][hist[valid]]
    return a_hist, b_hist, rhs_hist


def model_outputs_from_states_np(
    model: OperatorSpaceMoEROM,
    a_state: np.ndarray,
    b_state: np.ndarray,
    cur: int,
    a_hist: np.ndarray,
    b_hist: np.ndarray,
    rhs_hist: np.ndarray,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rhs_g = galerkin_rhs_by_label(
        tensors,
        a_state[None, :],
        b_state[None, :],
        arrays["label_id"][cur : cur + 1],
        arrays["labels"],
        args.r_u,
        args.r_p,
    )[0]
    base_x = make_features_np(
        a_state[None, :],
        b_state[None, :],
        rhs_g[None, :],
        arrays["re"][cur : cur + 1],
        arrays["phase"][cur : cur + 1],
        args.phase_harmonics,
    )
    x = make_history_features_from_states_np(
        base_x,
        a_state[None, :],
        b_state[None, :],
        rhs_g[None, :],
        a_hist,
        b_hist,
        rhs_hist,
    )
    xt = torch.tensor(scalers["x"].transform(x), dtype=torch.float32, device=device)
    with torch.no_grad():
        rhs_std, pressure_std, _, _ = model(xt, return_expert_stack=False)
    rhs_op = (
        rhs_std.detach().cpu().numpy()[0] * scalers["rhs_operator"].scale
        + scalers["rhs_operator"].mean
    )
    if args.rhs_target == "residual":
        rhs_op = rhs_g + rhs_op
    pressure_op = (
        pressure_std.detach().cpu().numpy()[0] * scalers["pressure_next"].scale
        + scalers["pressure_next"].mean
    )
    return rhs_op.astype(np.float32), pressure_op.astype(np.float32), rhs_g


def integrate_autonomous_step_np(
    model: OperatorSpaceMoEROM,
    a_state: np.ndarray,
    b_state: np.ndarray,
    cur: int,
    dt: float,
    a_hist: np.ndarray,
    b_hist: np.ndarray,
    rhs_hist: np.ndarray,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k1, pressure_op, rhs_g = model_outputs_from_states_np(
        model, a_state, b_state, cur, a_hist, b_hist, rhs_hist, arrays, scalers, tensors, args, device
    )
    if args.integrator == "rk4":
        k2, _, _ = model_outputs_from_states_np(
            model,
            a_state + 0.5 * dt * k1,
            b_state,
            cur,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            args,
            device,
        )
        k3, _, _ = model_outputs_from_states_np(
            model,
            a_state + 0.5 * dt * k2,
            b_state,
            cur,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            args,
            device,
        )
        k4, _, _ = model_outputs_from_states_np(
            model,
            a_state + dt * k3,
            b_state,
            cur,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            args,
            device,
        )
        a_next = a_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        a_next = a_state + dt * k1
    if args.pressure_target == "state":
        b_next = pressure_op
    else:
        b_base = pressure_surrogate_by_label(
            pressure_tensors,
            a_next[None, :],
            arrays["label_id"][cur : cur + 1],
            arrays["labels"],
            args.r_u,
            args.r_p,
        )[0]
        b_next = b_base + pressure_op
    return a_next.astype(np.float32), b_next.astype(np.float32), rhs_g.astype(np.float32)


def rollout_autonomous_pressure_eval_np(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    test_label_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    idx = np.where(arrays["label_id"] == test_label_id)[0]
    idx = idx[np.argsort(arrays["time"][idx])]
    valid_sample_set = set(arrays["sample_ids"].tolist())
    a_rel_errors: List[float] = []
    b_rel_errors: List[float] = []
    stride = max(1, args.rollout_steps)
    model.eval()
    for start_pos in range(1, len(idx) - args.rollout_steps - 1, stride):
        start = int(idx[start_pos])
        if start not in valid_sample_set:
            continue
        a_cur = arrays["a"][start].copy()
        b_cur = arrays["b"][start].copy()
        a_hist, b_hist, rhs_hist = init_history_states_np(start, arrays)
        pred_a = []
        pred_b = []
        cur = start
        ok = True
        for _ in range(args.rollout_steps):
            nxt = int(arrays["next_idx"][cur])
            if nxt < 0 or int(arrays["label_id"][nxt]) != int(test_label_id):
                ok = False
                break
            dt = float(arrays["time"][nxt] - arrays["time"][cur])
            a_next, b_next, rhs_g = integrate_autonomous_step_np(
                model,
                a_cur,
                b_cur,
                cur,
                dt,
                a_hist,
                b_hist,
                rhs_hist,
                arrays,
                scalers,
                tensors,
                pressure_tensors,
                args,
                device,
            )
            if not (np.all(np.isfinite(a_next)) and np.all(np.isfinite(b_next))):
                ok = False
                break
            pred_a.append(a_next.copy())
            pred_b.append(b_next.copy())
            a_hist = np.concatenate([a_next[None, None, :], a_hist[:, :-1, :]], axis=1)
            b_hist = np.concatenate([b_next[None, None, :], b_hist[:, :-1, :]], axis=1)
            rhs_hist = np.concatenate([rhs_g[None, None, :], rhs_hist[:, :-1, :]], axis=1)
            a_cur = a_next
            b_cur = b_next
            cur = nxt
        if ok and len(pred_a) == args.rollout_steps:
            true_a = arrays["a"][idx[start_pos + 1 : start_pos + args.rollout_steps + 1]]
            true_b = arrays["b"][idx[start_pos + 1 : start_pos + args.rollout_steps + 1]]
            a_rel_errors.append(relative_l2_np(true_a, np.asarray(pred_a)))
            b_rel_errors.append(relative_l2_np(true_b, np.asarray(pred_b)))
    if not a_rel_errors:
        return {
            "a_relative_l2_mean": float("nan"),
            "a_relative_l2_median": float("nan"),
            "a_relative_l2_p90": float("nan"),
            "a_relative_l2_max": float("nan"),
            "b_relative_l2_mean": float("nan"),
            "b_relative_l2_median": float("nan"),
            "b_relative_l2_p90": float("nan"),
            "b_relative_l2_max": float("nan"),
            "num_windows": 0,
        }
    return {
        "a_relative_l2_mean": float(np.mean(a_rel_errors)),
        "a_relative_l2_median": float(np.median(a_rel_errors)),
        "a_relative_l2_p90": float(np.percentile(a_rel_errors, 90)),
        "a_relative_l2_max": float(np.max(a_rel_errors)),
        "b_relative_l2_mean": float(np.mean(b_rel_errors)),
        "b_relative_l2_median": float(np.median(b_rel_errors)),
        "b_relative_l2_p90": float(np.percentile(b_rel_errors, 90)),
        "b_relative_l2_max": float(np.max(b_rel_errors)),
        "num_windows": int(len(a_rel_errors)),
    }


def one_step_integrator_eval_np(
    model: OperatorSpaceMoEROM,
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


def one_step_autonomous_pressure_eval_np(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    sample_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    pred_a = []
    true_a = []
    pred_b = []
    true_b = []
    for cur in sample_ids.tolist():
        nxt = int(arrays["next_idx"][cur])
        if nxt < 0 or int(arrays["label_id"][nxt]) != int(arrays["label_id"][cur]):
            continue
        dt = float(arrays["time"][nxt] - arrays["time"][cur])
        a_hist, b_hist, rhs_hist = init_history_states_np(int(cur), arrays)
        a_next, b_next, _ = integrate_autonomous_step_np(
            model,
            arrays["a"][cur].copy(),
            arrays["b"][cur].copy(),
            int(cur),
            dt,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            pressure_tensors,
            args,
            device,
        )
        pred_a.append(a_next)
        pred_b.append(b_next)
        true_a.append(arrays["a"][nxt])
        true_b.append(arrays["b"][nxt])
    if not pred_a:
        return {
            "a_relative_l2": float("nan"),
            "a_rmse": float("nan"),
            "b_relative_l2": float("nan"),
            "b_rmse": float("nan"),
            "num_samples": 0,
        }
    pred_a_arr = np.asarray(pred_a)
    true_a_arr = np.asarray(true_a)
    pred_b_arr = np.asarray(pred_b)
    true_b_arr = np.asarray(true_b)
    return {
        "a_relative_l2": relative_l2_np(true_a_arr, pred_a_arr),
        "a_rmse": rmse_np(true_a_arr, pred_a_arr),
        "b_relative_l2": relative_l2_np(true_b_arr, pred_b_arr),
        "b_rmse": rmse_np(true_b_arr, pred_b_arr),
        "num_samples": int(len(pred_a)),
    }


def gate_top1_for_ids_np(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if len(sample_ids) == 0:
        return np.asarray([], dtype=np.int64)
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        _, _, gates, _ = model(x, return_expert_stack=False)
    gate_np = np.mean(np.stack([g.detach().cpu().numpy() for g in gates], axis=0), axis=0)
    return np.argmax(gate_np, axis=1).astype(np.int64)


def expert_error_analysis_np(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    sample_ids: np.ndarray,
    test_label_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    top1 = gate_top1_for_ids_np(model, arrays, scalers, sample_ids, device)
    num_experts = int(max(model.num_experts, int(top1.max()) + 1 if len(top1) else 0))
    one_step = {
        e: {"pred_a": [], "true_a": [], "pred_b": [], "true_b": []}
        for e in range(num_experts)
    }
    for pos, cur in enumerate(sample_ids.tolist()):
        nxt = int(arrays["next_idx"][cur])
        if nxt < 0 or int(arrays["label_id"][nxt]) != int(arrays["label_id"][cur]):
            continue
        dt = float(arrays["time"][nxt] - arrays["time"][cur])
        a_hist, b_hist, rhs_hist = init_history_states_np(int(cur), arrays)
        a_next, b_next, _ = integrate_autonomous_step_np(
            model,
            arrays["a"][cur].copy(),
            arrays["b"][cur].copy(),
            int(cur),
            dt,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            pressure_tensors,
            args,
            device,
        )
        item = one_step[int(top1[pos])]
        item["pred_a"].append(a_next)
        item["true_a"].append(arrays["a"][nxt])
        item["pred_b"].append(b_next)
        item["true_b"].append(arrays["b"][nxt])

    idx = np.where(arrays["label_id"] == test_label_id)[0]
    idx = idx[np.argsort(arrays["time"][idx])]
    valid_sample_set = set(arrays["sample_ids"].tolist())
    rollout = {e: {"a": [], "b": []} for e in range(num_experts)}
    stride = max(1, args.rollout_steps)
    for start_pos in range(1, len(idx) - args.rollout_steps - 1, stride):
        start = int(idx[start_pos])
        if start not in valid_sample_set:
            continue
        start_top1 = int(gate_top1_for_ids_np(model, arrays, scalers, np.asarray([start]), device)[0])
        a_cur = arrays["a"][start].copy()
        b_cur = arrays["b"][start].copy()
        a_hist, b_hist, rhs_hist = init_history_states_np(start, arrays)
        pred_a = []
        pred_b = []
        cur = start
        ok = True
        for _ in range(args.rollout_steps):
            nxt = int(arrays["next_idx"][cur])
            if nxt < 0 or int(arrays["label_id"][nxt]) != int(test_label_id):
                ok = False
                break
            dt = float(arrays["time"][nxt] - arrays["time"][cur])
            a_next, b_next, rhs_g = integrate_autonomous_step_np(
                model,
                a_cur,
                b_cur,
                cur,
                dt,
                a_hist,
                b_hist,
                rhs_hist,
                arrays,
                scalers,
                tensors,
                pressure_tensors,
                args,
                device,
            )
            if not (np.all(np.isfinite(a_next)) and np.all(np.isfinite(b_next))):
                ok = False
                break
            pred_a.append(a_next.copy())
            pred_b.append(b_next.copy())
            a_hist = np.concatenate([a_next[None, None, :], a_hist[:, :-1, :]], axis=1)
            b_hist = np.concatenate([b_next[None, None, :], b_hist[:, :-1, :]], axis=1)
            rhs_hist = np.concatenate([rhs_g[None, None, :], rhs_hist[:, :-1, :]], axis=1)
            a_cur = a_next
            b_cur = b_next
            cur = nxt
        if ok and len(pred_a) == args.rollout_steps:
            true_a = arrays["a"][idx[start_pos + 1 : start_pos + args.rollout_steps + 1]]
            true_b = arrays["b"][idx[start_pos + 1 : start_pos + args.rollout_steps + 1]]
            rollout[start_top1]["a"].append(relative_l2_np(true_a, np.asarray(pred_a)))
            rollout[start_top1]["b"].append(relative_l2_np(true_b, np.asarray(pred_b)))

    per_expert: Dict[str, object] = {}
    for e in range(num_experts):
        item = one_step[e]
        if item["pred_a"]:
            pred_a_arr = np.asarray(item["pred_a"])
            true_a_arr = np.asarray(item["true_a"])
            pred_b_arr = np.asarray(item["pred_b"])
            true_b_arr = np.asarray(item["true_b"])
            one_a = relative_l2_np(true_a_arr, pred_a_arr)
            one_b = relative_l2_np(true_b_arr, pred_b_arr)
            count = int(len(item["pred_a"]))
        else:
            one_a = float("nan")
            one_b = float("nan")
            count = 0
        roll_a = rollout[e]["a"]
        roll_b = rollout[e]["b"]
        per_expert[str(e)] = {
            "top1_one_step_count": count,
            "one_step_a_relative_l2": float(one_a),
            "one_step_b_relative_l2": float(one_b),
            "rollout_window_count": int(len(roll_a)),
            "rollout_a_relative_l2_mean": float(np.mean(roll_a)) if roll_a else float("nan"),
            "rollout_b_relative_l2_mean": float(np.mean(roll_b)) if roll_b else float("nan"),
        }
    return per_expert


def build_arrays(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    pod_dir = args.data_root
    idx = load_snapshot_index(pod_dir / "pod_snapshot_index.csv")
    vel = np.load(pod_dir / "global_velocity_pod_weighted_l2.npz")
    pre = np.load(pod_dir / "global_pressure_pod_weighted_l2.npz")
    tensors = np.load(args.tensor_path)
    pressure_tensors = np.load(args.pressure_surrogate_path)

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
    b_next = b[nxt]
    pressure_base_next = pressure_surrogate_by_label(
        pressure_tensors, a_next, label_id, labels, args.r_u, args.r_p
    )
    pressure_residual = (b_next - pressure_base_next).astype(np.float32)
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
        "b_next": b_next.astype(np.float32),
        "pressure_base_next": pressure_base_next.astype(np.float32),
        "pressure_residual": pressure_residual,
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
    pressure_tensors: np.lib.npyio.NpzFile,
    test_label_id: int,
    device: torch.device,
) -> Dict[str, object]:
    sample_ids = arrays["sample_ids"]
    train_all = sample_ids[arrays["label_id"][sample_ids] != test_label_id]
    test_ids = sample_ids[arrays["label_id"][sample_ids] == test_label_id]
    print(
        json.dumps(
            {
                "event": "split_start",
                "experiment": args.experiment_name,
                "test_label": str(arrays["labels"][test_label_id]),
                "test_Re": float(arrays["re"][test_ids[0]]) if len(test_ids) else None,
            }
        ),
        flush=True,
    )

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
    pressure_target_array = (
        arrays["b_next"] if args.pressure_target == "state" else arrays["pressure_residual"]
    )
    rhs_target_array = (
        arrays["residual"] if args.rhs_target == "residual" else arrays["adot"]
    )

    scalers = {
        "x": Standardizer.fit(arrays["x"][train_ids]),
        "rhs_operator": Standardizer.fit(rhs_target_array[train_ids]),
        "alpha_next": Standardizer.fit(arrays["a_next"][train_ids]),
        "pressure_next": Standardizer.fit(pressure_target_array[train_ids]),
        "pressure_state": Standardizer.fit(arrays["b_next"][train_ids]),
    }
    alpha_rel_floor = (
        np.percentile(np.sum(arrays["a_next"][train_ids] ** 2, axis=1), 10)
        * args.relative_floor_frac
        + EPS
    )
    rhs_rel_floor = (
        np.percentile(np.sum(arrays["adot"][train_ids] ** 2, axis=1), 10)
        * args.relative_floor_frac
        + EPS
    )
    pressure_rel_floor = (
        np.percentile(np.sum(arrays["b_next"][train_ids] ** 2, axis=1), 10)
        * args.relative_floor_frac
        + EPS
    )
    pressure_energy_train = np.sum(arrays["b_next"][train_ids] ** 2, axis=1).astype(np.float32)
    pressure_energy_ref = float(np.median(pressure_energy_train) + EPS)

    x_train = torch.tensor(scalers["x"].transform(arrays["x"][train_ids]), device=device)
    rhs_train = torch.tensor(
        scalers["rhs_operator"].transform(rhs_target_array[train_ids]), device=device
    )
    pressure_train = torch.tensor(
        scalers["pressure_next"].transform(pressure_target_array[train_ids]),
        device=device,
    )
    pressure_weight_train = torch.ones(len(train_ids), dtype=torch.float32, device=device)
    if args.pressure_amplitude_weight_power > 0.0:
        p_energy = torch.tensor(pressure_energy_train, dtype=torch.float32, device=device)
        pressure_weight_train = (
            torch.tensor(pressure_energy_ref, dtype=torch.float32, device=device)
            / torch.clamp(p_energy, min=EPS)
        ) ** args.pressure_amplitude_weight_power
        pressure_weight_train = torch.clamp(
            pressure_weight_train,
            min=1.0 / max(args.pressure_amplitude_weight_max, 1.0),
            max=max(args.pressure_amplitude_weight_max, 1.0),
        )
        pressure_weight_train = pressure_weight_train / torch.mean(pressure_weight_train)
    x_all_t = torch.tensor(scalers["x"].transform(arrays["x"]), device=device)
    current_a_train = torch.tensor(arrays["a"][train_ids], device=device)
    rhs_g_train = torch.tensor(arrays["rhs_g"][train_ids], device=device)
    dt_train = torch.tensor(arrays["dt_next"][train_ids][:, None], device=device)
    train_ids_t = torch.tensor(train_ids, dtype=torch.long, device=device)
    phi_recon = torch.tensor(arrays["phi_recon"], device=device)

    arrays_t = {
        "a": torch.tensor(arrays["a"], dtype=torch.float32, device=device),
        "b": torch.tensor(arrays["b"], dtype=torch.float32, device=device),
        "b_next": torch.tensor(arrays["b_next"], dtype=torch.float32, device=device),
        "pressure_base_next": torch.tensor(
            arrays["pressure_base_next"], dtype=torch.float32, device=device
        ),
        "pressure_residual": torch.tensor(
            arrays["pressure_residual"], dtype=torch.float32, device=device
        ),
        "adot": torch.tensor(arrays["adot"], dtype=torch.float32, device=device),
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
        "rhs_op_mean": torch.tensor(
            scalers["rhs_operator"].mean, dtype=torch.float32, device=device
        ),
        "rhs_op_scale": torch.tensor(
            scalers["rhs_operator"].scale, dtype=torch.float32, device=device
        ),
        "alpha_mean": torch.tensor(scalers["alpha_next"].mean, dtype=torch.float32, device=device),
        "alpha_scale": torch.tensor(scalers["alpha_next"].scale, dtype=torch.float32, device=device),
        "pressure_mean": torch.tensor(
            scalers["pressure_next"].mean, dtype=torch.float32, device=device
        ),
        "pressure_scale": torch.tensor(
            scalers["pressure_next"].scale, dtype=torch.float32, device=device
        ),
        "pressure_state_scale": torch.tensor(
            scalers["pressure_state"].scale, dtype=torch.float32, device=device
        ),
        "alpha_rel_floor": torch.tensor(alpha_rel_floor, dtype=torch.float32, device=device),
        "rhs_rel_floor": torch.tensor(rhs_rel_floor, dtype=torch.float32, device=device),
        "pressure_rel_floor": torch.tensor(
            pressure_rel_floor, dtype=torch.float32, device=device
        ),
    }
    gal = build_galerkin_torch(tensors, arrays["labels"], args.r_u, args.r_p, device)
    sur = build_pressure_surrogate_torch(
        pressure_tensors, arrays["labels"], args.r_u, args.r_p, device
    )

    model = OperatorSpaceMoEROM(
        in_dim=arrays["x"].shape[1],
        out_dim=args.r_u,
        pressure_dim=args.r_p,
        hidden_dim=args.hidden_dim,
        expert_hidden=args.expert_hidden,
        num_blocks=args.num_blocks,
        num_experts=args.num_experts,
        num_operator_spaces=args.num_shared_experts,
        num_regime_groups=args.num_regime_groups,
        experts_per_group=args.experts_per_group,
        top_k=args.top_k,
        group_top_k=args.group_top_k,
        dropout=args.dropout,
        temperature=args.temperature,
        gate_floor=args.gate_floor,
        group_temperature=args.group_temperature,
        group_gate_floor=args.group_gate_floor,
        shared_scale=args.shared_scale,
        routed_scale=args.routed_scale,
        expert_blocks=args.expert_blocks,
        quadratic_rank=args.quadratic_rank,
        quadratic_scale=args.quadratic_scale,
        phase_harmonics=args.phase_harmonics,
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
        args.scheduled_sampling_probability = scheduled_sampling_probability(args, epoch)
        perm = torch.randperm(num_train, device=device)
        batch_losses = []
        for start in range(0, num_train, args.batch_size):
            bi = perm[start : start + args.batch_size]
            x = x_train[bi]
            rhs_target = rhs_train[bi]
            pressure_target = pressure_train[bi]
            batch_ids = train_ids_t[bi]
            next_train_ids = arrays_t["next_idx"][batch_ids]
            rhs_std, pressure_pred_std, gates, expert_stack = model(x)
            dyn_loss = F.mse_loss(rhs_std, rhs_target)
            pressure_mse = torch.mean((pressure_pred_std - pressure_target) ** 2, dim=1)
            pressure_loss = torch.mean(pressure_weight_train[bi] * pressure_mse)

            rhs_pred = rhs_std * scalers_t["rhs_op_scale"] + scalers_t["rhs_op_mean"]
            if args.rhs_target == "residual":
                rhs_pred = rhs_g_train[bi] + rhs_pred
            alpha_euler = current_a_train[bi] + dt_train[bi] * rhs_pred
            coeff_loss = F.mse_loss(
                (alpha_euler - arrays_t["a"][next_train_ids]) / scalers_t["alpha_scale"],
                torch.zeros_like(alpha_euler),
            )
            alpha_rel_loss = relative_vector_loss_torch(
                alpha_euler - arrays_t["a"][next_train_ids],
                arrays_t["a"][next_train_ids],
                scalers_t["alpha_rel_floor"],
            )
            rhs_rel_loss = relative_vector_loss_torch(
                rhs_pred - arrays_t["adot"][batch_ids],
                arrays_t["adot"][batch_ids],
                scalers_t["rhs_rel_floor"],
            )
            pressure_op_phy = (
                pressure_pred_std * scalers_t["pressure_scale"]
                + scalers_t["pressure_mean"]
            )
            if args.pressure_target == "state":
                pressure_next_phy = pressure_op_phy
            else:
                pressure_next_phy = arrays_t["pressure_base_next"][batch_ids] + pressure_op_phy
            pressure_rel_loss = relative_vector_loss_torch(
                pressure_next_phy - arrays_t["b_next"][batch_ids],
                arrays_t["b_next"][batch_ids],
                scalers_t["pressure_rel_floor"],
            )
            finite_step_rhs = (
                arrays_t["a"][next_train_ids] - current_a_train[bi]
            ) / torch.clamp(dt_train[bi], min=1.0e-8)
            consistency_loss = F.mse_loss(
                (rhs_pred - finite_step_rhs) / scalers_t["rhs_op_scale"],
                torch.zeros_like(rhs_pred),
            )

            recon_delta = (alpha_euler - arrays_t["a"][next_train_ids]) @ phi_recon
            recon_true = arrays_t["a"][next_train_ids] @ phi_recon
            recon_loss = torch.mean(recon_delta**2) / (torch.mean(recon_true**2) + EPS)

            balance_loss, entropy_loss, router_info = router_regularization(gates)
            group_gate, group_logits = model.group_router_outputs(x)
            group_balance_loss, group_entropy_loss, group_supervision_loss = (
                group_router_regularization(group_gate, group_logits, arrays_t["re"][batch_ids])
            )
            diversity_loss = expert_diversity_loss(expert_stack)
            regime_loss = regime_router_loss(gates, arrays_t["re"][batch_ids])
            prev_ids = arrays_t["prev_idx"][batch_ids]
            smooth_mask = (prev_ids >= 0) & (
                arrays_t["label_id"][prev_ids] == arrays_t["label_id"][batch_ids]
            )
            if bool(torch.any(smooth_mask)):
                _, _, prev_gates, _ = model(
                    x_all_t[prev_ids[smooth_mask]], return_expert_stack=False
                )
                smooth_loss = gate_smoothness_loss([g[smooth_mask] for g in gates], prev_gates)
            else:
                smooth_loss = torch.tensor(0.0, device=device)

            loss = (
                args.lambda_coeff * coeff_loss
                + args.lambda_dyn * dyn_loss
                + args.lambda_pressure * pressure_loss
                + args.lambda_alpha_rel * alpha_rel_loss
                + args.lambda_rhs_rel * rhs_rel_loss
                + args.lambda_pressure_rel * pressure_rel_loss
                + args.lambda_recon * recon_loss
                + args.lambda_consistency * consistency_loss
                + args.lambda_router_balance * balance_loss
                + args.lambda_router_entropy * entropy_loss
                + args.lambda_group_balance * group_balance_loss
                + args.lambda_group_entropy * group_entropy_loss
                + args.lambda_group_supervision * group_supervision_loss
                + args.lambda_router_smooth * smooth_loss
                + args.lambda_expert_diversity * diversity_loss
                + args.lambda_regime_router * regime_loss
            )

            batch_no = start // args.batch_size
            do_rollout = (
                args.lambda_rollout > 0
                and train_roll_starts_t.numel() > 0
                and batch_no % max(1, args.rollout_every_batches) == 0
            )
            if do_rollout:
                current_rollout_steps = curriculum_step_for_epoch(args, epoch)
                count = min(args.rollout_batch, int(train_roll_starts_t.numel()))
                rb = train_roll_starts_t[torch.randint(0, train_roll_starts_t.numel(), (count,), device=device)]
                roll_loss, energy_loss, trajectory_loss = rollout_loss_torch(
                    model, rb, arrays_t, scalers_t, gal, sur, args, current_rollout_steps
                )
                loss = (
                    loss
                    + args.lambda_rollout * roll_loss
                    + args.lambda_energy * energy_loss
                    + args.lambda_trajectory_consistency * trajectory_loss
                )
            else:
                current_rollout_steps = 0
                roll_loss = torch.tensor(0.0, device=device)
                energy_loss = torch.tensor(0.0, device=device)
                trajectory_loss = torch.tensor(0.0, device=device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "coeff": float(coeff_loss.detach().cpu()),
                    "dyn": float(dyn_loss.detach().cpu()),
                    "pressure": float(pressure_loss.detach().cpu()),
                    "alpha_rel": float(alpha_rel_loss.detach().cpu()),
                    "rhs_rel": float(rhs_rel_loss.detach().cpu()),
                    "pressure_rel": float(pressure_rel_loss.detach().cpu()),
                    "recon": float(recon_loss.detach().cpu()),
                    "rollout": float(roll_loss.detach().cpu()),
                    "energy": float(energy_loss.detach().cpu()),
                    "trajectory_consistency": float(trajectory_loss.detach().cpu()),
                    "rollout_steps": int(current_rollout_steps),
                    "scheduled_sampling": float(args.scheduled_sampling_probability),
                    "router_smooth": float(smooth_loss.detach().cpu()),
                    "group_balance": float(group_balance_loss.detach().cpu()),
                    "group_entropy": float(group_entropy_loss.detach().cpu()),
                    "group_supervision": float(group_supervision_loss.detach().cpu()),
                    "expert_diversity": float(diversity_loss.detach().cpu()),
                    "regime_router": float(regime_loss.detach().cpu()),
                }
            )
        sched.step()

        if epoch == 1 or epoch % max(1, args.eval_every) == 0:
            val = evaluate_model(model, arrays, scalers, val_ids, args, device)
            val_score = (
                val["rhs_relative_l2"]
                + val["alpha_head_relative_l2"]
                + 0.35 * val["pressure_head_relative_l2"]
            )
            mean_loss = float(np.mean([b["loss"] for b in batch_losses]))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "val_score": float(val_score),
                    "val_rhs_relative_l2": float(val["rhs_relative_l2"]),
                    "val_alpha_relative_l2": float(val["alpha_head_relative_l2"]),
                    "val_pressure_relative_l2": float(val["pressure_head_relative_l2"]),
                }
            )
            event = {
                "event": "epoch_eval",
                "experiment": args.experiment_name,
                "test_label": str(arrays["labels"][test_label_id]),
                "epoch": int(epoch),
                "train_loss": mean_loss,
                "val_score": float(val_score),
                "val_rhs_relative_l2": float(val["rhs_relative_l2"]),
                "val_alpha_relative_l2": float(val["alpha_head_relative_l2"]),
                "val_pressure_relative_l2": float(val["pressure_head_relative_l2"]),
                "scheduled_sampling": float(args.scheduled_sampling_probability),
                "rollout_steps": int(curriculum_step_for_epoch(args, epoch)),
            }
            if args.eval_routing_every > 0 and epoch % args.eval_routing_every == 0:
                route_ids = np.asarray(val_ids)
                if len(route_ids) > args.eval_routing_max_samples:
                    rng = np.random.default_rng(args.seed + epoch)
                    route_ids = np.sort(
                        rng.choice(
                            route_ids,
                            size=args.eval_routing_max_samples,
                            replace=False,
                        )
                    )
                route = routing_analysis(model, arrays, scalers, route_ids, args, device)
                group_route = group_routing_analysis(model, arrays, scalers, route_ids, device)
                event["eval_routing"] = {
                    "num_samples": int(len(route_ids)),
                    "group_mean_load": group_route.get("mean_load", []),
                    "group_top1_fraction": group_route.get("top1_fraction", []),
                    "active_experts_mean": route.get("active_experts_mean"),
                    "active_experts_std": route.get("active_experts_std"),
                    "dead_experts_threshold_1pct": route.get(
                        "dead_experts_threshold_1pct"
                    ),
                    "load_cv": route.get("load_cv"),
                }
                history[-1]["eval_routing"] = event["eval_routing"]
            print(json.dumps(event), flush=True)
            if val_score < best_val - args.early_stop_min_delta:
                best_val = float(val_score)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = args.output_dir / (
        f"{args.experiment_name}_{str(arrays['labels'][test_label_id])}_checkpoint.pt"
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "test_label_id": int(test_label_id),
            "test_Re": float(arrays["re"][test_ids[0]]),
            "test_Re_label": str(arrays["labels"][test_label_id]),
            "best_epoch": int(best_epoch),
            "best_val_score": float(best_val),
            "scalers": {
                key: {"mean": value.mean, "scale": value.scale}
                for key, value in scalers.items()
            },
        },
        ckpt_path,
    )

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
        "pressure_surrogate_relative_l2": relative_l2_np(
            arrays["b_next"][test_ids], arrays["pressure_base_next"][test_ids]
        ),
        "pressure_surrogate_rmse": rmse_np(
            arrays["b_next"][test_ids], arrays["pressure_base_next"][test_ids]
        ),
    }
    one_step_integrator = one_step_integrator_eval_np(
        model, arrays, scalers, tensors, test_ids, args, device
    )
    one_step_autonomous = one_step_autonomous_pressure_eval_np(
        model, arrays, scalers, tensors, pressure_tensors, test_ids, args, device
    )
    rollout = rollout_eval_np(model, arrays, scalers, tensors, test_label_id, args, device)
    rollout_autonomous = rollout_autonomous_pressure_eval_np(
        model, arrays, scalers, tensors, pressure_tensors, test_label_id, args, device
    )
    route_test = routing_analysis(model, arrays, scalers, test_ids, args, device)
    route_train = routing_analysis(model, arrays, scalers, train_ids, args, device)
    route_re_groups = routing_by_re_group_analysis(
        model, arrays, scalers, train_ids, device
    )
    group_route_test = group_routing_analysis(model, arrays, scalers, test_ids, device)
    group_route_train = group_routing_analysis(model, arrays, scalers, train_ids, device)
    shared_test = shared_operator_analysis(model, arrays, scalers, test_ids, device)
    shared_train = shared_operator_analysis(model, arrays, scalers, train_ids, device)
    expert_diversity = expert_operator_diversity_analysis(
        model, arrays, scalers, test_ids, device
    )
    expert_errors = expert_error_analysis_np(
        model,
        arrays,
        scalers,
        tensors,
        pressure_tensors,
        test_ids,
        test_label_id,
        args,
        device,
    )
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
        "checkpoint_path": str(ckpt_path),
        "baseline_galerkin": base_metrics,
        "deep_moe": test_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "one_step_integrator": one_step_integrator,
        "one_step_autonomous_pressure": one_step_autonomous,
        "rollout_teacher_forced_pressure": rollout,
        "rollout_autonomous_pressure": rollout_autonomous,
        "routing_analysis_test": route_test,
        "routing_analysis_train": route_train,
        "routing_by_re_group_train": route_re_groups,
        "group_routing_analysis_test": group_route_test,
        "group_routing_analysis_train": group_route_train,
        "shared_operator_analysis_test": shared_test,
        "shared_operator_analysis_train": shared_train,
        "expert_operator_diversity": expert_diversity,
        "expert_error_analysis_test": expert_errors,
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
        "# HPRS-MoE-ROM v14 Summary",
        "",
        "## Architecture",
        "",
        (
            f"Shared encoder + {settings['num_blocks']} latent refinement blocks, "
            f"hidden_dim={settings['hidden_dim']}, regime_groups={settings['num_regime_groups']}, "
            f"experts_per_group={settings['experts_per_group']}, "
            f"shared_experts_per_group={settings['shared_experts_per_group']}, "
            f"group_top_k={settings['group_top_k']}, in_group_top_k={settings['top_k']}, "
            f"expert_hidden={settings['expert_hidden']}, expert_blocks={settings['expert_blocks']}, "
            f"quadratic_rank={settings['quadratic_rank']}."
        ),
        "",
        (
            f"Shared/routed scales: {settings['shared_scale']:.3g} / "
            f"{settings['routed_scale']:.3g}; routed gate floor: "
            f"{settings['gate_floor']:.3g}."
        ),
        "",
        (
            f"A shared group router selects a physics regime, then group-local velocity/pressure "
            f"Top-2 routers mix a group-shared expert with routed physics-aware operator experts. "
            f"Experts output `{settings['rhs_target']}` velocity operator targets plus a pressure "
            f"`{settings['pressure_target']}` branch. For `residual`, the learned closure is added "
            "to the Galerkin RHS before RK4."
        ),
        "",
        "Losses: one-step coefficient, sampled reconstruction, full-RHS operator, closed-loop "
        "multi-step rollout, energy consistency, trajectory consistency, pressure closure, "
        "relative terms, group/router load-balance, entropy, temporal smoothness, expert diversity, "
        "and weak Re-regime supervision.",
        "",
        "## Metrics",
        "",
        f"Integrator: `{settings['integrator']}`.",
        "",
        "| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["results"]:
        base = item["baseline_galerkin"]
        deep = item["deep_moe"]
        roll = item["rollout_teacher_forced_pressure"]
        roll_auto = item["rollout_autonomous_pressure"]
        lines.append(
            f"| {item['test_Re']} | Galerkin only | {base['rhs_relative_l2']:.6g} | "
            f"{base['pressure_surrogate_relative_l2']:.6g} | - | "
            f"{base['one_step_euler_relative_l2']:.6g} | - | - | - | - | - | - | - | - | - |"
        )
        routing = item["routing_analysis_test"]
        one = item["one_step_integrator"]
        one_auto = item["one_step_autonomous_pressure"]
        lines.append(
            f"| {item['test_Re']} | HPRS-MoE | {deep['rhs_relative_l2']:.6g} | "
            f"{base['pressure_surrogate_relative_l2']:.6g} | "
            f"{deep['pressure_head_relative_l2']:.6g} | {one['relative_l2']:.6g} | "
            f"{one_auto['a_relative_l2']:.6g} | {one_auto['b_relative_l2']:.6g} | "
            f"{roll['relative_l2_mean']:.6g} | {roll_auto['a_relative_l2_mean']:.6g} | "
            f"{roll_auto['b_relative_l2_mean']:.6g} | {routing['active_experts_mean']:.3g} | "
            f"{routing['load_cv']:.6g} | "
            f"{routing['entropy_mean']:.6g} | {routing['dead_experts_threshold_1pct']} |"
        )
    lines.append("")
    lines.append("## Expert Diagnostics")
    lines.append("")
    lines.append("| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |")
    lines.append("|---:|---|---|---|---:|")
    for item in result["results"]:
        shared = item.get("shared_operator_analysis_test", {})
        weights = shared.get("group_mean_load", shared.get("mean_mixer_weight", []))
        weight_text = "[" + ", ".join(f"{float(v):.3f}" for v in weights) + "]" if weights else "-"
        top = shared.get("group_top1_fraction", [])
        top_text = "[" + ", ".join(f"{float(v):.3f}" for v in top) + "]" if top else "-"
        entropy = shared.get("mixer_entropy_mean", float("nan"))
        lines.append(
            f"| {item['test_Re']} | {shared.get('always_active_in_selected_group', False)} | "
            f"{weight_text} | {top_text} | {entropy:.6g} |"
        )
    lines.append("")
    lines.append(
        "| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |"
    )
    lines.append("|---:|---:|---|---|")
    for item in result["results"]:
        diversity = item.get("expert_operator_diversity", {})
        groups = item.get("routing_by_re_group_train", {})
        group_bits = []
        for key in ["low_Re_lt_80", "mid_Re_80_160", "high_Re_ge_160"]:
            info = groups.get(key, {})
            top = info.get("top1_fraction", [])
            if top:
                group_bits.append(f"{key}: e{int(np.argmax(top))}")
        lines.append(
            f"| {item['test_Re']} | {diversity.get('pairwise_cosine_max_abs', float('nan')):.6g} | "
            f"{diversity.get('collapse_flag_abs_cos_gt_0p95', False)} | "
            f"{'; '.join(group_bits)} |"
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
    pressure_tensors = np.load(args.pressure_surrogate_path)

    results = []
    for test_label_id in args.test_re_indices:
        results.append(
            train_one_split(
                args, arrays, meta, tensors, pressure_tensors, int(test_label_id), device
            )
        )

    settings = {
        "experiment_name": args.experiment_name,
        "r_u": args.r_u,
        "r_p": args.r_p,
        "phase_harmonics": args.phase_harmonics,
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "num_experts": int(args.num_regime_groups * (args.experts_per_group + 1)),
        "legacy_num_experts_arg": args.num_experts,
        "num_operator_spaces": args.num_regime_groups,
        "num_regime_groups": args.num_regime_groups,
        "experts_per_group": args.experts_per_group,
        "shared_experts_per_group": 1,
        "top_k": args.top_k,
        "group_top_k": args.group_top_k,
        "expert_hidden": args.expert_hidden,
        "expert_blocks": args.expert_blocks,
        "quadratic_rank": args.quadratic_rank,
        "quadratic_scale": args.quadratic_scale,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "gate_floor": args.gate_floor,
        "group_temperature": args.group_temperature,
        "group_gate_floor": args.group_gate_floor,
        "shared_scale": args.shared_scale,
        "routed_scale": args.routed_scale,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "eval_every": args.eval_every,
        "early_stop_min_delta": args.early_stop_min_delta,
        "curriculum_steps": parse_curriculum_steps(args.curriculum_steps),
        "train_rollout_steps": args.train_rollout_steps,
        "rollout_steps": args.rollout_steps,
        "rollout_batch": args.rollout_batch,
        "rollout_every_batches": args.rollout_every_batches,
        "recon_dim": args.recon_dim,
        "loss_weights": {
            "coeff": args.lambda_coeff,
            "dyn": args.lambda_dyn,
            "pressure": args.lambda_pressure,
            "recon": args.lambda_recon,
            "rollout": args.lambda_rollout,
            "pressure_rollout": args.lambda_pressure_rollout,
            "consistency": args.lambda_consistency,
            "router_balance": args.lambda_router_balance,
            "router_entropy": args.lambda_router_entropy,
            "group_balance": args.lambda_group_balance,
            "group_entropy": args.lambda_group_entropy,
            "group_supervision": args.lambda_group_supervision,
            "router_smooth": args.lambda_router_smooth,
            "expert_diversity": args.lambda_expert_diversity,
            "regime_router": args.lambda_regime_router,
            "energy": args.lambda_energy,
            "trajectory_consistency": args.lambda_trajectory_consistency,
            "alpha_rel": args.lambda_alpha_rel,
            "rhs_rel": args.lambda_rhs_rel,
            "pressure_rel": args.lambda_pressure_rel,
            "rollout_relative_mix": args.rollout_relative_mix,
            "relative_floor_frac": args.relative_floor_frac,
        },
        "scheduled_sampling": {
            "start": args.scheduled_sampling_start,
            "end": args.scheduled_sampling_end,
            "warmup_frac": args.scheduled_sampling_warmup_frac,
        },
        "history_len": args.history_len,
        "rhs_target": args.rhs_target,
        "pressure_target": args.pressure_target,
        "pressure_amplitude_weight_power": args.pressure_amplitude_weight_power,
        "pressure_amplitude_weight_max": args.pressure_amplitude_weight_max,
        "test_re_indices": args.test_re_indices,
        "integrator": args.integrator,
        "analysis_bins": args.analysis_bins,
        "eval_routing_every": args.eval_routing_every,
        "eval_routing_max_samples": args.eval_routing_max_samples,
        "pressure_surrogate_path": str(args.pressure_surrogate_path),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    out = {
        "scheme": "weighted_l2_hprs_moe_rom_v14_re50_300_closed_loop_rk4",
        "data_root": str(args.data_root),
        "tensor_path": str(args.tensor_path),
        "pressure_surrogate_path": str(args.pressure_surrogate_path),
        "settings": settings,
        "data_meta": meta,
        "results": results,
        "runtime_seconds": float(time.time() - started),
    }
    json_path = args.output_dir / f"{args.experiment_name}_metrics.json"
    md_path = args.output_dir / f"{args.experiment_name}_summary.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_summary_md(md_path, out)
    if args.print_full_json:
        print(json.dumps(out, indent=2))
    else:
        print(
            json.dumps(
                {
                    "scheme": out["scheme"],
                    "experiment_name": args.experiment_name,
                    "metrics_json": str(json_path),
                    "summary_md": str(md_path),
                    "runtime_seconds": out["runtime_seconds"],
                    "test_Re": [item["test_Re"] for item in results],
                }
            )
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
