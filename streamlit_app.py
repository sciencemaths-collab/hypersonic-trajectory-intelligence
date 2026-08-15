#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import html
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Trajectory Operations Console", page_icon="T", layout="wide", initial_sidebar_state="auto")

ENGINE = Path(__file__).resolve().parent / "alien_exit_cell_predictor_v6_3.py"
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
SIDE_NAMES = ["right", "left", "up", "down", "front", "back"]
SIDE_TO_AXIS_SIGN = {
    "right": (0, +1),
    "left": (0, -1),
    "up": (1, +1),
    "down": (1, -1),
    "front": (2, +1),
    "back": (2, -1),
}

st.markdown(
    """
<style>
:root{
  --bg:#0b0d10; --surface:#12161b; --raised:#171c22; --stroke:#2a3139;
  --txt:#f1f4f6; --muted:#98a2ad; --cyan:#35b8d0; --good:#45c486;
  --bad:#ef6262; --warn:#e9b44c; --blue:#5f8ee4;
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:var(--bg)!important;color:var(--txt)!important;}
[data-testid="stSidebar"]{background:#0e1115!important;border-right:1px solid var(--stroke);}
[data-testid="stSidebar"] .block-container{padding-top:1.25rem;}
.block-container{padding-top:.65rem;padding-bottom:1.2rem;max-width:1680px;}
h1,h2,h3,p{letter-spacing:0!important;}
.ops-head{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:8px 0 14px;border-bottom:1px solid var(--stroke);margin-bottom:12px;}
.ops-title{font-size:20px;font-weight:700;color:var(--txt);line-height:1.2}.ops-sub{font-size:12px;color:var(--muted);margin-top:4px}
.status-line{display:flex;flex-wrap:wrap;gap:7px;justify-content:flex-end}.status{font:600 11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;padding:4px 7px;border:1px solid var(--stroke);background:var(--surface);color:var(--muted);border-radius:3px}
.status.ok{color:var(--good);border-color:#245a42}.status.warn{color:var(--warn);border-color:#624d25}.status.info{color:var(--cyan);border-color:#245461}
.section-title{font-size:12px;font-weight:700;color:#c9d0d7;text-transform:uppercase;margin:15px 0 8px;padding-bottom:6px;border-bottom:1px solid var(--stroke)}
.kpi{border-left:3px solid var(--blue);background:var(--surface);padding:10px 12px;min-height:88px}.kpi.good{border-left-color:var(--good)}.kpi.warn{border-left-color:var(--warn)}.kpi.bad{border-left-color:var(--bad)}
.label{font:600 10px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);text-transform:uppercase}.value{font-size:23px;font-weight:700;color:var(--txt);margin-top:3px}.meta{font-size:11px;color:var(--muted);margin-top:3px}
.decision{background:var(--surface);border:1px solid var(--stroke);padding:12px 14px}.decision-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.decision-cell{min-width:0}.decision-main{font:700 16px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--txt);overflow-wrap:anywhere}.decision .pass{color:var(--good)}.decision .fail{color:var(--bad)}
.callout{border-left:3px solid var(--cyan);background:var(--surface);padding:9px 12px;font-size:12px;color:#c5cbd2;margin:9px 0 12px}.callout.warn{border-left-color:var(--warn)}
[data-testid="stPlotlyChart"]{border:1px solid var(--stroke);background:var(--surface);}
[data-testid="stDataFrame"]{border:1px solid var(--stroke);}
button[kind="primary"]{border-radius:3px!important;font-weight:700!important;}
[data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--stroke)}[data-baseweb="tab"]{border-radius:0!important;padding:8px 12px!important;}
@media(max-width:900px){.ops-head{align-items:flex-start;flex-direction:column}.status-line{justify-content:flex-start}.decision-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.value{font-size:20px}}
</style>
""",
    unsafe_allow_html=True,
)

PLOT_CONFIG = {"displaylogo": False, "responsive": True, "scrollZoom": True}
PLOT_BG = "#12161b"
GRID = "#2a3139"


def industrial_plot(fig: go.Figure, title: str, height: int | None = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#dce2e7"), x=0.015, y=0.985),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Arial, sans-serif", size=11, color="#aeb7c0"),
        margin=dict(l=42, r=24, t=82, b=38),
        legend=dict(orientation="h", yanchor="top", y=1.10, x=0),
        hoverlabel=dict(bgcolor="#171c22", bordercolor="#39434d", font_color="#f1f4f6"),
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# ------------------------------
# IO helpers
# ------------------------------
def run_engine(out_prefix: str, quick: bool, force_cpu: bool) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(ENGINE), "--output-prefix", out_prefix]
    if quick:
        cmd.append("--quick")
    if force_cpu:
        cmd.append("--no-gpu")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900)


def load_trace(prefix: str):
    p = Path(prefix + "_online_trace.npz")
    if not p.exists():
        return None
    return np.load(p, allow_pickle=False)


def load_metrics(prefix: str):
    p = Path(prefix + "_metrics.json")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------
# Physics / geometry helpers
# ------------------------------
def ecef_to_lla(pos_xyz_m: np.ndarray) -> pd.DataFrame:
    x = np.asarray(pos_xyz_m[:, 0], dtype=float)
    y = np.asarray(pos_xyz_m[:, 1], dtype=float)
    z = np.asarray(pos_xyz_m[:, 2], dtype=float)
    lon = np.arctan2(y, x)
    p = np.sqrt(x * x + y * y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(7):
        sin_lat = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + alt + 1e-12)))
    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / np.cos(lat) - N
    return pd.DataFrame({"lat": np.degrees(lat), "lon": np.degrees(lon), "alt_km": alt / 1000.0})


def estimate_velocity(pos: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(pos)
    if len(pos) > 1:
        vel[1:] = np.diff(pos, axis=0) / max(dt, 1e-9)
    return vel


def estimate_acceleration(vel: np.ndarray, dt: float) -> np.ndarray:
    acc = np.zeros_like(vel)
    if len(vel) > 1:
        acc[1:] = np.diff(vel, axis=0) / max(dt, 1e-9)
    return acc


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def make_local_frame(vel: np.ndarray, acc: np.ndarray | None = None) -> np.ndarray:
    """Return rotation matrix with columns [right, up, forward]."""
    f = normalize(np.asarray(vel, dtype=float))
    if not np.any(f):
        f = np.array([1.0, 0.0, 0.0])

    world_up = np.array([0.0, 0.0, 1.0])
    # If acceleration has a perpendicular component, use it to stabilize roll.
    if acc is not None:
        a = np.asarray(acc, dtype=float)
        a_perp = a - np.dot(a, f) * f
        if np.linalg.norm(a_perp) > 1e-6:
            world_up = normalize(a_perp)

    r = np.cross(f, world_up)
    if np.linalg.norm(r) < 1e-8:
        world_up = np.array([0.0, 1.0, 0.0])
        r = np.cross(f, world_up)
    r = normalize(r)
    u = normalize(np.cross(r, f))
    return np.column_stack([r, u, f])


def world_to_local(points: np.ndarray, center: np.ndarray, R: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float) - center[None, :]
    return pts @ R


def local_to_world(points_local: np.ndarray, center: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.asarray(points_local, dtype=float) @ R.T + center[None, :]


def compute_half_lengths(speed: float, lookahead_s: float, base_scale: float = 0.42) -> np.ndarray:
    # Curvature-aware shell: larger at higher speed / longer horizon, still clamped.
    L = float(np.clip(base_scale * max(speed, 1.0) * max(lookahead_s, 0.2), 35.0, 450.0))
    return np.array([L, L, L], dtype=float)


def gate_center_local(side: str, row: int, col: int, N: int, half_lengths: np.ndarray) -> np.ndarray:
    Lx, Ly, Lz = half_lengths
    def cell_coord(idx: int, L: float) -> float:
        cell = 2.0 * L / N
        return -L + (idx + 0.5) * cell

    # rows go top->bottom, cols left->right in face coordinates
    if side in ("right", "left"):
        y = cell_coord(col, Ly)
        z = cell_coord(N - 1 - row, Lz)
        x = Lx if side == "right" else -Lx
        return np.array([x, y, z])
    if side in ("up", "down"):
        x = cell_coord(col, Lx)
        z = cell_coord(N - 1 - row, Lz)
        y = Ly if side == "up" else -Ly
        return np.array([x, y, z])
    # front/back
    x = cell_coord(col, Lx)
    y = cell_coord(N - 1 - row, Ly)
    z = Lz if side == "front" else -Lz
    return np.array([x, y, z])


def project_to_face(q: np.ndarray, side: str, half_lengths: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Return face uv coords in [-1,1]x[-1,1] and crossing point in local coords."""
    Lx, Ly, Lz = half_lengths
    x, y, z = q.astype(float)
    axis, sign = SIDE_TO_AXIS_SIGN[side]
    L = [Lx, Ly, Lz][axis]
    coord = [x, y, z][axis]
    if abs(coord) < 1e-9:
        coord = sign * 1e-9
    s = (sign * L) / coord
    cross = np.array([x, y, z], dtype=float) * s
    cross[axis] = sign * L

    if side in ("right", "left"):
        u = np.clip(cross[1] / Ly, -0.999, 0.999)
        v = np.clip(cross[2] / Lz, -0.999, 0.999)
    elif side in ("up", "down"):
        u = np.clip(cross[0] / Lx, -0.999, 0.999)
        v = np.clip(cross[2] / Lz, -0.999, 0.999)
    else:
        u = np.clip(cross[0] / Lx, -0.999, 0.999)
        v = np.clip(cross[1] / Ly, -0.999, 0.999)
    return float(u), float(v), cross


def uv_to_gate(u: float, v: float, N: int) -> Tuple[int, int, int]:
    col = int(np.clip(np.floor((u + 1.0) * 0.5 * N), 0, N - 1))
    row_from_bottom = int(np.clip(np.floor((v + 1.0) * 0.5 * N), 0, N - 1))
    row = N - 1 - row_from_bottom
    gid = row * N + col
    return row, col, gid


def side_scores_from_local(q: np.ndarray, half_lengths: np.ndarray, temp: float = 5.0) -> Dict[str, float]:
    x, y, z = q / (half_lengths + 1e-12)
    raw = np.array([
        max(x, 0.0), max(-x, 0.0), max(y, 0.0), max(-y, 0.0), max(z, 0.0), max(-z, 0.0)
    ], dtype=float)
    logits = temp * raw
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum() + 1e-12
    return {name: float(p) for name, p in zip(SIDE_NAMES, probs)}


def gate_probabilities_for_side(side: str, q: np.ndarray, half_lengths: np.ndarray, N: int, sigma_cells: float = 0.85) -> np.ndarray:
    u, v, _ = project_to_face(q, side, half_lengths)
    row_f = (1 - (v + 1) * 0.5) * N - 0.5
    col_f = ((u + 1) * 0.5) * N - 0.5
    rr, cc = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    d2 = (rr - row_f) ** 2 + (cc - col_f) ** 2
    logits = -0.5 * d2 / max(sigma_cells ** 2, 1e-6)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum() + 1e-12
    return probs


def full_distribution(q: np.ndarray, half_lengths: np.ndarray, N: int) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    sp = side_scores_from_local(q, half_lengths)
    per_side = {s: sp[s] * gate_probabilities_for_side(s, q, half_lengths, N) for s in SIDE_NAMES}
    return sp, per_side


def actual_side_gate(q_true: np.ndarray, half_lengths: np.ndarray, N: int) -> Tuple[str, int, int, np.ndarray]:
    ratios = np.abs(q_true) / (half_lengths + 1e-12)
    axis = int(np.argmax(ratios))
    sign = 1 if q_true[axis] >= 0 else -1
    side = {
        (0, 1): "right", (0, -1): "left",
        (1, 1): "up", (1, -1): "down",
        (2, 1): "front", (2, -1): "back",
    }[(axis, sign)]
    u, v, cross = project_to_face(q_true, side, half_lengths)
    row, col, gid = uv_to_gate(u, v, N)
    return side, row, gid, cross


def first_boundary_crossing(path_local: np.ndarray, half_lengths: np.ndarray,
                            N: int) -> Tuple[str, int, int, np.ndarray] | None:
    """Return the first linearly interpolated crossing of an axis-aligned box."""
    path = np.asarray(path_local, dtype=float)
    for k in range(1, len(path)):
        p0, p1 = path[k - 1], path[k]
        delta = p1 - p0
        candidates = []
        for axis in range(3):
            if abs(delta[axis]) < 1e-12:
                continue
            for sign in (-1, 1):
                lam = (sign * half_lengths[axis] - p0[axis]) / delta[axis]
                if 0.0 <= lam <= 1.0:
                    cross = p0 + lam * delta
                    other = [j for j in range(3) if j != axis]
                    if all(abs(cross[j]) <= half_lengths[j] + 1e-8 for j in other):
                        candidates.append((float(lam), axis, sign, cross))
        if candidates:
            _, axis, sign, cross = min(candidates, key=lambda item: item[0])
            side = {
                (0, 1): "right", (0, -1): "left",
                (1, 1): "up", (1, -1): "down",
                (2, 1): "front", (2, -1): "back",
            }[(axis, sign)]
            u, v, _ = project_to_face(cross, side, half_lengths)
            row, _, gid = uv_to_gate(u, v, N)
            return side, row, gid, cross
    return None


@dataclass
class FramePrediction:
    frame: int
    time_s: float
    center: np.ndarray
    R: np.ndarray
    half_lengths: np.ndarray
    true_local: np.ndarray
    est_local: np.ndarray
    actual_side: str
    actual_gate: int
    pred_side: str
    pred_gate: int
    actual_cross_local: np.ndarray
    pred_cross_local: np.ndarray
    side_probs: Dict[str, float]
    gate_probs: Dict[str, np.ndarray]
    exact_hit: bool
    side_hit: bool
    confidence: float
    unresolved_mass: float
    maneuver_mass: float


def analyze_dynamic_side_gate(trace, lookahead_frames: int, N: int) -> Tuple[List[FramePrediction], pd.DataFrame]:
    pos_true = np.asarray(trace["pos_true"], dtype=float)
    pos_est = np.asarray(trace["pos_est"], dtype=float)
    dt = float(trace["dt"][0]) if "dt" in trace else 1.0
    vel_est = np.asarray(trace["vel_est"], dtype=float) if "vel_est" in trace else estimate_velocity(pos_est, dt)
    acc_est = estimate_acceleration(vel_est, dt)
    physics_pos_pred = np.asarray(trace["physics_pos_pred"], dtype=float) if "physics_pos_pred" in trace else None
    sigma_pos_pred = np.asarray(trace["physics_sigma_pos_pred"], dtype=float) if "physics_sigma_pos_pred" in trace else None
    sigma_weights = np.asarray(trace["physics_sigma_weights"], dtype=float) if "physics_sigma_weights" in trace else None
    maneuver_pos_pred = np.asarray(trace["maneuver_pos_pred"], dtype=float) if "maneuver_pos_pred" in trace else None
    maneuver_event_step = np.asarray(trace["maneuver_event_step"], dtype=int) if "maneuver_event_step" in trace else None
    maneuver_step_probability = float(trace["maneuver_step_probability"][0]) if "maneuver_step_probability" in trace else 0.0

    frames: List[FramePrediction] = []
    rows = []

    T = len(pos_true)
    horizon = int(max(1, lookahead_frames))
    # Begin after two observations so velocity and acceleration use history only.
    for t in range(2, T - horizon):
        center = pos_est[t]
        v = vel_est[t]
        a = acc_est[t]
        speed = float(np.linalg.norm(v))
        lookahead_s = horizon * dt
        half_lengths = compute_half_lengths(speed, lookahead_s)
        R = make_local_frame(v, a)

        true_path_local = world_to_local(pos_true[t:t + horizon + 1], center, R)
        true_crossing = first_boundary_crossing(true_path_local, half_lengths, N)
        q_true = true_path_local[-1]
        # Prefer the engine's causal 6-DOF rollout. Legacy traces retain a
        # constant-acceleration fallback and never read pos_est[t+horizon].
        if physics_pos_pred is not None and horizon < physics_pos_pred.shape[1]:
            pred_world = physics_pos_pred[t, horizon]
        else:
            pred_world = center + v * lookahead_s + 0.5 * a * lookahead_s**2
        q_est = world_to_local(pred_world[None, :], center, R)[0]

        if true_crossing is None:
            actual_side, actual_row, actual_gate, actual_cross = actual_side_gate(q_true, half_lengths, N)
        else:
            actual_side, actual_row, actual_gate, actual_cross = true_crossing

        unresolved_mass = 0.0
        maneuver_mass = 0.0
        if sigma_pos_pred is not None and sigma_weights is not None and horizon < sigma_pos_pred.shape[2]:
            side_mass = {s: 0.0 for s in SIDE_NAMES}
            gate_mass = {s: np.zeros((N, N), dtype=float) for s in SIDE_NAMES}
            eligible = np.array([], dtype=int)
            if maneuver_pos_pred is not None and maneuver_event_step is not None:
                eligible = np.flatnonzero(maneuver_event_step <= horizon)
            if eligible.size:
                maneuver_mass = 1.0 - (1.0 - maneuver_step_probability) ** horizon
            nominal_mass = 1.0 - maneuver_mass
            for path_world, weight in zip(sigma_pos_pred[t, :, :horizon + 1], sigma_weights):
                weight = float(weight) * nominal_mass
                crossing = first_boundary_crossing(world_to_local(path_world, center, R), half_lengths, N)
                if crossing is None:
                    unresolved_mass += weight
                    continue
                side, row, gate, _ = crossing
                _, col = divmod(gate, N)
                side_mass[side] += weight
                gate_mass[side][row, col] += weight
            if eligible.size:
                hypothesis_weight = maneuver_mass / len(eligible)
                for idx in eligible:
                    path_world = maneuver_pos_pred[t, idx, :horizon + 1]
                    crossing = first_boundary_crossing(
                        world_to_local(path_world, center, R), half_lengths, N)
                    if crossing is None:
                        unresolved_mass += hypothesis_weight
                        continue
                    side, row, gate, _ = crossing
                    _, col = divmod(gate, N)
                    side_mass[side] += hypothesis_weight
                    gate_mass[side][row, col] += hypothesis_weight
            sp, gp = side_mass, gate_mass
            if sum(sp.values()) <= 1e-12:
                sp, gp = full_distribution(q_est, half_lengths, N)
                unresolved_mass = 1.0
        else:
            sp, gp = full_distribution(q_est, half_lengths, N)
        pred_side = max(sp, key=sp.get)
        pred_gate = int(np.argmax(gp[pred_side]))
        pred_row, pred_col = divmod(pred_gate, N)
        pred_cross = gate_center_local(pred_side, pred_row, pred_col, N, half_lengths)
        confidence = float(gp[pred_side].reshape(-1)[pred_gate])

        exact_hit = (pred_side == actual_side) and (pred_gate == actual_gate)
        side_hit = pred_side == actual_side

        fp = FramePrediction(
            frame=t,
            time_s=t * dt,
            center=center,
            R=R,
            half_lengths=half_lengths,
            true_local=q_true,
            est_local=q_est,
            actual_side=actual_side,
            actual_gate=actual_gate,
            pred_side=pred_side,
            pred_gate=pred_gate,
            actual_cross_local=actual_cross,
            pred_cross_local=pred_cross,
            side_probs=sp,
            gate_probs=gp,
            exact_hit=exact_hit,
            side_hit=side_hit,
            confidence=confidence,
            unresolved_mass=unresolved_mass,
            maneuver_mass=maneuver_mass,
        )
        frames.append(fp)
        rows.append({
            "frame": t,
            "time_s": t * dt,
            "actual_side": actual_side,
            "pred_side": pred_side,
            "actual_gate": actual_gate,
            "pred_gate": pred_gate,
            "exact_hit": int(exact_hit),
            "side_hit": int(side_hit),
            "confidence": confidence,
            "unresolved_mass": unresolved_mass,
            "maneuver_mass": maneuver_mass,
            "speed_mps": speed,
            "box_half_length_m": half_lengths[0],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["cum_exact_acc"] = df["exact_hit"].cumsum() / np.arange(1, len(df) + 1)
        df["cum_side_acc"] = df["side_hit"].cumsum() / np.arange(1, len(df) + 1)
    return frames, df


def face_matrix(frame: FramePrediction, side: str) -> np.ndarray:
    return frame.gate_probs[side]


def side_gate_confusion(frames: List[FramePrediction], N: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    exact_rows = []
    gate_rows = []
    for fp in frames:
        exact_rows.append((f"{fp.actual_side}:{fp.actual_gate}", f"{fp.pred_side}:{fp.pred_gate}"))
        gate_rows.append((fp.actual_gate, fp.pred_gate))
    if not exact_rows:
        return pd.DataFrame(), pd.DataFrame()
    exact = pd.crosstab(
        pd.Series([a for a, _ in exact_rows], name="actual"),
        pd.Series([p for _, p in exact_rows], name="pred"),
        dropna=False,
    )
    gate = pd.crosstab(
        pd.Series([a for a, _ in gate_rows], name="actual_gate"),
        pd.Series([p for _, p in gate_rows], name="pred_gate"),
        dropna=False,
    )
    return exact, gate


# ------------------------------
# Plot builders
# ------------------------------
def cube_vertices(half_lengths: np.ndarray) -> np.ndarray:
    Lx, Ly, Lz = half_lengths
    return np.array([
        [-Lx, -Ly, -Lz], [-Lx, -Ly, +Lz], [-Lx, +Ly, -Lz], [-Lx, +Ly, +Lz],
        [+Lx, -Ly, -Lz], [+Lx, -Ly, +Lz], [+Lx, +Ly, -Lz], [+Lx, +Ly, +Lz],
    ], dtype=float)


def cube_edges() -> List[Tuple[int, int]]:
    return [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]


def face_outline(side: str, half_lengths: np.ndarray) -> np.ndarray:
    Lx, Ly, Lz = half_lengths
    if side in ("right", "left"):
        x = Lx if side == "right" else -Lx
        pts = np.array([[x,-Ly,-Lz],[x,+Ly,-Lz],[x,+Ly,+Lz],[x,-Ly,+Lz],[x,-Ly,-Lz]])
    elif side in ("up", "down"):
        y = Ly if side == "up" else -Ly
        pts = np.array([[-Lx,y,-Lz],[+Lx,y,-Lz],[+Lx,y,+Lz],[-Lx,y,+Lz],[-Lx,y,-Lz]])
    else:
        z = Lz if side == "front" else -Lz
        pts = np.array([[-Lx,-Ly,z],[+Lx,-Ly,z],[+Lx,+Ly,z],[-Lx,+Ly,z],[-Lx,-Ly,z]])
    return pts


def plot_mission_view(frame: FramePrediction, pos_true: np.ndarray, pos_est: np.ndarray, around: int = 10) -> go.Figure:
    i = frame.frame
    lo = max(0, i - around)
    hi = min(len(pos_true), i + around + 1)
    true_local = world_to_local(pos_true[lo:hi], frame.center, frame.R)
    est_local = world_to_local(pos_est[lo:hi], frame.center, frame.R)
    fig = go.Figure()

    # cube edges
    verts = cube_vertices(frame.half_lengths)
    for a, b in cube_edges():
        seg = np.vstack([verts[a], verts[b]])
        fig.add_trace(go.Scatter3d(x=seg[:,0], y=seg[:,1], z=seg[:,2], mode="lines", showlegend=False,
                                   line=dict(width=4)))

    # true / estimated local trajectory
    fig.add_trace(go.Scatter3d(x=true_local[:,0], y=true_local[:,1], z=true_local[:,2], mode="lines",
                               name="True trail", line=dict(width=7)))
    fig.add_trace(go.Scatter3d(x=est_local[:,0], y=est_local[:,1], z=est_local[:,2], mode="lines",
                               name="Estimated trail", line=dict(width=5, dash="dot")))

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="Ship now",
                               marker=dict(size=7, symbol="diamond")))
    fig.add_trace(go.Scatter3d(x=[frame.actual_cross_local[0]], y=[frame.actual_cross_local[1]], z=[frame.actual_cross_local[2]],
                               mode="markers", name="Actual gate", marker=dict(size=8, symbol="circle")))
    fig.add_trace(go.Scatter3d(x=[frame.pred_cross_local[0]], y=[frame.pred_cross_local[1]], z=[frame.pred_cross_local[2]],
                               mode="markers", name="Predicted gate", marker=dict(size=8, symbol="x")))

    # highlight actual and predicted sides
    for side, name in [(frame.actual_side, "Actual side"), (frame.pred_side, "Pred side")]:
        pts = face_outline(side, frame.half_lengths)
        fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode="lines", name=name,
                                   line=dict(width=9 if name == "Actual side" else 5, dash="dash" if name == "Pred side" else None)))

    fig.update_layout(
        height=660,
        scene=dict(
            xaxis_title="Right / Left",
            yaxis_title="Up / Down",
            zaxis_title="Forward / Back",
            bgcolor=PLOT_BG,
            xaxis=dict(backgroundcolor=PLOT_BG, gridcolor=GRID, zerolinecolor=GRID),
            yaxis=dict(backgroundcolor=PLOT_BG, gridcolor=GRID, zerolinecolor=GRID),
            zaxis=dict(backgroundcolor=PLOT_BG, gridcolor=GRID, zerolinecolor=GRID),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.5, z=1.2)),
        ),
    )
    fig = industrial_plot(fig, f"LOCAL TRAJECTORY / FRAME {frame.frame}", 660)
    fig.update_layout(
        margin=dict(l=20, r=20, t=112, b=28),
        title=dict(text=f"LOCAL TRAJECTORY / FRAME {frame.frame}", x=0.015, y=0.985,
                   font=dict(size=12, color="#dce2e7")),
        legend=dict(orientation="h", yanchor="top", y=1.08, x=0, font=dict(size=10)),
    )
    return fig


def plot_side_heatmap(frame: FramePrediction, side: str, N: int) -> go.Figure:
    mat = face_matrix(frame, side)
    actual_row, actual_col = divmod(frame.actual_gate, N)
    pred_row, pred_col = divmod(frame.pred_gate, N)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=mat, coloraxis="coloraxis", zmin=0.0, zmax=float(mat.max())))
    if side == frame.actual_side:
        fig.add_trace(go.Scatter(x=[actual_col], y=[actual_row], mode="markers+text", name="Actual",
                                 text=["Actual"], textposition="bottom center", marker=dict(size=14, symbol="circle")))
    if side == frame.pred_side:
        fig.add_trace(go.Scatter(x=[pred_col], y=[pred_row], mode="markers+text", name="Predicted",
                                 text=["Pred"], textposition="top center", marker=dict(size=14, symbol="x")))
    fig.update_layout(coloraxis=dict(colorscale="Viridis", colorbar=dict(title="mass")))
    fig.update_yaxes(autorange="reversed", title="row")
    fig.update_xaxes(title="col")
    return industrial_plot(fig, f"{side.upper()} FACE / GATE MASS", 360)


def plot_side_probabilities(frame: FramePrediction) -> go.Figure:
    masses = dict(frame.side_probs)
    if frame.unresolved_mass > 1e-9:
        masses["unresolved"] = frame.unresolved_mass
    s = pd.Series(masses).sort_values(ascending=False)
    colors = ["#45c486" if side == frame.pred_side else "#5f8ee4" for side in s.index]
    fig = go.Figure(go.Bar(x=s.index.tolist(), y=s.values.tolist(), marker_color=colors))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="score")
    return industrial_plot(fig, "FIRST-PASSAGE SIDE MASS", 260)


def plot_accuracy_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["cum_side_acc"], mode="lines", name="Side accuracy"))
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["cum_exact_acc"], mode="lines", name="Exact side+gate accuracy"))
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["confidence"], mode="lines", name="Predicted joint mass", yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="agreement", tickformat=".0%", range=[0, 1]),
        yaxis2=dict(title="joint mass", overlaying="y", side="right"),
    )
    return industrial_plot(fig, "AGREEMENT AND SCORE TREND", 350)


def plot_earth_track(df_track: pd.DataFrame, metrics_df: pd.DataFrame) -> go.Figure:
    merged = df_track.iloc[:len(metrics_df)].copy()
    merged["exact_hit"] = metrics_df["exact_hit"].values
    merged["confidence"] = metrics_df["confidence"].values
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=merged["lon_true"], lat=merged["lat_true"], mode="lines+markers", name="True track",
        marker=dict(size=5, color=merged["confidence"], colorbar=dict(title="confidence")),
    ))
    miss = merged[merged["exact_hit"] == 0]
    hit = merged[merged["exact_hit"] == 1]
    fig.add_trace(go.Scattergeo(lon=hit["lon_true"], lat=hit["lat_true"], mode="markers", name="Exact hits", marker=dict(size=7)))
    fig.add_trace(go.Scattergeo(lon=miss["lon_true"], lat=miss["lat_true"], mode="markers", name="Misses", marker=dict(size=7, symbol="x")))
    fig.update_geos(showland=True, landcolor="#20262c", oceancolor="#0d171c", bgcolor=PLOT_BG,
                    showcountries=True, countrycolor="#4a545e", showocean=True, projection_type="natural earth")
    return industrial_plot(fig, "EARTH-REFERENCED TRACK / AGREEMENT OVERLAY", 500)


# ------------------------------
# App
# ------------------------------
with st.sidebar:
    st.markdown("<div class='section-title'>Run control</div>", unsafe_allow_html=True)
    out_prefix = st.text_input("Artifact prefix", value="alien_ui", help="Path prefix for generated model and trace artifacts.")
    quick = st.toggle("Quick integration run", value=True, help="On: fast system check. Off: full 48-group performance benchmark.")
    force_cpu = st.toggle("CPU execution", value=True)
    run_label = "Run quick verification" if quick else "Run full benchmark"
    auto_run = st.button(run_label, type="primary", icon=":material/play_arrow:", width="stretch")
    st.caption("Runtime limit: 15 minutes")
    st.markdown("<div class='section-title'>Projection setup</div>", unsafe_allow_html=True)
    lookahead_frames = st.slider("Lookahead", 1, 18, 8, format="%d frames")
    gate_N = st.selectbox("Gate matrix", [4, 5, 6], index=0, format_func=lambda n: f"{n} x {n} per face")
    st.markdown("<div class='section-title'>Method boundary</div>", unsafe_allow_html=True)
    st.caption("Six-face output: UKF sigma-point first-passage mixture with causal future maneuver impulses. Forward-face Transformer probabilities use validation-only temperature scaling.")

if auto_run:
    with st.spinner("Running engine..."):
        proc = run_engine(out_prefix, quick=quick, force_cpu=force_cpu)
    if proc.returncode != 0:
        st.error("Engine run failed")
        with st.expander("Runtime log"):
            st.code(proc.stderr or proc.stdout or "No diagnostic output", language="text")
    else:
        st.toast("Trace generated", icon=":material/check_circle:")

trace = load_trace(out_prefix)
metrics = load_metrics(out_prefix)

trace_path = Path(out_prefix + "_online_trace.npz")
trace_time = "NO TRACE"
if trace_path.exists():
    trace_time = datetime.fromtimestamp(trace_path.stat().st_mtime, tz=timezone.utc).strftime("%H:%M:%S UTC")
run_mode = "QUICK VERIFY" if quick else "FULL TRAIN"
st.markdown(
    f"""
<div class='ops-head'>
  <div><div class='ops-title'>TRAJECTORY INFERENCE CONSOLE <span style='color:#68727c'>/ v6.4P</span></div>
  <div class='ops-sub'>Causal six-face boundary projection and held-out forward-face inference</div></div>
  <div class='status-line'>
    <span class='status ok'>ENGINE ONLINE</span><span class='status info'>CAUSAL PATH</span>
    <span class='status {'warn' if quick else 'ok'}'>{run_mode}</span><span class='status'>TRACE {trace_time}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if trace is None:
    st.warning("No trace is loaded. Generate a trace from Run control.", icon=":material/database:")
    st.stop()

pos_true = np.asarray(trace["pos_true"], dtype=float)
pos_est = np.asarray(trace["pos_est"], dtype=float)
dt = float(trace["dt"][0]) if "dt" in trace else 1.0

frames, dyn_df = analyze_dynamic_side_gate(trace, lookahead_frames=lookahead_frames, N=int(gate_N))
if not frames:
    st.error("The trace is too short for the chosen lookahead. Lower the lookahead frames.")
    st.stop()

lla_true = ecef_to_lla(pos_true)
track_df = pd.DataFrame({
    "lat_true": lla_true["lat"].iloc[:len(dyn_df)],
    "lon_true": lla_true["lon"].iloc[:len(dyn_df)],
    "alt_true_km": lla_true["alt_km"].iloc[:len(dyn_df)],
})

exact_acc = float(dyn_df["exact_hit"].mean())
side_acc = float(dyn_df["side_hit"].mean())
conf_mean = float(dyn_df["confidence"].mean())
finite_trace = bool(np.isfinite(pos_true).all() and np.isfinite(pos_est).all())
gate_rej = metrics.get("dataset", {}).get("gate_rejection_rate", np.nan) if isinstance(metrics, dict) else np.nan
legacy_acc = metrics.get("test", {}).get("test_acc_primary", np.nan) if isinstance(metrics, dict) else np.nan
primary_h = metrics.get("test", {}).get("primary_horizon_steps", "?") if isinstance(metrics, dict) else "?"
baseline_key = f"test_majority_baseline_h{primary_h}"
legacy_baseline = metrics.get("test", {}).get(baseline_key, np.nan) if isinstance(metrics, dict) else np.nan
ece_key = f"test_ece_h{primary_h}"
model_ece = metrics.get("test", {}).get(ece_key, np.nan) if isinstance(metrics, dict) else np.nan

health_class = "good" if finite_trace and (pd.isna(gate_rej) or gate_rej < 0.05) else "bad"
all_horizons = metrics.get("config", {}).get("horizon_steps", []) if isinstance(metrics, dict) else []
model_ready = bool(not quick and all_horizons and all(
    not pd.isna(metrics.get("test", {}).get(f"test_acc_h{h}", np.nan))
    and metrics.get("test", {}).get(f"test_acc_h{h}", -np.inf)
        > metrics.get("test", {}).get(f"test_majority_baseline_h{h}", np.inf)
    and metrics.get("test", {}).get(f"test_ece_h{h}", np.inf) < 0.10
    for h in all_horizons
))
model_class = "good" if model_ready else "warn"
model_value = "N/A" if pd.isna(legacy_acc) else f"{100 * legacy_acc:.1f}%"
model_meta = "no held-out score" if pd.isna(legacy_acc) else (
    f"baseline {100 * legacy_baseline:.1f}% / ECE {100 * model_ece:.1f}%"
    if not pd.isna(legacy_baseline) and not pd.isna(model_ece) else "baseline or calibration unavailable")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='kpi {'good' if side_acc >= .9 else 'warn'}'><div class='label'>Observed side agreement</div><div class='value'>{100*side_acc:.1f}%</div><div class='meta'>{int(dyn_df['side_hit'].sum())} / {len(dyn_df)} evaluated frames</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi {'good' if exact_acc >= .5 else 'warn'}'><div class='label'>Observed side + gate agreement</div><div class='value'>{100*exact_acc:.1f}%</div><div class='meta'>{int(dyn_df['exact_hit'].sum())} exact matches</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><div class='label'>Mean predicted joint mass</div><div class='value'>{100*conf_mean:.1f}%</div><div class='meta'>UKF sigma quadrature</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi {model_class}'><div class='label'>Transformer / H{primary_h}</div><div class='value'>{model_value}</div><div class='meta'>{model_meta}</div></div>", unsafe_allow_html=True)

if quick:
    st.markdown("<div class='callout warn'><b>Verification run:</b> system integration is valid; four-epoch Transformer scores are not release-quality performance evidence.</div>", unsafe_allow_html=True)
if not model_ready:
    st.markdown("<div class='callout warn'><b>MODEL NOT RELEASE READY:</b> every horizon must come from a full run, beat its train-derived majority baseline, and keep ECE below 10%.</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Frame inspection</div>", unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns([2.2, 1, 1])
with fc1:
    frame_idx = st.slider("Timeline", 0, len(frames) - 1, min(10, len(frames) - 1), label_visibility="collapsed")
frame = frames[frame_idx]
with fc2:
    selected_side = st.selectbox("Face", SIDE_NAMES, index=SIDE_NAMES.index(frame.pred_side), label_visibility="collapsed")
with fc3:
    st.markdown(f"<div class='status-line' style='justify-content:flex-start;padding-top:4px'><span class='status {health_class}'>TRACE {'VALID' if finite_trace else 'INVALID'}</span></div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class='decision'><div class='decision-grid'>
  <div class='decision-cell'><div class='label'>Prediction at T + {lookahead_frames * dt:.2f}s</div><div class='decision-main'>{frame.pred_side.upper()} / GATE {frame.pred_gate:02d}</div></div>
  <div class='decision-cell'><div class='label'>Observed outcome</div><div class='decision-main'>{frame.actual_side.upper()} / GATE {frame.actual_gate:02d}</div></div>
  <div class='decision-cell'><div class='label'>Frame result</div><div class='decision-main {'pass' if frame.exact_hit else 'fail'}'>{'EXACT MATCH' if frame.exact_hit else ('SIDE MATCH' if frame.side_hit else 'MISS')}</div></div>
  <div class='decision-cell'><div class='label'>First-passage mass</div><div class='decision-main'>{100 * frame.confidence:.1f}%</div><div class='meta'>maneuver prior {100 * frame.maneuver_mass:.1f}% / unresolved {100 * frame.unresolved_mass:.1f}%</div></div>
</div></div>
""",
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1.65, 1.0])
with c1:
    st.plotly_chart(plot_mission_view(frame, pos_true, pos_est, around=12), width="stretch", config=PLOT_CONFIG)
with c2:
    st.plotly_chart(plot_side_probabilities(frame), width="stretch", config=PLOT_CONFIG)
    st.plotly_chart(plot_side_heatmap(frame, selected_side, N=int(gate_N)), width="stretch", config=PLOT_CONFIG)

st.markdown("<div class='section-title'>Analysis</div>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["Trend", "Confusion", "Geospatial", "Data export"])
with tab1:
    st.plotly_chart(plot_accuracy_timeline(dyn_df), width="stretch", config=PLOT_CONFIG)
    side_counts = dyn_df.groupby(["actual_side", "pred_side"]).size().reset_index(name="count")
    st.dataframe(side_counts, width="stretch", hide_index=True)

with tab2:
    exact_cm, gate_cm = side_gate_confusion(frames, int(gate_N))
    cc1, cc2 = st.columns(2)
    with cc1:
        if not exact_cm.empty:
            exact_fig = industrial_plot(go.Figure(go.Heatmap(z=exact_cm.values, x=exact_cm.columns, y=exact_cm.index, colorscale="Blues")), "SIDE + GATE CONFUSION", 480)
            st.plotly_chart(exact_fig, width="stretch", config=PLOT_CONFIG)
    with cc2:
        if not gate_cm.empty:
            gate_fig = industrial_plot(go.Figure(go.Heatmap(z=gate_cm.values, x=gate_cm.columns, y=gate_cm.index, colorscale="Viridis")), "GATE-ONLY CONFUSION", 480)
            st.plotly_chart(gate_fig, width="stretch", config=PLOT_CONFIG)

with tab3:
    st.plotly_chart(plot_earth_track(track_df, dyn_df), width="stretch", config=PLOT_CONFIG)
    alt_fig = go.Figure()
    alt_fig.add_trace(go.Scatter(x=dyn_df["time_s"], y=track_df["alt_true_km"], mode="lines", name="Altitude km"))
    industrial_plot(alt_fig, "ALTITUDE PROFILE", 300)
    st.plotly_chart(alt_fig, width="stretch", config=PLOT_CONFIG)

with tab4:
    st.dataframe(dyn_df, width="stretch", hide_index=True)
    export = dyn_df.copy()
    st.download_button("Export frame metrics", data=export.to_csv(index=False).encode("utf-8"), file_name="dynamic_side_gate_metrics.csv", mime="text/csv", icon=":material/download:")
