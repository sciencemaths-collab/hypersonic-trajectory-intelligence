#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Informed Trajectory Inference v6.4 Probabilistic (Self-contained bundle)

What you get:
1) Multimodal Transformer: interleaved [feature_t, control_t] tokens + token-type embeddings
2) Long-horizon planning: predicts exit-cell for K horizons AND predicts next-feature (autoregressive head)
3) Numerical stability: robust Square-Root UKF-style update (SPD enforcement + gating + safe inflation)
   - Implemented in vectorized NumPy for speed on CPU
4) Visualization: aesthetic Plotly HTML animation (3D trajectory + exit-cell probability heatmap),
   plus PNG heatmaps and class-over-time plots, and metrics JSON.

Run quick demo:
  python3 alien_exit_cell_predictor_v6_3.py --quick --no-gpu --output-prefix demo

Run bigger:
  python3 alien_exit_cell_predictor_v6_3.py --output-prefix run1

Dependencies:
  numpy, torch, matplotlib, plotly (install via pip if needed)

Notes:
- State x = [pos(3), vel(3), quat(wxyz)(4), omega(3)]  -> 13D
- Measurement z = [pos(3), vel(3)] -> 6D
- Exit-cell labels are computed from a velocity-aligned "virtual box" and its front-face N×N grid (C=N^2 classes).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# =========================
# Config / CLI
# =========================

@dataclass
class Config:
    # simulation
    dt: float = 0.05
    traj_steps: int = 200
    init_altitude_m: float = 200e3
    max_mach: float = 35.0
    maneuver_prob: float = 0.06
    force_std: float = 10_000.0      # N (body frame control)
    torque_std: float = 8_000.0      # N*m (body frame control)

    # environment noise (realistic physics noise)
    wind_accel_std: float = 25.0     # m/s^2 random accel in inertial frame
    gravity_jitter_std: float = 0.03 # relative jitter (3%)
    rho_jitter_std: float = 0.06     # relative jitter
    temp_jitter_std: float = 2.0     # Kelvin jitter

    # measurement noise
    meas_pos_std: float = 35.0       # m
    meas_vel_std: float = 4.0        # m/s
    meas_bias_pos_std: float = 1.5   # m random-walk bias per traj
    meas_bias_vel_std: float = 0.25  # m/s random-walk bias per traj
    meas_bias_rw: float = 0.015      # bias random walk factor

    # UKF
    # NOTE: alpha too small makes Merwe weights explode (c = n + lambda becomes ~0)
    # which can overflow covariance math. Use a moderate alpha for numerical stability.
    ukf_alpha: float = 1.0
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    P_scale: float = 1e2
    Q_scale: float = 2e-1
    R_scale: float = 1.0

    # gating
    gate_threshold: float = 24.0
    gate_inflate: float = 3.0
    P_trace_cap: float = 2e6

    # dataset
    num_trajectories: int = 250
    offline_traj_fraction: float = 0.35
    offline_stride: int = 2
    min_samples: int = 800

    # stress-test / runtime overrides (optional)
    # If set, generate_dataset will use these instead of the default offline trajectory logic.
    # Useful for fast CI-style stability tests.
    offline_traj_min: int = 30           # default behavior preserved
    offline_traj_max: int = 10_000       # safety cap
    offline_traj_override: int = 0       # if >0, use exactly this many trajectories
    min_trajectory_groups: int = 48      # trajectory diversity before early stop

    # box / labels
    box_N: int = 4
    box_min_L: float = 60.0
    box_max_L: float = 900.0
    box_L_scale: float = 0.35
    box_cross_scale: float = 0.025
    box_cross_min_m: float = 25.0
    box_cross_max_m: float = 250.0

    # horizons
    # The forward face is about box_L_scale seconds away at constant speed.
    # Keep every default horizon beyond that crossing time so labels are defined.
    horizon_steps: Tuple[int, ...] = (8, 10, 12)  # 0.4, 0.5, 0.6 s at dt=0.05

    # transformer
    window: int = 10              # number of timesteps in context
    d_model: int = 96
    nhead: int = 4
    nlayers: int = 3
    ff_mult: int = 4
    dropout: float = 0.12

    # training
    epochs: int = 18
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    feature_loss_w: float = 0.25   # weight for autoregressive feature prediction
    horizon_loss_w: float = 1.0

    # runtime
    seed: int = 42
    gpu: bool = True
    quick: bool = False
    output_prefix: str = "v6_3"

    # visualization
    viz_frames: int = 140
    viz_stride: int = 1
    no_viz: bool = False


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name, action="store_true")
    group.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: default})


def parse_tuple_ints(s: str) -> Tuple[int, ...]:
    s = s.strip()
    if not s:
        return tuple()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    # a compact selection of args; advanced users can edit the dataclass defaults
    p.add_argument("--output-prefix", type=str, default="v6_3")
    p.add_argument("--quick", action="store_true", help="Fast demo mode (small data + fewer steps)")
    add_bool_flag(p, "gpu", True)
    add_bool_flag(p, "no_viz", False)
    # Convenience alias (people naturally type "--no-viz")
    p.add_argument("--no-viz", dest="no_viz", action="store_true", help="Alias for --no_viz")

    p.add_argument("--num-trajectories", type=int, default=250)
    p.add_argument("--traj-steps", type=int, default=200)
    p.add_argument("--offline-stride", type=int, default=2)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--horizon-steps", type=parse_tuple_ints, default=(8,10,12))
    p.add_argument("--box-N", type=int, default=4)

    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--nhead", type=int, default=4)

    p.add_argument("--log-level", type=str, default="INFO")
    a = p.parse_args()

    cfg = Config()
    cfg.output_prefix = a.output_prefix
    cfg.quick = bool(a.quick)
    cfg.gpu = bool(a.gpu)
    cfg.no_viz = bool(a.no_viz)

    cfg.num_trajectories = int(a.num_trajectories)
    cfg.traj_steps = int(a.traj_steps)
    cfg.offline_stride = int(a.offline_stride)
    cfg.window = int(a.window)
    cfg.horizon_steps = tuple(a.horizon_steps)
    cfg.box_N = int(a.box_N)

    cfg.epochs = int(a.epochs)
    cfg.batch_size = int(a.batch_size)
    cfg.d_model = int(a.d_model)
    cfg.nlayers = int(a.nlayers)
    cfg.nhead = int(a.nhead)

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s: %(message)s",
        level=getattr(logging, str(a.log_level).upper(), logging.INFO),
    )

    if cfg.quick:
        # fast mode knobs (aim: laptop-friendly and Streamlit-friendly)
        cfg.traj_steps = min(cfg.traj_steps, 90)
        cfg.num_trajectories = min(cfg.num_trajectories, 45)
        cfg.offline_traj_fraction = 0.30
        cfg.min_samples = 200
        cfg.offline_stride = max(cfg.offline_stride, 6)
        cfg.epochs = min(cfg.epochs, 4)
        cfg.window = min(cfg.window, 8)
        cfg.d_model = min(cfg.d_model, 64)
        cfg.nlayers = min(cfg.nlayers, 2)
        cfg.nhead = min(cfg.nhead, 4)
        cfg.viz_frames = 90
        cfg.min_trajectory_groups = 12

    return cfg


def seed_everything(seed: int):
    np.random.seed(seed)
    random = np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return random


# =========================
# Quaternion + Rotation utils (wxyz)
# =========================

def q_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return (q / n).astype(float)


def q_mult(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    # Hamilton product (wxyz)
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ], dtype=float)


def q_to_R(q: np.ndarray) -> np.ndarray:
    # Convert unit quaternion (wxyz) to rotation matrix
    q = q_normalize(q)
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    return np.array([
        [ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz],
    ], dtype=float)


# =========================
# Physics models (fast + noisy)
# =========================

class EarthModel:
    def __init__(self, radius: float = 6_371e3, omega: float = 7.292115e-5,
                 mu: float = 3.986004418e14):
        self.radius = float(radius)
        self.omega = float(omega)
        self.mu = float(mu)

    def gravity(self, pos: np.ndarray, g_jitter: float = 0.0) -> np.ndarray:
        r = float(np.linalg.norm(pos))
        if r < 1e-9:
            return np.zeros(3)
        return -(self.mu / r**3) * (1.0 + g_jitter) * pos

    def rotating_frame_acceleration(self, vel: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Coriolis plus centrifugal acceleration in Earth-fixed coordinates."""
        omega = np.array([0.0, 0.0, self.omega], dtype=float)
        coriolis = -2.0 * np.cross(omega, vel)
        centrifugal = -np.cross(omega, np.cross(omega, pos))
        return coriolis + centrifugal


class Atmosphere:
    # simplified ISA layer model (enough for diversity)
    layers = [
        (0.0,   288.15, -0.0065),
        (11e3,  216.65,  0.0),
        (20e3,  216.65,  0.001),
        (32e3,  228.65,  0.0028),
        (47e3,  270.65,  0.0),
        (51e3,  270.65, -0.0028),
        (71e3,  214.65, -0.002),
    ]
    def __init__(self, Rg: float = 287.05):
        self.Rg = float(Rg)

    def properties(self, h: float) -> Tuple[float, float, float]:
        # Continuous ISA-style integration through 86 km, followed by a
        # documented exponential upper-atmosphere approximation.
        h = float(max(h, 0.0))
        target = min(h, 86e3)
        p_base = 101325.0
        g0 = 9.80665
        T = self.layers[0][1]
        for i, (h0, T0, lapse) in enumerate(self.layers):
            h1 = self.layers[i + 1][0] if i + 1 < len(self.layers) else 86e3
            if target <= h0:
                break
            dh = min(target, h1) - h0
            if abs(lapse) < 1e-12:
                T = T0
                p_top = p_base * np.exp(-g0 * dh / (self.Rg * T0))
            else:
                T = T0 + lapse * dh
                p_top = p_base * (T / T0) ** (-g0 / (lapse * self.Rg))
            if target <= h1:
                p_base = float(p_top)
                break
            p_base = float(p_top)

        if h > 86e3:
            T = 186.0
            rho_86 = p_base / (self.Rg * max(T, 150.0))
            Hs = 7000.0  # explicit approximation, not a full thermosphere model
            rho = float(rho_86 * np.exp(-(h - 86e3) / Hs))
            p = float(rho * self.Rg * T)
            return rho, T, p
        rho = p_base / (self.Rg * max(T, 150.0))
        return float(rho), float(T), float(p_base)

    @staticmethod
    def a_sound(T: float) -> float:
        # Clamp T to avoid absurdly low a_sound that would explode Mach.
        return float(np.sqrt(1.4 * 287.05 * max(T, 150.0)))

    @staticmethod
    def mu_air(T: float) -> float:
        # Sutherland approx
        mu0, T0, S = 1.716e-5, 273.15, 111.0
        T = float(max(T, 1e-6))
        return float(mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S))


class AeroModel:
    def __init__(self, Cd0=0.02, k=0.04, S_ref=1.0, L_char=1.0):
        self.Cd0 = float(Cd0)
        self.k = float(k)
        self.S_ref = float(S_ref)
        self.L_char = float(L_char)

    def Cd(self, Mach: float, Re: float) -> float:
        Re = float(max(Re, 1e-6))
        return float(self.Cd0 + self.k * Mach**2 * (1.0 + 1.0 / (Re**0.2)))


class Projectile6DOF:
    def __init__(self, mass: float, I_body: np.ndarray, earth: EarthModel, atm: Atmosphere, aero: AeroModel):
        self.mass = float(mass)
        self.I = np.array(I_body, dtype=float)
        self.I_inv = np.linalg.inv(self.I)
        self.earth = earth
        self.atm = atm
        self.aero = aero

    def deriv(self, x: np.ndarray, u: np.ndarray, env: Dict[str, float], add_wind: bool = True) -> np.ndarray:
        # x: 13D, u: 6D (body frame force/torque)
        pos = x[:3]
        vel = x[3:6]
        q = q_normalize(x[6:10])
        w = x[10:13]

        # environment jitters
        g_jit = float(env.get("g_jit", 0.0))
        rho_jit = float(env.get("rho_jit", 0.0))
        T_jit = float(env.get("T_jit", 0.0))
        wind_std = float(env.get("wind_std", 0.0))

        # Central gravity plus non-inertial terms for the Earth-fixed frame.
        Fg = self.mass * self.earth.gravity(pos, g_jit)
        Fc = self.mass * self.earth.rotating_frame_acceleration(vel, pos)

        alt = float(np.linalg.norm(pos) - self.earth.radius)
        rho, T, _ = self.atm.properties(alt)
        rho = max(0.0, rho * (1.0 + rho_jit))
        T = max(150.0, T + T_jit)
        a = self.atm.a_sound(T)
        mu = self.atm.mu_air(T)

        Rm = q_to_R(q)
        v_b = Rm.T @ vel
        speed = float(np.linalg.norm(v_b))

        if speed < 1e-9:
            Fd_b = np.zeros(3)
        else:
            Mach = float(np.clip(speed / max(a, 1e-6), 0.0, 80.0))
            Re = float(np.clip(rho * speed * self.aero.L_char / max(mu, 1e-12), 1e-6, 1e12))
            Cd = float(np.clip(self.aero.Cd(Mach, Re), 0.0, 5.0))
            dir_b = v_b / speed
            Fd_b = -0.5 * Cd * rho * speed**2 * dir_b * self.aero.S_ref

        Fd = Rm @ Fd_b
        F_ctrl = Rm @ u[:3]
        Ftot = Fg + Fc + Fd + F_ctrl
        acc = Ftot / self.mass

        if add_wind and wind_std > 0:
            acc = acc + np.asarray(env.get("wind_accel", np.zeros(3)), dtype=float)

        # rotational dynamics
        torque = u[3:] - np.cross(w, self.I @ w)
        w_dot = self.I_inv @ torque

        q_dot = 0.5 * q_mult(q, np.array([0.0, w[0], w[1], w[2]], dtype=float))
        return np.hstack([vel, acc, q_dot, w_dot])

    def step_rk2(self, x: np.ndarray, u: np.ndarray, dt: float, env: Dict[str, float]) -> np.ndarray:
        # RK2 (midpoint) for speed
        k1 = self.deriv(x, u, env)
        xm = x + 0.5 * dt * k1
        k2 = self.deriv(xm, u, env)
        xn = x + dt * k2
        xn[6:10] = q_normalize(xn[6:10])
        xn = np.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0)
        if np.linalg.norm(xn[6:10]) < 1e-9:
            xn[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
        return xn


# =========================
# UKF (robust + gating)
# =========================

def enforce_spd(P: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    P = np.nan_to_num(P, nan=eps, posinf=eps, neginf=eps)
    P = 0.5 * (P + P.T)
    try:
        w, V = np.linalg.eigh(P)
        w = np.clip(w, eps, None)
        return (V * w) @ V.T
    except np.linalg.LinAlgError:
        return np.eye(P.shape[0]) * eps


def sqrt_spd(P: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    # Prefer Cholesky; fallback to eigen sqrt if needed
    P = enforce_spd(P, eps)
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(P)
        w = np.clip(w, eps, None)
        return V @ np.diag(np.sqrt(w))


class MerweSigmaPoints:
    def __init__(self, n: int, alpha: float, beta: float, kappa: float):
        self.n = int(n)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        # Merwe scaled sigma points.
        # Guard against tiny c which creates huge weights and numeric overflow.
        self.lmbda = self.alpha**2 * (self.n + self.kappa) - self.n
        c = self.n + self.lmbda
        if abs(c) < 1e-3:
            # Prefer adjusting alpha rather than letting weights explode.
            # Keep sign of c if negative.
            c = 1e-3 if c >= 0 else -1e-3
        self.Wm = np.full(2*self.n + 1, 1.0/(2*c), dtype=float)
        self.Wc = np.full(2*self.n + 1, 1.0/(2*c), dtype=float)
        self.Wm[0] = self.lmbda / c
        self.Wc[0] = self.lmbda / c + (1.0 - self.alpha**2 + self.beta)
        self.c = c

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        P = enforce_spd(P)
        S = sqrt_spd(self.c * P)
        sig = np.zeros((2*self.n + 1, self.n), dtype=float)
        sig[0] = x
        for i in range(self.n):
            sig[i+1]     = x + S[:, i]
            sig[self.n+i+1] = x - S[:, i]
        return sig


class RobustUKF:
    def __init__(self, dim_x: int, dim_z: int, dt: float, points: MerweSigmaPoints, fx, hx):
        self.n = int(dim_x)
        self.m = int(dim_z)
        self.dt = float(dt)
        self.points = points
        self.fx = fx
        self.hx = hx

        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)
        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)

        self._sigmas_f = None
        self.total_updates = 0
        self.gate_rejections = 0

    def _sanity_clip_state(self):
        """Hard safety rails.

        The measurement does not observe attitude/omega directly. Numerical error can
        occasionally inject absurd angular rates via the cross-covariance term.
        This keeps the filter from exploding and taking the dataset with it.
        """
        self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        # pos (m) and vel (m/s)
        self.x[:3] = np.clip(self.x[:3], -1.0e7, 1.0e7)
        self.x[3:6] = np.clip(self.x[3:6], -5.0e4, 5.0e4)
        # omega (rad/s)
        self.x[10:13] = np.clip(self.x[10:13], -500.0, 500.0)
        # keep quaternion normalized
        self.x[6:10] = q_normalize(self.x[6:10])

    def predict(self):
        sigmas = self.points.sigma_points(self.x, self.P)
        # propagate each sigma (vectorized loop; n=13 so ok)
        sig_f = np.zeros_like(sigmas)
        for i in range(sigmas.shape[0]):
            sig_f[i] = self.fx(sigmas[i], self.dt)
        self._sigmas_f = sig_f

        Wm, Wc = self.points.Wm, self.points.Wc
        x_pred = np.sum(sig_f * Wm[:, None], axis=0)

        # covariance
        dX = sig_f - x_pred[None, :]
        P_pred = (dX.T * Wc) @ dX + self.Q
        self.x = x_pred
        self.P = enforce_spd(P_pred)
        self._sanity_clip_state()

    def update(self, z: np.ndarray, gate_threshold: float, gate_inflate: float, P_trace_cap: float):
        z = np.asarray(z, dtype=float)
        sig_f = self._sigmas_f
        if sig_f is None:
            # Allow update() at t=0 or after cache reset without a prior predict()
            sig_f = self.points.sigma_points(self.x, self.P)
            self._sigmas_f = sig_f

        # transform sigma points to measurement space
        Zsig = np.zeros((sig_f.shape[0], self.m), dtype=float)
        for i in range(sig_f.shape[0]):
            Zsig[i] = self.hx(sig_f[i])

        Wm, Wc = self.points.Wm, self.points.Wc
        z_pred = np.sum(Zsig * Wm[:, None], axis=0)

        dZ = Zsig - z_pred[None, :]
        S = (dZ.T * Wc) @ dZ + self.R
        S = enforce_spd(S)

        y = z - z_pred

        # gating using Mahalanobis distance via Cholesky
        try:
            L = np.linalg.cholesky(S)
            # solve L * v = y
            v = np.linalg.solve(L, y)
            d2 = float(v @ v)
        except np.linalg.LinAlgError:
            d2 = float(1e12)

        self.total_updates += 1
        if d2 > gate_threshold:
            self.gate_rejections += 1
            # safe inflation using Q (n×n), not R (m×m)
            self.P = enforce_spd(self.P + gate_inflate * self.Q)
            if float(np.trace(self.P)) > P_trace_cap or not np.isfinite(np.trace(self.P)):
                self.P = np.eye(self.n) * 1e2
            return False  # rejected

        # cross covariance Pxz
        dX = sig_f - self.x[None, :]
        Pxz = (dX.T * Wc) @ dZ

        # Kalman gain K = Pxz S^-1 (use solve)
        try:
            K = np.linalg.solve(S.T, Pxz.T).T  # Pxz @ inv(S)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(S)

        self.x = self.x + K @ y
        self.P = enforce_spd(self.P - K @ S @ K.T)

        self._sanity_clip_state()

        if float(np.trace(self.P)) > P_trace_cap or not np.isfinite(np.trace(self.P)):
            self.P = np.eye(self.n) * 1e2

        return True


# =========================
# Virtual Box exit cell
# =========================

class VirtualBox:
    def __init__(self, center: np.ndarray, direction: np.ndarray, up: np.ndarray,
                 L: float, N: int, half_width: float | None = None):
        self.center = np.asarray(center, dtype=float)
        v = np.asarray(direction, dtype=float)
        self.v = v / (np.linalg.norm(v) + 1e-12)
        up = np.asarray(up, dtype=float)
        u = np.cross(up, self.v)
        if np.linalg.norm(u) < 1e-9:
            self.valid = False
            return
        self.u = u / np.linalg.norm(u)
        self.w = np.cross(self.v, self.u)
        self.L = float(L)
        self.half_width = float(L if half_width is None else half_width)
        self.N = int(N)
        self.valid = True

    def compute_exit_cell(self, p: np.ndarray) -> Optional[Tuple[int, int]]:
        if not self.valid:
            return None
        p = np.asarray(p, dtype=float)
        d = float(np.dot(p - self.center, self.v))
        if d <= self.L:
            return None
        face_center = self.center + self.v * self.L
        # intersection point on the plane scaled along ray from center
        inter = self.center + (p - self.center) * (self.L / max(d, 1e-12))
        off = inter - face_center
        # normalize to [-1, 1] across face using L as half-width/height
        s = float(np.dot(off, self.u) / max(self.half_width, 1e-12))
        t = float(np.dot(off, self.w) / max(self.half_width, 1e-12))
        if not (np.isfinite(s) and np.isfinite(t)):
            return None
        i = int(np.clip((s + 1) * self.N / 2, 0, self.N - 1))
        j = int(np.clip((t + 1) * self.N / 2, 0, self.N - 1))
        return i, j


# =========================
# Feature extraction
# =========================

def extract_features(x: np.ndarray, P: np.ndarray | float, earth: EarthModel, atm: Atmosphere, env: Dict[str, float]) -> np.ndarray:
    # features from estimate state
    pos = x[:3]
    vel = x[3:6]
    q = q_normalize(x[6:10])
    w = x[10:13]

    alt = float(np.linalg.norm(pos) - earth.radius)

    # Features use the nominal atmosphere. Simulator-only perturbations are
    # intentionally excluded because they are not observable online.
    rho, T, _ = atm.properties(alt)
    a = atm.a_sound(T)

    speed = float(np.linalg.norm(vel))
    Mach = float(speed / max(a, 1e-6))
    dp = float(0.5 * rho * speed * speed)
    cov_tr = float(P if np.ndim(P) == 0 else np.trace(P))

    Rm = q_to_R(q)
    vhat = vel / (speed + 1e-12)
    rhat = pos / (np.linalg.norm(pos) + 1e-12)
    angles = np.array([np.arccos(np.clip(vhat @ Rm[:, i], -1.0, 1.0)) for i in range(3)], dtype=float)

    feat = np.hstack([
        speed,
        alt,
        Mach,
        dp,
        angles,        # 3
        rhat,          # 3: gravity/curvature direction in ECEF
        vhat,          # 3: trajectory direction in ECEF
        q,             # 4
        w,             # 3
        cov_tr
    ]).astype(np.float32)

    return np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def feature_dim() -> int:
    return 21


def control_features(u: np.ndarray) -> np.ndarray:
    # control token features (6D)
    u = np.asarray(u, dtype=float)
    return np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# =========================
# Dataset generation
# =========================

def init_state(cfg: Config, earth: EarthModel) -> np.ndarray:
    # place at +altitude along Y
    pos = np.array([0.0, earth.radius + cfg.init_altitude_m, 0.0], dtype=float)

    # velocity from Mach
    mach = np.random.uniform(0.3, cfg.max_mach)
    speed = mach * Atmosphere.a_sound(288.15)

    theta = np.random.uniform(0.0, np.pi / 7)     # elevation
    phi = np.random.uniform(0.0, 2.0 * np.pi)     # azimuth

    vel = np.array([
        speed * np.cos(phi) * np.cos(theta),
        speed * np.sin(phi) * np.cos(theta),
        speed * np.sin(theta),
    ], dtype=float)

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    w = np.zeros(3, dtype=float)

    return np.hstack([pos, vel, q, w])


def make_env(cfg: Config) -> Dict[str, float]:
    return {
        "wind_std": float(cfg.wind_accel_std),
        "wind_accel": np.random.randn(3) * float(cfg.wind_accel_std),
        "g_jit": float(np.random.randn() * cfg.gravity_jitter_std),
        "rho_jit": float(np.random.randn() * cfg.rho_jitter_std),
        "T_jit": float(np.random.randn() * cfg.temp_jitter_std),
    }


def nominal_filter_env() -> Dict[str, float]:
    """Environment available to estimation and projection without simulator truth."""
    return {
        "wind_std": 0.0,
        "wind_accel": np.zeros(3),
        "g_jit": 0.0,
        "rho_jit": 0.0,
        "T_jit": 0.0,
    }


def make_controls(cfg: Config, steps: int) -> np.ndarray:
    U = np.zeros((steps, 6), dtype=float)
    for t in range(steps):
        if np.random.rand() < cfg.maneuver_prob:
            U[t, :3] = np.random.randn(3) * cfg.force_std
            U[t, 3:] = np.random.randn(3) * cfg.torque_std
    return U


def rollout_truth(cfg: Config, proj: Projectile6DOF, x0: np.ndarray, U: np.ndarray, env: Dict[str, float]) -> np.ndarray:
    X = np.zeros((len(U) + 1, 13), dtype=float)
    X[0] = x0
    x = x0.copy()
    for t in range(len(U)):
        x = proj.step_rk2(x, U[t], cfg.dt, env)
        X[t + 1] = x
    return X


def make_measurements(cfg: Config, X_truth: np.ndarray) -> np.ndarray:
    # z=[pos,vel] + noise + slow bias drift per trajectory
    T = X_truth.shape[0]
    Z = np.zeros((T, 6), dtype=float)

    bpos = np.random.randn(3) * cfg.meas_bias_pos_std
    bvel = np.random.randn(3) * cfg.meas_bias_vel_std

    for t in range(T):
        bpos = (1.0 - cfg.meas_bias_rw) * bpos + cfg.meas_bias_rw * (np.random.randn(3) * cfg.meas_bias_pos_std)
        bvel = (1.0 - cfg.meas_bias_rw) * bvel + cfg.meas_bias_rw * (np.random.randn(3) * cfg.meas_bias_vel_std)

        z = np.hstack([X_truth[t, :3], X_truth[t, 3:6]])
        z[:3] += bpos + np.random.randn(3) * cfg.meas_pos_std
        z[3:6] += bvel + np.random.randn(3) * cfg.meas_vel_std
        Z[t] = z
    return Z


def compute_box_L(cfg: Config, speed: float) -> float:
    L = cfg.box_L_scale * speed
    L = float(np.clip(L, cfg.box_min_L, cfg.box_max_L))
    return L


def compute_box_cross_width(cfg: Config, speed: float) -> float:
    return float(np.clip(cfg.box_cross_scale * speed,
                         cfg.box_cross_min_m, cfg.box_cross_max_m))


def compute_labels(cfg: Config, X_est: np.ndarray, X_truth: np.ndarray) -> np.ndarray:
    # returns labels[t, h] for horizons h in cfg.horizon_steps, with -1 if undefined
    steps = X_est.shape[0] - 1
    Hs = list(cfg.horizon_steps)
    C = cfg.box_N * cfg.box_N
    labels = -np.ones((steps, len(Hs)), dtype=int)

    for t in range(steps):
        pos = X_est[t, :3]
        vel = X_est[t, 3:6]
        speed = float(np.linalg.norm(vel))
        if speed < 1e-6:
            continue
        L = compute_box_L(cfg, speed)
        half_width = compute_box_cross_width(cfg, speed)
        vb = VirtualBox(pos, vel, np.array([0.0, 0.0, 1.0]), L, cfg.box_N,
                        half_width=half_width)
        for hi, h in enumerate(Hs):
            tp = t + h
            if tp >= X_truth.shape[0]:
                continue
            cell = vb.compute_exit_cell(X_truth[tp, :3])
            if cell is None:
                continue
            labels[t, hi] = cell[0] * cfg.box_N + cell[1]
    return labels


def run_ukf(cfg: Config, proj: Projectile6DOF, earth: EarthModel, atm: Atmosphere,
            Z: np.ndarray, U: np.ndarray, env: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    # Returns state means, full covariance history, and gating statistics.
    pts = MerweSigmaPoints(13, cfg.ukf_alpha, cfg.ukf_beta, cfg.ukf_kappa)

    # fx uses control; we close over current u
    u_cur = np.zeros(6, dtype=float)

    # Hidden simulator disturbances must not be exposed to the estimator.
    filter_env = nominal_filter_env()

    def fx(x, dt):
        return proj.step_rk2(x, u_cur, dt, filter_env)

    def hx(x):
        return x[:6]

    ukf = RobustUKF(13, 6, cfg.dt, pts, fx=fx, hx=hx)

    # init near first measurement
    ukf.x = np.zeros(13, dtype=float)
    ukf.x[:6] = Z[0]
    ukf.x[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
    ukf.x[10:13] = 0.0

    # Prior covariance in the correct units (position ~ meters, omega ~ rad/s).
    P0 = np.diag([
        80.0**2, 80.0**2, 80.0**2,        # pos
        15.0**2, 15.0**2, 15.0**2,        # vel
        0.2**2, 0.2**2, 0.2**2, 0.2**2,   # quat
        3.0**2, 3.0**2, 3.0**2            # omega
    ]).astype(float)
    ukf.P = P0 * float(cfg.P_scale / 1e2)  # keep legacy scaling roughly consistent
    # Process/measurement covariances should be in the *right units*.
    # Using identity here makes gating behave wildly because position is ~6e6 m.
    Q = np.diag([
        5.0**2, 5.0**2, 5.0**2,          # pos (m)
        15.0**2, 15.0**2, 15.0**2,       # vel (m/s)
        1e-3, 1e-3, 1e-3, 1e-3,          # quat
        1.0**2, 1.0**2, 1.0**2           # omega (rad/s)
    ]).astype(float)
    R = np.diag([
        cfg.meas_pos_std**2, cfg.meas_pos_std**2, cfg.meas_pos_std**2,
        cfg.meas_vel_std**2, cfg.meas_vel_std**2, cfg.meas_vel_std**2,
    ]).astype(float)
    ukf.Q = Q * float(cfg.Q_scale)
    ukf.R = R * float(cfg.R_scale)

    X_est = np.zeros((Z.shape[0], 13), dtype=float)
    P_hist = np.zeros((Z.shape[0], 13, 13), dtype=float)

    # Initialize sigma cache for the very first update (t=0) using the prior.
    # This avoids requiring predict() before the first update.
    ukf._sigmas_f = pts.sigma_points(ukf.x, ukf.P)

    for t in range(Z.shape[0]):
        if t > 0:
            u_cur = U[t-1].copy()
            ukf.predict()
        accepted = ukf.update(Z[t], cfg.gate_threshold, cfg.gate_inflate, cfg.P_trace_cap)
        X_est[t] = ukf.x
        P_hist[t] = ukf.P

    stats = {
        "total_updates": int(ukf.total_updates),
        "gate_rejections": int(ukf.gate_rejections),
    }
    return X_est, P_hist, stats


def build_windows(cfg: Config, feats: np.ndarray, ctrls: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    feats: (T, F)
    ctrls: (T-1, 6) controls applied between t and t+1 (we align ctrl_t with feat_t for token stream)
    labels: (T-1, H) labels at each t for H horizons

    Returns:
      X_seq: (N, Ltok, D) where tokens interleave [feat, ctrl] over window
      y_cls: (N, H) classification labels for each horizon
      y_nextfeat: (N, F) the next feature target for autoregressive head
    """
    W = cfg.window
    H = labels.shape[1]
    Fdim = feats.shape[1]
    Cdim = ctrls.shape[1]
    # token length = 2*W (feat+ctrl per step), and each token dim is max(Fdim, Cdim) via separate linear embeddings later
    # For simplicity, we store as two separate arrays and combine in model; but for dataset, store (2W, Draw) where Draw=Fdim+Cdim with padding.
    # We'll create raw tokens as concatenation: token_feat = [feat, zeros(Cdim)], token_ctrl=[zeros(Fdim), ctrl]
    Draw = Fdim + Cdim

    X_list, y_list, nf_list = [], [], []
    # t runs over labels index (0..T-2)
    for t in range(W-1, labels.shape[0]):
        # ensure all horizons defined (optional: allow partial; here require at least one defined)
        y = labels[t]
        if np.all(y < 0):
            continue

        # build window ending at t (inclusive) using feats indices [t-W+1..t]
        fwin = feats[t-W+1:t+1]               # (W, F)
        cwin = ctrls[t-W+1:t+1]               # (W, 6)  (ctrl aligned with same t for token stream)
        tok = np.zeros((2*W, Draw), dtype=np.float32)
        tok[0::2, :Fdim] = fwin
        tok[1::2, Fdim:] = cwin

        # next feature target (t+1) if exists
        if t+1 < feats.shape[0]:
            next_f = feats[t+1]
        else:
            next_f = feats[t]

        X_list.append(tok)
        y_list.append(y.copy())
        nf_list.append(next_f.copy())

    if len(X_list) == 0:
        return np.zeros((0, 2*W, Draw), dtype=np.float32), np.zeros((0, H), dtype=np.int64), np.zeros((0, Fdim), dtype=np.float32)

    return np.stack(X_list), np.stack(y_list).astype(np.int64), np.stack(nf_list).astype(np.float32)


def generate_dataset(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    earth, atm, aero = EarthModel(), Atmosphere(), AeroModel()
    I = np.diag([500.0, 800.0, 1000.0])
    proj = Projectile6DOF(1000.0, I, earth, atm, aero)

    # Determine how many trajectories to generate.
    # Default preserves legacy behavior: max(cfg.offline_traj_min, int(num_trajectories * offline_traj_fraction))
    # Optional override enables fast stress tests.
    if int(getattr(cfg, "offline_traj_override", 0)) > 0:
        n_traj = int(cfg.offline_traj_override)
    else:
        n_traj = max(int(getattr(cfg, "offline_traj_min", 30)),
                     int(cfg.num_trajectories * cfg.offline_traj_fraction))
    n_traj = int(min(n_traj, int(getattr(cfg, "offline_traj_max", 10_000))))

    stride = max(1, int(cfg.offline_stride))
    H = len(cfg.horizon_steps)

    groups = []
    class_counts = np.zeros(cfg.box_N*cfg.box_N, dtype=int)

    gate_total, gate_rej = 0, 0

    for _ in range(n_traj):
        env = make_env(cfg)
        x0 = init_state(cfg, earth)
        U = make_controls(cfg, cfg.traj_steps)
        X_truth = rollout_truth(cfg, proj, x0, U, env)
        Z = make_measurements(cfg, X_truth)

        X_est, P_tr, gate_stats = run_ukf(cfg, proj, earth, atm, Z, U, env)
        gate_total += gate_stats["total_updates"]
        gate_rej += gate_stats["gate_rejections"]

        # build features at each time in X_est
        feats = np.zeros((X_est.shape[0], feature_dim()), dtype=np.float32)
        for t in range(X_est.shape[0]):
            feats[t] = extract_features(X_est[t], P_tr[t], earth, atm, env)

        # labels computed at each t (0..T-2)
        labels = compute_labels(cfg, X_est, X_truth)

        # subsample with stride (reduce redundancy)
        feats_s = feats[::stride]
        # controls: align ctrl token with same t index; make ctrls length match feats_s
        # We'll sample ctrls from U; for ctrl at time t corresponds to U[t] (between t and t+1)
        # When subsampling, we take U indices also at stride.
        U_pad = np.vstack([U, np.zeros((1,6), dtype=float)])  # length traj_steps+1
        ctrls_s = U_pad[::stride]
        labels_s = labels[::stride]

        X_seq, y_cls, y_nf = build_windows(cfg, feats_s, ctrls_s, labels_s)

        if X_seq.shape[0] == 0:
            continue

        # update class counts using first horizon only (rough)
        for yi in y_cls[:, 0]:
            if yi >= 0:
                class_counts[int(yi)] += 1

        groups.append((X_seq, y_cls, y_nf))

        if len(groups) >= cfg.min_trajectory_groups and sum(len(g[0]) for g in groups) >= cfg.min_samples:
            break

    if len(groups) < 3:
        raise RuntimeError("No samples generated. Try increasing trajectories or lowering thresholds.")

    order = np.random.permutation(len(groups))
    n_g = len(order)
    n_tr_g = max(1, int(0.70 * n_g))
    n_val_g = max(1, int(0.15 * n_g))
    if n_tr_g + n_val_g >= n_g:
        n_tr_g, n_val_g = n_g - 2, 1

    def join(ix):
        chosen = [groups[int(i)] for i in ix]
        return tuple(np.concatenate([g[k] for g in chosen], axis=0) for k in range(3))

    tr = join(order[:n_tr_g])
    va = join(order[n_tr_g:n_tr_g+n_val_g])
    te = join(order[n_tr_g+n_val_g:])
    n = sum(len(g[0]) for g in groups)

    split_valid = {}
    split_diversity = {}
    for split_name, split in (("train", tr), ("validation", va), ("test", te)):
        counts = np.sum(split[1] >= 0, axis=0).astype(int)
        split_valid[split_name] = counts.tolist()
        empty = [cfg.horizon_steps[i] for i, count in enumerate(counts) if count == 0]
        if empty:
            raise RuntimeError(
                f"{split_name} split has no valid labels for horizon steps {empty}. "
                "Choose horizons beyond the virtual-box crossing time or generate more trajectories.")
        split_diversity[split_name] = [
            int(np.unique(split[1][split[1][:, i] >= 0, i]).size)
            for i in range(len(cfg.horizon_steps))
        ]

    stats = {
        "samples": float(n),
        "gate_total": float(gate_total),
        "gate_rejections": float(gate_rej),
        "gate_rejection_rate": float(gate_rej / max(gate_total, 1)),
        "trajectory_groups": float(n_g),
        "split_by_trajectory": 1.0,
        "valid_labels_by_split": split_valid,
        "unique_classes_by_split": split_diversity,
    }

    return (*tr, *va, *te, stats)


# =========================
# Multimodal Transformer
# =========================

class MultiModalTransformer(nn.Module):
    """
    Input: raw tokens (B, L, Draw) where each token is either [feat, 0] or [0, ctrl]
    We embed with two separate linear layers (feat and ctrl portions) then add token-type embedding.
    Output:
      - logits for each horizon: (B, H, C)
      - next_feature prediction: (B, F)
    """
    def __init__(self, Fdim: int, Cdim: int, H: int, C: int, d_model: int, nhead: int, nlayers: int, ff_mult: int, dropout: float):
        super().__init__()
        self.Fdim = Fdim
        self.Cdim = Cdim
        self.Draw = Fdim + Cdim
        self.H = H
        self.C = C
        self.d_model = d_model

        self.feat_embed = nn.Linear(Fdim, d_model)
        self.ctrl_embed = nn.Linear(Cdim, d_model)

        self.type_embed = nn.Embedding(2, d_model)  # 0=feat token, 1=ctrl token
        pe = torch.zeros(512, d_model)
        position = torch.arange(512, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=nlayers, enable_nested_tensor=False)

        self.norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, H * C)
        self.nextfeat_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, Fdim),
        )

    def forward(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_raw: (B, L, Draw)
        Fdim = self.Fdim
        feat_part = x_raw[:, :, :Fdim]
        ctrl_part = x_raw[:, :, Fdim:]

        # token type: feat tokens are even indices, ctrl tokens are odd indices
        L = x_raw.shape[1]
        idx = torch.arange(L, device=x_raw.device)
        tok_type = (idx % 2).long()  # 0/1
        tok_type = tok_type.unsqueeze(0).expand(x_raw.shape[0], -1)  # (B, L)

        # embed each token
        # For feat tokens, ctrl_part is zero; for ctrl tokens, feat_part is zero by construction.
        h = self.feat_embed(feat_part) + self.ctrl_embed(ctrl_part)
        h = h + self.type_embed(tok_type) + self.pos_embed[:, :L, :]

        h = self.encoder(h)
        h_last = self.norm(h[:, -1, :])

        logits = self.cls_head(h_last).view(-1, self.H, self.C)
        next_feat = self.nextfeat_head(h_last)
        return logits, next_feat


def compute_class_weights(y: np.ndarray, C: int) -> np.ndarray:
    counts = np.bincount(y[y >= 0], minlength=C).astype(np.float64)
    # effective number of samples trick (helps heavy imbalance)
    beta = 0.999
    eff = (1 - np.power(beta, counts)) / (1 - beta + 1e-12)
    w = np.zeros(C, dtype=np.float64)
    seen = counts > 0
    # Square-root weighting avoids the instability of combining extreme inverse
    # weights with naturally imbalanced trajectory data.
    w[seen] = 1.0 / np.sqrt(eff[seen] + 1e-12)
    w[seen] = w[seen] / np.mean(w[seen])
    return w.astype(np.float32)


def train_model(cfg: Config, device: torch.device, X_tr, y_tr, nf_tr, X_val, y_val, nf_val, stats: Dict[str, float]) -> Tuple[MultiModalTransformer, Dict[str, float]]:
    C = cfg.box_N * cfg.box_N
    H = len(cfg.horizon_steps)
    Draw = X_tr.shape[-1]
    # infer Fdim/Cdim
    # Draw = Fdim + Cdim with Cdim=6 fixed
    Cdim = 6
    Fdim = Draw - Cdim

    model = MultiModalTransformer(
        Fdim=Fdim, Cdim=Cdim, H=H, C=C,
        d_model=cfg.d_model, nhead=cfg.nhead, nlayers=cfg.nlayers,
        ff_mult=cfg.ff_mult, dropout=cfg.dropout
    ).to(device)

    # data loaders
    Xtr_t = torch.from_numpy(X_tr).float()
    ytr_t = torch.from_numpy(y_tr).long()
    nftr_t = torch.from_numpy(nf_tr).float()

    Xv_t = torch.from_numpy(X_val).float()
    yv_t = torch.from_numpy(y_val).long()
    nfv_t = torch.from_numpy(nf_val).float()

    ds_tr = TensorDataset(Xtr_t, ytr_t, nftr_t)
    ds_val = TensorDataset(Xv_t, yv_t, nfv_t)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cls_w = compute_class_weights(y_tr, C)
    cls_w_t = torch.from_numpy(cls_w).to(device)

    best = float("inf")
    best_state = None
    hist = {"train_loss": [], "val_loss": []}

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb, nfb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            nfb = nfb.to(device)

            logits, nextf = model(xb)
            # horizon classification loss (sum over horizons)
            loss_cls = 0.0
            for hi in range(H):
                mask = yb[:, hi] >= 0
                if mask.any():
                    loss_cls = loss_cls + F.cross_entropy(
                        logits[mask, hi, :], yb[mask, hi], weight=cls_w_t,
                        label_smoothing=cfg.label_smoothing)
            loss_cls = loss_cls / H

            loss_nf = F.mse_loss(nextf, nfb)

            loss = cfg.horizon_loss_w * loss_cls + cfg.feature_loss_w * loss_nf

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += float(loss.item()) * xb.size(0)
            tr_n += xb.size(0)

        tr_loss = tr_loss / max(tr_n, 1)

        # val
        model.eval()
        vl_loss = 0.0
        vl_n = 0
        with torch.no_grad():
            for xb, yb, nfb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                nfb = nfb.to(device)
                logits, nextf = model(xb)
                loss_cls = 0.0
                for hi in range(H):
                    mask = yb[:, hi] >= 0
                    if mask.any():
                        loss_cls = loss_cls + F.cross_entropy(
                            logits[mask, hi, :], yb[mask, hi], weight=cls_w_t)
                loss_cls = loss_cls / H
                loss_nf = F.mse_loss(nextf, nfb)
                loss = cfg.horizon_loss_w * loss_cls + cfg.feature_loss_w * loss_nf
                vl_loss += float(loss.item()) * xb.size(0)
                vl_n += xb.size(0)
        vl_loss = vl_loss / max(vl_n, 1)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(vl_loss)

        logging.info(f"Epoch {ep:03d}/{cfg.epochs}  train={tr_loss:.4f}  val={vl_loss:.4f}")

        if vl_loss < best - 1e-6:
            best = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out = dict(stats)
    out.update({
        "best_val_loss": float(best),
        "epochs": int(cfg.epochs),
        "train_loss_last": float(hist["train_loss"][-1]) if hist["train_loss"] else None,
        "val_loss_last": float(hist["val_loss"][-1]) if hist["val_loss"] else None,
    })
    return model, out


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Top-label expected calibration error with fixed-width confidence bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def eval_model(cfg: Config, model: MultiModalTransformer, device: torch.device,
               X_te, y_te, y_train=None, temperatures=None) -> Dict[str, float]:
    C = cfg.box_N * cfg.box_N
    H = len(cfg.horizon_steps)

    Xt = torch.from_numpy(X_te).float().to(device)
    yt = torch.from_numpy(y_te).long().to(device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(Xt)
        # per-horizon metrics
        metrics = {}
        for hi in range(H):
            mask = yt[:, hi] >= 0
            if not mask.any():
                for name in ("acc", "top3", "nll", "brier", "ece", "balanced_acc", "majority_baseline"):
                    metrics[f"test_{name}_h{cfg.horizon_steps[hi]}"] = float("nan")
                continue
            temperature = float(temperatures[hi]) if temperatures is not None else 1.0
            valid_logits = logits[mask, hi, :] / max(temperature, 1e-6)
            valid_y = yt[mask, hi]
            probs = F.softmax(valid_logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            acc = float((pred == valid_y).float().mean().item())
            # top3
            topk = torch.topk(probs, k=min(3, C), dim=-1).indices
            top3 = float((topk == valid_y.unsqueeze(-1)).any(dim=-1).float().mean().item())
            nll = float(F.nll_loss(torch.log(probs.clamp_min(1e-12)), valid_y).item())
            one_hot = F.one_hot(valid_y, num_classes=C).float()
            brier = float(torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1)).item())
            conf, _ = torch.max(probs, dim=-1)
            ece = expected_calibration_error(
                conf.cpu().numpy(), (pred == valid_y).float().cpu().numpy())
            class_recalls = []
            for cls in torch.unique(valid_y):
                cls_mask = valid_y == cls
                class_recalls.append(float((pred[cls_mask] == valid_y[cls_mask]).float().mean().item()))
            balanced_acc = float(np.mean(class_recalls)) if class_recalls else float("nan")
            train_h = y_train[:, hi] if y_train is not None else np.empty(0, dtype=int)
            train_h = train_h[train_h >= 0]
            majority = int(np.bincount(train_h, minlength=C).argmax()) if train_h.size else -1
            majority_acc = float((valid_y == majority).float().mean().item()) if majority >= 0 else float("nan")
            metrics[f"test_acc_h{cfg.horizon_steps[hi]}"] = acc
            metrics[f"test_top3_h{cfg.horizon_steps[hi]}"] = top3
            metrics[f"test_nll_h{cfg.horizon_steps[hi]}"] = nll
            metrics[f"test_brier_h{cfg.horizon_steps[hi]}"] = brier
            metrics[f"test_ece_h{cfg.horizon_steps[hi]}"] = ece
            metrics[f"test_balanced_acc_h{cfg.horizon_steps[hi]}"] = balanced_acc
            metrics[f"test_majority_baseline_h{cfg.horizon_steps[hi]}"] = majority_acc

        # overall first-horizon
        metrics["test_acc_primary"] = metrics.get(f"test_acc_h{cfg.horizon_steps[0]}", float("nan"))
        metrics["primary_horizon_steps"] = int(cfg.horizon_steps[0])
        if temperatures is not None:
            metrics["temperature_by_horizon"] = {
                str(h): float(temperatures[i]) for i, h in enumerate(cfg.horizon_steps)}
    return metrics


def fit_temperature_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a validation-only temperature with NLL and ECE acceptance gates."""
    if labels.numel() == 0:
        return 1.0
    candidates = torch.unique(torch.cat([
        torch.logspace(np.log10(0.25), np.log10(4.0), 81,
                       device=logits.device),
        torch.ones(1, device=logits.device),
    ])).sort().values
    losses = torch.stack([F.cross_entropy(logits / t, labels) for t in candidates])
    base_probs = F.softmax(logits, dim=-1)
    base_pred = torch.argmax(base_probs, dim=-1)
    base_ece = expected_calibration_error(
        torch.max(base_probs, dim=-1).values.cpu().numpy(),
        (base_pred == labels).float().cpu().numpy())
    base_nll = float(F.cross_entropy(logits, labels).item())

    # Deploy a non-identity fit only when both validation criteria support it.
    accepted = []
    for i, temperature in enumerate(candidates):
        probs = F.softmax(logits / temperature, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        ece = expected_calibration_error(
            torch.max(probs, dim=-1).values.cpu().numpy(),
            (pred == labels).float().cpu().numpy())
        nll = float(losses[i].item())
        if nll < base_nll - 1e-4 and ece <= base_ece + 1e-4:
            accepted.append((nll, abs(float(torch.log(temperature).item())), i))
    if not accepted:
        return 1.0
    _, _, best_i = min(accepted)
    return float(candidates[best_i].item())


def fit_model_temperatures(cfg: Config, model: MultiModalTransformer,
                           device: torch.device, X_val, y_val) -> np.ndarray:
    xv = torch.from_numpy(X_val).float().to(device)
    yv = torch.from_numpy(y_val).long().to(device)
    model.eval()
    temperatures = []
    with torch.no_grad():
        logits, _ = model(xv)
        for hi in range(len(cfg.horizon_steps)):
            mask = yv[:, hi] >= 0
            temperatures.append(fit_temperature_from_logits(
                logits[mask, hi, :], yv[mask, hi]))
    return np.asarray(temperatures, dtype=np.float32)


# =========================
# Online run + visualization
# =========================

def build_causal_token_buffer(feats: np.ndarray, ctrls: np.ndarray, t: int, window: int) -> np.ndarray:
    """Build the same interleaved training window using only samples at or before t."""
    if t < window - 1:
        raise ValueError("A complete causal window is not available yet")
    if t >= len(feats):
        raise IndexError("t is outside the feature sequence")
    fdim = feats.shape[1]
    cdim = ctrls.shape[1]
    start = t - window + 1
    tok = np.zeros((2 * window, fdim + cdim), dtype=np.float32)
    tok[0::2, :fdim] = feats[start:t + 1]
    tok[1::2, fdim:] = ctrls[start:t + 1]
    return tok


def project_state_horizons(proj: Projectile6DOF, x_est: np.ndarray, u_current: np.ndarray,
                           dt: float, env: Dict[str, float], horizons: int) -> np.ndarray:
    """Causal 6-DOF rollout: apply current impulse once, then zero-mean control."""
    states = np.zeros((horizons + 1, len(x_est)), dtype=float)
    states[0] = np.asarray(x_est, dtype=float)
    x = states[0].copy()
    for h in range(1, horizons + 1):
        u = u_current if h == 1 else np.zeros_like(u_current)
        x = proj.step_rk2(x, u, dt, env)
        states[h] = x
    return states


def project_sigma_horizons(proj: Projectile6DOF, x_est: np.ndarray, P_est: np.ndarray,
                           u_current: np.ndarray, dt: float, env: Dict[str, float],
                           horizons: int) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate 2n+1 positive-weight sigma hypotheses through the 6-DOF model."""
    points = MerweSigmaPoints(len(x_est), alpha=1.0, beta=2.0, kappa=0.0)
    sigmas = points.sigma_points(x_est, P_est)
    weights = np.clip(points.Wm, 0.0, None)
    weights /= weights.sum()
    paths = np.zeros((len(sigmas), horizons + 1, len(x_est)), dtype=float)
    for i, sigma in enumerate(sigmas):
        sigma = sigma.copy()
        sigma[6:10] = q_normalize(sigma[6:10])
        paths[i] = project_state_horizons(
            proj, sigma, u_current, dt, env, horizons)
    return paths, weights


def project_maneuver_hypotheses(proj: Projectile6DOF, x_est: np.ndarray,
                                u_current: np.ndarray, dt: float,
                                env: Dict[str, float], horizons: int,
                                force_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic one-impulse future maneuver hypotheses and their event steps."""
    event_steps = [2, 6, 12]
    paths, events = [], []
    for event_step in event_steps:
        for axis in range(3):
            for sign in (-1.0, 1.0):
                states = np.zeros((horizons + 1, len(x_est)), dtype=float)
                states[0] = x_est
                x = x_est.copy()
                for h in range(1, horizons + 1):
                    if h == 1:
                        u = u_current
                    elif h == event_step:
                        u = np.zeros(6, dtype=float)
                        u[axis] = sign * force_scale
                    else:
                        u = np.zeros(6, dtype=float)
                    x = proj.step_rk2(x, u, dt, env)
                    states[h] = x
                paths.append(states)
                events.append(event_step)
    return np.stack(paths), np.asarray(events, dtype=int)

def online_run(cfg: Config, model: MultiModalTransformer, device: torch.device, out_prefix: Path,
               token_mu: np.ndarray, token_sd: np.ndarray,
               temperatures: np.ndarray) -> Dict[str, str]:
    earth, atm, aero = EarthModel(), Atmosphere(), AeroModel()
    I = np.diag([500.0, 800.0, 1000.0])
    proj = Projectile6DOF(1000.0, I, earth, atm, aero)

    env = make_env(cfg)
    x0 = init_state(cfg, earth)
    U = make_controls(cfg, cfg.traj_steps)
    X_truth = rollout_truth(cfg, proj, x0, U, env)
    Z = make_measurements(cfg, X_truth)
    X_est, P_tr, gate_stats = run_ukf(cfg, proj, earth, atm, Z, U, env)

    # features
    feats = np.zeros((X_est.shape[0], feature_dim()), dtype=np.float32)
    for t in range(X_est.shape[0]):
        feats[t] = extract_features(X_est[t], P_tr[t], earth, atm, env)

    # labels for truth overlay (first horizon)
    labels = compute_labels(cfg, X_est, X_truth)  # (steps, H)
    # build streaming token buffer
    W = cfg.window
    Cdim = 6
    Fdim = feats.shape[1]
    Draw = Fdim + Cdim
    C = cfg.box_N * cfg.box_N
    H = len(cfg.horizon_steps)

    # create token stream: for each t we have feat token then ctrl token
    # We'll make tokens for subsampled indices in visualization
    viz_stride = max(1, int(cfg.viz_stride))
    last_t = min(cfg.viz_frames, feats.shape[0] - 1)
    indices = np.arange(W - 1, last_t, viz_stride)
    if len(indices) == 0:
        raise RuntimeError("Online trace is shorter than the configured causal window")

    P_preds = []
    y_true = []
    pos_true = []
    pos_est = []
    vel_est = []
    physics_pos_pred = []
    physics_sigma_pos_pred = []
    maneuver_pos_pred = []
    max_projection_horizon = 18
    projection_env = nominal_filter_env()

    model.eval()
    for t in indices:
        buf = build_causal_token_buffer(feats, U, int(t), W)
        buf_norm = (buf - token_mu[None, :]) / token_sd[None, :]
        xb = torch.from_numpy(buf_norm[None, :, :].astype(np.float32)).to(device)
        with torch.no_grad():
            logits, nextf = model(xb)
            temp = torch.from_numpy(temperatures).to(device).view(-1, 1)
            probs = F.softmax(logits[0] / temp, dim=-1).cpu().numpy()  # (H, C)
        P_preds.append(probs)

        # truth label (h=first)
        if t < labels.shape[0] and labels[t, 0] >= 0:
            y_true.append(int(labels[t, 0]))
        else:
            y_true.append(-1)

        pos_true.append(X_truth[t, :3].copy())
        pos_est.append(X_est[t, :3].copy())
        vel_est.append(X_est[t, 3:6].copy())
        u_current = U[t].copy() if t < len(U) else np.zeros(6, dtype=float)
        projection = project_state_horizons(
            proj, X_est[t], u_current, cfg.dt, projection_env,
            max_projection_horizon * viz_stride)
        physics_pos_pred.append(projection[::viz_stride, :3])
        sigma_paths, sigma_weights = project_sigma_horizons(
            proj, X_est[t], P_tr[t], u_current, cfg.dt, projection_env,
            max_projection_horizon * viz_stride)
        physics_sigma_pos_pred.append(sigma_paths[:, ::viz_stride, :3])
        maneuver_paths, maneuver_events = project_maneuver_hypotheses(
            proj, X_est[t], u_current, cfg.dt, projection_env,
            max_projection_horizon * viz_stride, cfg.force_std)
        maneuver_pos_pred.append(maneuver_paths[:, ::viz_stride, :3])

    P_preds = np.stack(P_preds, axis=0)  # (Tvis, H, C)
    y_true = np.array(y_true, dtype=int)
    pos_true = np.stack(pos_true, axis=0)
    pos_est = np.stack(pos_est, axis=0)
    vel_est = np.stack(vel_est, axis=0)
    physics_pos_pred = np.stack(physics_pos_pred, axis=0)
    physics_sigma_pos_pred = np.stack(physics_sigma_pos_pred, axis=0)
    maneuver_pos_pred = np.stack(maneuver_pos_pred, axis=0)

    # Save summary PNG heatmaps (true counts vs pred counts for horizon 1)
    pred_cls = np.argmax(P_preds[:, 0, :], axis=1)

    # Save raw online trace for the Streamlit/Three.js GUI
    trace_npz = out_prefix.with_suffix("").as_posix() + "_online_trace.npz"
    try:
        np.savez(
            trace_npz,
            pos_true=pos_true,
            pos_est=pos_est,
            vel_est=vel_est,
            physics_pos_pred=physics_pos_pred,
            physics_sigma_pos_pred=physics_sigma_pos_pred,
            physics_sigma_weights=sigma_weights,
            maneuver_pos_pred=maneuver_pos_pred,
            maneuver_event_step=np.maximum(1, maneuver_events // viz_stride),
            maneuver_step_probability=np.array([cfg.maneuver_prob], dtype=float),
            transformer_temperature=temperatures,
            source_frame=indices.astype(int),
            projection_control_policy=np.array(["current_then_zero"]),
            probs_h1=P_preds[:, 0, :],
            y_true=y_true,
            y_pred=pred_cls,
            dt=np.array([cfg.dt * viz_stride], dtype=float),
            box_N=np.array([cfg.box_N], dtype=int),
            primary_horizon_steps=np.array([cfg.horizon_steps[0]], dtype=int),
        )
    except Exception:
        trace_npz = ""
    tc = np.bincount(y_true[y_true >= 0], minlength=C).reshape(cfg.box_N, cfg.box_N)
    pc = np.bincount(pred_cls, minlength=C).reshape(cfg.box_N, cfg.box_N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.2))
    primary_h = cfg.horizon_steps[0]
    ax1.imshow(tc, cmap="Reds"); ax1.set_title(f"True Counts (h={primary_h})")
    ax2.imshow(pc, cmap="Blues"); ax2.set_title(f"Pred Counts (h={primary_h})")
    for ax, mat in ((ax1, tc), (ax2, pc)):
        for i in range(cfg.box_N):
            for j in range(cfg.box_N):
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="white", fontsize=9)
        ax.set_xticks(range(cfg.box_N)); ax.set_yticks(range(cfg.box_N))
    plt.tight_layout()
    heat_png = out_prefix.with_suffix("").as_posix() + "_online_heatmaps.png"
    fig.savefig(heat_png, dpi=220); plt.close(fig)

    # class over time
    fig = plt.figure(figsize=(10.5, 3.0))
    t_axis = np.arange(len(pred_cls)) * cfg.dt * viz_stride
    plt.plot(t_axis, pred_cls, linewidth=1.5)
    plt.plot(t_axis, y_true, linewidth=1.5, alpha=0.7)
    plt.yticks(range(C))
    plt.xlabel("Time (s)")
    plt.ylabel("Class")
    plt.title(f"Predicted vs True class (h={primary_h})")
    plt.tight_layout()
    cls_png = out_prefix.with_suffix("").as_posix() + "_online_class_over_time.png"
    fig.savefig(cls_png, dpi=220); plt.close(fig)

    html_path = None
    if (not cfg.no_viz) and go is not None:
        html_path = out_prefix.with_suffix("").as_posix() + "_online_animation.html"
        make_plotly_animation(cfg, pos_true, P_preds[:, 0, :], y_true, pred_cls, html_path)

    # Save probs vs truth grid (optional simplified: only h=1 line plot)
    fig = plt.figure(figsize=(10.5, 3.0))
    plt.plot(t_axis, np.max(P_preds[:, 0, :], axis=1), linewidth=1.5)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Time (s)")
    plt.ylabel(f"Max P (h={primary_h})")
    plt.title("Confidence over time (max probability)")
    plt.tight_layout()
    conf_png = out_prefix.with_suffix("").as_posix() + "_online_confidence.png"
    fig.savefig(conf_png, dpi=220); plt.close(fig)

    return {
        "online_heatmaps_png": heat_png,
        "online_class_png": cls_png,
        "online_conf_png": conf_png,
        "online_trace_npz": trace_npz,
        "online_html": html_path or "",
        "online_gate_total": str(gate_stats["total_updates"]),
        "online_gate_rejections": str(gate_stats["gate_rejections"]),
    }


def make_plotly_animation(cfg: Config, pos: np.ndarray, probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out_html: str):
    """
    Aesthetic plotly animation:
      - 3D trajectory with moving marker
      - Heatmap of predicted probabilities (NxN) with true cell highlighted in title
    """
    if go is None:
        return

    N = cfg.box_N
    C = N*N
    T = pos.shape[0]

    # Normalize positions for display (km)
    X = pos[:, 0] / 1000.0
    Y = pos[:, 1] / 1000.0
    Z = pos[:, 2] / 1000.0

    # base traces
    traj = go.Scatter3d(
        x=X, y=Y, z=Z,
        mode="lines",
        line=dict(width=4, color="rgba(0,255,200,0.35)"),
        name="Trajectory"
    )
    marker = go.Scatter3d(
        x=[X[0]], y=[Y[0]], z=[Z[0]],
        mode="markers",
        marker=dict(size=6, color="rgba(255,80,200,0.95)"),
        name="Now"
    )

    def heat_for(t):
        mat = probs[t].reshape(N, N)
        return mat

    heat0 = heat_for(0)
    heat = go.Heatmap(
        z=heat0,
        colorscale="Viridis",
        zmin=0.0, zmax=1.0,
        showscale=True,
        colorbar=dict(title="P(cell)", len=0.75),
        name="P(exit)"
    )

    layout = go.Layout(
        template="plotly_dark",
        title=f"Alien Exit-Cell Prediction (h={cfg.horizon_steps[0]})  |  true={y_true[0]}  pred={y_pred[0]}",
        width=1100,
        height=560,
        margin=dict(l=10, r=10, t=55, b=10),
        scene=dict(
            xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)",
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(domain=[0.55, 1.0]),  # heatmap area
        yaxis=dict(domain=[0.0, 1.0]),
        scene_domain=dict(x=[0.0, 0.52], y=[0.0, 1.0]),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.58, y=1.08,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ],
            )
        ],
        sliders=[dict(
            x=0.58, y=-0.02, len=0.40,
            steps=[]
        )],
    )

    frames = []
    slider_steps = []
    for t in range(T):
        title = f"Alien Exit-Cell Prediction (h={cfg.horizon_steps[0]})  |  true={int(y_true[t])}  pred={int(y_pred[t])}"
        fr = go.Frame(
            data=[
                marker.update(x=[X[t]], y=[Y[t]], z=[Z[t]]),
                heat.update(z=heat_for(t)),
            ],
            layout=go.Layout(title=title),
            name=str(t),
        )
        frames.append(fr)
        slider_steps.append(dict(method="animate", args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=str(t)))

    layout.sliders[0]["steps"] = slider_steps

    fig = go.Figure(data=[traj, marker, heat], layout=layout, frames=frames)
    fig.write_html(out_html, include_plotlyjs="cdn")


# =========================
# Main
# =========================

def main():
    cfg = parse_args()
    seed_everything(cfg.seed)

    out_prefix = Path(cfg.output_prefix).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if cfg.gpu and torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Device: {device}")

    logging.info("Generating dataset...")
    X_tr, y_tr, nf_tr, X_val, y_val, nf_val, X_te, y_te, nf_te, stats = generate_dataset(cfg)
    logging.info(f"Samples: train={len(y_tr)} val={len(y_val)} test={len(y_te)}  gate_rej_rate={stats['gate_rejection_rate']:.3f}")

    # Standardize raw tokens (feature and control scale) using train statistics
    # We'll compute mean/std over the raw Draw dim across all tokens in train set.
    mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    sd = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0) + 1e-6

    def norm(X):
        return ((X - mu[None, None, :]) / sd[None, None, :]).astype(np.float32)

    X_trn, X_valn, X_ten = norm(X_tr), norm(X_val), norm(X_te)
    nf_mu = nf_tr.mean(axis=0)
    nf_sd = nf_tr.std(axis=0) + 1e-6

    def norm_nf(nf):
        return ((nf - nf_mu[None, :]) / nf_sd[None, :]).astype(np.float32)

    nf_trn, nf_valn, nf_ten = norm_nf(nf_tr), norm_nf(nf_val), norm_nf(nf_te)

    logging.info("Training multimodal Transformer...")
    model, train_stats = train_model(cfg, device, X_trn, y_tr, nf_trn, X_valn, y_val, nf_valn, stats)

    logging.info("Evaluating...")
    temperatures = fit_model_temperatures(cfg, model, device, X_valn, y_val)
    test_stats = eval_model(cfg, model, device, X_ten, y_te, y_train=y_tr,
                            temperatures=temperatures)

    # Online run + visuals
    online_paths = online_run(cfg, model, device, out_prefix, mu, sd, temperatures)

    weights_path = out_prefix.with_suffix("").as_posix() + "_model_state.pt"
    norm_path = out_prefix.with_suffix("").as_posix() + "_normalization.npz"
    torch.save(model.state_dict(), weights_path)
    np.savez(norm_path, token_mu=mu, token_sd=sd, next_feature_mu=nf_mu,
             next_feature_sd=nf_sd, transformer_temperature=temperatures)

    metrics = {
        "config": {
            "dt": cfg.dt,
            "traj_steps": cfg.traj_steps,
            "num_trajectories": cfg.num_trajectories,
            "offline_stride": cfg.offline_stride,
            "window": cfg.window,
            "horizon_steps": list(cfg.horizon_steps),
            "box_N": cfg.box_N,
            "d_model": cfg.d_model,
            "nlayers": cfg.nlayers,
            "nhead": cfg.nhead,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "quick": cfg.quick,
        },
        "dataset": stats,
        "train": train_stats,
        "test": test_stats,
        "online": online_paths,
        "artifacts": {
            "model_state": weights_path,
            "normalization": norm_path,
            "load_note": "Load model_state with torch.load(..., weights_only=True).",
        },
    }

    metrics_path = out_prefix.with_suffix("").as_posix() + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Done.")
    logging.info(f"Metrics JSON: {metrics_path}")
    for k, v in online_paths.items():
        if k.endswith("_png") or k.endswith("_html"):
            logging.info(f"{k}: {v}")


if __name__ == "__main__":
    main()
