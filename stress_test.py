#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stress test for Alien Exit-Cell Predictor v6.2

Goals:
- Catch NaNs/Infs in physics + UKF + feature pipeline
- Measure run-to-run stability across seeds
- Give a quick sanity check on predictability (beats random baseline)

This script intentionally keeps workloads small so you can run it on CPU.

Usage:
  python3 stress_test.py
  python3 stress_test.py --seeds 0,1,2,3,4 --device cpu

Outputs:
  stress_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

# Load the engine file as a module without requiring packaging
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "alien_exit_cell_predictor_v6_3.py"

spec = importlib.util.spec_from_file_location("alien_v6_2", ENGINE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)  # type: ignore


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", type=str, default="stress_report.json")
    p.add_argument("--traj", type=int, default=24, help="override offline trajectories for fast stress tests")
    return p.parse_args()


def finite_ok(arr: np.ndarray) -> bool:
    return bool(np.isfinite(arr).all())


def main():
    a = parse()
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]

    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    results = {
        "engine": str(ENGINE.name),
        "device": str(device),
        "runs": [],
    }

    for seed in seeds:
        t0 = time.time()

        cfg = mod.Config()
        # small but meaningful
        cfg.seed = seed
        cfg.gpu = (device.type == "cuda")
        cfg.quick = True
        cfg.no_viz = True
        # make stress a bit harsher
        cfg.wind_accel_std = 35.0
        cfg.maneuver_prob = 0.10
        cfg.force_std = 12_000.0
        cfg.torque_std = 10_000.0
        # keep dataset tiny
        cfg.num_trajectories = 30
        cfg.traj_steps = 80
        cfg.offline_stride = 6
        cfg.epochs = 1
        cfg.d_model = 48
        cfg.nlayers = 2
        cfg.nhead = 4
        cfg.batch_size = 128

        mod.seed_everything(cfg.seed)

        # dataset (override traj count for fast stress tests)
        cfg.offline_traj_override = int(a.traj)
        X_tr, y_tr, nf_tr, X_val, y_val, nf_val, X_te, y_te, nf_te, ds_stats = mod.generate_dataset(cfg)

        # finite checks
        checks = {
            "X_tr_finite": finite_ok(X_tr),
            "nf_tr_finite": finite_ok(nf_tr),
            "X_val_finite": finite_ok(X_val),
            "X_te_finite": finite_ok(X_te),
        }

        # train 1 epoch and eval
        mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
        sd = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0) + 1e-6
        X_trn = ((X_tr - mu[None, None, :]) / sd[None, None, :]).astype(np.float32)
        X_valn = ((X_val - mu[None, None, :]) / sd[None, None, :]).astype(np.float32)
        X_ten = ((X_te - mu[None, None, :]) / sd[None, None, :]).astype(np.float32)

        nf_mu = nf_tr.mean(axis=0)
        nf_sd = nf_tr.std(axis=0) + 1e-6
        nf_trn = ((nf_tr - nf_mu[None, :]) / nf_sd[None, :]).astype(np.float32)
        nf_valn = ((nf_val - nf_mu[None, :]) / nf_sd[None, :]).astype(np.float32)

        model, tr_stats = mod.train_model(cfg, device, X_trn, y_tr, nf_trn, X_valn, y_val, nf_valn, ds_stats)
        te_stats = mod.eval_model(cfg, model, device, X_ten, y_te)

        # random baseline for horizon-0 accuracy
        C = cfg.box_N * cfg.box_N
        y0 = y_te[:, 0]
        mask = y0 >= 0
        if mask.any():
            rand_acc = float((np.random.randint(0, C, size=mask.sum()) == y0[mask]).mean())
        else:
            rand_acc = None

        dt = time.time() - t0
        results["runs"].append({
            "seed": seed,
            "seconds": dt,
            "dataset": {
                "train": int(len(y_tr)),
                "val": int(len(y_val)),
                "test": int(len(y_te)),
                "gate_rejection_rate": float(ds_stats.get("gate_rejection_rate", 0.0)),
            },
            "checks": checks,
            "train": tr_stats,
            "test": te_stats,
            "baseline": {"random_acc_h0": rand_acc},
        })

    outp = Path(a.out).resolve()
    outp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote: {outp}")


if __name__ == "__main__":
    main()
