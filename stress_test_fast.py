#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast stress test for Alien Exit-Cell Predictor v6.2

Analytics:
- Numerical stability (NaN/Inf)
- UKF gating health
- Label diversity (entropy, unique classes) per horizon
- Simple predictability probes:
  - random baseline
  - majority baseline
  - linear softmax probe on snapshot features (last token + nf)

Usage:
  python3 stress_test_fast.py --seeds 0,1,2 --traj 18 --mode diversity --out stress_fast.json

Modes:
  - stability: default-ish, catches NaN/Inf and UKF issues
  - diversity: smaller virtual boxes + more maneuvering for richer class distribution
  - brutal: higher noise/maneuvering (may reduce learnability but should stay stable)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch

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
    p.add_argument("--traj", type=int, default=18)
    p.add_argument("--mode", type=str, default="stability", choices=["stability", "diversity", "brutal"])
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", type=str, default="stress_fast.json")
    p.add_argument("--min-samples", type=int, default=140, help="minimum dataset samples to target (passed into cfg.min_samples)")
    p.add_argument("--topk", type=str, default="1,3,5", help="comma-separated top-k values for accuracy (e.g., 1,3,5)")
    p.add_argument("--cm-dir", type=str, default="", help="optional directory to write confusion matrices CSVs")
    p.add_argument("--metrics-dir", type=str, default="", help="optional directory to write extended metrics CSV/JSON")
    p.add_argument("--ece-bins", type=int, default=15, help="number of bins for Expected Calibration Error (ECE)")
    p.add_argument("--probe-steps", type=int, default=25, help="SGD steps for linear probe")
    p.add_argument("--probe-lr", type=float, default=0.2, help="SGD learning rate for linear probe")
    p.add_argument("--max-retries", type=int, default=5, help="max retries if dataset generation yields zero windows")
    p.add_argument("--horizon-cap", type=int, default=24, help="during retries, cap horizon steps at this value")
    return p.parse_args()


def finite_ok(a: np.ndarray) -> bool:
    return bool(np.isfinite(a).all())



def softmax(logits: np.ndarray) -> np.ndarray:
    # stable softmax
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (ez.sum(axis=1, keepdims=True) + 1e-12)

def topk_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if k <= 1:
        pred = logits.argmax(axis=1)
        return float((pred == y_true).mean())
    k = int(min(k, logits.shape[1]))
    topk = np.argpartition(-logits, kth=k-1, axis=1)[:, :k]
    hit = np.any(topk == y_true[:, None], axis=1)
    return float(hit.mean())

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> np.ndarray:
    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def normalize_cm(cm: np.ndarray, mode: str) -> np.ndarray:
    cm = cm.astype(np.float64)
    if mode == "row":
        denom = cm.sum(axis=1, keepdims=True) + 1e-12
        return cm / denom
    if mode == "col":
        denom = cm.sum(axis=0, keepdims=True) + 1e-12
        return cm / denom
    if mode == "all":
        denom = cm.sum() + 1e-12
        return cm / denom
    raise ValueError("mode must be row|col|all")

def per_class_prf(cm: np.ndarray) -> dict:
    # cm: rows true, cols pred
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)

    precision = tp / (pred_count + 1e-12)
    recall = tp / (support + 1e-12)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "pred_count": pred_count,
    }

def aggregate_f1(cm: np.ndarray, prf: dict) -> dict:
    precision = prf["precision"]
    recall = prf["recall"]
    f1 = prf["f1"]
    support = prf["support"]
    total = float(support.sum() + 1e-12)

    macro_f1 = float(np.nanmean(f1))
    weighted_f1 = float(np.nansum(f1 * support) / total)

    # micro-F1 for multiclass = accuracy
    # (because micro precision = micro recall = accuracy)
    micro_f1 = float(np.sum(np.diag(cm)) / (np.sum(cm) + 1e-12))

    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "micro_f1": micro_f1}

def nll_and_brier(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    probs = softmax(logits)
    p_true = probs[np.arange(len(y_true)), y_true]
    nll = float(-np.mean(np.log(p_true + 1e-12)))
    # Brier for multiclass
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    return nll, brier

def expected_calibration_error(logits: np.ndarray, y_true: np.ndarray, bins: int = 15) -> tuple[float, list]:
    probs = softmax(logits)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    table = []
    n = len(y_true)

    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if not np.any(m):
            table.append({"bin": i, "lo": float(lo), "hi": float(hi), "count": 0, "acc": float("nan"), "conf": float("nan")})
            continue
        bin_acc = float(acc[m].mean())
        bin_conf = float(conf[m].mean())
        w = float(m.sum() / max(n, 1))
        ece += abs(bin_acc - bin_conf) * w
        table.append({"bin": i, "lo": float(lo), "hi": float(hi), "count": int(m.sum()), "acc": bin_acc, "conf": bin_conf})

    return float(ece), table

def balanced_accuracy(cm: np.ndarray) -> float:
    # mean recall across classes
    support = cm.sum(axis=1).astype(np.float64)
    recall = np.diag(cm).astype(np.float64) / (support + 1e-12)
    return float(np.nanmean(recall))

def entropy_bits(labels: np.ndarray, C: int) -> float:
    labels = labels[labels >= 0]
    if labels.size == 0:
        return float("nan")
    c = np.bincount(labels, minlength=C).astype(float)
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def linear_probe_fit_predict(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    C: int,
    steps: int = 25,
    lr: float = 0.2,
):
    """Fit a simple softmax linear classifier (CPU) and return logits + preds."""
    device = torch.device("cpu")
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    W = torch.zeros((Xtr_t.shape[1], C), device=device, requires_grad=True)
    b = torch.zeros((C,), device=device, requires_grad=True)
    opt = torch.optim.SGD([W, b], lr=lr)

    for _ in range(steps):
        logits = Xtr_t @ W + b
        loss = torch.nn.functional.cross_entropy(logits, ytr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        te_logits = (Xte_t @ W + b).cpu().numpy()
        te_pred = te_logits.argmax(axis=1)
    return te_logits, te_pred


def configure(cfg: "mod.Config", mode: str, traj_override: int, device: torch.device, min_samples: int):
    cfg.gpu = (device.type == "cuda")
    cfg.no_viz = True
    cfg.quick = True

    # allow small/fast dataset generation (requires patched engine)
    cfg.offline_traj_override = int(traj_override)
    cfg.offline_traj_min = 1
    cfg.offline_traj_max = int(traj_override)
    cfg.min_samples = 50

    # keep run bounded
    cfg.traj_steps = 45
    cfg.offline_stride = 8

    if mode == "stability":
        cfg.wind_accel_std = 35.0
        cfg.maneuver_prob = 0.10
        cfg.force_std = 12_000.0
        cfg.torque_std = 10_000.0

    elif mode == "diversity":
        cfg.wind_accel_std = 45.0
        cfg.maneuver_prob = 0.22
        cfg.force_std = 16_000.0
        cfg.torque_std = 14_000.0

        # Smaller virtual box => more exits + more varied cells
        cfg.box_L_scale = max(0.12, float(cfg.box_L_scale) * 0.45)
        cfg.box_min_L = max(25.0, float(cfg.box_min_L) * 0.60)
        cfg.box_max_L = max(180.0, float(cfg.box_max_L) * 0.35)

        # longer horizons also diversify direction patterns
        cfg.horizon_steps = (8, 10, 12)

    else:  # brutal
        cfg.wind_accel_std = 70.0
        cfg.maneuver_prob = 0.28
        cfg.force_std = 22_000.0
        cfg.torque_std = 20_000.0

        cfg.box_L_scale = max(0.10, float(cfg.box_L_scale) * 0.40)
        cfg.box_min_L = max(20.0, float(cfg.box_min_L) * 0.55)
        cfg.box_max_L = max(150.0, float(cfg.box_max_L) * 0.30)
        cfg.horizon_steps = (8, 15, 24)


def main():
    a = parse()
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]

    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    topk_list = []
    for part in str(a.topk).split(","):
        part = part.strip()
        if part:
            try:
                topk_list.append(int(part))
            except ValueError:
                pass
    if not topk_list:
        topk_list = [1, 3, 5]

    cm_dir = Path(a.cm_dir).resolve() if str(a.cm_dir).strip() else None
    metrics_dir = Path(a.metrics_dir).resolve() if str(a.metrics_dir).strip() else None
    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
    if cm_dir is not None:
        cm_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "engine": str(ENGINE.name),
        "device": str(device),
        "mode": a.mode,
        "traj_override": int(a.traj),
        "runs": [],
    }

    for seed in seeds:
        t0 = time.time()
        cfg = mod.Config()
        cfg.seed = seed
        configure(cfg, a.mode, a.traj, device, a.min_samples)
        mod.seed_everything(cfg.seed)

        # dataset generation can occasionally yield zero windows if window/horizon/stride don't fit.
        # Retry with progressively more permissive settings.
        last_err = None
        for attempt in range(int(a.max_retries)):
            try:
                X_tr, y_tr, nf_tr, X_val, y_val, nf_val, X_te, y_te, nf_te, ds_stats = mod.generate_dataset(cfg)
                break
            except RuntimeError as e:
                last_err = str(e)
                # Make windows easier to form + increase exits
                cfg.offline_stride = max(1, int(cfg.offline_stride) // 2)
                cfg.window_T = max(4, int(getattr(cfg, "window_T", 8)) - 1)
                # Clamp horizons so window_T + max_h fits better
                cfg.horizon_steps = tuple(int(min(h, int(a.horizon_cap))) for h in cfg.horizon_steps)
                # Shrink box a bit more to encourage exits
                cfg.box_L_scale = max(0.08, float(cfg.box_L_scale) * 0.85)
                cfg.box_max_L = max(80.0, float(cfg.box_max_L) * 0.90)
        else:
            raise RuntimeError(f"generate_dataset failed after retries: {last_err}")

        checks = {
            "X_tr_finite": finite_ok(X_tr),
            "nf_tr_finite": finite_ok(nf_tr),
            "X_val_finite": finite_ok(X_val),
            "X_te_finite": finite_ok(X_te),
        }

        C = cfg.box_N * cfg.box_N
        H = len(cfg.horizon_steps)

        # snapshot features
        feat_tr = np.concatenate([X_tr[:, -1, :], nf_tr], axis=1).astype(np.float32)
        feat_te = np.concatenate([X_te[:, -1, :], nf_te], axis=1).astype(np.float32)

        mu = feat_tr.mean(axis=0)
        sd = feat_tr.std(axis=0) + 1e-6
        feat_tr = (feat_tr - mu) / sd
        feat_te = (feat_te - mu) / sd

        per_h = []
        for hi in range(H):
            y_tr_h = y_tr[:, hi]
            y_te_h = y_te[:, hi]
            mtr = y_tr_h >= 0
            mte = y_te_h >= 0
            if (not mtr.any()) or (not mte.any()):
                per_h.append({"h": int(cfg.horizon_steps[hi]), "entropy_bits": float("nan")})
                continue

            counts = np.bincount(y_te_h[mte], minlength=C)
            maj = float(counts.max() / counts.sum())
            rand = float(1.0 / C)
            ent = entropy_bits(y_te_h[mte], C)
            uniq = int((counts > 0).sum())
            te_logits, te_pred = linear_probe_fit_predict(feat_tr[mtr], y_tr_h[mtr], feat_te[mte], C, steps=a.probe_steps, lr=a.probe_lr)
            topk_acc = {f"top{k}": topk_accuracy(te_logits, y_te_h[mte], k) for k in topk_list}
            cm = confusion_matrix(y_te_h[mte], te_pred, C)
            # Extended metrics
            nll, brier = nll_and_brier(te_logits, y_te_h[mte])
            ece, ece_table = expected_calibration_error(te_logits, y_te_h[mte], bins=int(a.ece_bins))
            bal_acc = balanced_accuracy(cm)
            prf = per_class_prf(cm)
            agg = aggregate_f1(cm, prf)

            if metrics_dir is not None:
                # Per-class metrics
                pc = np.arange(C, dtype=int)
                pc_df = np.stack([pc, prf["support"].astype(int), prf["pred_count"].astype(int),
                                  prf["precision"], prf["recall"], prf["f1"]], axis=1)
                pc_path = metrics_dir / f"per_class_seed{seed}_h{int(cfg.horizon_steps[hi])}.csv"
                np.savetxt(pc_path, pc_df, delimiter=",", fmt=["%d","%d","%d","%.6f","%.6f","%.6f"],
                           header="class,support,pred_count,precision,recall,f1", comments="")

                # Normalized confusion matrices
                np.savetxt(metrics_dir / f"confusion_row_norm_seed{seed}_h{int(cfg.horizon_steps[hi])}.csv",
                           normalize_cm(cm, "row"), delimiter=",", fmt="%.6f")
                np.savetxt(metrics_dir / f"confusion_col_norm_seed{seed}_h{int(cfg.horizon_steps[hi])}.csv",
                           normalize_cm(cm, "col"), delimiter=",", fmt="%.6f")

                # Calibration table
                cal_path = metrics_dir / f"calibration_seed{seed}_h{int(cfg.horizon_steps[hi])}.json"
                Path(cal_path).write_text(json.dumps(ece_table, indent=2), encoding="utf-8")
            if cm_dir is not None:
                cm_path = cm_dir / f"confusion_seed{seed}_h{int(cfg.horizon_steps[hi])}.csv"
                np.savetxt(cm_path, cm, fmt="%d", delimiter=",")

            per_h.append(
                {
                    "h": int(cfg.horizon_steps[hi]),
                    "unique_classes": uniq,
                    "entropy_bits": ent,
                    "majority_acc": maj,
                    "random_acc": rand,
                    "linear_probe_acc": float(topk_acc.get("top1", np.nan)),
                    "topk": topk_acc,
                    "confusion_matrix_written": bool(cm_dir is not None),
                    "nll": float(nll),
                    "brier": float(brier),
                    "ece": float(ece),
                    "balanced_acc": float(bal_acc),
                    "macro_f1": float(agg["macro_f1"]),
                    "weighted_f1": float(agg["weighted_f1"]),
                    "micro_f1": float(agg["micro_f1"]),
                    "metrics_written": bool(metrics_dir is not None),
                    "counts": counts.tolist(),
                }
            )

        out["runs"].append(
            {
                "seed": seed,
                "seconds": float(time.time() - t0),
                "dataset": {"train": int(len(y_tr)), "val": int(len(y_val)), "test": int(len(y_te))},
                "gate_rejection_rate": float(ds_stats.get("gate_rejection_rate", 0.0)),
                "checks": checks,
                "horizons": per_h,
            }
        )

    # Write summary table if requested
    if metrics_dir is not None:
        rows = []
        for run in out["runs"]:
            seed = run["seed"]
            base_row = {"seed": seed, "seconds": run["seconds"], "train": run["dataset"]["train"], "val": run["dataset"]["val"], "test": run["dataset"]["test"], "gate_rejection_rate": run.get("gate_rejection_rate", 0.0)}
            for h in run["horizons"]:
                r = dict(base_row)
                r.update({
                    "h": h.get("h"),
                    "unique_classes": h.get("unique_classes"),
                    "entropy_bits": h.get("entropy_bits"),
                    "majority_acc": h.get("majority_acc"),
                    "random_acc": h.get("random_acc"),
                    "top1": h.get("topk", {}).get("top1", h.get("linear_probe_acc")),
                    "top3": h.get("topk", {}).get("top3"),
                    "top5": h.get("topk", {}).get("top5"),
                    "nll": h.get("nll"),
                    "brier": h.get("brier"),
                    "ece": h.get("ece"),
                    "balanced_acc": h.get("balanced_acc"),
                    "macro_f1": h.get("macro_f1"),
                    "weighted_f1": h.get("weighted_f1"),
                    "micro_f1": h.get("micro_f1"),
                })
                rows.append(r)
        import csv
        summary_path = metrics_dir / "summary_metrics.csv"
        if rows:
            cols = list(rows[0].keys())
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow(r)

    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
