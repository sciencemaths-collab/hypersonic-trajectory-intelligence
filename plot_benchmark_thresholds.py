#!/usr/bin/env python3
"""Create a standalone benchmark pass/fail chart from the saved full-run metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_METRICS = ROOT / "benchmark_v64_maneuver_calibrated_metrics.json"
DEFAULT_OUTPUT = ROOT / "benchmark_pass_thresholds.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    test = metrics["test"]
    dt = float(metrics["config"]["dt"])
    steps = metrics["config"]["horizon_steps"]
    horizons = np.asarray(steps, dtype=float) * dt
    accuracy = np.asarray([test[f"test_acc_h{h}"] for h in steps])
    baseline = np.asarray([test[f"test_majority_baseline_h{h}"] for h in steps])
    ece = np.asarray([test[f"test_ece_h{h}"] for h in steps])
    accuracy_pass = accuracy > baseline
    ece_pass = ece < 0.10

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), constrained_layout=True)
    fig.patch.set_facecolor("#0b0d10")
    labels = [f"{h:.1f}s" for h in horizons]
    x = np.arange(len(labels))

    ax = axes[0]
    bars = ax.bar(x, accuracy, width=0.58,
                  color=np.where(accuracy_pass, "#45c486", "#ef6262"), label="Model accuracy")
    ax.plot(x, baseline, color="#e9b44c", marker="o", linewidth=2.5,
            linestyle="--", label="Required: exceed majority baseline")
    for bar, value, passed in zip(bars, accuracy, accuracy_pass):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.009,
                f"{100 * value:.1f}%\n{'PASS' if passed else 'FAIL'}",
                ha="center", va="bottom", color="#f1f4f6", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(0.34, float(max(accuracy.max(), baseline.max())) * 1.38))
    ax.set_title("HELD-OUT ACCURACY", loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (higher is better)")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    bars = ax.bar(x, ece, width=0.58, color=np.where(ece_pass, "#45c486", "#ef6262"), label="ECE")
    ax.axhline(0.10, color="#e9b44c", linewidth=2.5, linestyle="--", label="Maximum allowed: 10%")
    for bar, value, passed in zip(bars, ece, ece_pass):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006,
                f"{100 * value:.1f}%\n{'PASS' if passed else 'FAIL'}",
                ha="center", va="bottom", color="#f1f4f6", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(0.17, float(ece.max()) * 1.38))
    ax.set_title("EXPECTED CALIBRATION ERROR", loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel("ECE (lower is better)")
    ax.legend(loc="upper left", frameon=False)

    for ax in axes:
        ax.set_facecolor("#12161b")
        ax.set_xticks(x, labels)
        ax.set_xlabel("Prediction horizon")
        ax.grid(axis="y", color="#2a3139", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#46515c")
        ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")

    full_pass = bool(np.all(accuracy_pass & ece_pass))
    fig.suptitle("TRAJECTORY MODEL / FULL BENCHMARK RELEASE GATES", fontsize=18,
                 fontweight="bold", x=0.01, ha="left", color="#f1f4f6")
    fig.text(0.99, 0.99, "MODEL PASS" if full_pass else "MODEL NOT RELEASE READY",
             ha="right", va="top", fontsize=12, fontweight="bold",
             color="#45c486" if full_pass else "#ef6262")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor=fig.get_facecolor())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
