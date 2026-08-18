#!/usr/bin/env python3
"""Evaluate frozen HTI 0.8 probability bundles on complete trajectory events.

The evaluator consumes already-produced probabilities. It does not train or tune
on final-test data. Fusion weights and other fitted quantities must be selected
on validation data before this script is used for a final-test report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hti.benchmarking import (  # noqa: E402
    event_bootstrap_delta,
    evaluate_probabilities,
    nll_contributions,
    selective_risk_curve,
)
from hti.fusion import log_linear_pool  # noqa: E402


def _as_horizon_matrix(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 1:
        return value[:, None]
    if value.ndim == 2:
        return value
    raise ValueError("labels must have shape (samples,) or (samples, horizons)")


def _as_probability_cube(array: np.ndarray, horizons: int) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.ndim == 2:
        value = value[:, None, :]
    if value.ndim != 3 or value.shape[1] != horizons:
        raise ValueError("variant probabilities must have shape (samples, horizons, classes)")
    return value


def _self_test_bundle() -> dict[str, np.ndarray]:
    samples = 120
    horizons = 3
    classes = 4
    labels = np.tile(np.arange(classes, dtype=int), samples // classes)
    labels = np.column_stack([labels, np.roll(labels, 1), np.roll(labels, 2)])
    event_ids = np.repeat(np.arange(12, dtype=int), 10)

    def distributions(correct_mass: float) -> np.ndarray:
        cube = np.full((samples, horizons, classes), (1.0 - correct_mass) / (classes - 1))
        for horizon in range(horizons):
            cube[np.arange(samples), horizon, labels[:, horizon]] = correct_mass
        return cube

    core = distributions(0.58)
    structural = distributions(0.68)
    combined = np.empty_like(core)
    for horizon in range(horizons):
        combined[:, horizon, :] = log_linear_pool(
            core[:, horizon, :], structural[:, horizon, :], structural_weight=0.50
        )
    return {
        "labels": labels,
        "event_ids": event_ids,
        "probs__core_hti": core,
        "probs__core_plus_structural": structural,
        "probs__hti_08_combined": combined,
    }


def _load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def evaluate(bundle: dict[str, np.ndarray], config: dict[str, object]) -> dict[str, object]:
    if "labels" not in bundle or "event_ids" not in bundle:
        raise ValueError("bundle must contain labels and event_ids")
    labels = _as_horizon_matrix(bundle["labels"])
    event_ids = np.asarray(bundle["event_ids"]).reshape(-1)
    if len(event_ids) != len(labels):
        raise ValueError("event_ids must match the sample count")

    horizons = [float(value) for value in config["horizons_seconds"]]
    if labels.shape[1] != len(horizons):
        raise ValueError("label horizon count does not match frozen configuration")
    metric_cfg = config["metrics"]
    ece_bins = int(metric_cfg["ece_bins"])
    credible_level = float(metric_cfg["credible_level"])
    bootstrap_iterations = int(metric_cfg["bootstrap_iterations"])

    variants: dict[str, np.ndarray] = {}
    for key, value in bundle.items():
        if key.startswith("probs__"):
            variants[key.removeprefix("probs__")] = _as_probability_cube(value, len(horizons))
    if not variants:
        raise ValueError("bundle must contain at least one probs__<variant> array")
    for name, cube in variants.items():
        if cube.shape[:2] != labels.shape:
            raise ValueError(f"variant {name!r} does not match label shape")

    report: dict[str, object] = {
        "schema": "hti.ablation-report.v0.8",
        "protocol_status": config.get("status"),
        "samples": int(len(labels)),
        "unique_events": int(len(np.unique(event_ids))),
        "horizons_seconds": horizons,
        "variants": {},
        "paired_core_vs_combined": {},
        "scientific_note": (
            "This evaluator reports frozen-test evidence only. It does not establish external "
            "domain validity or operational readiness."
        ),
    }

    variant_report: dict[str, object] = {}
    for name, cube in sorted(variants.items()):
        horizon_report = []
        for index, horizon in enumerate(horizons):
            probabilities = cube[:, index, :]
            y = labels[:, index]
            metrics = evaluate_probabilities(
                probabilities,
                y,
                ece_bins=ece_bins,
                credible_level=credible_level,
            )
            horizon_report.append(
                {
                    "horizon_seconds": horizon,
                    "metrics": metrics.to_dict(),
                    "selective_risk": selective_risk_curve(probabilities, y),
                }
            )
        variant_report[name] = horizon_report
    report["variants"] = variant_report

    if "core_hti" in variants and "hti_08_combined" in variants:
        comparisons: dict[str, object] = {}
        for index, horizon in enumerate(horizons):
            core = variants["core_hti"][:, index, :]
            combined = variants["hti_08_combined"][:, index, :]
            y = labels[:, index]
            comparisons[str(horizon)] = {
                "nll_delta_combined_minus_core": event_bootstrap_delta(
                    nll_contributions(combined, y),
                    nll_contributions(core, y),
                    event_ids,
                    iterations=bootstrap_iterations,
                    seed=20260818 + index,
                    lower_is_better=True,
                )
            }
        report["paired_core_vs_combined"] = comparisons
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/hti_08_ablation_frozen.json")
    )
    parser.add_argument("--report", type=Path, default=Path("hti08_ablation_report.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test == (args.bundle is not None):
        raise SystemExit("choose exactly one of --self-test or --bundle")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    bundle = _self_test_bundle() if args.self_test else _load_bundle(args.bundle)
    report = evaluate(bundle, config)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
