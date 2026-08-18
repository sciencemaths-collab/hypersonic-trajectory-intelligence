"""Multi-seed aggregation for frozen HTI 0.8 Phase 3 reports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

METRIC_NAMES = (
    "class_coverage",
    "accuracy_top1",
    "accuracy_top3",
    "accuracy_top5",
    "nll",
    "brier",
    "ece",
    "credible_coverage",
    "credible_mean_size",
    "credible_median_size",
    "mean_confidence",
    "mean_entropy_concentration",
)


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("aggregate metric values must be finite and non-empty")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_reports(
    reports: Sequence[dict[str, object]],
    protocol: dict[str, object],
    *,
    protocol_sha256: str,
    execution_config_sha256: str,
) -> dict[str, object]:
    """Aggregate all frozen seeds without selecting favorable subsets."""

    expected_seeds = sorted(int(value) for value in protocol["seeds"])
    if len(reports) != len(expected_seeds):
        raise ValueError("aggregate requires exactly one report per frozen seed")
    by_seed: dict[int, dict[str, object]] = {}
    for report in reports:
        contract = report.get("contract")
        if not isinstance(contract, dict):
            raise ValueError("every report must contain a validated contract")
        seed = int(contract["seed"])
        if seed in by_seed:
            raise ValueError(f"duplicate report for seed {seed}")
        if str(report.get("protocol_sha256")) != protocol_sha256:
            raise ValueError("report protocol hash does not match frozen protocol")
        if str(report.get("execution_config_sha256")) != execution_config_sha256:
            raise ValueError("report execution hash does not match frozen execution config")
        if not all(
            bool(contract.get(key))
            for key in (
                "required_variants_present",
                "event_splits_disjoint",
                "core_fusion_reproducible",
                "structural_fusion_reproducible",
                "topology_application_reproducible",
                "combined_reproducible",
            )
        ):
            raise ValueError(f"report for seed {seed} failed its reproducibility contract")
        by_seed[seed] = report
    if sorted(by_seed) != expected_seeds:
        raise ValueError("report seed set does not equal frozen seed set")

    horizons = [float(value) for value in protocol["horizons_seconds"]]
    variants = [str(value) for value in protocol["variants"]]
    metric_summary: dict[str, list[dict[str, object]]] = {}
    for variant in variants:
        horizon_records: list[dict[str, object]] = []
        for horizon_index, horizon in enumerate(horizons):
            per_metric: dict[str, object] = {}
            for metric in METRIC_NAMES:
                values = []
                for seed in expected_seeds:
                    report = by_seed[seed]
                    variant_report = report["variants"][variant]
                    values.append(float(variant_report[horizon_index]["metrics"][metric]))
                per_metric[metric] = _summary(values)
            horizon_records.append(
                {
                    "horizon_seconds": horizon,
                    "metrics": per_metric,
                }
            )
        metric_summary[variant] = horizon_records

    gates = protocol["claim_gates"]
    minimum_coverage = float(gates["minimum_test_class_coverage"])
    maximum_ece_regression = float(gates["maximum_ece_regression"])
    credible_low, credible_high = (float(value) for value in gates["credible_coverage_interval"])
    horizon_gates: list[dict[str, object]] = []
    for horizon_index, horizon in enumerate(horizons):
        seed_records = []
        for seed in expected_seeds:
            report = by_seed[seed]
            core = report["variants"]["core_hti"][horizon_index]["metrics"]
            combined = report["variants"]["hti_08_combined"][horizon_index]["metrics"]
            bootstrap = report["paired_core_vs_combined"][str(horizon)][
                "nll_delta_combined_minus_core"
            ]
            seed_records.append(
                {
                    "seed": seed,
                    "class_coverage_pass": float(combined["class_coverage"]) >= minimum_coverage,
                    "combined_nll_better": float(combined["nll"]) < float(core["nll"]),
                    "ece_regression_pass": (
                        float(combined["ece"]) - float(core["ece"])
                    )
                    <= maximum_ece_regression,
                    "credible_coverage_pass": credible_low
                    <= float(combined["credible_coverage"])
                    <= credible_high,
                    "bootstrap_interval_supports_improvement": bool(
                        float(bootstrap["interval_supports_direction"]) == 1.0
                    ),
                    "nll_delta_combined_minus_core": float(bootstrap["delta"]),
                    "nll_delta_ci95": [
                        float(bootstrap["ci95_low"]),
                        float(bootstrap["ci95_high"]),
                    ],
                }
            )
        horizon_gates.append(
            {
                "horizon_seconds": horizon,
                "per_seed": seed_records,
                "all_seeds_class_coverage_pass": all(
                    bool(record["class_coverage_pass"]) for record in seed_records
                ),
                "all_seeds_nll_better": all(
                    bool(record["combined_nll_better"]) for record in seed_records
                ),
                "all_seeds_ece_regression_pass": all(
                    bool(record["ece_regression_pass"]) for record in seed_records
                ),
                "all_seeds_credible_coverage_pass": all(
                    bool(record["credible_coverage_pass"]) for record in seed_records
                ),
                "bootstrap_support_count": int(
                    sum(
                        bool(record["bootstrap_interval_supports_improvement"])
                        for record in seed_records
                    )
                ),
            }
        )

    return {
        "schema": "hti.phase3-multiseed-summary.v0.8",
        "seeds": expected_seeds,
        "horizons_seconds": horizons,
        "protocol_sha256": protocol_sha256,
        "execution_config_sha256": execution_config_sha256,
        "variants": metric_summary,
        "claim_gate_evidence": horizon_gates,
        "scientific_note": (
            "This is internal synthetic multi-seed evidence. Passing these gates does not "
            "establish external domain validity, flightworthiness, or operational readiness."
        ),
    }
