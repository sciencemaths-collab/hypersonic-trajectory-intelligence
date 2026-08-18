#!/usr/bin/env python3
"""Evaluate frozen HTI 0.8 probability bundles on complete trajectory events.

The evaluator consumes already-produced probabilities. It never trains or tunes
on final-test data. It independently reconstructs every fused variant from the
frozen bundle inputs before reporting metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hti.benchmarking import (  # noqa: E402
    assert_disjoint_event_splits,
    conformal_set_stats,
    evaluate_probabilities,
    event_bootstrap_delta,
    nll_contributions,
    reliability_bins,
    selective_risk_curve,
)
from hti.fusion import apply_cell_prior, log_linear_pool  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


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
        raise ValueError("probabilities must have shape (samples, horizons, classes)")
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    totals = value.sum(axis=2, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("every probability row must have positive mass")
    return value / totals


def _scalar(bundle: dict[str, np.ndarray], key: str) -> object:
    if key not in bundle:
        raise ValueError(f"bundle is missing required field {key!r}")
    values = np.asarray(bundle[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"bundle field {key!r} must be scalar")
    return values[0].item() if hasattr(values[0], "item") else values[0]


def _sha256_value(bundle: dict[str, np.ndarray], key: str) -> str:
    value = str(_scalar(bundle, key)).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"bundle field {key!r} must be a lowercase SHA-256 digest")
    return value


def _cell_partition_sha256(cell_ids: np.ndarray) -> str:
    encoded = json.dumps(
        [str(value) for value in np.asarray(cell_ids).reshape(-1)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _horizon_values(bundle: dict[str, np.ndarray], key: str, horizons: int) -> np.ndarray | None:
    if key not in bundle:
        return None
    values = np.asarray(bundle[key], dtype=float).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, horizons)
    if values.size != horizons or not np.isfinite(values).all():
        raise ValueError(f"{key!r} must contain one value or one value per horizon")
    return values


def _apply_topology_cube(
    probabilities: np.ndarray,
    topology_prior: np.ndarray,
    *,
    strength: float,
    floor: float,
) -> np.ndarray:
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("topology prior strength must be finite and non-negative")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("topology prior floor must be finite and positive")
    if probabilities.shape != topology_prior.shape:
        raise ValueError("topology prior must match probability cube shape")
    output = np.empty_like(probabilities, dtype=float)
    for horizon in range(probabilities.shape[1]):
        output[:, horizon, :] = apply_cell_prior(
            probabilities[:, horizon, :],
            np.maximum(topology_prior[:, horizon, :], float(floor)),
            strength=float(strength),
        )
    return output


def _synthetic_cube(labels: np.ndarray, classes: int, correct_mass: float) -> np.ndarray:
    samples, horizons = labels.shape
    cube = np.full(
        (samples, horizons, classes),
        (1.0 - correct_mass) / (classes - 1),
        dtype=float,
    )
    for horizon in range(horizons):
        cube[np.arange(samples), horizon, labels[:, horizon]] = correct_mass
    return cube


def _self_test_bundle(
    *,
    protocol_sha256: str,
    execution_config_sha256: str,
    execution: dict[str, object],
) -> dict[str, np.ndarray]:
    samples = 120
    horizons = 3
    classes = 4
    base = np.tile(np.arange(classes, dtype=int), samples // classes)
    labels = np.column_stack([base, np.roll(base, 1), np.roll(base, 2)])
    test_event_ids = np.arange(300, 312, dtype=int)
    event_ids = np.repeat(test_event_ids, 10)
    candidates = np.linspace(0.0, 1.0, 21)

    physics = _synthetic_cube(labels, classes, 0.52)
    transformer = _synthetic_cube(labels, classes, 0.62)
    core_weights = np.full(horizons, candidates[6], dtype=float)
    core = np.empty_like(transformer)
    for horizon, weight in enumerate(core_weights):
        core[:, horizon, :] = log_linear_pool(
            transformer[:, horizon, :],
            physics[:, horizon, :],
            structural_weight=float(weight),
        )

    raw_structural = _synthetic_cube(labels, classes, 0.72)
    structural_weights = np.full(horizons, candidates[10], dtype=float)
    core_plus_structural = np.empty_like(core)
    for horizon, weight in enumerate(structural_weights):
        core_plus_structural[:, horizon, :] = log_linear_pool(
            core[:, horizon, :],
            raw_structural[:, horizon, :],
            structural_weight=float(weight),
        )

    topology_prior = _synthetic_cube(labels, classes, 0.62)
    topology_cfg = execution["topology_branch"]
    topology_strength = float(topology_cfg["cell_prior_strength"])
    topology_floor = float(topology_cfg["cell_prior_floor"])
    core_plus_topology = _apply_topology_cube(
        core, topology_prior, strength=topology_strength, floor=topology_floor
    )
    combined = _apply_topology_cube(
        core_plus_structural,
        topology_prior,
        strength=topology_strength,
        floor=topology_floor,
    )
    variants = {
        "constant_velocity": _synthetic_cube(labels, classes, 0.35),
        "filter_direct": _synthetic_cube(labels, classes, 0.40),
        "physics_only": physics,
        "learned_only": _synthetic_cube(labels, classes, 0.48),
        "core_hti": core,
        "core_plus_structural": core_plus_structural,
        "core_plus_topology": core_plus_topology,
        "hti_08_combined": combined,
    }

    cell_ids = np.array([f"cell-{index}" for index in range(classes)])
    validation_digest = hashlib.sha256(b"synthetic-validation-selection").hexdigest()
    learned_digest = hashlib.sha256(b"synthetic-learned-model").hexdigest()
    core_digest = hashlib.sha256(b"synthetic-core-transformer-model").hexdigest()
    structural_digest = hashlib.sha256(b"synthetic-structural-model").hexdigest()
    topology_definition = {
        "kind": topology_cfg["kind"],
        "history_direction_source": topology_cfg["history_direction_source"],
        "path_distance": topology_cfg["path_distance"],
        "path_reweighting": "hti.phase3_models.history_consistency_weights",
        "cell_prior_application": "hti.fusion.apply_cell_prior",
    }
    topology_definition_digest = _sha256_json(topology_definition)
    topology_coefficients_digest = _sha256_json(topology_cfg)
    partition_digest = _cell_partition_sha256(cell_ids)

    bundle: dict[str, np.ndarray] = {
        "labels": labels,
        "event_ids": event_ids,
        "seed": np.array([101], dtype=int),
        "train_event_ids": np.arange(100, 120, dtype=int),
        "validation_event_ids": np.arange(200, 205, dtype=int),
        "test_event_ids": test_event_ids,
        "orientation_source": np.array([str(execution["structural_branch"]["orientation_source"])]),
        "fusion_selection_data": np.array(["validation_only"]),
        "core_physics_weights": core_weights,
        "fusion_structural_weights": structural_weights,
        "validation_selection_sha256": np.array([validation_digest]),
        "source_commit": np.array(["0" * 40]),
        "protocol_sha256": np.array([protocol_sha256]),
        "execution_config_sha256": np.array([execution_config_sha256]),
        "topology_definition_sha256": np.array([topology_definition_digest]),
        "topology_coefficients_sha256": np.array([topology_coefficients_digest]),
        "cell_partition_sha256": np.array([partition_digest]),
        "model_sha256__learned_only": np.array([learned_digest]),
        "model_sha256__core_hti": np.array([core_digest]),
        "model_sha256__structural_branch": np.array([structural_digest]),
        "core_transformer_probabilities": transformer,
        "structural_branch_probabilities": raw_structural,
        "topology_cell_prior": topology_prior,
        "topology_true_path_suppressed": np.zeros((samples, horizons), dtype=np.uint8),
    }
    for name, probabilities in variants.items():
        bundle[f"probs__{name}"] = probabilities
        bundle[f"cell_ids__{name}"] = cell_ids
        bundle[f"conformal_qhat__{name}"] = np.full(horizons, 0.70, dtype=float)
        bundle[f"conformal_calibration_sha256__{name}"] = np.array([validation_digest])
    return bundle


def _load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _validate_fusion_weights(
    values: np.ndarray | None,
    *,
    candidates: np.ndarray,
    field_name: str,
) -> np.ndarray:
    if values is None or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError(f"{field_name} must contain frozen values in [0, 1]")
    if any(
        not np.any(np.isclose(weight, candidates, rtol=0.0, atol=1e-12))
        for weight in values
    ):
        raise ValueError(f"{field_name} must come from the frozen candidate grid")
    return values


def _matches_serialized_probabilities(
    reproduced: np.ndarray,
    recorded: np.ndarray,
) -> bool:
    """Compare a reconstruction at the precision recorded by the producer.

    The Transformer path is float32, so ``_select_pool`` serializes the core and
    structural fusion cubes as float32 even though ``log_linear_pool`` computes
    in float64. Casting the reconstruction to the recorded dtype preserves a
    strict composition check without rejecting the producer's own rounding.
    """

    recorded_array = np.asarray(recorded)
    reproduced_array = np.asarray(reproduced, dtype=recorded_array.dtype)
    return bool(
        np.allclose(
            reproduced_array,
            recorded_array,
            rtol=1e-10,
            atol=1e-12,
        )
    )


def _validate_contract(
    bundle: dict[str, np.ndarray],
    config: dict[str, object],
    execution: dict[str, object],
    *,
    protocol_sha256: str,
    execution_config_sha256: str,
    labels: np.ndarray,
    event_ids: np.ndarray,
    variants: dict[str, np.ndarray],
) -> dict[str, object]:
    seed = int(_scalar(bundle, "seed"))
    if seed not in {int(value) for value in config["seeds"]}:
        raise ValueError(f"seed {seed} is not present in the frozen protocol")

    bundle_source_commit = str(_scalar(bundle, "source_commit")).lower()
    if len(bundle_source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in bundle_source_commit
    ):
        raise ValueError("source_commit must identify the immutable bundle-producing commit")

    if _sha256_value(bundle, "protocol_sha256") != protocol_sha256:
        raise ValueError("protocol_sha256 does not match the supplied frozen protocol")
    if _sha256_value(bundle, "execution_config_sha256") != execution_config_sha256:
        raise ValueError("execution_config_sha256 does not match the supplied frozen execution config")

    train_ids = np.asarray(bundle.get("train_event_ids", [])).reshape(-1)
    validation_ids = np.asarray(bundle.get("validation_event_ids", [])).reshape(-1)
    test_ids = np.asarray(bundle.get("test_event_ids", [])).reshape(-1)
    if min(len(train_ids), len(validation_ids), len(test_ids)) == 0:
        raise ValueError("bundle must contain non-empty train/validation/test event-id manifests")
    assert_disjoint_event_splits(train_ids, validation_ids, test_ids)
    if set(np.unique(event_ids).tolist()) != set(np.unique(test_ids).tolist()):
        raise ValueError("sample event_ids must represent exactly the frozen test-event set")

    required_variants = {str(name) for name in config["variants"]}
    missing_variants = sorted(required_variants - set(variants))
    if missing_variants:
        raise ValueError(f"bundle is missing frozen variants: {missing_variants}")

    reference_cells: np.ndarray | None = None
    for name in sorted(required_variants):
        key = f"cell_ids__{name}"
        if key not in bundle:
            raise ValueError(f"bundle is missing ordered cell partition {key!r}")
        cells = np.asarray(bundle[key]).reshape(-1)
        if cells.size != variants[name].shape[2] or len(np.unique(cells)) != cells.size:
            raise ValueError(f"{key!r} must contain one unique ID per probability class")
        if reference_cells is None:
            reference_cells = cells
        elif not np.array_equal(cells, reference_cells):
            raise ValueError("variant cell partitions or ordered cell identities differ")
    partition_sha256 = _sha256_value(bundle, "cell_partition_sha256")
    if partition_sha256 != _cell_partition_sha256(reference_cells):
        raise ValueError("cell_partition_sha256 does not match the ordered cell IDs")

    if str(_scalar(bundle, "fusion_selection_data")) != "validation_only":
        raise ValueError("fusion_selection_data must be 'validation_only'")
    validation_sha256 = _sha256_value(bundle, "validation_selection_sha256")
    candidates = np.asarray(config["fusion_policy"]["candidate_structural_weights"], dtype=float)
    core_weights = _validate_fusion_weights(
        _horizon_values(bundle, "core_physics_weights", labels.shape[1]),
        candidates=candidates,
        field_name="core_physics_weights",
    )
    structural_weights = _validate_fusion_weights(
        _horizon_values(bundle, "fusion_structural_weights", labels.shape[1]),
        candidates=candidates,
        field_name="fusion_structural_weights",
    )

    if "core_transformer_probabilities" not in bundle:
        raise ValueError("bundle is missing core_transformer_probabilities")
    transformer = _as_probability_cube(bundle["core_transformer_probabilities"], labels.shape[1])
    if transformer.shape != variants["core_hti"].shape:
        raise ValueError("core_transformer_probabilities must match the core probability cube")
    for horizon, weight in enumerate(core_weights):
        reproduced_core = log_linear_pool(
            transformer[:, horizon, :],
            variants["physics_only"][:, horizon, :],
            structural_weight=float(weight),
        )
        if not _matches_serialized_probabilities(
            reproduced_core,
            bundle["probs__core_hti"][:, horizon, :],
        ):
            raise ValueError("core_hti does not reproduce from Transformer, physics-only, and frozen core weights")

    if "structural_branch_probabilities" not in bundle:
        raise ValueError("bundle is missing raw structural_branch_probabilities")
    raw_structural = _as_probability_cube(
        bundle["structural_branch_probabilities"], labels.shape[1]
    )
    if raw_structural.shape != variants["core_hti"].shape:
        raise ValueError("structural_branch_probabilities must match the core probability cube")
    for horizon, weight in enumerate(structural_weights):
        reproduced_structural = log_linear_pool(
            variants["core_hti"][:, horizon, :],
            raw_structural[:, horizon, :],
            structural_weight=float(weight),
        )
        if not _matches_serialized_probabilities(
            reproduced_structural,
            bundle["probs__core_plus_structural"][:, horizon, :],
        ):
            raise ValueError("core_plus_structural does not reproduce from core, raw structural probabilities, and frozen weights")

    if "topology_cell_prior" not in bundle:
        raise ValueError("bundle is missing topology_cell_prior")
    topology_prior = _as_probability_cube(bundle["topology_cell_prior"], labels.shape[1])
    if topology_prior.shape != variants["core_hti"].shape:
        raise ValueError("topology_cell_prior must match the core probability cube")
    topology_cfg = execution["topology_branch"]
    topology_strength = float(topology_cfg["cell_prior_strength"])
    topology_floor = float(topology_cfg["cell_prior_floor"])
    reproduced_topology = _apply_topology_cube(
        variants["core_hti"],
        topology_prior,
        strength=topology_strength,
        floor=topology_floor,
    )
    if not _matches_serialized_probabilities(
        reproduced_topology,
        bundle["probs__core_plus_topology"],
    ):
        raise ValueError("core_plus_topology does not reproduce from core and frozen topology prior")
    reproduced_combined = _apply_topology_cube(
        variants["core_plus_structural"],
        topology_prior,
        strength=topology_strength,
        floor=topology_floor,
    )
    if not _matches_serialized_probabilities(
        reproduced_combined,
        bundle["probs__hti_08_combined"],
    ):
        raise ValueError("hti_08_combined does not reproduce from structural fusion plus frozen topology prior")

    orientation_source = str(_scalar(bundle, "orientation_source"))
    expected_orientation = str(execution["structural_branch"]["orientation_source"])
    if orientation_source != expected_orientation:
        raise ValueError("orientation_source does not match the frozen structural observation source")

    expected_topology_definition = _sha256_json(
        {
            "kind": topology_cfg["kind"],
            "history_direction_source": topology_cfg["history_direction_source"],
            "path_distance": topology_cfg["path_distance"],
            "path_reweighting": "hti.phase3_models.history_consistency_weights",
            "cell_prior_application": "hti.fusion.apply_cell_prior",
        }
    )
    topology_definition_sha256 = _sha256_value(bundle, "topology_definition_sha256")
    if topology_definition_sha256 != expected_topology_definition:
        raise ValueError("topology_definition_sha256 does not match the frozen topology definition")
    topology_coefficients_sha256 = _sha256_value(bundle, "topology_coefficients_sha256")
    if topology_coefficients_sha256 != _sha256_json(topology_cfg):
        raise ValueError("topology_coefficients_sha256 does not match frozen topology coefficients")

    model_sha256 = {
        "learned_only": _sha256_value(bundle, "model_sha256__learned_only"),
        "core_hti": _sha256_value(bundle, "model_sha256__core_hti"),
        "structural_branch": _sha256_value(bundle, "model_sha256__structural_branch"),
    }
    for name in required_variants:
        if f"conformal_qhat__{name}" in bundle:
            _sha256_value(bundle, f"conformal_calibration_sha256__{name}")

    if bool(config["topology_policy"].get("report_true_path_suppression_failures", False)):
        if "topology_true_path_suppressed" not in bundle:
            raise ValueError("frozen topology policy requires topology_true_path_suppressed evidence")
    if "topology_true_path_suppressed" in bundle:
        suppressed = np.asarray(bundle["topology_true_path_suppressed"])
        if suppressed.shape != labels.shape:
            raise ValueError("topology_true_path_suppressed must match labels shape")

    return {
        "seed": seed,
        "bundle_source_commit": bundle_source_commit,
        "orientation_source": orientation_source,
        "protocol_sha256": protocol_sha256,
        "execution_config_sha256": execution_config_sha256,
        "cell_partition_sha256": partition_sha256,
        "validation_selection_sha256": validation_sha256,
        "topology_definition_sha256": topology_definition_sha256,
        "topology_coefficients_sha256": topology_coefficients_sha256,
        "model_sha256": model_sha256,
        "core_physics_weights": [float(value) for value in core_weights],
        "fusion_structural_weights": [float(value) for value in structural_weights],
        "topology_cell_prior_strength": topology_strength,
        "topology_cell_prior_floor": topology_floor,
        "split_event_counts": {
            "train": int(len(np.unique(train_ids))),
            "validation": int(len(np.unique(validation_ids))),
            "test": int(len(np.unique(test_ids))),
        },
        "required_variants_present": True,
        "event_splits_disjoint": True,
        "core_fusion_reproducible": True,
        "structural_fusion_reproducible": True,
        "topology_application_reproducible": True,
        "combined_reproducible": True,
    }


def _topology_stats(
    core: np.ndarray,
    topology: np.ndarray,
    labels: np.ndarray,
    suppressed: np.ndarray | None,
) -> list[dict[str, float | int]]:
    reports: list[dict[str, float | int]] = []
    for horizon in range(labels.shape[1]):
        core_h = core[:, horizon, :]
        topology_h = topology[:, horizon, :]
        y = labels[:, horizon]
        total_variation = 0.5 * np.sum(np.abs(topology_h - core_h), axis=1)
        true_delta = topology_h[np.arange(len(y)), y] - core_h[np.arange(len(y)), y]
        record: dict[str, float | int] = {
            "mean_total_variation": float(np.mean(total_variation)),
            "median_total_variation": float(np.median(total_variation)),
            "max_total_variation": float(np.max(total_variation)),
            "mean_true_cell_probability_delta": float(np.mean(true_delta)),
            "fraction_true_cell_probability_decreased": float(np.mean(true_delta < 0.0)),
        }
        if suppressed is not None:
            flags = np.asarray(suppressed[:, horizon], dtype=bool)
            record["true_path_suppression_count"] = int(np.count_nonzero(flags))
            record["true_path_suppression_fraction"] = float(np.mean(flags))
        reports.append(record)
    return reports


def evaluate(
    bundle: dict[str, np.ndarray],
    config: dict[str, object],
    execution: dict[str, object],
    *,
    protocol_sha256: str,
    execution_config_sha256: str,
) -> dict[str, object]:
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

    contract = _validate_contract(
        bundle,
        config,
        execution,
        protocol_sha256=protocol_sha256,
        execution_config_sha256=execution_config_sha256,
        labels=labels,
        event_ids=event_ids,
        variants=variants,
    )
    report: dict[str, object] = {
        "schema": "hti.ablation-report.v0.8",
        "protocol_status": config.get("status"),
        "samples": int(len(labels)),
        "unique_events": int(len(np.unique(event_ids))),
        "horizons_seconds": horizons,
        "contract": contract,
        "variants": {},
        "paired_core_vs_combined": {},
        "topology_effect": {},
        "scientific_note": (
            "This evaluator reports frozen-test evidence only. It does not establish external "
            "domain validity or operational readiness."
        ),
    }

    variant_report: dict[str, object] = {}
    for name, cube in sorted(variants.items()):
        qhat = _horizon_values(bundle, f"conformal_qhat__{name}", len(horizons))
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
            record: dict[str, object] = {
                "horizon_seconds": horizon,
                "metrics": metrics.to_dict(),
                "reliability_bins": reliability_bins(probabilities, y, bins=ece_bins),
                "selective_risk": selective_risk_curve(probabilities, y),
            }
            if qhat is not None:
                record["conformal"] = conformal_set_stats(probabilities, y, float(qhat[index]))
            horizon_report.append(record)
        variant_report[name] = horizon_report
    report["variants"] = variant_report

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

    suppressed = bundle.get("topology_true_path_suppressed")
    report["topology_effect"] = {
        str(horizon): values
        for horizon, values in zip(
            horizons,
            _topology_stats(
                variants["core_hti"],
                variants["core_plus_topology"],
                labels,
                np.asarray(suppressed) if suppressed is not None else None,
            ),
            strict=True,
        )
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/hti_08_ablation_frozen.json")
    )
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=Path("configs/hti_08_phase3_execution_frozen.json"),
    )
    parser.add_argument("--report", type=Path, default=Path("hti08_ablation_report.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test == (args.bundle is not None):
        raise SystemExit("choose exactly one of --self-test or --bundle")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    execution = json.loads(args.execution_config.read_text(encoding="utf-8"))
    protocol_sha256 = _sha256_file(args.config)
    execution_sha256 = _sha256_file(args.execution_config)
    bundle = (
        _self_test_bundle(
            protocol_sha256=protocol_sha256,
            execution_config_sha256=execution_sha256,
            execution=execution,
        )
        if args.self_test
        else _load_bundle(args.bundle)
    )
    report = evaluate(
        bundle,
        config,
        execution,
        protocol_sha256=protocol_sha256,
        execution_config_sha256=execution_sha256,
    )
    bundle_source_commit = str(report["contract"]["bundle_source_commit"])
    report["source_commit"] = bundle_source_commit
    report["bundle_source_commit"] = bundle_source_commit
    report["evaluator_source_commit"] = _source_commit()
    report["protocol_sha256"] = protocol_sha256
    report["execution_config_sha256"] = execution_sha256
    report["bundle_sha256"] = "self-test-generated" if args.self_test else _sha256_file(args.bundle)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
