"""Shared sample-eligibility rules for the frozen HTI 0.8 Phase 3 study."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .phase3 import Phase3Event, coverage_report_from_event_labels, split_event_ids


def minimum_source_frame(execution: dict[str, object]) -> int:
    """Return the earliest frame usable by every frozen Phase 3 variant."""

    learned = execution["learned_only"]
    structural = execution["structural_branch"]
    learned_min = (int(learned["history_points"]) - 1) * int(
        learned["history_stride_frames"]
    )
    structural_min = int(structural["history_points"]) - 1
    return max(learned_min, structural_min)


def trim_event(event: Phase3Event, *, min_source_frame: int) -> Phase3Event:
    """Restrict one event to samples eligible for every frozen comparison branch."""

    mask = np.asarray(event.source_frames, dtype=int) >= int(min_source_frame)
    if not np.any(mask):
        raise ValueError(f"event {event.event_id} has no jointly eligible samples")
    return replace(
        event,
        tokens=event.tokens[mask],
        labels=event.labels[mask],
        next_features=event.next_features[mask],
        source_frames=event.source_frames[mask],
    )


def trim_events(
    events: list[Phase3Event], *, min_source_frame: int
) -> list[Phase3Event]:
    return [trim_event(event, min_source_frame=min_source_frame) for event in events]


def joint_valid_coverage_report(
    events: list[Phase3Event],
    splits: dict[str, np.ndarray],
    *,
    num_classes: int,
) -> dict[str, dict[str, object]]:
    """Report class coverage on rows valid at every frozen horizon.

    This matches the final evaluator bundle, which compares all variants on the
    same sample rows and therefore excludes rows lacking any horizon label.
    """

    labels: dict[int, np.ndarray] = {}
    for event in events:
        values = np.asarray(event.labels, dtype=int)
        joint = np.all(values >= 0, axis=1)
        if not np.any(joint):
            raise ValueError(f"event {event.event_id} has no all-horizon-valid samples")
        labels[int(event.event_id)] = values[joint]
    return coverage_report_from_event_labels(labels, splits, num_classes=num_classes)


def development_support_audit(
    events: list[Phase3Event],
    *,
    num_classes: int,
    train_fraction: float,
    validation_fraction: float,
    minimum_coverage: float,
    split_trials: int = 32,
) -> dict[str, object]:
    """Diagnose generator support and random complete-event split stability.

    This is a development-only calculation audit. It never trains a model and
    never consumes prediction scores. Labels are used only to measure whether
    the simulator/grid supplies enough class support for a fair experiment.
    """

    if split_trials < 2:
        raise ValueError("split_trials must be at least 2")
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must lie in (0, 1]")

    event_labels: dict[int, np.ndarray] = {}
    for event in events:
        values = np.asarray(event.labels, dtype=int)
        joint = np.all(values >= 0, axis=1)
        if not np.any(joint):
            raise ValueError(f"event {event.event_id} has no all-horizon-valid samples")
        event_labels[int(event.event_id)] = values[joint]

    event_ids = np.asarray(sorted(event_labels), dtype=np.int64)
    joined = np.concatenate([event_labels[int(event_id)] for event_id in event_ids], axis=0)
    horizons = joined.shape[1]
    pooled_unique: list[int] = []
    pooled_coverage: list[float] = []
    pooled_classes: list[list[int]] = []
    class_event_counts: list[list[int]] = []
    per_event_unique: list[list[int]] = []
    for horizon in range(horizons):
        classes = np.unique(joined[:, horizon])
        pooled_classes.append([int(value) for value in classes])
        pooled_unique.append(int(classes.size))
        pooled_coverage.append(float(classes.size / num_classes))
        class_event_counts.append(
            [
                int(sum(class_id in np.unique(labels[:, horizon]) for labels in event_labels.values()))
                for class_id in range(num_classes)
            ]
        )
        per_event_unique.append(
            [int(np.unique(labels[:, horizon]).size) for labels in event_labels.values()]
        )

    trial_values = {name: [[] for _ in range(horizons)] for name in ("validation", "test")}
    trial_passes = {name: 0 for name in trial_values}
    for trial in range(split_trials):
        splits = split_event_ids(
            event_ids,
            seed=17_000 + trial,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        coverage = coverage_report_from_event_labels(
            event_labels, splits, num_classes=num_classes
        )
        for name in trial_values:
            values = [float(value) for value in coverage[name]["class_coverage"]]
            for horizon, value in enumerate(values):
                trial_values[name][horizon].append(value)
            trial_passes[name] += int(min(values) >= minimum_coverage)

    stability: dict[str, object] = {}
    for name, horizon_values in trial_values.items():
        stability[name] = {
            "coverage_by_horizon": [
                {
                    "min": float(np.min(values)),
                    "median": float(np.median(values)),
                    "max": float(np.max(values)),
                }
                for values in horizon_values
            ],
            "all_horizons_gate_pass_fraction": float(trial_passes[name] / split_trials),
        }

    return {
        "model_training_used": False,
        "prediction_scores_used": False,
        "events": int(len(event_ids)),
        "joint_valid_samples": int(len(joined)),
        "pooled_unique_classes": pooled_unique,
        "pooled_grid_coverage": pooled_coverage,
        "pooled_class_ids": pooled_classes,
        "class_event_counts": class_event_counts,
        "per_event_unique_classes": [
            {
                "min": int(np.min(values)),
                "median": float(np.median(values)),
                "max": int(np.max(values)),
            }
            for values in per_event_unique
        ],
        "split_stability": {
            "trials": int(split_trials),
            "seed_range": [17_000, 17_000 + split_trials - 1],
            **stability,
        },
    }
