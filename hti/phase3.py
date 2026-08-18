"""Phase 3 event-preserving synthetic evaluation utilities for HTI 0.8.

This module reuses the existing non-sensitive research simulator and estimator,
but preserves complete trajectory-event identity and source-frame provenance so
all later ablations can share the same events and event-level splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alien_exit_cell_predictor_v6_3 import (
    AeroModel,
    Atmosphere,
    Config,
    EarthModel,
    Projectile6DOF,
    compute_labels,
    extract_features,
    feature_dim,
    init_state,
    make_controls,
    make_env,
    make_measurements,
    rollout_truth,
    run_ukf,
    seed_everything,
)


@dataclass(frozen=True)
class Phase3Event:
    event_id: int
    tokens: np.ndarray
    labels: np.ndarray
    next_features: np.ndarray
    source_frames: np.ndarray
    estimated_states: np.ndarray
    covariances: np.ndarray
    truth_states: np.ndarray
    measurements: np.ndarray
    controls: np.ndarray


@dataclass(frozen=True)
class Phase3GenerationStats:
    requested_events: int
    generated_events: int
    attempts: int
    samples: int
    gate_updates: int
    gate_rejections: int

    @property
    def gate_rejection_rate(self) -> float:
        return float(self.gate_rejections / max(self.gate_updates, 1))


def build_windows_with_frames(
    cfg: Config,
    features: np.ndarray,
    controls: np.ndarray,
    labels: np.ndarray,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build core HTI windows while retaining their original event frame."""

    stride = max(1, int(stride))
    if features.ndim != 2 or controls.ndim != 2 or labels.ndim != 2:
        raise ValueError("features, controls, and labels must be two-dimensional")
    if features.shape[0] != labels.shape[0] + 1:
        raise ValueError("features must contain one more frame than labels")
    if controls.shape[0] != labels.shape[0]:
        raise ValueError("controls must align with label frames")

    features_s = features[::stride]
    control_pad = np.vstack([controls, np.zeros((1, controls.shape[1]), dtype=float)])
    controls_s = control_pad[::stride]
    labels_s = labels[::stride]
    frame_s = np.arange(labels.shape[0], dtype=int)[::stride]

    window = int(cfg.window)
    feature_dimensionality = features_s.shape[1]
    control_dimensionality = controls_s.shape[1]
    raw_dimensionality = feature_dimensionality + control_dimensionality
    horizons = labels_s.shape[1]

    tokens: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    next_features: list[np.ndarray] = []
    source_frames: list[int] = []

    for local_t in range(window - 1, labels_s.shape[0]):
        target = labels_s[local_t]
        if np.all(target < 0):
            continue
        feature_window = features_s[local_t - window + 1 : local_t + 1]
        control_window = controls_s[local_t - window + 1 : local_t + 1]
        token = np.zeros((2 * window, raw_dimensionality), dtype=np.float32)
        token[0::2, :feature_dimensionality] = feature_window
        token[1::2, feature_dimensionality:] = control_window
        if local_t + 1 < features_s.shape[0]:
            next_feature = features_s[local_t + 1]
        else:
            next_feature = features_s[local_t]
        tokens.append(token)
        targets.append(target.copy())
        next_features.append(next_feature.copy())
        source_frames.append(int(frame_s[local_t]))

    if not tokens:
        return (
            np.zeros((0, 2 * window, raw_dimensionality), dtype=np.float32),
            np.zeros((0, horizons), dtype=np.int64),
            np.zeros((0, feature_dimensionality), dtype=np.float32),
            np.zeros(0, dtype=int),
        )
    return (
        np.stack(tokens),
        np.stack(targets).astype(np.int64),
        np.stack(next_features).astype(np.float32),
        np.asarray(source_frames, dtype=int),
    )


def generate_phase3_events(
    cfg: Config,
    *,
    event_target: int,
    seed: int,
    max_attempt_factor: int = 3,
) -> tuple[list[Phase3Event], Phase3GenerationStats]:
    """Generate complete synthetic events without discarding event identity."""

    if event_target < 3:
        raise ValueError("event_target must be at least 3")
    if max_attempt_factor < 1:
        raise ValueError("max_attempt_factor must be positive")
    seed_everything(int(seed))

    earth, atmosphere, aero = EarthModel(), Atmosphere(), AeroModel()
    inertia = np.diag([500.0, 800.0, 1000.0])
    projectile = Projectile6DOF(1000.0, inertia, earth, atmosphere, aero)
    events: list[Phase3Event] = []
    attempts = 0
    gate_updates = 0
    gate_rejections = 0
    max_attempts = int(event_target * max_attempt_factor)

    while len(events) < event_target and attempts < max_attempts:
        attempts += 1
        environment = make_env(cfg)
        initial_state = init_state(cfg, earth)
        controls = make_controls(cfg, cfg.traj_steps)
        truth = rollout_truth(cfg, projectile, initial_state, controls, environment)
        measurements = make_measurements(cfg, truth)
        estimated, covariance, gate_stats = run_ukf(
            cfg, projectile, earth, atmosphere, measurements, controls, environment
        )
        gate_updates += int(gate_stats["total_updates"])
        gate_rejections += int(gate_stats["gate_rejections"])

        features = np.zeros((estimated.shape[0], feature_dim()), dtype=np.float32)
        for frame in range(estimated.shape[0]):
            features[frame] = extract_features(
                estimated[frame], covariance[frame], earth, atmosphere, environment
            )
        labels = compute_labels(cfg, estimated, truth)
        tokens, window_labels, next_features, source_frames = build_windows_with_frames(
            cfg,
            features,
            controls,
            labels,
            stride=max(1, int(cfg.offline_stride)),
        )
        if len(tokens) == 0:
            continue

        event_id = int(seed) * 1_000_000 + len(events)
        events.append(
            Phase3Event(
                event_id=event_id,
                tokens=tokens,
                labels=window_labels,
                next_features=next_features,
                source_frames=source_frames,
                estimated_states=estimated,
                covariances=covariance,
                truth_states=truth,
                measurements=measurements,
                controls=controls,
            )
        )

    if len(events) != event_target:
        raise RuntimeError(
            f"generated {len(events)} usable events after {attempts} attempts; "
            f"target was {event_target}"
        )
    stats = Phase3GenerationStats(
        requested_events=int(event_target),
        generated_events=int(len(events)),
        attempts=int(attempts),
        samples=int(sum(len(event.tokens) for event in events)),
        gate_updates=int(gate_updates),
        gate_rejections=int(gate_rejections),
    )
    return events, stats


def split_event_ids(
    event_ids: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, np.ndarray]:
    """Create a deterministic complete-event train/validation/test split."""

    unique = np.unique(np.asarray(event_ids, dtype=np.int64).reshape(-1))
    if unique.size < 3:
        raise ValueError("at least three complete events are required")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be below 1")

    rng = np.random.default_rng(int(seed) + 104729)
    order = rng.permutation(unique)
    count = len(order)
    train_count = max(1, int(train_fraction * count))
    validation_count = max(1, int(validation_fraction * count))
    if train_count + validation_count >= count:
        train_count = count - 2
        validation_count = 1
    return {
        "train": np.sort(order[:train_count]),
        "validation": np.sort(order[train_count : train_count + validation_count]),
        "test": np.sort(order[train_count + validation_count :]),
    }


def coverage_report_from_event_labels(
    event_labels: dict[int, np.ndarray],
    splits: dict[str, np.ndarray],
    *,
    num_classes: int,
) -> dict[str, dict[str, object]]:
    """Report valid-label counts and class coverage by split and horizon."""

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if not event_labels:
        raise ValueError("event_labels cannot be empty")
    first = next(iter(event_labels.values()))
    if np.asarray(first).ndim != 2:
        raise ValueError("event label arrays must be two-dimensional")
    horizons = np.asarray(first).shape[1]
    report: dict[str, dict[str, object]] = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in splits:
            raise ValueError(f"missing split {split_name!r}")
        arrays = []
        for event_id in np.asarray(splits[split_name]).reshape(-1):
            key = int(event_id)
            if key not in event_labels:
                raise ValueError(f"split references unknown event {key}")
            value = np.asarray(event_labels[key], dtype=int)
            if value.ndim != 2 or value.shape[1] != horizons:
                raise ValueError("all event label arrays must share the same horizon count")
            arrays.append(value)
        if not arrays:
            raise ValueError(f"split {split_name!r} is empty")
        joined = np.concatenate(arrays, axis=0)
        valid_counts: list[int] = []
        unique_classes: list[int] = []
        class_coverage: list[float] = []
        for horizon in range(horizons):
            valid = joined[:, horizon]
            valid = valid[valid >= 0]
            valid_counts.append(int(len(valid)))
            count = int(np.unique(valid).size) if len(valid) else 0
            unique_classes.append(count)
            class_coverage.append(float(count / num_classes))
        report[split_name] = {
            "events": int(len(np.asarray(splits[split_name]).reshape(-1))),
            "samples": int(len(joined)),
            "valid_labels": valid_counts,
            "unique_classes": unique_classes,
            "class_coverage": class_coverage,
        }
    return report


def coverage_report(
    events: list[Phase3Event],
    splits: dict[str, np.ndarray],
    *,
    num_classes: int,
) -> dict[str, dict[str, object]]:
    labels = {int(event.event_id): event.labels for event in events}
    return coverage_report_from_event_labels(labels, splits, num_classes=num_classes)
