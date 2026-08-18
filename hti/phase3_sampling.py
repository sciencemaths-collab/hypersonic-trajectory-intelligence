"""Shared sample-eligibility rules for the frozen HTI 0.8 Phase 3 study."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .phase3 import Phase3Event, coverage_report_from_event_labels


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
