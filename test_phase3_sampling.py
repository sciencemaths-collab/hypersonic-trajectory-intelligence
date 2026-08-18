import unittest

import numpy as np

from hti.phase3 import Phase3Event
from hti.phase3_sampling import (
    development_support_audit,
    joint_valid_coverage_report,
    minimum_source_frame,
    trim_event,
)


def _event() -> Phase3Event:
    frames = np.array([14, 16, 18, 20, 22], dtype=int)
    labels = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [0, 0, -1],
        ],
        dtype=int,
    )
    return Phase3Event(
        event_id=17000001,
        tokens=np.zeros((5, 2, 2), dtype=np.float32),
        labels=labels,
        next_features=np.zeros((5, 1), dtype=np.float32),
        source_frames=frames,
        estimated_states=np.zeros((30, 13), dtype=float),
        covariances=np.repeat(np.eye(13)[None, :, :], 30, axis=0),
        truth_states=np.zeros((30, 13), dtype=float),
        measurements=np.zeros((30, 6), dtype=float),
        controls=np.zeros((29, 6), dtype=float),
    )


class Phase3SamplingTests(unittest.TestCase):
    def test_minimum_source_frame_covers_all_frozen_histories(self):
        execution = {
            "learned_only": {"history_points": 10, "history_stride_frames": 2},
            "structural_branch": {"history_points": 8},
        }
        self.assertEqual(minimum_source_frame(execution), 18)

    def test_trim_event_applies_shared_eligibility(self):
        trimmed = trim_event(_event(), min_source_frame=18)
        np.testing.assert_array_equal(trimmed.source_frames, np.array([18, 20, 22]))
        self.assertEqual(len(trimmed.tokens), 3)
        self.assertEqual(len(trimmed.labels), 3)

    def test_joint_coverage_uses_all_horizon_valid_rows(self):
        event = trim_event(_event(), min_source_frame=18)
        splits = {
            "train": np.array([event.event_id]),
            "validation": np.array([event.event_id]),
            "test": np.array([event.event_id]),
        }
        report = joint_valid_coverage_report([event], splits, num_classes=4)
        self.assertEqual(report["test"]["samples"], 2)
        self.assertEqual(report["test"]["unique_classes"], [2, 2, 2])

    def test_support_audit_separates_generator_support_from_split_stability(self):
        events = []
        for event_id in range(12):
            event = _event()
            labels = np.full((5, 3), event_id % 4, dtype=int)
            events.append(
                Phase3Event(
                    event_id=17_000_000 + event_id,
                    tokens=event.tokens,
                    labels=labels,
                    next_features=event.next_features,
                    source_frames=event.source_frames,
                    estimated_states=event.estimated_states,
                    covariances=event.covariances,
                    truth_states=event.truth_states,
                    measurements=event.measurements,
                    controls=event.controls,
                )
            )
        audit = development_support_audit(
            events,
            num_classes=4,
            train_fraction=0.5,
            validation_fraction=0.25,
            minimum_coverage=0.5,
            split_trials=8,
        )
        self.assertFalse(audit["model_training_used"])
        self.assertFalse(audit["prediction_scores_used"])
        self.assertEqual(audit["pooled_unique_classes"], [4, 4, 4])
        self.assertEqual(audit["pooled_grid_coverage"], [1.0, 1.0, 1.0])
        self.assertEqual(audit["split_stability"]["trials"], 8)


if __name__ == "__main__":
    unittest.main()
