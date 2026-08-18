import unittest

import numpy as np

from hti.phase3 import Phase3Event
from hti.phase3_sampling import (
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


if __name__ == "__main__":
    unittest.main()
