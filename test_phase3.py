import unittest

import numpy as np

from alien_exit_cell_predictor_v6_3 import Config
from hti.phase3 import (
    build_windows_with_frames,
    coverage_report_from_event_labels,
    split_event_ids,
)


class Phase3Tests(unittest.TestCase):
    def test_window_builder_preserves_original_source_frames(self):
        cfg = Config()
        cfg.window = 2
        features = np.arange(7 * 3, dtype=np.float32).reshape(7, 3)
        controls = np.zeros((6, 6), dtype=float)
        labels = np.tile(np.array([[0, 1, 2]], dtype=int), (6, 1))
        tokens, targets, next_features, frames = build_windows_with_frames(
            cfg,
            features,
            controls,
            labels,
            stride=2,
        )
        np.testing.assert_array_equal(frames, np.array([2, 4]))
        self.assertEqual(tokens.shape[0], 2)
        self.assertEqual(targets.shape, (2, 3))
        self.assertEqual(next_features.shape, (2, 3))

    def test_event_split_is_disjoint_and_exhaustive(self):
        event_ids = np.arange(20, dtype=int)
        splits = split_event_ids(event_ids, seed=17)
        train = set(splits["train"].tolist())
        validation = set(splits["validation"].tolist())
        test = set(splits["test"].tolist())
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(train | validation | test, set(event_ids.tolist()))
        self.assertEqual((len(train), len(validation), len(test)), (14, 3, 3))

    def test_coverage_report_uses_complete_event_membership(self):
        event_labels = {
            10: np.array([[0, 0], [1, 1]], dtype=int),
            11: np.array([[2, 2], [3, 3]], dtype=int),
            20: np.array([[0, 1], [0, 1]], dtype=int),
            21: np.array([[1, 2], [1, 2]], dtype=int),
            30: np.array([[2, 3], [2, 3]], dtype=int),
            31: np.array([[3, 0], [3, 0]], dtype=int),
        }
        splits = {
            "train": np.array([10, 11]),
            "validation": np.array([20, 21]),
            "test": np.array([30, 31]),
        }
        report = coverage_report_from_event_labels(event_labels, splits, num_classes=4)
        self.assertEqual(report["train"]["events"], 2)
        self.assertEqual(report["train"]["class_coverage"][0], 1.0)
        self.assertEqual(report["validation"]["class_coverage"][0], 0.5)
        self.assertEqual(report["test"]["class_coverage"][0], 0.5)


if __name__ == "__main__":
    unittest.main()
