import unittest

import numpy as np

from hti.phase3_models import (
    RidgeProbabilisticClassifier,
    history_consistency_weights,
    path_cell_distribution,
    select_temperature,
    smoothed_cell_distribution,
)


class Phase3ModelTests(unittest.TestCase):
    def test_ridge_classifier_outputs_normalized_probabilities(self):
        features = np.array(
            [[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.2, 0.9]], dtype=float
        )
        labels = np.array([0, 0, 1, 1], dtype=int)
        model = RidgeProbabilisticClassifier.fit(
            features, labels, classes=2, ridge_lambda=0.05
        )
        probabilities = model.probabilities(features)
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(4))
        self.assertEqual(probabilities.shape, (4, 2))

    def test_temperature_selection_uses_candidate_grid(self):
        features = np.array(
            [[0.0], [0.2], [1.0], [1.2], [0.1], [1.1]], dtype=float
        )
        labels = np.array([0, 0, 1, 1, 0, 1], dtype=int)
        model = RidgeProbabilisticClassifier.fit(
            features, labels, classes=2, ridge_lambda=0.05
        )
        candidates = np.array([0.5, 1.0, 2.0])
        temperature, nll = select_temperature(model, features, labels, candidates)
        self.assertIn(temperature, candidates.tolist())
        self.assertTrue(np.isfinite(nll))

    def test_history_consistency_downweights_turning_away(self):
        paths = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [-1.0, 0.0]],
            ]
        )
        weights = history_consistency_weights(
            np.array([0.5, 0.5]),
            paths,
            origin=np.array([0.0, 0.0]),
            travel_direction=np.array([1.0, 0.0]),
            gamma=2.0,
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[0], weights[1])

    def test_path_cell_distribution_preserves_unresolved_mass(self):
        result = path_cell_distribution(
            np.array([0.4, 0.3, 0.3]),
            np.array([0, 1, -1]),
            classes=3,
            smoothing=0.0,
        )
        self.assertAlmostEqual(float(result.sum()), 1.0)
        np.testing.assert_allclose(result, np.array([0.5, 0.4, 0.1]))

    def test_smoothed_cell_distribution_handles_unresolved(self):
        uniform = smoothed_cell_distribution(None, classes=4, smoothing=0.02)
        np.testing.assert_allclose(uniform, np.full(4, 0.25))
        resolved = smoothed_cell_distribution(2, classes=4, smoothing=0.04)
        self.assertEqual(int(np.argmax(resolved)), 2)
        self.assertAlmostEqual(float(resolved.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
