import unittest

import numpy as np

from hti.phase3 import Phase3Event
from hti.phase3_models import (
    RidgeProbabilisticClassifier,
    history_consistency_weights,
    measurement_history_vector,
    noisy_truth_endpoint_structure,
    path_cell_distribution,
    select_temperature,
    smoothed_cell_distribution,
)


def _event() -> Phase3Event:
    frames = 24
    states = np.zeros((frames, 13), dtype=float)
    states[:, 0] = np.arange(frames, dtype=float) * 10.0
    states[:, 1] = np.sin(np.arange(frames, dtype=float) / 5.0)
    states[:, 3] = 10.0
    states[:, 6] = 1.0
    measurements = np.column_stack([states[:, :3], states[:, 3:6]])
    return Phase3Event(
        event_id=17000001,
        tokens=np.zeros((1, 2, 2), dtype=np.float32),
        labels=np.zeros((1, 3), dtype=np.int64),
        next_features=np.zeros((1, 1), dtype=np.float32),
        source_frames=np.array([20], dtype=int),
        estimated_states=states.copy(),
        covariances=np.repeat(np.eye(13)[None, :, :], frames, axis=0),
        truth_states=states.copy(),
        measurements=measurements,
        controls=np.zeros((frames - 1, 6), dtype=float),
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

    def test_measurement_history_is_causal_and_frozen_dimension(self):
        event = _event()
        vector = measurement_history_vector(
            event, 20, history_points=10, history_stride_frames=2
        )
        self.assertEqual(vector.shape, (60,))
        np.testing.assert_allclose(vector[-6:], event.measurements[20])
        np.testing.assert_allclose(vector[:6], event.measurements[2])

    def test_noisy_truth_endpoint_structure_is_deterministic(self):
        event = _event()
        first, first_direction = noisy_truth_endpoint_structure(
            event,
            20,
            seed=17,
            dt=0.05,
            history_points=8,
            body_length=20.0,
            endpoint_noise_std=1.0,
        )
        second, second_direction = noisy_truth_endpoint_structure(
            event,
            20,
            seed=17,
            dt=0.05,
            history_points=8,
            body_length=20.0,
            endpoint_noise_std=1.0,
        )
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first_direction, second_direction)
        self.assertEqual(first.shape, (19,))
        self.assertAlmostEqual(float(np.linalg.norm(first_direction)), 1.0, places=12)

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
