import unittest

import numpy as np

from hti.assurance import (
    assess_prediction,
    confidence_margin,
    conformal_prediction_set,
    conformal_quantile,
    empirical_set_coverage,
    normalized_entropy,
)


class AssuranceTests(unittest.TestCase):
    def test_entropy_bounds(self):
        self.assertAlmostEqual(normalized_entropy(np.array([0.25, 0.25, 0.25, 0.25])), 1.0)
        self.assertAlmostEqual(normalized_entropy(np.array([1.0, 0.0, 0.0, 0.0])), 0.0)

    def test_margin(self):
        self.assertAlmostEqual(confidence_margin(np.array([0.70, 0.20, 0.10])), 0.50)

    def test_low_information_prediction_abstains(self):
        decision = assess_prediction(np.array([0.26, 0.25, 0.25, 0.24]))
        self.assertFalse(decision.accept)
        self.assertIn("low confidence", decision.reason)
        self.assertIn("high predictive entropy", decision.reason)

    def test_clear_prediction_is_accepted(self):
        decision = assess_prediction(np.array([0.80, 0.10, 0.06, 0.04]))
        self.assertTrue(decision.accept)
        self.assertEqual(decision.prediction, 0)
        self.assertEqual(decision.reason, "accepted")

    def test_conformal_set_is_nonempty(self):
        selected = conformal_prediction_set(np.array([0.45, 0.30, 0.25]), qhat=0.20)
        self.assertEqual(selected, [0])

    def test_conformal_quantile_and_empirical_coverage(self):
        calibration = np.array(
            [
                [0.80, 0.10, 0.10],
                [0.15, 0.75, 0.10],
                [0.10, 0.10, 0.80],
                [0.70, 0.20, 0.10],
                [0.15, 0.70, 0.15],
            ]
        )
        labels = np.array([0, 1, 2, 0, 1])
        qhat = conformal_quantile(calibration, labels, alpha=0.20)
        self.assertGreaterEqual(qhat, 0.0)
        self.assertLessEqual(qhat, 1.0)
        self.assertGreaterEqual(empirical_set_coverage(calibration, labels, qhat), 0.80)

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            normalized_entropy(np.array([0.5, -0.5, 1.0]))
        with self.assertRaises(ValueError):
            normalized_entropy(np.array([0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
