import json
import unittest
from pathlib import Path

import numpy as np

from hti.benchmarking import (
    assert_disjoint_event_splits,
    conformal_set_stats,
    evaluate_probabilities,
    event_bootstrap_delta,
    nll_contributions,
    reliability_bins,
    selective_risk_curve,
)
from hti.fusion import apply_cell_prior, log_linear_pool, select_structural_weight
from scripts.hti_08_evaluate_predictions import _self_test_bundle, evaluate


class FusionBenchmarkingTests(unittest.TestCase):
    def frozen_config(self):
        return json.loads(Path("configs/hti_08_ablation_frozen.json").read_text(encoding="utf-8"))

    def test_log_pool_endpoints_recover_sources(self):
        core = np.array([[0.7, 0.2, 0.1]])
        structural = np.array([[0.2, 0.7, 0.1]])
        np.testing.assert_allclose(
            log_linear_pool(core, structural, structural_weight=0.0), core
        )
        np.testing.assert_allclose(
            log_linear_pool(core, structural, structural_weight=1.0), structural
        )

    def test_validation_selection_prefers_better_structural_branch(self):
        labels = np.array([0, 1, 0, 1])
        core = np.array(
            [[0.55, 0.45], [0.45, 0.55], [0.52, 0.48], [0.48, 0.52]]
        )
        structural = np.array(
            [[0.90, 0.10], [0.10, 0.90], [0.85, 0.15], [0.15, 0.85]]
        )
        selection = select_structural_weight(
            core,
            structural,
            labels,
            candidates=np.array([0.0, 0.5, 1.0]),
        )
        self.assertEqual(selection.structural_weight, 1.0)
        self.assertEqual(selection.candidate_count, 3)

    def test_cell_prior_reweights_and_normalizes(self):
        probabilities = np.array([[0.5, 0.5], [0.2, 0.8]])
        adjusted = apply_cell_prior(probabilities, np.array([1.0, 0.1]))
        np.testing.assert_allclose(adjusted.sum(axis=1), np.ones(2))
        self.assertGreater(adjusted[0, 0], adjusted[0, 1])

    def test_event_split_leakage_is_rejected(self):
        with self.assertRaises(ValueError):
            assert_disjoint_event_splits(
                np.array([1, 2]), np.array([3, 4]), np.array([4, 5])
            )
        assert_disjoint_event_splits(
            np.array([1, 2]), np.array([3, 4]), np.array([5, 6])
        )

    def test_perfect_probabilities_have_perfect_top1(self):
        probabilities = np.array(
            [[0.99, 0.01], [0.01, 0.99], [0.98, 0.02], [0.02, 0.98]]
        )
        labels = np.array([0, 1, 0, 1])
        metrics = evaluate_probabilities(probabilities, labels, credible_level=0.95)
        self.assertEqual(metrics.accuracy_top1, 1.0)
        self.assertEqual(metrics.class_coverage, 1.0)
        self.assertLess(metrics.nll, 0.03)
        self.assertEqual(metrics.credible_coverage, 1.0)
        self.assertEqual(metrics.credible_median_size, 1.0)

    def test_reliability_bins_account_for_every_sample(self):
        probabilities = np.array(
            [[0.90, 0.10], [0.80, 0.20], [0.55, 0.45], [0.40, 0.60]]
        )
        labels = np.array([0, 0, 1, 1])
        bins = reliability_bins(probabilities, labels, bins=5)
        self.assertEqual(sum(int(record["count"]) for record in bins), len(labels))
        self.assertEqual(len(bins), 5)

    def test_precalibrated_conformal_stats_report_coverage_and_size(self):
        probabilities = np.array(
            [[0.80, 0.20], [0.15, 0.85], [0.70, 0.30], [0.20, 0.80]]
        )
        labels = np.array([0, 1, 0, 1])
        stats = conformal_set_stats(probabilities, labels, qhat=0.40)
        self.assertEqual(stats["coverage"], 1.0)
        self.assertEqual(stats["mean_size"], 1.0)
        self.assertEqual(stats["median_size"], 1.0)

    def test_selective_risk_prefers_concentrated_correct_forecasts(self):
        probabilities = np.array(
            [
                [0.95, 0.05],
                [0.90, 0.10],
                [0.55, 0.45],
                [0.51, 0.49],
                [0.45, 0.55],
            ]
        )
        labels = np.array([0, 0, 0, 1, 0])
        curve = selective_risk_curve(probabilities, labels, coverages=(0.4, 1.0))
        self.assertEqual(curve[0]["selective_risk"], 0.0)
        self.assertGreater(curve[-1]["selective_risk"], curve[0]["selective_risk"])

    def test_event_bootstrap_detects_lower_nll_candidate(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        events = np.array([10, 10, 20, 20, 30, 30])
        baseline = np.array(
            [[0.60, 0.40], [0.40, 0.60], [0.55, 0.45], [0.45, 0.55], [0.60, 0.40], [0.40, 0.60]]
        )
        candidate = np.array(
            [[0.80, 0.20], [0.20, 0.80], [0.75, 0.25], [0.25, 0.75], [0.80, 0.20], [0.20, 0.80]]
        )
        result = event_bootstrap_delta(
            nll_contributions(candidate, labels),
            nll_contributions(baseline, labels),
            events,
            iterations=200,
            seed=7,
            lower_is_better=True,
        )
        self.assertLess(result["delta"], 0.0)
        self.assertLess(result["ci95_high"], 0.0)
        self.assertEqual(result["favorable_direction"], 1.0)
        self.assertEqual(result["interval_supports_direction"], 1.0)

    def test_frozen_evaluator_rejects_missing_variant(self):
        bundle = _self_test_bundle()
        del bundle["probs__physics_only"]
        with self.assertRaises(ValueError):
            evaluate(bundle, self.frozen_config())

    def test_frozen_evaluator_rejects_split_leakage(self):
        bundle = _self_test_bundle()
        bundle["validation_event_ids"] = np.array([100, 200, 201], dtype=int)
        with self.assertRaises(ValueError):
            evaluate(bundle, self.frozen_config())


if __name__ == "__main__":
    unittest.main()
