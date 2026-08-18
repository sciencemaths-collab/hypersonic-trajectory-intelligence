import hashlib
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
from scripts.hti_08_evaluate_predictions import (
    _apply_topology_cube,
    _self_test_bundle,
    evaluate,
)


class FusionBenchmarkingTests(unittest.TestCase):
    protocol_path = Path("configs/hti_08_ablation_frozen.json")
    execution_path = Path("configs/hti_08_phase3_execution_frozen.json")

    def frozen_config(self):
        return json.loads(self.protocol_path.read_text(encoding="utf-8"))

    def frozen_execution(self):
        return json.loads(self.execution_path.read_text(encoding="utf-8"))

    def protocol_sha256(self):
        return hashlib.sha256(self.protocol_path.read_bytes()).hexdigest()

    def execution_sha256(self):
        return hashlib.sha256(self.execution_path.read_bytes()).hexdigest()

    def self_test_bundle(self):
        return _self_test_bundle(
            protocol_sha256=self.protocol_sha256(),
            execution_config_sha256=self.execution_sha256(),
            execution=self.frozen_execution(),
        )

    def evaluate_bundle(self, bundle):
        return evaluate(
            bundle,
            self.frozen_config(),
            self.frozen_execution(),
            protocol_sha256=self.protocol_sha256(),
            execution_config_sha256=self.execution_sha256(),
        )

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

    def test_self_test_bundle_satisfies_full_composition_contract(self):
        report = self.evaluate_bundle(self.self_test_bundle())
        contract = report["contract"]
        self.assertTrue(contract["core_fusion_reproducible"])
        self.assertTrue(contract["structural_fusion_reproducible"])
        self.assertTrue(contract["topology_application_reproducible"])
        self.assertTrue(contract["combined_reproducible"])

    def test_float32_serialized_fusions_satisfy_composition_contract(self):
        bundle = self.self_test_bundle()
        transformer = bundle["core_transformer_probabilities"].astype(np.float32)
        physics = bundle["probs__physics_only"]
        core = np.empty(transformer.shape, dtype=np.float32)
        for horizon, weight in enumerate(bundle["core_physics_weights"]):
            core[:, horizon, :] = log_linear_pool(
                transformer[:, horizon, :],
                physics[:, horizon, :],
                structural_weight=float(weight),
            )

        structural = bundle["structural_branch_probabilities"]
        core_plus_structural = np.empty(core.shape, dtype=np.float32)
        for horizon, weight in enumerate(bundle["fusion_structural_weights"]):
            core_plus_structural[:, horizon, :] = log_linear_pool(
                core[:, horizon, :],
                structural[:, horizon, :],
                structural_weight=float(weight),
            )

        topology = self.frozen_execution()["topology_branch"]
        bundle["core_transformer_probabilities"] = transformer
        bundle["probs__core_hti"] = core
        bundle["probs__core_plus_structural"] = core_plus_structural
        bundle["probs__core_plus_topology"] = _apply_topology_cube(
            core,
            bundle["topology_cell_prior"],
            strength=float(topology["cell_prior_strength"]),
            floor=float(topology["cell_prior_floor"]),
        )
        bundle["probs__hti_08_combined"] = _apply_topology_cube(
            core_plus_structural,
            bundle["topology_cell_prior"],
            strength=float(topology["cell_prior_strength"]),
            floor=float(topology["cell_prior_floor"]),
        )

        report = self.evaluate_bundle(bundle)
        self.assertTrue(report["contract"]["core_fusion_reproducible"])

    def test_float32_contract_still_rejects_one_ulp_composition_change(self):
        bundle = self.self_test_bundle()
        core = bundle["probs__core_hti"].astype(np.float32)
        core[0, 0, 0] = np.nextafter(core[0, 0, 0], np.float32(1.0))
        bundle["probs__core_hti"] = core
        with self.assertRaisesRegex(ValueError, "core_hti"):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_rejects_unreproducible_core_fusion(self):
        bundle = self.self_test_bundle()
        bundle["probs__core_hti"] = bundle["core_transformer_probabilities"].copy()
        with self.assertRaisesRegex(ValueError, "core_hti"):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_rejects_missing_variant(self):
        bundle = self.self_test_bundle()
        del bundle["probs__physics_only"]
        with self.assertRaises(ValueError):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_rejects_split_leakage(self):
        bundle = self.self_test_bundle()
        bundle["validation_event_ids"] = np.array([100, 200, 201], dtype=int)
        with self.assertRaises(ValueError):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_rejects_unreproducible_structural_fusion(self):
        bundle = self.self_test_bundle()
        bundle["probs__core_plus_structural"] = bundle["probs__core_hti"].copy()
        with self.assertRaisesRegex(ValueError, "core_plus_structural"):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_rejects_combined_without_topology(self):
        bundle = self.self_test_bundle()
        bundle["probs__hti_08_combined"] = bundle["probs__core_plus_structural"].copy()
        with self.assertRaisesRegex(ValueError, "hti_08_combined"):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_rejects_unreproducible_topology_variant(self):
        bundle = self.self_test_bundle()
        bundle["probs__core_plus_topology"] = bundle["probs__core_hti"].copy()
        with self.assertRaisesRegex(ValueError, "core_plus_topology"):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_rejects_mismatched_cell_partition(self):
        bundle = self.self_test_bundle()
        bundle["cell_ids__physics_only"] = bundle["cell_ids__physics_only"][::-1]
        with self.assertRaisesRegex(ValueError, "cell partitions"):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_requires_calibration_provenance(self):
        bundle = self.self_test_bundle()
        del bundle["conformal_calibration_sha256__core_hti"]
        with self.assertRaisesRegex(ValueError, "conformal_calibration_sha256"):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_requires_topology_provenance(self):
        bundle = self.self_test_bundle()
        del bundle["topology_coefficients_sha256"]
        with self.assertRaisesRegex(ValueError, "topology_coefficients_sha256"):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_rejects_wrong_protocol_digest(self):
        bundle = self.self_test_bundle()
        bundle["protocol_sha256"] = np.array(["0" * 64])
        with self.assertRaisesRegex(ValueError, "protocol_sha256"):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_rejects_wrong_execution_config_digest(self):
        bundle = self.self_test_bundle()
        bundle["execution_config_sha256"] = np.array(["0" * 64])
        with self.assertRaisesRegex(ValueError, "execution_config_sha256"):
            self.evaluate_bundle(bundle)

    def test_frozen_evaluator_requires_model_artifact_digests(self):
        bundle = self.self_test_bundle()
        del bundle["model_sha256__structural_branch"]
        with self.assertRaisesRegex(ValueError, "model_sha256__structural_branch"):
            self.evaluate_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
