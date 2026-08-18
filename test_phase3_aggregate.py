import hashlib
import json
import unittest
from pathlib import Path

from hti.phase3_aggregate import METRIC_NAMES, aggregate_reports


class Phase3AggregateTests(unittest.TestCase):
    evidence_dir = Path("evidence/phase3_run_32117123938")

    def protocol(self):
        return {
            "seeds": [101, 211, 307, 401, 503],
            "horizons_seconds": [0.4, 0.5, 0.6],
            "variants": ["core_hti", "hti_08_combined"],
            "claim_gates": {
                "minimum_test_class_coverage": 0.5,
                "maximum_ece_regression": 0.02,
                "credible_coverage_interval": [0.93, 0.97],
            },
        }

    def report(self, seed):
        def metrics(nll, ece):
            record = {name: 0.5 for name in METRIC_NAMES}
            record.update(
                {
                    "class_coverage": 0.75,
                    "accuracy_top1": 0.4,
                    "accuracy_top3": 0.8,
                    "accuracy_top5": 0.9,
                    "nll": nll,
                    "brier": 0.7,
                    "ece": ece,
                    "credible_coverage": 0.95,
                    "credible_mean_size": 4.0,
                    "credible_median_size": 4.0,
                    "mean_confidence": 0.45,
                    "mean_entropy_concentration": 0.3,
                }
            )
            return record

        variants = {}
        for variant in ("core_hti", "hti_08_combined"):
            records = []
            for horizon in (0.4, 0.5, 0.6):
                is_combined = variant == "hti_08_combined"
                records.append(
                    {
                        "horizon_seconds": horizon,
                        "metrics": metrics(1.0 if not is_combined else 0.9, 0.04),
                    }
                )
            variants[variant] = records
        paired = {
            str(horizon): {
                "nll_delta_combined_minus_core": {
                    "delta": -0.1,
                    "ci95_low": -0.15,
                    "ci95_high": -0.03,
                    "favorable_direction": 1.0,
                    "interval_supports_direction": 1.0,
                }
            }
            for horizon in (0.4, 0.5, 0.6)
        }
        return {
            "protocol_sha256": "a" * 64,
            "execution_config_sha256": "b" * 64,
            "contract": {
                "seed": seed,
                "required_variants_present": True,
                "event_splits_disjoint": True,
                "core_fusion_reproducible": True,
                "structural_fusion_reproducible": True,
                "topology_application_reproducible": True,
                "combined_reproducible": True,
            },
            "variants": variants,
            "paired_core_vs_combined": paired,
        }

    def test_aggregate_requires_all_frozen_seeds(self):
        reports = [self.report(seed) for seed in self.protocol()["seeds"]]
        summary = aggregate_reports(
            reports,
            self.protocol(),
            protocol_sha256="a" * 64,
            execution_config_sha256="b" * 64,
        )
        self.assertEqual(summary["seeds"], [101, 211, 307, 401, 503])
        self.assertTrue(summary["claim_gate_evidence"][0]["all_seeds_nll_better"])
        self.assertEqual(summary["claim_gate_evidence"][0]["bootstrap_support_count"], 5)

    def test_aggregate_rejects_missing_seed(self):
        reports = [self.report(seed) for seed in self.protocol()["seeds"][:-1]]
        with self.assertRaises(ValueError):
            aggregate_reports(
                reports,
                self.protocol(),
                protocol_sha256="a" * 64,
                execution_config_sha256="b" * 64,
            )

    def test_aggregate_rejects_hash_mismatch(self):
        reports = [self.report(seed) for seed in self.protocol()["seeds"]]
        reports[0]["execution_config_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            aggregate_reports(
                reports,
                self.protocol(),
                protocol_sha256="a" * 64,
                execution_config_sha256="b" * 64,
            )

    def test_published_phase3_evidence_matches_provenance(self):
        provenance = json.loads(
            (self.evidence_dir / "provenance.json").read_text(encoding="utf-8")
        )
        for filename, expected in provenance["derived_reports"].items():
            digest = hashlib.sha256((self.evidence_dir / filename).read_bytes()).hexdigest()
            self.assertEqual(digest, expected)

        summary = json.loads(
            (self.evidence_dir / "hti08_phase3_multiseed_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["seeds"], [101, 211, 307, 401, 503])
        self.assertTrue(
            all(
                not gate["all_seeds_nll_better"]
                and gate["bootstrap_support_count"] == 0
                for gate in summary["claim_gate_evidence"]
            )
        )


if __name__ == "__main__":
    unittest.main()
