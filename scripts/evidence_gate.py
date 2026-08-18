#!/usr/bin/env python3
"""Audit recorded benchmark evidence without retraining the model.

The script separates engineering-integrity checks from scientific-readiness
checks. CI should fail on engineering-integrity defects. Scientific-readiness
failures remain visible until stronger evidence is generated and independently
validated; they should not be hidden by tuning the gate after seeing test data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def add_check(checks: list[dict[str, Any]], check_id: str, tier: str, passed: bool, detail: str) -> None:
    checks.append({"id": check_id, "tier": tier, "passed": bool(passed), "detail": detail})


def iter_strings(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)
    elif isinstance(value, str):
        yield value


def looks_absolute_path(value: str) -> bool:
    return value.startswith("/") or bool(WINDOWS_ABSOLUTE.match(value))


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def audit(metrics: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    config = metrics.get("config", {})
    dataset = metrics.get("dataset", {})
    test = metrics.get("test", {})
    provenance = metrics.get("provenance", {})

    horizons = [int(x) for x in config.get("horizon_steps", [])]
    class_count = int(config.get("box_N", 0)) ** 2

    add_check(
        checks,
        "ENG-001",
        "engineering",
        bool(horizons) and class_count >= 4,
        f"configured horizons={horizons}, classes={class_count}",
    )
    add_check(
        checks,
        "ENG-002",
        "engineering",
        float(dataset.get("split_by_trajectory", 0.0)) == 1.0,
        "train/validation/test must be isolated by trajectory group",
    )
    groups = int(float(dataset.get("trajectory_groups", 0.0)))
    add_check(
        checks,
        "ENG-003",
        "engineering",
        groups >= 48,
        f"trajectory groups={groups}; minimum recorded-evidence target=48",
    )

    absolute_paths = sorted({s for s in iter_strings(metrics) if looks_absolute_path(s)})
    add_check(
        checks,
        "ENG-004",
        "engineering",
        not absolute_paths,
        "recorded evidence must not contain workstation-specific absolute paths"
        if not absolute_paths
        else f"absolute paths found: {absolute_paths}",
    )

    required_numeric = []
    for h in horizons:
        required_numeric.extend(
            [
                test.get(f"test_acc_h{h}"),
                test.get(f"test_majority_baseline_h{h}"),
                test.get(f"test_ece_h{h}"),
                test.get(f"test_nll_h{h}"),
                test.get(f"test_brier_h{h}"),
            ]
        )
    add_check(
        checks,
        "ENG-005",
        "engineering",
        bool(required_numeric) and all(finite_number(x) for x in required_numeric),
        "recorded test metrics must be present and finite for every configured horizon",
    )

    valid_by_split = dataset.get("valid_labels_by_split", {})
    unique_by_split = dataset.get("unique_classes_by_split", {})
    for index, h in enumerate(horizons):
        accuracy = float(test.get(f"test_acc_h{h}", float("nan")))
        baseline = float(test.get(f"test_majority_baseline_h{h}", float("nan")))
        ece = float(test.get(f"test_ece_h{h}", float("nan")))
        add_check(
            checks,
            f"SCI-ACC-H{h}",
            "scientific",
            math.isfinite(accuracy) and math.isfinite(baseline) and accuracy > baseline,
            f"accuracy={accuracy:.4f}, majority baseline={baseline:.4f}",
        )
        add_check(
            checks,
            f"SCI-CAL-H{h}",
            "scientific",
            math.isfinite(ece) and ece <= 0.10,
            f"ECE={ece:.4f}; threshold<=0.10",
        )

        test_valid = list(valid_by_split.get("test", []))
        n_valid = int(test_valid[index]) if index < len(test_valid) else 0
        add_check(
            checks,
            f"SCI-N-H{h}",
            "scientific",
            n_valid >= 500,
            f"valid held-out labels={n_valid}; minimum evidence target=500",
        )

        for split in ("train", "validation", "test"):
            values = list(unique_by_split.get(split, []))
            unique = int(values[index]) if index < len(values) else 0
            coverage = (unique / class_count) if class_count else 0.0
            add_check(
                checks,
                f"SCI-COVERAGE-{split.upper()}-H{h}",
                "scientific",
                coverage >= 0.50,
                f"{split} class coverage={unique}/{class_count} ({coverage:.1%}); target>=50%",
            )

    seeds = provenance.get("multi_seed_seeds", [])
    add_check(
        checks,
        "SCI-MULTISEED",
        "scientific",
        isinstance(seeds, list) and len(seeds) >= 5,
        f"independent benchmark seeds={len(seeds) if isinstance(seeds, list) else 0}; target>=5",
    )
    add_check(
        checks,
        "SCI-EXTERNAL",
        "scientific",
        bool(provenance.get("external_validation", False)),
        "independent external or domain-specific validation evidence is required",
    )

    engineering = [c for c in checks if c["tier"] == "engineering"]
    scientific = [c for c in checks if c["tier"] == "scientific"]
    return {
        "schema": "hti.evidence-audit.v1",
        "engineering_integrity_pass": all(c["passed"] for c in engineering),
        "scientific_readiness_pass": all(c["passed"] for c in scientific),
        "operational_certification_claim": False,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--fail-on",
        choices=("none", "engineering", "scientific"),
        default="none",
        help="Select which tier should produce a non-zero exit status.",
    )
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    report = audit(metrics)
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for check in report["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"[{mark}] {check['id']} ({check['tier']}): {check['detail']}")
    print(
        "engineering_integrity_pass=",
        report["engineering_integrity_pass"],
        " scientific_readiness_pass=",
        report["scientific_readiness_pass"],
        sep="",
    )

    if args.fail_on == "engineering" and not report["engineering_integrity_pass"]:
        return 2
    if args.fail_on == "scientific" and not report["scientific_readiness_pass"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
