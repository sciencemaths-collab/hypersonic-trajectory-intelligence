#!/usr/bin/env python3
"""Run a development-only coverage pilot before the five final HTI 0.8 seeds.

The pilot seed must not be one of the frozen final-test seeds. It uses the
candidate execution configuration to test complete-event support on exactly the
sample rows eligible for every frozen variant and valid at every horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alien_exit_cell_predictor_v6_3 import Config  # noqa: E402
from hti.phase3 import generate_phase3_events, split_event_ids  # noqa: E402
from hti.phase3_sampling import (  # noqa: E402
    joint_valid_coverage_report,
    minimum_source_frame,
    trim_events,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _configure(
    protocol: dict[str, object], execution: dict[str, object], seed: int
) -> tuple[Config, int, int]:
    generation = execution["trajectory_generation"]
    cfg = Config()
    cfg.seed = int(seed)
    cfg.gpu = False
    cfg.no_viz = True
    cfg.traj_steps = int(generation["traj_steps"])
    cfg.offline_stride = max(1, int(generation["offline_stride"]))
    cfg.window = int(generation["window"])
    cfg.maneuver_prob = float(generation["maneuver_prob"])
    cfg.force_std = float(generation["force_std_n"])
    cfg.torque_std = float(generation["torque_std_nm"])
    cfg.max_mach = float(generation["max_mach"])
    cfg.box_cross_scale = float(generation["box_cross_scale"])
    cfg.box_cross_min_m = float(generation["box_cross_min_m"])
    cfg.box_cross_max_m = float(generation["box_cross_max_m"])
    cfg.horizon_steps = tuple(
        int(round(float(horizon) / cfg.dt)) for horizon in protocol["horizons_seconds"]
    )
    return (
        cfg,
        int(generation["target_event_groups"]),
        int(generation["minimum_windows"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/hti_08_ablation_frozen.json")
    )
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=Path("configs/hti_08_phase3_execution_frozen.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("hti08_phase3_pilot.json"))
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    execution = json.loads(args.execution_config.read_text(encoding="utf-8"))
    frozen_seeds = {int(value) for value in protocol["seeds"]}
    if int(args.seed) in frozen_seeds:
        raise SystemExit("development pilot must not use a frozen final-test seed")

    cfg, event_target, minimum_windows = _configure(protocol, execution, int(args.seed))
    raw_events, generation = generate_phase3_events(
        cfg,
        event_target=event_target,
        seed=int(args.seed),
    )
    min_frame = minimum_source_frame(execution)
    events = trim_events(raw_events, min_source_frame=min_frame)
    eligible_samples = int(sum(len(event.tokens) for event in events))

    split_cfg = protocol["split_policy"]
    event_ids = np.asarray([event.event_id for event in events], dtype=np.int64)
    splits = split_event_ids(
        event_ids,
        seed=int(args.seed),
        train_fraction=float(split_cfg["train_fraction"]),
        validation_fraction=float(split_cfg["validation_fraction"]),
    )
    coverage = joint_valid_coverage_report(
        events, splits, num_classes=cfg.box_N * cfg.box_N
    )
    sample_counts = np.asarray([len(event.tokens) for event in events], dtype=int)
    minimum_coverage = float(protocol["claim_gates"]["minimum_test_class_coverage"])
    test_coverage = [float(value) for value in coverage["test"]["class_coverage"]]
    coverage_gate_pass = bool(min(test_coverage) >= minimum_coverage)
    window_gate_pass = bool(eligible_samples >= minimum_windows)

    report = {
        "schema": "hti.phase3-development-pilot.v3",
        "development_only": True,
        "final_test_seed_used": False,
        "source_commit": _source_commit(),
        "protocol_sha256": _sha256_file(args.protocol),
        "execution_config_sha256": _sha256_file(args.execution_config),
        "seed": int(args.seed),
        "requested_events": int(event_target),
        "generated_events": int(generation.generated_events),
        "generation_attempts": int(generation.attempts),
        "raw_generated_windows": int(generation.samples),
        "minimum_joint_source_frame": int(min_frame),
        "eligible_windows": eligible_samples,
        "samples": eligible_samples,
        "minimum_windows": int(minimum_windows),
        "window_gate_pass": window_gate_pass,
        "minimum_test_class_coverage": minimum_coverage,
        "coverage_gate_pass": coverage_gate_pass,
        "pilot_gate_pass": bool(coverage_gate_pass and window_gate_pass),
        "samples_per_event_after_eligibility": {
            "min": int(sample_counts.min()),
            "median": float(np.median(sample_counts)),
            "max": int(sample_counts.max()),
        },
        "gate_updates": int(generation.gate_updates),
        "gate_rejections": int(generation.gate_rejections),
        "gate_rejection_rate": generation.gate_rejection_rate,
        "configuration": {
            "dt": float(cfg.dt),
            "traj_steps": int(cfg.traj_steps),
            "offline_stride": int(cfg.offline_stride),
            "window": int(cfg.window),
            "horizon_steps": list(cfg.horizon_steps),
            "horizons_seconds": [float(value) for value in protocol["horizons_seconds"]],
            "box_N": int(cfg.box_N),
            "box_cross_scale": float(cfg.box_cross_scale),
            "box_cross_min_m": float(cfg.box_cross_min_m),
            "box_cross_max_m": float(cfg.box_cross_max_m),
            "maneuver_probability": float(cfg.maneuver_prob),
            "force_std_N": float(cfg.force_std),
            "torque_std_Nm": float(cfg.torque_std),
            "max_mach": float(cfg.max_mach),
        },
        "split_event_ids": {
            name: [int(value) for value in ids] for name, ids in splits.items()
        },
        "coverage": coverage,
        "scientific_note": (
            "This development pilot uses only seed 17, which is not a frozen final-test seed. "
            "Coverage is measured after shared sample eligibility and all-horizon label validity, "
            "matching the final comparison bundle."
        ),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
