#!/usr/bin/env python3
"""Run a development-only event-coverage pilot before the five final HTI 0.8 seeds.

The pilot seed must not be one of the frozen final-test seeds. It generates
complete synthetic events, preserves event identity, and reports label/class
coverage without training models or inspecting any final-seed result.
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
from hti.phase3 import coverage_report, generate_phase3_events, split_event_ids  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--events", type=int, default=80)
    parser.add_argument("--traj-steps", type=int, default=200)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/hti_08_ablation_frozen.json")
    )
    parser.add_argument("--out", type=Path, default=Path("hti08_phase3_pilot.json"))
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frozen_seeds = {int(value) for value in protocol["seeds"]}
    if int(args.seed) in frozen_seeds:
        raise SystemExit("development pilot must not use a frozen final-test seed")

    cfg = Config()
    cfg.seed = int(args.seed)
    cfg.gpu = False
    cfg.no_viz = True
    cfg.traj_steps = int(args.traj_steps)
    cfg.offline_stride = max(1, int(args.stride))
    cfg.horizon_steps = tuple(int(round(float(h) / cfg.dt)) for h in protocol["horizons_seconds"])

    events, generation = generate_phase3_events(
        cfg,
        event_target=int(args.events),
        seed=int(args.seed),
    )
    split_cfg = protocol["split_policy"]
    event_ids = np.asarray([event.event_id for event in events], dtype=np.int64)
    splits = split_event_ids(
        event_ids,
        seed=int(args.seed),
        train_fraction=float(split_cfg["train_fraction"]),
        validation_fraction=float(split_cfg["validation_fraction"]),
    )
    coverage = coverage_report(events, splits, num_classes=cfg.box_N * cfg.box_N)
    sample_counts = np.asarray([len(event.tokens) for event in events], dtype=int)

    report = {
        "schema": "hti.phase3-development-pilot.v1",
        "development_only": True,
        "final_test_seed_used": False,
        "source_commit": _source_commit(),
        "protocol_sha256": _sha256_file(args.protocol),
        "seed": int(args.seed),
        "requested_events": int(args.events),
        "generated_events": int(generation.generated_events),
        "generation_attempts": int(generation.attempts),
        "samples": int(generation.samples),
        "samples_per_event": {
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
        },
        "split_event_ids": {
            name: [int(value) for value in ids] for name, ids in splits.items()
        },
        "coverage": coverage,
        "scientific_note": (
            "This development pilot may inform pre-final configuration design. It is not a final-test "
            "performance result and uses no frozen final-test seed."
        ),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
