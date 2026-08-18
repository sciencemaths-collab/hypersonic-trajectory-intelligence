#!/usr/bin/env python3
"""Create a portable manifest for a research evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_FILES = (
    "alien_exit_cell_predictor_v6_3.py",
    "streamlit_app.py",
    "hti/assurance.py",
    "hti/topological_entropy.py",
    "hti/fusion.py",
    "hti/benchmarking.py",
    "hti/phase3.py",
    "test_assurance.py",
    "test_topological_entropy.py",
    "test_fusion_benchmarking.py",
    "test_phase3.py",
    "scripts/hti_08_evaluate_predictions.py",
    "scripts/hti_08_phase3_pilot.py",
    "configs/hti_08_ablation_frozen.json",
    "requirements.txt",
    "benchmark_v64_maneuver_calibrated_metrics.json",
    "docs/HTI_0_8_ARCHITECTURE.md",
    "docs/TOPOLOGICAL_ENTROPY_VALIDATION.md",
    "docs/HTI_0_8_PHASE2_BENCHMARKING.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
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
    parser.add_argument("--out", type=Path, default=Path("evidence_manifest.json"))
    parser.add_argument("files", nargs="*", default=list(DEFAULT_FILES))
    args = parser.parse_args()

    records = []
    for item in args.files:
        path = Path(item)
        if not path.is_file():
            raise SystemExit(f"required evidence file not found: {path}")
        records.append(
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "schema": "hti.evidence-manifest.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "files": records,
        "claim_scope": "independent research prototype; not operational certification",
    }
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
