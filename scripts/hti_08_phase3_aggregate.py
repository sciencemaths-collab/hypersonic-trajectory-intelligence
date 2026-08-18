#!/usr/bin/env python3
"""Aggregate the complete frozen five-seed HTI 0.8 Phase 3 reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hti.phase3_aggregate import aggregate_reports  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/hti_08_ablation_frozen.json")
    )
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=Path("configs/hti_08_phase3_execution_frozen.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("hti08_phase3_multiseed_summary.json"))
    args = parser.parse_args()

    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    summary = aggregate_reports(
        reports,
        protocol,
        protocol_sha256=_sha256_file(args.config),
        execution_config_sha256=_sha256_file(args.execution_config),
    )
    summary["input_reports"] = [
        {"path": path.as_posix(), "sha256": _sha256_file(path)} for path in args.reports
    ]
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
