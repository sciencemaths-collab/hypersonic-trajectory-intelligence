#!/usr/bin/env python3
"""Execute one frozen HTI 0.8 Phase 3 synthetic ablation seed.

The implementation lives in :mod:`hti.phase3_runner` so the same execution
logic can be imported and tested without invoking a subprocess. This CLI does
not evaluate final-test metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hti.phase3_runner import execute_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/hti_08_ablation_frozen.json")
    )
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=Path("configs/hti_08_phase3_execution_frozen.json"),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selection-out", type=Path)
    parser.add_argument("--receipt-out", type=Path)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    execution = json.loads(args.execution_config.read_text(encoding="utf-8"))
    seed = int(args.seed)
    output = args.out or Path(f"hti08_ablation_{seed}.npz")
    selection_output = args.selection_out or Path(f"hti08_selection_{seed}.json")
    receipt_output = args.receipt_out or Path(f"hti08_seed_receipt_{seed}.json")

    receipt = execute_seed(
        seed=seed,
        protocol=protocol,
        execution=execution,
        protocol_path=args.protocol,
        execution_path=args.execution_config,
        output=output,
        selection_output=selection_output,
    )
    receipt_output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(receipt_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
