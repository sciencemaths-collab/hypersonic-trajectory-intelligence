#!/usr/bin/env bash
set -e
python3 alien_exit_cell_predictor_v6_3.py --quick --no-gpu --output-prefix demo
echo "Done. See demo_metrics.json and demo_* outputs."
