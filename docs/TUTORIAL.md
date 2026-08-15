# Tutorial

This tutorial runs the research prototype, explains its outputs, and reproduces
the included benchmark chart.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run a Quick Verification

```bash
streamlit run streamlit_app.py
```

Keep **Quick integration run** enabled and select **Run quick verification**.
This verifies simulation, filtering, inference, trace generation, and dashboard
rendering. Its four-epoch model scores are not performance evidence.

## 3. Inspect a Prediction

Use **Timeline** to choose a source frame and **Lookahead** to select the projection
horizon. The decision strip reports the predicted and observed side/gate pair.

The trajectory plot is expressed in a local right/up/forward frame. The adjacent
plots show first-passage side mass and the selected face's gate distribution.
Unresolved mass is reported separately rather than silently renormalized.

## 4. Run the Full Benchmark

Turn off **Quick integration run**. The primary button changes to **Run full
benchmark**. Start the run and allow training and online trace generation to
complete.

The release rule is evaluated independently at each prediction horizon:

1. held-out accuracy must exceed the train-derived majority baseline; and
2. expected calibration error must remain below 10%.

Every horizon must satisfy both conditions before the model is marked ready.

## 5. Run from the Command Line

Quick integration run:

```bash
python alien_exit_cell_predictor_v6_3.py \
  --quick --no-gpu --no-viz --output-prefix demo
```

Full CPU benchmark:

```bash
python alien_exit_cell_predictor_v6_3.py \
  --no-gpu --no-viz --output-prefix full_run
```

Each run writes metrics, normalization parameters, model weights, plots, and an
online trace using the chosen output prefix.

## 6. Reproduce the Standalone Benchmark Plot

```bash
python plot_benchmark_thresholds.py
```

The script reads `benchmark_v64_maneuver_calibrated_metrics.json` and writes
`benchmark_pass_thresholds.png`.

## 7. Run Verification Tests

```bash
python -m unittest -v test_v63.py
python stress_test_fast.py --seeds 0 --traj 6 --mode diversity
```

## Troubleshooting

- If `streamlit` is unavailable, confirm that the virtual environment is active
  and rerun `pip install -r requirements.txt`.
- If the GUI has no trace, select the run button or generate one from the CLI with
  the `alien_ui` output prefix.
- Use `--no-gpu` when CUDA or Metal acceleration is unavailable.
- A quick-run release warning is expected; quick mode verifies integration only.

