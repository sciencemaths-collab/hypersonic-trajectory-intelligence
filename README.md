# Hypersonic Trajectory Intelligence

![Development status](https://img.shields.io/badge/status-research%20%26%20development-e9b44c)
![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)
![License](https://img.shields.io/badge/license-MIT-45c486)

Physics-informed 6DOF tracking, maneuver inference, and uncertainty-aware
flight-path prediction for high-speed aerospace research.

![Hypersonic Trajectory Intelligence operations console](assets/project_promo.png)

## Development Status

This project is an **independent research prototype under active development**.
It demonstrates an integrated trajectory-inference workflow using synthetic
simulation data. It is not certified, validated, or approved for operational,
safety-critical, targeting, weapons-control, or flight-control use.

The project is not affiliated with, endorsed by, or developed on behalf of NASA,
the United States Air Force, the United States Government, or any aerospace or
defense organization.

The current full benchmark passes its accuracy and calibration requirements at
0.4 and 0.5 seconds. The 0.6-second horizon does not pass. See
[Validation](#validation) and [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

## What It Does

The system receives noisy simulated position and velocity measurements and
estimates:

- the side of a local virtual safety volume that a vehicle will cross;
- the gate cell on that boundary;
- uncertainty over possible first-boundary crossings;
- the effect of plausible future maneuver impulses; and
- calibrated forward-face class probabilities at multiple prediction horizons.

The implementation combines:

- rotating-Earth ECEF equations and 6DOF rigid-body propagation;
- a 13-state unscented Kalman filter with innovation gating;
- 27 propagated UKF sigma-point trajectories;
- 18 signed body-axis maneuver hypotheses at three causal event times;
- interpolated first-passage side and gate probabilities;
- a causal multimodal Transformer; and
- validation-only temperature scaling with NLL and ECE acceptance gates.

## Possible Applications

These are research directions, not claims of operational readiness.

- **NASA / spaceflight research:** launch-vehicle telemetry monitoring, reentry
  trajectory assessment, spacecraft approach monitoring, range-safety research,
  and identifying departures from predicted flight corridors.
- **Hypersonic research:** estimating the next trajectory region when aerodynamic
  forces, sensor noise, and maneuvers create uncertainty.
- **Missile-warning research:** short-term trajectory classification from radar
  observations. This is dual-use and requires authorized, safety-controlled,
  legally compliant development and evaluation.
- **Air and space traffic:** predicting whether an object will cross a protected
  volume or safety boundary.
- **Autonomous systems:** collision-avoidance research and dynamic keep-out zones
  for aircraft, drones, or high-speed robots.
- Aerospace trajectory monitoring.
- Autonomous-vehicle interception or collision-avoidance research.
- Drone and robotics safety-boundary prediction.
- Sensor-fusion and tracking experiments.
- Testing maneuver-detection and probabilistic forecasting methods.

## Quick Start

### Requirements

- Python 3.10 or newer
- macOS, Linux, or Windows
- CPU execution is supported; a compatible GPU is optional

```bash
git clone https://github.com/sciencemaths-collab/hypersonic-trajectory-intelligence.git
cd hypersonic-trajectory-intelligence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

### Start the GUI

```bash
streamlit run streamlit_app.py
```

Open `http://127.0.0.1:8501` if Streamlit does not open it automatically.

In the sidebar:

1. Keep **Quick integration run** enabled for a fast end-to-end system check.
2. Select **Run quick verification**.
3. Use the timeline and lookahead controls to inspect boundary predictions.
4. Turn off **Quick integration run** to expose **Run full benchmark**.
5. Run the full benchmark when performance evidence is required.

The full run trains and evaluates the model using trajectory-isolated data
splits. It takes longer than the quick integration check.

### Command-Line Run

```bash
python alien_exit_cell_predictor_v6_3.py \
  --quick --no-gpu --no-viz --output-prefix demo
```

Remove `--quick` for the full configuration.

## Reading the Console

- **Prediction:** the most probable side and gate at the selected lookahead.
- **First-passage mass:** probability assigned to the displayed first boundary
  crossing by the sigma-point and maneuver mixture.
- **Maneuver prior:** aggregate probability assigned to future maneuver
  hypotheses over the selected horizon.
- **Unresolved mass:** probability that did not cross the virtual volume within
  the selected projection window.
- **Accuracy:** must exceed the train-derived majority baseline.
- **ECE:** expected calibration error; lower is better and the release limit is
  10%.

## Tutorials

The guided walkthrough in [docs/TUTORIAL.md](docs/TUTORIAL.md) covers installation,
the GUI, CLI execution, benchmark interpretation, output artifacts, and common
problems. New tutorials should be proposed through a GitHub issue or pull request.

## Validation

The recorded full benchmark contains 48 disjoint trajectory groups and 4,224
windows. Current results are:

| Horizon | Accuracy | Majority baseline | ECE | Status |
|---|---:|---:|---:|---|
| 0.4 s | 26.6% | 20.9% | 3.7% | Pass |
| 0.5 s | 24.0% | 18.5% | 6.1% | Pass |
| 0.6 s | 16.7% | 17.7% | 13.1% | Fail |

![Full benchmark release gates](benchmark_pass_thresholds.png)

Regenerate the standalone chart with:

```bash
python plot_benchmark_thresholds.py
```

Automated checks:

```bash
python -m unittest -v test_v63.py
python stress_test_fast.py --seeds 0 --traj 6 --mode diversity
```

## Known Limitations

- Training and validation use synthetic trajectories only.
- Aerodynamics and upper-atmosphere behavior are simplified.
- Quaternion uncertainty uses Euclidean UKF coordinates rather than a manifold
  error-state formulation.
- Future maneuvers use a transparent single-event hypothesis mixture rather than
  a learned operational intent model.
- The longest benchmark horizon currently fails the release criteria.
- External sensor data, scenario-shift testing, and multi-seed performance
  evidence are still required.

## Responsible Use

This software is dual-use. Users are responsible for complying with applicable
laws, export controls, organizational policies, safety requirements, and data
permissions. Do not use this prototype as the sole basis for decisions affecting
human safety, navigation, targeting, interception, or operational control.

Research contributions that improve verification, uncertainty quantification,
simulation realism, robustness, interpretability, and safety are encouraged.

## Contributing

Open an issue before substantial changes so the proposed experiment, evidence,
and validation criteria can be discussed. Pull requests should include focused
tests and should not describe experimental results as operational capability.

## License

Released under the [MIT License](LICENSE).

