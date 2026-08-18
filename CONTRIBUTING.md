# Contributing

## Principle

Scientific claims must move with evidence. A code change that improves a headline metric but weakens holdout discipline, class coverage, calibration, traceability, or reproducibility is not an acceptable improvement.

## Before changing behavior

1. Identify the affected project requirement IDs in `docs/REQUIREMENTS_TRACEABILITY.md`.
2. State the hypothesis being tested.
3. Define acceptance criteria before inspecting final test outcomes.
4. Keep whole trajectories isolated across train/validation/test splits.
5. Add focused unit or numerical tests.
6. Update the validation report when a claim changes.

## Pull requests

A behavior-changing PR should include:

- affected requirement IDs;
- what changed and why;
- new/updated tests;
- benchmark configuration;
- whether calibration or assurance thresholds changed;
- whether any final test data influenced development decisions; and
- updated evidence artifacts when metrics are cited.

Do not lower evidence thresholds after seeing a failing final test result.

## Safety and data

Do not contribute credentials, private telemetry, controlled technical data, proprietary datasets without permission, or instructions that turn the project into operational targeting/interception software.

## Style

- Python 3.10+.
- Prefer explicit units in variable names or documentation.
- Keep numerical constants documented.
- Avoid hidden global state in scientific calculations.
- Prefer deterministic test fixtures.
- Keep the assurance/evidence layer independent from model-training code where practical.

## Local verification

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m unittest -v test_v63.py test_assurance.py
python scripts/evidence_gate.py benchmark_v64_maneuver_calibrated_metrics.json --fail-on engineering
ruff check hti scripts test_assurance.py
pip-audit -r requirements.txt
```
