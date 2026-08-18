## Summary

Describe the change and why it is needed.

## Requirements affected

List project requirement IDs from `docs/REQUIREMENTS_TRACEABILITY.md`.

## Scientific claim impact

- [ ] No scientific claim changes
- [ ] Claim changes are documented in `VALIDATION_REPORT.md`
- [ ] Acceptance criteria were defined before final-test inspection
- [ ] Final test data did not tune thresholds/hyperparameters

## Verification

- [ ] `python -m unittest -v test_v63.py test_assurance.py`
- [ ] Evidence engineering-integrity gate passes
- [ ] New behavior has focused tests
- [ ] Benchmark/evidence artifacts are portable and contain no workstation-specific paths

## Data and safety

- [ ] No credentials, private telemetry, or controlled/proprietary data were committed
- [ ] The change does not introduce operational targeting/interception or safety-critical control functionality

## Evidence

Attach or link the controlled benchmark/evidence output used to support any new quantitative statement.
