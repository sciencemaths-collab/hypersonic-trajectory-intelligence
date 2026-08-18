# External Validation Protocol

## Objective

Determine whether the trajectory-inference pipeline produces useful, calibrated, and appropriately uncertain short-horizon boundary predictions outside the synthetic distribution used during development.

This protocol is designed for non-sensitive civil aerospace research, an independently configured high-fidelity simulator, or appropriately authorized telemetry.

## 1. Freeze before evaluation

Before opening the final external test set, freeze:

- source commit;
- dependency/environment manifest;
- state estimator configuration;
- virtual-volume geometry;
- model weights;
- normalization constants;
- calibration temperatures;
- assurance thresholds;
- conformal calibration split and `alpha`;
- primary metrics and pass/fail thresholds; and
- random seeds used for any stochastic evaluation.

No final-test result should be used to retune these values.

## 2. Data requirements

Each external trajectory should provide, or permit derivation of:

- timestamp;
- coordinate frame and datum;
- position with units;
- velocity with units or sufficient sampling to estimate it;
- measurement uncertainty or sensor-quality metadata where available;
- trajectory/scenario identifier; and
- provenance and permission for research use.

If ground truth is derived rather than directly measured, document the derivation and its uncertainty.

## 3. Scenario strata

External evaluation should report performance separately across meaningful strata rather than only as one aggregate number. Examples:

- altitude bands;
- speed bands;
- sensor-noise/measurement-quality bands;
- nominal versus maneuvering segments;
- atmospheric-model mismatch;
- starting-condition shifts;
- vehicle-parameter shifts; and
- regions of the gate/class space.

## 4. Leakage controls

- Keep complete trajectories in one split only.
- Do not create train and test windows from the same trajectory.
- Do not fit normalization, calibration, conformal thresholds, or abstention thresholds on final test trajectories.
- Do not select scenarios after seeing which ones improve the headline metric.

## 5. Metrics

Report at minimum for each horizon:

- top-1 accuracy;
- balanced accuracy;
- top-k accuracy where scientifically justified;
- majority and simple-physics baselines;
- negative log likelihood;
- multiclass Brier score;
- expected calibration error with bin definition;
- per-class support and recall;
- class-coverage fraction;
- abstention rate;
- selective accuracy/risk on accepted predictions;
- conformal prediction-set empirical coverage and average set size; and
- unresolved first-passage probability mass.

Every metric should include sample count. Multi-seed or resampling intervals should be reported where applicable.

## 6. Shift and robustness tests

Evaluate controlled deviations from the development assumptions, including:

- measurement-noise inflation;
- timing jitter;
- missing observations;
- atmospheric-density/temperature mismatch;
- vehicle mass/inertia mismatch;
- force/maneuver distribution shift; and
- coordinate-frame perturbation tests.

The goal is not to find a setting that makes the model look strongest. The goal is to locate the envelope where evidence remains credible and the regions where the system should abstain.

## 7. Baselines and ablations

Compare against at least:

- majority-class baseline;
- constant-velocity propagation;
- physics-only rollout without Transformer contribution; and
- learned model without physics-derived probability features where practical.

Ablations should be specified before final evaluation.

## 8. Acceptance framework

A future external-readiness release should require all of the following:

1. engineering-integrity gate passes;
2. all predeclared primary horizons beat the frozen baseline criterion;
3. all primary horizons satisfy the frozen calibration criterion;
4. class coverage meets the predeclared minimum;
5. conformal sets achieve the predeclared empirical coverage tolerance;
6. selective risk does not degrade catastrophically as abstention decreases;
7. multi-seed evidence is stable enough to support the stated conclusion; and
8. independent/domain validation has been reviewed by someone other than the model developer.

These are project acceptance rules and may need to be strengthened for a particular organization or mission.

## 9. Evidence bundle

Archive:

- source commit and tag;
- environment lock/container digest;
- frozen configuration;
- dataset provenance manifest;
- split manifest;
- model checksum;
- calibration/assurance parameters;
- raw predictions;
- metrics JSON/CSV;
- plots generated from the raw predictions;
- evidence-gate report; and
- reviewer notes.

The evidence bundle should be reproducible without access to the developer's workstation paths.
