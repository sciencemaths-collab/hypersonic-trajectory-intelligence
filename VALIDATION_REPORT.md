# Validation and Credibility Report

## Status

**Recorded evidence status: research prototype; not externally validated; not operationally certified.**

This report distinguishes code verification, numerical checks, model-performance evidence, and external validation. Passing a software test is not treated as proof that the physical model is valid for an operational environment.

## Intended use of this report

The report supports independent technical review of the repository. It is not a flightworthiness, range-safety, weapons-control, navigation, or safety-critical certification.

## Physics and mathematics currently implemented

- State: ECEF position and velocity, body-to-ECEF quaternion, and body angular velocity.
- Gravity: central gravity using `mu = 3.986004418e14 m^3/s^2`.
- Rotating frame: fixed ECEF Earth-rate vector with Coriolis and centrifugal acceleration.
- Translation and rotation: midpoint RK2; quaternion normalized every step.
- Aerodynamics: simplified drag with bounded Mach, Reynolds number, and drag coefficient.
- Atmosphere: continuous ISA-style approximation through 86 km and an explicitly simplified exponential approximation above 86 km.
- Estimation: 13-state UKF with SPD covariance repair, innovation gating, and dimensioned covariance matrices.
- Probabilistic propagation: 27 sigma-point trajectories plus a transparent future-maneuver hypothesis mixture.
- Boundary event: first linearly interpolated crossing of a local six-face volume; unresolved mass is reported separately.
- Learning: causal multimodal Transformer using only observations/control context available through the prediction time.
- Calibration: one temperature per horizon fitted using validation logits only; the test set is not used to select the temperature.

## Independent assurance layer

Release 0.7 added `hti.assurance`, which is deliberately downstream from the predictor. It provides:

- normalized predictive entropy;
- top-1 confidence and top-1/top-2 margin;
- transparent abstention criteria; and
- split-conformal prediction sets fitted on a separate calibration set.

This layer does not convert an invalid physical model into a valid one. It reduces the risk of presenting a diffuse probability vector as a confident research prediction.

## HTI 0.8 structural/topological layer

Release 0.8 adds `hti.topological_entropy`, an independent structural-motion and topology layer based on the supplied *Unified Topological-Entropy Trajectory Inference* construction.

It adds implementation support for:

- invertible nose/rear to center/body-axis geometry in 2D or 3D;
- circular or directional smoothing;
- body/travel slip angle;
- signed turn rate and curvature;
- local swept-area descriptors;
- alternating-turn/zigzag diagnostics;
- body-length/deformation rate;
- transparent explanatory motion-mode evidence;
- monotone non-negative topology/reachability path weighting;
- 3D spherical safety-volume intersection penalties;
- separate occupancy and terminal-cell probabilities;
- entropy concentration and Bayesian credible-cell sets; and
- backward conditioning of retained candidate paths for terminal-cell explanation.

These additions are **architecture and implementation claims only** at this stage. The recorded 0.7 benchmark predates the 0.8 layer and is not evidence that the structural/topological features improve held-out prediction.

The structural CLI may analyze existing HTI `*_online_trace.npz` artifacts, but those artifacts do not currently store attitude. In trace mode, velocity direction is therefore used only as an explicitly labeled orientation proxy. It must not be presented as independently observed body attitude.

## Causality checks

Existing automated tests verify that:

- Transformer windows end at the prediction frame;
- changing future observations does not change the current input window;
- six-face prediction does not read future estimated or true position;
- future truth is used only for scoring after prediction;
- future simulator perturbations are not injected into nominal prospective filter rollouts; and
- saved NumPy traces are loaded with `allow_pickle=False`.

## Recorded single-seed benchmark

Configuration: 48 disjoint trajectory groups, 4,224 windows, 18 epochs, CPU, and horizons of 0.4, 0.5, and 0.6 seconds.

| Horizon | Accuracy | Majority baseline | Balanced accuracy | Top-3 | ECE | Original gate |
|---|---:|---:|---:|---:|---:|---|
| 0.4 s | 26.6% | 20.9% | 26.7% | 72.3% | 3.7% | Pass |
| 0.5 s | 24.0% | 18.5% | 24.4% | 70.3% | 6.1% | Pass |
| 0.6 s | 16.7% | 17.7% | 17.3% | 65.8% | 13.1% | Fail |

Validation-only temperatures were `0.812`, `1.000`, and `0.901` for the three horizons. Identity calibration was retained at 0.5 s when a non-identity candidate did not satisfy the validation selection rule.

## Class-coverage audit

A new audit found that aggregate accuracy was being reported over a narrow subset of the nominal 16 gate classes.

| Horizon | Train classes | Validation classes | Test classes |
|---|---:|---:|---:|
| 0.4 s | 11/16 | 4/16 | 4/16 |
| 0.5 s | 8/16 | 4/16 | 4/16 |
| 0.6 s | 5/16 | 4/16 | 4/16 |

Consequences:

1. The recorded benchmark does **not** demonstrate broad 16-class generalization.
2. The 0.4 s and 0.5 s passes remain valid only for the recorded held-out distribution.
3. New data-generation and external-validation work should increase scenario and class support without using final test outcomes to choose thresholds.

The evidence gate now requires at least 50% class coverage per split/horizon before it will mark that scientific-readiness check as passing. This is a project research gate, not a NASA requirement.

## Engineering-integrity gate

`scripts/evidence_gate.py` checks:

- configured horizon/class schema;
- trajectory-group isolation;
- minimum recorded trajectory-group count;
- absence of workstation-specific absolute paths in versioned evidence;
- presence and finiteness of required metrics; and
- portable provenance metadata.

These checks may fail CI because they concern repository integrity.

## Scientific-readiness gate

The same script separately reports:

- accuracy above the recorded majority baseline at every configured horizon;
- ECE at or below 10% at every configured horizon;
- adequate held-out sample count;
- at least 50% class coverage in train/validation/test for every horizon;
- at least five independent benchmark seeds; and
- independent external/domain validation.

The current artifact is expected to fail this scientific tier. CI records the failure but does not relabel the prototype as broken software.

## Automated verification inventory

The core, assurance, and HTI 0.8 suites cover:

- atmosphere positivity/monotonicity;
- central-gravity inverse-square behavior;
- ECEF rotation directions;
- deterministic propagation;
- quaternion normalization;
- SPD covariance repair;
- hidden-jitter isolation;
- causal windows and future-perturbation isolation;
- virtual-box gate geometry;
- horizon validity;
- calibration behavior;
- probability validation;
- entropy and confidence-margin behavior;
- abstention behavior;
- split-conformal set construction/coverage utilities;
- endpoint center/axis round-trip recovery;
- centerline-to-endpoint bridge consistency;
- circular angle-wrap invariance;
- structural motion-mode normalization;
- topology-weight monotonicity;
- topology path-weight renormalization;
- terminal distribution normalization;
- cell-prior renormalization;
- credible-cell mass behavior;
- occupancy probability semantics;
- forward/backward conditional consistency;
- conditional mode/state explanation; and
- 3D spherical safety-volume intersection counting.

GitHub Actions runs these checks on Python 3.10, 3.11, and 3.12, plus linting, dependency auditing, evidence-integrity checks, environment capture, and SHA-256 evidence manifests.

## Required HTI 0.8 empirical experiment

Before any statement that 0.8 improves predictive performance, use identical frozen event-level splits and seeds to compare at least:

1. constant velocity;
2. filter/direct extrapolation;
3. physics-only;
4. learned-only where scientifically fair;
5. core HTI;
6. core + structural features;
7. core + topology; and
8. combined HTI 0.8.

The experiment must include component ablations for slip/orientation, curvature/turn rate, zigzag, swept area, deformation, and mode evidence; topology failure analysis; entropy-concentration versus empirical error/selective-risk deciles; 95% Bayesian credible-region coverage/size; conformal coverage/size; and robustness to endpoint noise, endpoint label swaps, missing frames, timing shifts, and abrupt turns.

See [docs/TOPOLOGICAL_ENTROPY_VALIDATION.md](docs/TOPOLOGICAL_ENTROPY_VALIDATION.md).

## Remaining scientific work before a strong external validation claim

- Generate multiple independent frozen benchmark seeds.
- Improve scenario support so train, calibration/validation, and test sets cover materially more of the nominal class space while preserving trajectory isolation.
- Run the frozen HTI 0.8 baseline/ablation experiment.
- Add scenario-shift tests across sensor noise, endpoint quality, atmospheric assumptions, vehicle parameters, maneuver regimes, and starting conditions.
- Evaluate probability calibration, credible-region coverage, conformal prediction-set coverage, and selective risk under distribution shift.
- Validate against independent public civil telemetry or an independently configured high-fidelity simulator.
- Replace the Euclidean quaternion covariance treatment with a manifold/error-state estimator if attitude uncertainty becomes a material part of the intended use.
- Establish configuration-controlled benchmark manifests and independent review of the final evidence bundle.

See [docs/EXTERNAL_VALIDATION_PROTOCOL.md](docs/EXTERNAL_VALIDATION_PROTOCOL.md).

## Bottom line

The current system is a credible **research prototype with a stronger assurance, evidence, and structural/topological architecture**. HTI 0.8 is suitable for technical evaluation, collaboration discussions, and the next controlled validation phase. It is not yet evidence-complete for operational, safety-critical, certified, or externally validated use.
