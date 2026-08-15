# Validation Report

## Scope

This bundle is simulation and research software. The audit covers numerical
stability, causal data flow, coordinate-frame equations, target geometry,
trajectory-isolated evaluation, calibration, UI integrity, and packaging. It is
not certification for operational or safety-critical use.

## Physics and mathematics

- State: ECEF position and velocity, body-to-ECEF quaternion, and body angular
  velocity.
- Gravity: central gravity using `mu = 3.986004418e14 m^3/s^2`.
- Rotating frame: fixed ECEF Earth-rate vector with Coriolis and centrifugal
  acceleration.
- Translation and rotation: midpoint RK2; quaternion normalized every step.
- Aerodynamics: simplified drag with bounded Mach, Reynolds number, and drag
  coefficient. Atmosphere uses a continuous ISA-style model through 86 km and a
  documented exponential approximation above it.
- Estimation: 13-state UKF with SPD covariance repair, innovation gating, and
  dimensioned covariance matrices.
- Six-face projection: 27 positive-weight UKF sigma hypotheses propagated through
  deterministic 6-DOF rollouts. Side and gate masses use each hypothesis's first
  linearly interpolated box crossing. The current control impulse is applied once;
  unknown future control and hidden environment perturbations use zero-mean nominal
assumptions. Unresolved mass is reported separately.

The maneuver mixture supplements the no-future-maneuver sigma paths with 18
signed body-axis impulse hypotheses at three causal event times. Their aggregate
mass is `1 - (1 - p)^h`, where `p` is the configured per-step maneuver hazard and
`h` is the selected lookahead. This is a transparent single-event approximation,
not a learned operational intent model.

Transformer logits use one temperature per horizon selected solely from validation
data. A non-identity temperature is accepted only when it improves validation
negative log likelihood without worsening validation ECE; otherwise that horizon
falls back to `T = 1`. Saved temperatures are reused for test evaluation and
online probabilities; the test set is never used to fit or select calibration.

Hidden simulator gravity, density, temperature, and wind perturbations are not
provided to the UKF, Transformer features, or prospective physics rollout.

## Causality

- Transformer windows end at the prediction frame.
- Changing future observations does not change the current input window.
- Six-face prediction does not read future estimated or true position.
- Future truth is used only to score predictions after they are generated.
- Future random controls are not provided to the prospective rollout.

## Final single-seed benchmark

Configuration: 48 disjoint trajectory groups, 4,224 windows, 18 epochs, CPU,
and horizons of 0.4, 0.5, and 0.6 seconds.

| Horizon | Accuracy | Majority baseline | Balanced accuracy | Top-3 | ECE |
|---|---:|---:|---:|---:|---:|
| 0.4 s | 26.6% | 20.9% | 26.7% | 72.3% | 3.7% |
| 0.5 s | 24.0% | 18.5% | 24.4% | 70.3% | 6.1% |
| 0.6 s | 16.7% | 17.7% | 17.3% | 65.8% | 13.1% |

The validation-only calibration gate deployed temperatures `0.812`, `1.000`,
and `0.901` for the three horizons. Identity calibration was retained at 0.5 s
because no non-identity candidate improved validation NLL while preserving ECE.

The model does not pass the release gate because every horizon must beat its
train-derived majority baseline and keep ECE below 10%. The UI reports this as
`MODEL NOT RELEASE READY`.

## Automated verification

- 18 unit tests pass.
- Tests cover atmosphere monotonicity, central-gravity inverse-square behavior,
  ECEF rotation directions, deterministic propagation, quaternion normalization,
  SPD repair, hidden-jitter isolation, causal windows, future perturbation,
  forward-face gate geometry, horizon validity, and calibration behavior.
- Diversity stress tests report finite train/validation/test arrays and increased
  target entropy after separating forward distance from transverse gate width.
- Desktop and mobile layouts have no horizontal overflow. Plot title and legend
  bounding boxes were checked and do not overlap.

## Remaining limitations

- Synthetic trajectories only; no external or real sensor validation.
- Simplified drag and upper-atmosphere models.
- Quaternion uncertainty is represented in Euclidean UKF coordinates rather than
  a manifold error-state formulation.
- One final benchmark seed is insufficient for a production claim.
- Unknown future maneuvers limit longer-horizon class predictability.
- Geometric score is a normalized heuristic, not a calibrated posterior.

Required next evidence for release: multiple independent seeds, frozen test sets,
scenario-shift tests, ablations, uncertainty calibration on validation data, and
domain-specific external validation.
