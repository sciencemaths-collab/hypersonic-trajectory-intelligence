# Model Assumptions and Applicability Limits

This register is part of the credibility evidence. An assumption is not a defect merely because it is simplified, but a scientific claim must remain inside the envelope supported by the assumptions and validation.

## Dynamics and Earth model

- Central point-mass gravity is used for propagation.
- A constant Earth rotation rate is used for ECEF Coriolis and centrifugal terms.
- Higher-order gravity, oblateness/J2, tides, and geophysical variation are not modeled in the core propagator.
- Integration uses midpoint RK2 with a nominal `dt = 0.05 s`; no formal time-step convergence study is versioned yet.

**Implication:** numerical and force-model error must be quantified before longer-duration or precision trajectory claims.

## Atmosphere

- The atmosphere is ISA-like through 86 km.
- Above 86 km, density follows a simple exponential approximation with a fixed scale height and temperature.
- No empirical space-weather/thermosphere model is used.

**Implication:** upper-atmosphere results are useful for synthetic diversity, not as a validated high-altitude atmosphere prediction.

## Aerodynamics

- Aerodynamic force is currently represented primarily by drag.
- The drag coefficient is a simplified analytic function of Mach and Reynolds number rather than a validated vehicle coefficient table, wind-tunnel model, CFD surrogate, or flight-derived aerodynamic database.
- Lift, side force, control-surface aerodynamics, aeroelasticity, shock interactions, and attitude-dependent coefficient surfaces are not represented as a validated vehicle model.
- Heating, ablation, mass loss, and changing inertia are not modeled.

**Implication:** the simulator is a research dynamics generator, not a high-fidelity hypersonic vehicle digital twin.

## Maneuvers

- Future unknown maneuvers are represented by a transparent single-event hypothesis mixture over signed body axes and selected event times.
- The maneuver prior is not an operational intent model.
- The model does not know future controls when producing a prospective prediction.

**Implication:** longer-horizon predictability is intentionally limited when future control is unobserved.

## Sensors

- Measurements are synthetic position and velocity with configurable noise/bias behavior.
- A full radar/optical measurement equation, line-of-sight geometry, detection probability, clutter, track association, latency, packet loss, and sensor scheduling are not currently modeled as external sensor systems.

**Implication:** external sensor performance cannot be inferred from the current synthetic measurement model.

## State estimation

- The filter state contains position, velocity, a quaternion, and body angular rate.
- Quaternion uncertainty is handled in Euclidean UKF coordinates and normalized after propagation rather than using a dedicated manifold/error-state attitude formulation.
- Covariance repair and hard state sanity rails are used for numerical robustness.

**Implication:** attitude covariance should not be interpreted as a rigorously derived manifold uncertainty distribution.

## Learning and calibration

- Training data are synthetic.
- The Transformer uses strictly causal feature/control windows.
- Temperature calibration is fitted on validation data only.
- Current recorded validation/test class support is incomplete.

**Implication:** generalization outside the synthetic development distribution remains unproven.

## Virtual boundary geometry

- The six-face local volume and gate cells are an analytic research construct aligned to a local trajectory frame.
- They are not a certified range-safety volume, airspace rule, or mission-specific hazard corridor.

**Implication:** any external use must define its own geometry and acceptance criteria.
