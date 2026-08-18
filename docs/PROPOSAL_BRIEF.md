# Technical Evaluation Proposal Brief

## Working title

**Physics-Informed Short-Horizon Trajectory Intelligence for Uncertainty-Aware Aerospace Monitoring**

## One-sentence concept

An independent research prototype combines nonlinear state estimation, 6DOF physics propagation, causal learned representations, probabilistic calibration, and explicit abstention to study short-horizon boundary-crossing predictions under uncertain high-speed dynamics.

## Why it may be worth evaluating

Many trajectory-monitoring problems are not only point-estimation problems. A useful research system should communicate what it expects to happen, how uncertain that expectation is, when the evidence is outside the development envelope, and when it should decline to make a confident statement.

This project provides a compact test bed for that question.

## Current technical ingredients

- rotating-Earth ECEF dynamics;
- simplified atmospheric/aerodynamic model;
- 13-state robust UKF;
- sigma-point prospective propagation;
- transparent future-maneuver hypotheses;
- causal multimodal Transformer;
- validation-only temperature calibration;
- independent entropy/confidence/margin abstention; and
- split-conformal set-valued predictions.

## What has been demonstrated

- end-to-end synthetic trajectory generation, filtering, prediction, calibration, and visualization;
- trajectory-isolated train/validation/test construction;
- deterministic numerical tests for major physics/estimation invariants;
- a versioned single-seed benchmark with explicit 0.4/0.5/0.6-second results; and
- automated evidence auditing that now exposes inadequate class coverage, single-seed evidence, and missing external validation.

## What is **not** being claimed

- flightworthiness;
- operational range-safety authority;
- certified navigation or guidance;
- production readiness;
- operational targeting/interception capability; or
- NASA or government endorsement.

## Proposed collaboration/evaluation scope

A high-value technical evaluation could focus on four questions:

1. **Credibility:** Which modeling assumptions dominate short-horizon error?
2. **Robustness:** How does performance change under sensor and model mismatch?
3. **Uncertainty:** Are calibration, abstention, and conformal prediction sets reliable under distribution shift?
4. **External validity:** Does the method retain value on independent civil telemetry or a separately configured high-fidelity simulation environment?

## Suggested evaluation deliverables

- frozen external-validation protocol;
- multi-seed benchmark matrix;
- scenario-shift and sensitivity report;
- baseline/ablation comparison;
- uncertainty and selective-risk report;
- independently reproducible evidence bundle; and
- go/no-go recommendation for any next research phase.

## Best framing for outreach

Position the repository as an **evaluation-ready research prototype with transparent gaps**, not as a finished operational product. The strongest invitation is to test whether the architecture's combination of physics, learning, and explicit assurance earns credibility on independent data.
