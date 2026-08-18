# NASA-Oriented Research Readiness Mapping

## Important scope statement

This document is a **research mapping**, not a statement that NASA has reviewed, endorsed, certified, or approved the software. It organizes the repository so a technical evaluator can locate evidence relevant to common NASA modeling/simulation, software-engineering, and software-assurance concerns.

Primary official references:

- NASA-STD-7009B, *Standard for Models and Simulations*: https://standards.nasa.gov/standard/nasa/nasa-std-7009
- NASA-HDBK-7009B, *NASA Handbook for Models and Simulations*: https://standards.nasa.gov/standard/nasa/nasa-hdbk-7009
- NPR 7150.2D, *NASA Software Engineering Requirements*: https://nodis3.gsfc.nasa.gov/displayDir.cfm?c=7150&s=2D&t=NPR
- NASA-STD-8739.8, *Software Assurance and Software Safety Standard*: https://standards.nasa.gov/standard/nasa/nasa-std-87398

## Intended-use statement

Current intended use is independent research into short-horizon trajectory-state estimation, probabilistic boundary crossing, uncertainty characterization, and credibility methods using synthetic or appropriately authorized civil aerospace data.

Excluded uses include operational flight control, navigation, targeting, interception, weapons control, and any safety-critical decision for which the software has not been independently validated and certified.

## Credibility evidence map

The labels below are project categories, not NASA requirement identifiers.

| Credibility area | Current repository evidence | Status | Next evidence |
|---|---|---|---|
| Intended use and limitations | README, validation report, this document | Documented | External reviewer concurrence |
| Conceptual model | Architecture + physics description | Partial | Independent domain review of assumptions |
| Input/data pedigree | Synthetic generator configuration recorded | Partial | External dataset provenance and data card |
| Code verification | Unit tests, deterministic checks, CI matrix | Strong for prototype | Coverage expansion and independent IV&V |
| Numerical solution verification | gravity, rotation, covariance, causality tests | Partial | convergence/order studies and reference solutions |
| Validation | trajectory-isolated synthetic holdout | Limited | independent external/high-fidelity validation |
| Uncertainty characterization | UKF covariance, calibration, assurance module | Partial | coverage under shift; sensitivity to assumptions |
| Sensitivity / robustness | stress scripts | Limited | formal parameter sweeps and ranked sensitivities |
| Calibration | validation-only temperature selection | Implemented | multi-seed calibration confidence intervals |
| Configuration control | versioned benchmark + provenance + CI | Improved | frozen release manifest and signed release |
| Reproducibility | seeded runs, CLI, requirements, evidence audit | Improved | locked environment/container and archived artifacts |
| Independent review | none recorded | Missing | external review or IV&V-style assessment |

## Software-engineering readiness map

### Requirements and traceability

See [REQUIREMENTS_TRACEABILITY.md](REQUIREMENTS_TRACEABILITY.md). Each project requirement is mapped to implementation evidence or an explicit gap.

### Verification

CI verifies supported Python versions, source compilation, unit tests, linting of the assurance/evidence layer, dependency audit, and integrity of the recorded benchmark artifact.

### Configuration management

The benchmark artifact records the source commit and explicitly distinguishes recorded evidence from external validation. Future benchmark releases should add environment fingerprints, dependency lock hashes, and immutable artifact digests.

### Risk and assurance

The assurance layer is independent of the predictor and can abstain when probability outputs are diffuse. This is not a substitute for model validation, but it creates an explicit fail-safe path for research demonstrations.

## Evidence gaps that block an operational-readiness claim

1. No external/domain validation.
2. Single-seed versioned benchmark.
3. Incomplete class coverage in validation and test.
4. Simplified aerodynamics and upper-atmosphere model.
5. No independent verification/validation organization or reviewer.
6. No flight-hardware, sensor-interface, real-time, fault-tolerance, cybersecurity, or certification evidence.
7. No configuration-controlled operational requirements baseline.

## Appropriate proposal framing

A defensible proposal is **not** “this system is ready for NASA operations.” A defensible proposal is:

> An independent physics-informed trajectory-inference research prototype is available for technical evaluation. The proposed collaboration would test credibility, uncertainty, robustness, and external validity on non-sensitive civil aerospace scenarios, with explicit abstention and evidence gates.

That framing invites evaluation instead of asking an external organization to accept an unsupported deployment claim.
