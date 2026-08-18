# Scientific Gap Register

The goal of this register is to make closure criteria explicit. A gap is closed only when evidence is produced, not when wording is softened.

| Priority | Gap | Why it matters | Closure criterion | Current status |
|---|---|---|---|---|
| P0 | External validation | Synthetic holdout cannot establish domain validity | Frozen evaluation on independent civil telemetry or independently configured high-fidelity simulation, reviewed independently | Open |
| P0 | Validation/test class coverage | Current recorded test covers only 4/16 classes | Predeclared class-coverage target met in every split/horizon without trajectory leakage | Open |
| P0 | Multi-seed stability | One seed can hide variance | >=5 frozen independent seeds with summary intervals and unchanged test criteria | Open |
| P0 | 0.6 s calibration/performance failure | Longest recorded horizon fails original gates | Either meet frozen criteria on new independent evidence or remove 0.6 s from the supported claim envelope before evaluation | Open |
| P0 | Aerodynamic fidelity | Simplified drag cannot represent a specific high-speed vehicle | Replace with a documented, validated coefficient source or explicitly restrict claims to abstract dynamics research | Open |
| P0 | Numerical convergence evidence | RK2/0.05 s accuracy has not been convergence-qualified | Time-step refinement/reference-solution study with bounded error on representative scenarios | Open |
| P1 | Quaternion uncertainty geometry | Euclidean covariance is not manifold-consistent | Error-state/manifold estimator implemented and regression-validated, or attitude uncertainty shown immaterial to intended use | Open |
| P1 | Sensitivity analysis | Dominant assumptions are not ranked | Controlled parameter study with sensitivity measures for primary outputs | Open |
| P1 | Distribution-shift assurance | Calibration can fail outside development distribution | Shift matrix reporting calibration, abstention/selective risk, and conformal coverage | Open |
| P1 | Baseline/ablation completeness | Model value attribution is incomplete | Frozen constant-velocity, physics-only, learned-only, and combined comparisons | Open |
| P1 | Reproducible environment lock | Broad dependency ranges can drift | Release-specific lock/container digest archived with evidence bundle | Open |
| P2 | Real-time performance characterization | Runtime suitability is unknown | Hardware-defined latency/throughput benchmark with percentile reporting | Open |
| P2 | Independent IV&V-style review | Developer-only review can miss systematic errors | External technical review with findings tracked to closure | Open |

## Claim envelope today

The defensible claim today is narrow:

> The repository demonstrates a physics-informed, uncertainty-aware short-horizon trajectory-inference research workflow on synthetic data, with explicit evidence and assurance mechanisms and known validation limitations.

Anything stronger requires closure evidence from the table above.
