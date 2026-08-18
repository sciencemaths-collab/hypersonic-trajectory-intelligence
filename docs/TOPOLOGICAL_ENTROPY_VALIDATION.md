# HTI 0.8 Topological-Entropy Validation and Falsification Plan

## Purpose

This plan converts the new structural/topological architecture into falsifiable experiments. The implementation is not considered scientifically superior merely because its algebraic tests pass or its visualizations look plausible.

## Dataset split rule

All training, calibration, validation, and test partitions must be split by complete trajectory/event identity. Adjacent windows from one trajectory must not cross split boundaries.

## Primary questions

| Question | Required evidence |
|---|---|
| Is the terminal distribution calibrated? | NLL/log loss, Brier score, reliability plot, ECE |
| Does a 95% credible cell set cover at approximately 95%? | empirical coverage and mean/median set size |
| Is the peak cell useful? | top-1/top-3/top-5 accuracy plus geometric cell-distance error |
| Does endpoint geometry add information? | endpoint/two-point model vs center-only ablation on identical splits/seeds |
| Do topology and swept area add information? | component ablations with identical path banks and random seeds |
| Does entropy concentration predict error? | error/selective risk by entropy-concentration decile |
| Does the model beat simpler approaches? | constant velocity, Kalman, IMM and particle-filter baselines where applicable |
| Is it robust? | noise, missing frames, endpoint label swaps, deformation, timing shift, abrupt turns, scenario shift |
| Does backward explanation remain consistent? | conditional weights sum to one and recover forward terminal mass |

## Pre-registered comparison matrix

At minimum, the final experiment should contain:

1. **CV**: constant velocity;
2. **KF/UKF baseline**: state estimate with direct extrapolation;
3. **Physics-only**: current 6DOF probability rollout without Transformer;
4. **Learned-only**: causal learned branch where a scientifically fair learned-only configuration is possible;
5. **Core HTI**: present combined architecture;
6. **Core + structural**: adds endpoint/curvature/slip/zigzag/area features;
7. **Core + topology**: adds topology/reachability weighting only; and
8. **HTI 0.8 combined**: structural + topology + calibrated assurance.

The same event-level splits, seeds, horizons, grid definition, and evaluation code must be used across comparable variants.

## Endpoint-data integrity tests

Before evaluating predictive value:

- nose/rear identity must be stable or explicit label-switching scenarios must be evaluated;
- timestamps must be strictly increasing;
- endpoint coordinates must be in a documented common frame;
- body-length distributions and missing endpoint samples must be reported;
- when orientation is reconstructed from a state estimate, the source must be identified; and
- a velocity-derived orientation proxy must be labeled as proxy evidence and analyzed separately from true endpoint/attitude input.

## Calibration metrics

For every supported horizon report:

- negative log likelihood;
- Brier score;
- top-label ECE;
- reliability bins with counts;
- peak confidence distribution;
- entropy concentration distribution;
- conformal prediction-set coverage and size; and
- Bayesian credible-cell coverage and size.

Entropy concentration must never be presented as accuracy.

## Spatial metrics

Because cell accuracy can be harsh near cell boundaries, report both classification and geometric error:

- top-k cell accuracy;
- row/column or physical cell-center distance error;
- probability mass within one-cell and two-cell neighborhoods, when meaningful for the grid; and
- first-boundary/terminal side accuracy separately from cell-within-side accuracy when using six-face volumes.

## Topology falsification

Topology is useful only if the penalty or prior represents justified reachability information.

Required tests:

- zero-penalty topology must reproduce base path probabilities;
- increasing one violation penalty must not increase that path's relative support, all else equal;
- barrier/corridor definitions must be fixed before final evaluation;
- penalty coefficients must be selected from physical, engineering, validation-only, or pre-registered rationale, not final-test tuning;
- report the fraction of probability mass changed by topology; and
- report failure cases in which topology incorrectly suppresses the true path.

## Structural-feature ablations

Run individual or grouped ablations for:

- body orientation/slip;
- turn rate/curvature;
- zigzag statistic;
- swept area;
- body-length/deformation rate; and
- mode-evidence features.

The purpose is to determine whether the new features provide information beyond the existing state estimate rather than merely increasing model complexity.

## Entropy as an assurance signal

Sort predictions by `Q = 1 - H/log K` and evaluate error/selective risk by decile.

A useful assurance signal should generally show lower error among more concentrated accepted forecasts after calibration. A flat or reversed relationship is a finding, not something to hide.

Also evaluate the existing confidence/margin abstention and conformal-set layer so entropy is not the sole rejection mechanism.

## Robustness matrix

At minimum include controlled shifts in:

- measurement position noise;
- measurement velocity noise;
- endpoint noise;
- endpoint label swaps;
- random frame loss;
- timestamp jitter;
- atmosphere/density assumptions;
- vehicle mass/inertia or generic-vehicle parameters;
- maneuver frequency and magnitude;
- deformation rate; and
- abrupt turn behavior.

For each shift report accuracy, NLL, Brier, ECE, credible/conformal coverage, abstention rate, and selective risk.

## Backward inference validation

Backward explanation is evaluated as conditional consistency, not unique history recovery.

For every queried terminal cell with positive forward probability:

- conditional path weights must be finite, non-negative, and sum to one;
- summing joint path/cell mass over paths must recover the forward terminal-cell probability;
- conditional expected histories must remain within the convex support of retained path states;
- mode-support totals must sum to one; and
- zero-support cells must fail explicitly rather than return fabricated explanations.

## Example pre-registration targets

The exact release thresholds should be frozen before final-test inspection. A defensible example family is:

- held-out NLL lower than every pre-registered baseline;
- 95% credible-region empirical coverage within a narrow predeclared interval around 95%;
- credible region smaller than the best equally calibrated baseline;
- statistically supported top-k improvement under event-level bootstrap intervals;
- no material calibration regression at horizons claimed as improved; and
- no improvement claim when the held-out class-coverage requirement is not met.

These are examples, not post-hoc release thresholds for the current benchmark.

## Current status

HTI 0.8 adds architecture, code, and algebraic verification only. The versioned 0.7 benchmark is not retroactively relabeled as a 0.8 performance result.

A new frozen multi-seed experiment and external/domain validation are still required before claiming practical predictive improvement from the topological-entropy layer.
