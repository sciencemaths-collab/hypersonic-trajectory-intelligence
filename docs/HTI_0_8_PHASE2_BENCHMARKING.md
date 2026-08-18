# HTI 0.8 Phase 2 — Frozen Fusion and Ablation Benchmarking

## Objective

Phase 2 turns the 0.8 structural/topological architecture into a controlled comparison framework. It does **not** change the 13-state UKF or declare a new accuracy result. Instead, it creates one frozen path from probability outputs to event-level evidence.

## Why this phase is separate

The core predictor already has a validated engineering path. Injecting new structural dimensions directly into the UKF or Transformer before proving value would make it difficult to determine whether a change helped, hurt, or merely increased complexity.

Phase 2 therefore keeps four surfaces separate:

1. **core HTI probability** from the existing calibrated predictor;
2. **structural/topology probability** from the 0.8 path bank and feasibility layer;
3. **validation-only fusion** through a normalized logarithmic opinion pool; and
4. **frozen test evaluation** with no fitted parameter selected on final-test data.

## Frozen protocol

`configs/hti_08_ablation_frozen.json` records:

- five independent seeds;
- 0.4, 0.5, and 0.6 second horizons;
- complete-trajectory event splitting;
- the eight required baseline/ablation variants;
- the validation-only fusion-weight grid;
- NLL, Brier, ECE, top-k, credible-region, entropy/selective-risk, and bootstrap requirements; and
- the minimum evidence gates required before a predictive-improvement claim is allowed.

Changing those settings after final-test inspection creates a new protocol version and requires a new final-test evaluation.

## Validation-only fusion

`hti.fusion.log_linear_pool` combines two probability sources:

```text
log P_fused = (1-w) log P_core + w log P_structural + constant
```

followed by normalization.

`select_structural_weight` chooses `w` using validation negative log likelihood only. Ties favor the smaller structural weight. The selected value must then be frozen and reused on the final test set.

A topology/cell prior can be applied independently with `apply_cell_prior`, again followed by normalization.

## Evidence bundle contract

The evaluator accepts a NumPy `.npz` file loaded with `allow_pickle=False`.

Required arrays:

```text
labels                 shape (samples, horizons) or (samples,)
event_ids              shape (samples,)
probs__<variant>       shape (samples, horizons, classes) or (samples, classes)
```

Examples:

```text
probs__core_hti
probs__core_plus_structural
probs__core_plus_topology
probs__hti_08_combined
```

All variants compared in one report must use the same samples, labels, horizon definitions, event identities, and cell partition.

## Evaluation

Run:

```bash
python scripts/hti_08_evaluate_predictions.py \
  --bundle frozen_test_predictions.npz \
  --config configs/hti_08_ablation_frozen.json \
  --report hti08_ablation_report.json
```

The report contains, per variant and horizon:

- class coverage;
- top-1/top-3/top-5 accuracy;
- negative log likelihood;
- multiclass Brier score;
- top-label ECE;
- 95% Bayesian credible-region coverage and mean size;
- mean confidence;
- mean entropy concentration; and
- selective risk at increasing retained-forecast coverage.

When both `core_hti` and `hti_08_combined` are present, the evaluator also reports a paired **trajectory-event bootstrap** interval for NLL change. The bootstrap samples complete event identities, not adjacent windows.

## CI smoke test

Research CI executes the evaluator on a deterministic synthetic probability bundle. The smoke test verifies the evaluation machinery, schema handling, metrics, and report generation. It is **not** scientific performance evidence for HTI 0.8.

## Required final interpretation

A favorable 0.8 result requires more than higher top-1 accuracy. At a claimed horizon the project should show, under the frozen protocol:

- sufficient held-out class coverage;
- lower NLL than core HTI;
- no unacceptable calibration regression;
- credible-region coverage consistent with its nominal level;
- event-level uncertainty intervals;
- transparent structural/topology ablations; and
- no use of final-test data for fusion, topology, calibration, or assurance tuning.

External/domain validation is still required before an external-readiness claim.
