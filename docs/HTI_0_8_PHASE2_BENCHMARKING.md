# HTI 0.8 Phase 2 — Frozen Fusion and Ablation Benchmarking

## Objective

Phase 2 turns the 0.8 structural/topological architecture into a controlled comparison framework. It does **not** change the 13-state UKF or declare a new accuracy result. Instead, it creates one frozen path from probability outputs to event-level evidence.

## Why this phase is separate

The core predictor already has a verified engineering path. Injecting new structural dimensions directly into the UKF or Transformer before proving value would make it difficult to determine whether a change helped, hurt, or merely increased complexity.

Phase 2 therefore keeps these surfaces distinct:

1. **core HTI probability** from the existing calibrated predictor;
2. **raw structural probability** from the two-point structural branch;
3. **topology cell prior** from the frozen reachability/history-consistency construction;
4. **validation-only structural fusion** through a normalized logarithmic opinion pool;
5. **topology application** through prior multiplication and renormalization; and
6. **frozen test evaluation** with no fitted parameter selected on final-test data.

## Frozen protocol

`configs/hti_08_ablation_frozen.json` records:

- five independent seeds;
- 0.4, 0.5, and 0.6 second horizons;
- complete-trajectory event splitting;
- the eight required baseline/ablation variants;
- the validation-only structural-fusion weight grid;
- the required composition order: structural fusion first, frozen topology prior second;
- NLL, Brier, ECE, top-k, credible-region, entropy/selective-risk, and bootstrap requirements; and
- the minimum evidence gates required before a predictive-improvement claim is allowed.

`configs/hti_08_phase3_execution_frozen.json` records the concrete Phase 3 generator, model, structural, topology, conformal, and execution settings. Its SHA-256 is part of every final prediction bundle.

Changing these settings after final-test inspection creates a new protocol version and requires a new final-test evaluation.

## Validation-only structural fusion

`hti.fusion.log_linear_pool` combines core and **raw structural-branch** probabilities:

```text
log P_struct_fused = (1-w) log P_core + w log P_structural + constant
```

followed by normalization.

`select_structural_weight` chooses `w` using validation negative log likelihood only. Ties favor the smaller structural weight. The selected value is frozen per horizon and reused unchanged on the final test set.

The evaluator independently reconstructs `core_plus_structural` from `core_hti`, `structural_branch_probabilities`, and those frozen weights. A bundle that cannot be reconstructed is rejected.

## Topology composition

The topology branch supplies a positive class/cell prior `P_topology`. The prior strength is frozen in the Phase 3 execution config.

The two topology-bearing scored variants are defined as:

```text
core_plus_topology = normalize(P_core * P_topology^s)
hti_08_combined    = normalize(P_struct_fused * P_topology^s)
```

where `s` is `topology_branch.cell_prior_strength` in the frozen execution config.

This means **HTI 0.8 combined really includes both structural and topology information**. The evaluator reconstructs both topology-bearing variants and rejects a bundle in which `hti_08_combined` is merely the structural fusion without topology.

## Evidence bundle contract

The evaluator accepts a NumPy `.npz` file loaded with `allow_pickle=False`.

Required arrays/fields for a final frozen bundle include:

```text
labels                              (samples, horizons) or (samples,)
event_ids                           (samples,)
seed                                scalar frozen seed
train_event_ids                     complete-event training manifest
validation_event_ids                complete-event validation manifest
test_event_ids                      complete-event test manifest
orientation_source                  scalar Unicode provenance label
probs__<variant>                    scored variant probability cubes
cell_ids__<variant>                 ordered unique cell IDs, identical across variants
cell_partition_sha256               SHA-256 of the ordered frozen partition
structural_branch_probabilities     raw structural-branch probability cube
topology_cell_prior                 frozen topology-prior probability cube
fusion_selection_data               literal `validation_only`
fusion_structural_weights           one frozen candidate-grid weight per horizon
validation_selection_sha256         SHA-256 of the retained validation-selection artifact
execution_config_sha256             SHA-256 of the frozen Phase 3 execution config
model_sha256__core_hti              core learned-model artifact digest
model_sha256__learned_only          learned-only artifact digest
model_sha256__structural_branch     structural-model artifact digest
topology_definition_sha256          SHA-256 of frozen topology definitions
topology_coefficients_sha256        SHA-256 of frozen topology coefficients
topology_true_path_suppressed       shape matching labels when required by policy
```

The evaluator rejects overlapping train/validation/test event identities. The unique `event_ids` attached to prediction rows must exactly equal the declared test-event set.

All eight scored variants must use the same samples, labels, horizons, event identities, class count, and ordered cell partition. The evaluator rejects inconsistent cell IDs, execution-config hashes, fusion weights outside the frozen candidate grid, missing learned-model digests, and any structural/topology/combined probabilities that cannot be independently reconstructed.

Optional pre-calibrated conformal thresholds may be supplied as:

```text
conformal_qhat__<variant>                    scalar or one value per horizon
conformal_calibration_sha256__<variant>      SHA-256 of retained calibration evidence
```

These thresholds are **evaluated only**. The evaluator never fits them on final-test labels and rejects thresholds without calibration-artifact provenance.

## Evaluation

Run:

```bash
python scripts/hti_08_evaluate_predictions.py \
  --bundle frozen_test_predictions.npz \
  --config configs/hti_08_ablation_frozen.json \
  --execution-config configs/hti_08_phase3_execution_frozen.json \
  --report hti08_ablation_report.json
```

The report contains, per variant and horizon:

- class coverage;
- top-1/top-3/top-5 accuracy;
- negative log likelihood;
- multiclass Brier score;
- top-label ECE;
- reliability-bin boundaries, counts, mean confidence, and empirical accuracy;
- 95% Bayesian credible-region coverage, mean size, and median size;
- mean confidence;
- mean entropy concentration;
- selective risk at increasing retained-forecast coverage; and
- conformal coverage plus mean/median set size when a frozen `qhat` is supplied.

For `core_hti` versus `hti_08_combined`, the evaluator reports a paired **complete-event bootstrap** interval for NLL change. For topology, it reports total-variation probability change, true-cell support change, and path-level suppression flags when supplied.

The report records SHA-256 identifiers for the protocol, execution config, and probability bundle, plus the source checkout used to execute the evaluator.

## CI smoke test

Research CI executes the evaluator on a deterministic synthetic bundle containing all eight scored variants, raw structural probabilities, a topology prior, disjoint event manifests, conformal thresholds, orientation provenance, model/protocol digests, and topology-suppression evidence. The smoke test verifies schema handling, composition reproducibility, metrics, rejection paths, and report generation. It is **not** scientific performance evidence.

CI also directly launches the structural-analysis and Phase 3 pilot CLIs so packaging/import failures cannot hide behind module-only tests.

## Required final interpretation

A favorable 0.8 result requires more than higher top-1 accuracy. At a claimed horizon the project must show, under the frozen protocol:

- sufficient held-out class coverage;
- lower NLL than core HTI;
- no unacceptable calibration regression;
- credible-region coverage consistent with its nominal level;
- event-level uncertainty intervals;
- transparent structural and topology ablations;
- conformal coverage when thresholds were calibrated;
- explicit topology failure evidence;
- explicit orientation provenance;
- reproducible structural and topology composition; and
- no use of final-test data for fusion, topology, calibration, or assurance tuning.

External/domain validation is still required before an external-readiness claim.
