# Changelog

## 0.8.0-research

### Added

- `hti.topological_entropy`, an independent structural-motion and topology layer derived from the Unified Topological-Entropy Trajectory Inference construction.
- Invertible nose/rear endpoint geometry with 2D/3D support.
- Circular/directional smoothing, slip angle, signed turn rate, curvature, swept area, zigzag diagnostics, body-length change, and transparent motion-mode evidence.
- Topology/reachability path weighting with explicit monotonic non-negative penalties.
- 3D spherical keep-out-volume intersection scoring for non-sensitive corridor and safety research.
- Separate occupancy and terminal-cell probability utilities.
- Entropy concentration and shortest-prefix credible-cell summaries.
- Bayesian backward conditioning of stored path banks for terminal-cell explanation.
- `scripts/hti_structural_analysis.py` for observed endpoint CSVs or existing HTI online traces.
- `docs/HTI_0_8_ARCHITECTURE.md` and `docs/TOPOLOGICAL_ENTROPY_VALIDATION.md`.
- Thirteen unit tests covering the new mathematical and structural invariants.

### Changed

- Research CI now verifies the HTI 0.8 structural/topological test suite on Python 3.10–3.12.
- Evidence manifests now hash the 0.8 structural module, test suite, and validation/architecture documents.
- Requirements traceability now includes geometry, topology, entropy, backward inference, and proxy-orientation disclosure requirements.
- README architecture now separates the core 6DOF estimator, structural-motion evidence, topology weighting, calibrated assurance, and backward explanation.

### Scientific status

- HTI 0.8 is an architecture/verification upgrade, not a retroactive performance improvement claim.
- The existing versioned benchmark remains the recorded 0.7 evidence: 0.4 s and 0.5 s pass the original aggregate gates; 0.6 s fails.
- No claim is made that structural/topological features improve held-out accuracy until the pre-registered ablation and external-validation program is completed.

## 0.7.0-research

### Added

- Independent assurance utilities for normalized entropy, confidence margin, abstention, and split-conformal prediction sets.
- Machine-readable engineering/scientific evidence gate.
- Python 3.10–3.12 GitHub Actions verification.
- Dependency audit and lint checks for the assurance/evidence layer.
- NASA-oriented research-readiness mapping.
- Project requirements traceability.
- External validation protocol.
- Architecture and security documentation.
- Technical evaluation proposal brief.

### Changed

- Benchmark evidence now records portable provenance rather than developer-workstation absolute paths.
- README and validation report now distinguish software integrity, recorded benchmark performance, class coverage, external validation, and operational certification.

### Evidence status

- 0.4 s and 0.5 s retain their recorded original benchmark passes.
- 0.6 s retains its recorded original benchmark failure.
- Validation/test class coverage is explicitly reported as 4/16 at every recorded horizon.
- Multi-seed and external validation remain incomplete and are now explicit scientific-readiness failures rather than undocumented caveats.
