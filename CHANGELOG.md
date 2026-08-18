# Changelog

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
