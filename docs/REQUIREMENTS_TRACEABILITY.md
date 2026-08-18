# Project Requirements Traceability

These are **project requirements**, not NASA requirement numbers. They create an auditable path from intended behavior to verification evidence.

| ID | Requirement | Primary implementation | Verification / evidence | Status |
|---|---|---|---|---|
| HTI-PHY-001 | Central gravity shall follow inverse-square magnitude for the configured point-mass Earth model. | `EarthModel.gravity` | `test_central_gravity_matches_inverse_square_law` | Verified |
| HTI-PHY-002 | ECEF propagation shall include Coriolis and centrifugal terms with tested sign convention. | `EarthModel.rotating_frame_acceleration` | `test_ecef_rotation_terms_have_expected_direction` | Verified |
| HTI-PHY-003 | Propagated quaternions shall be normalized after integration. | `Projectile6DOF.step_rk2` | `test_physics_projection_is_finite_and_normalized` | Verified |
| HTI-EST-001 | State covariance shall be repaired to symmetric positive definite form when numerically required. | `enforce_spd` | `test_spd_repair` | Verified |
| HTI-EST-002 | Hidden simulator environment jitter shall not be injected into the nominal filter environment. | `nominal_filter_env` | `test_filter_environment_contains_no_hidden_truth_jitter` | Verified |
| HTI-CAU-001 | Online model tokens shall use no observations or controls after the prediction time. | `build_causal_token_buffer` | causal-window tests | Verified |
| HTI-GEO-001 | Virtual-box cell mapping shall be internally consistent for cell centers. | `VirtualBox` | cell-center round-trip test | Verified |
| HTI-GEO-002 | Transverse gate width shall be independent of forward distance. | `VirtualBox` | cross-width independence test | Verified |
| HTI-CAL-001 | Temperature calibration shall be selected using validation data only. | calibration functions | calibration unit test + validation report | Verified for implementation |
| HTI-AST-001 | Probability outputs shall be rejected if non-finite, negative, or zero-mass. | `hti.assurance` | `test_invalid_probabilities_are_rejected` | Verified |
| HTI-AST-002 | Assurance shall support transparent abstention based on confidence, entropy, and top-class margin. | `assess_prediction` | acceptance/abstention tests | Verified |
| HTI-AST-003 | Set-valued predictions shall support split-conformal calibration on a separate calibration set. | conformal utilities | conformal utility tests | Verified for implementation |
| HTI-EVD-001 | Train/validation/test benchmark evidence shall be isolated by trajectory group. | dataset split + evidence audit | `ENG-002` | Verified for recorded artifact |
| HTI-EVD-002 | Versioned benchmark evidence shall not contain workstation-specific absolute paths. | evidence serialization policy | `ENG-004` | Verified for 0.7 artifact |
| HTI-EVD-003 | External scientific-readiness evidence shall include >=50% class coverage in every split/horizon. | evidence audit | `SCI-COVERAGE-*` | **Not met** |
| HTI-EVD-004 | External scientific-readiness evidence shall include at least five independent frozen benchmark seeds. | evidence audit | `SCI-MULTISEED` | **Not met** |
| HTI-EVD-005 | External scientific-readiness evidence shall include independent domain/external validation. | evidence audit | `SCI-EXTERNAL` | **Not met** |
| HTI-SEC-001 | NumPy trace loading shall disable pickle execution. | `streamlit_app.py` | existing UI test | Verified |
| HTI-SEC-002 | Model-state loading guidance shall prefer non-executable weights-only loading where supported. | benchmark artifact / security policy | artifact inspection | Documented |

## Traceability rules for future changes

- A behavior-changing pull request should identify the affected requirement IDs.
- New scientific claims should add a requirement and an objective acceptance criterion before final test evaluation.
- A requirement may be marked verified only when the evidence is repeatable and linked to code or a controlled external artifact.
- Failing requirements remain visible. They are not deleted to create a cleaner release narrative.
