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
| HTI-GEO-003 | Nose/rear endpoint geometry shall be invertible to center/body-axis coordinates whenever body length is non-zero. | `endpoint_geometry`, `recover_endpoints` | `test_endpoint_transform_round_trip` | Verified |
| HTI-GEO-004 | A centerline/state trace may be bridged into endpoint form only with an explicit source for body direction and body length. | `endpoint_observations_from_centerline` | centerline bridge test + CLI provenance field | Verified for implementation |
| HTI-STR-001 | Wrapped angular smoothing shall be invariant to integer multiples of 2π. | `circular_smooth` | `test_circular_smoothing_is_wrap_invariant` | Verified |
| HTI-STR-002 | Structural motion output shall expose slip, turn, curvature, swept area, zigzag, length change, and normalized explanatory mode evidence. | `estimate_structural_motion` | structural-mode normalization test | Verified for implementation |
| HTI-TOP-001 | A topology factor using non-negative penalties shall be bounded in `(0,1]` and decrease as a violation penalty increases. | `topology_weight` | `test_topology_weight_is_monotone` | Verified |
| HTI-TOP-002 | Topology weighting shall multiply candidate-path support and renormalize to unit total mass. | `reweight_paths` | topology transfer test | Verified |
| HTI-TOP-003 | 3D safety-volume intersection counting shall identify candidate paths entering configured non-sensitive keep-out spheres. | `spherical_zone_penalties` | keep-out intersection test | Verified |
| HTI-PRB-001 | Retained terminal-path probabilities shall normalize to one when every retained path has one valid terminal cell. | `terminal_cell_probabilities` | terminal normalization test | Verified |
| HTI-PRB-002 | Occupancy probability shall remain separate from terminal probability and may sum above one across cells. | `occupancy_probabilities` | occupancy semantics test | Verified |
| HTI-PRB-003 | A cell-level prior shall be applied by multiplication followed by renormalization. | `terminal_cell_probabilities(..., cell_prior=...)` | cell-prior test | Verified |
| HTI-ENT-001 | Entropy concentration shall be reported as a forecast-distribution concentration diagnostic, not as accuracy or a cell-specific multiplier. | `summarize_distribution` | entropy/credible-set test + 0.8 architecture doc | Verified for implementation |
| HTI-BWD-001 | Backward conditioning on a supported terminal cell shall produce finite non-negative path weights summing to one. | `condition_paths_on_terminal_cell` | backward conditioning test | Verified |
| HTI-BWD-002 | Backward explanation shall support conditional mode support and expected state histories without claiming unique stochastic reversal. | `explain_terminal_cell` | backward explanation test | Verified for implementation |
| HTI-CAL-001 | Temperature calibration shall be selected using validation data only. | calibration functions | calibration unit test + validation report | Verified for implementation |
| HTI-AST-001 | Probability outputs shall be rejected if non-finite, negative, or zero-mass. | `hti.assurance` | `test_invalid_probabilities_are_rejected` | Verified |
| HTI-AST-002 | Assurance shall support transparent abstention based on confidence, entropy, and top-class margin. | `assess_prediction` | acceptance/abstention tests | Verified |
| HTI-AST-003 | Set-valued predictions shall support split-conformal calibration on a separate calibration set. | conformal utilities | conformal utility tests | Verified for implementation |
| HTI-EVD-001 | Train/validation/test benchmark evidence shall be isolated by trajectory group. | dataset split + evidence audit | `ENG-002` | Verified for recorded artifact |
| HTI-EVD-002 | Versioned benchmark evidence shall not contain workstation-specific absolute paths. | evidence serialization policy | `ENG-004` | Verified for 0.7 artifact |
| HTI-EVD-003 | External scientific-readiness evidence shall include >=50% class coverage in every split/horizon. | evidence audit | `SCI-COVERAGE-*` | **Not met** |
| HTI-EVD-004 | External scientific-readiness evidence shall include at least five independent frozen benchmark seeds. | evidence audit | `SCI-MULTISEED` | **Not met** |
| HTI-EVD-005 | External scientific-readiness evidence shall include independent domain/external validation. | evidence audit | `SCI-EXTERNAL` | **Not met** |
| HTI-EVD-006 | Any claim that HTI 0.8 improves prediction shall include pre-registered center-only/two-point, topology, swept-area/zigzag, and combined ablations on identical event-level splits and seeds. | validation protocol | `docs/TOPOLOGICAL_ENTROPY_VALIDATION.md` | **Not yet evaluated** |
| HTI-EVD-007 | Entropy usefulness shall be evaluated against empirical error/selective risk by concentration decile before being presented as an assurance signal. | validation protocol | entropy-error stratification | **Not yet evaluated** |
| HTI-SEC-001 | NumPy trace loading shall disable pickle execution. | `streamlit_app.py`, structural CLI | existing UI test + `allow_pickle=False` | Verified |
| HTI-SEC-002 | Model-state loading guidance shall prefer non-executable weights-only loading where supported. | benchmark artifact / security policy | artifact inspection | Documented |

## Traceability rules for future changes

- A behavior-changing pull request should identify the affected requirement IDs.
- New scientific claims should add a requirement and an objective acceptance criterion before final test evaluation.
- A requirement may be marked verified only when the evidence is repeatable and linked to code or a controlled external artifact.
- Failing requirements remain visible. They are not deleted to create a cleaner release narrative.
- Proxy orientation, synthetic topology, and heuristic mode evidence must be labeled as such in reports and demonstrations.
