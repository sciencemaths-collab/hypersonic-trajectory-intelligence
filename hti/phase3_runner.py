"""Frozen single-seed execution for the HTI 0.8 synthetic ablation study.

All fit/selection operations use complete-event train or validation data only.
The returned test bundle contains probabilities and provenance, but this module
never evaluates final-test metrics or changes frozen parameters from test data.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alien_exit_cell_predictor_v6_3 import (
    AeroModel,
    Atmosphere,
    Config,
    EarthModel,
    Projectile6DOF,
    VirtualBox,
    compute_box_cross_width,
    compute_box_L,
    fit_model_temperatures,
    nominal_filter_env,
    project_maneuver_hypotheses,
    project_sigma_horizons,
    project_state_horizons,
    seed_everything,
    train_model,
)

from .assurance import conformal_quantile
from .fusion import apply_cell_prior, log_linear_pool, select_structural_weight
from .phase3 import Phase3Event, generate_phase3_events, split_event_ids
from .phase3_models import (
    RidgeProbabilisticClassifier,
    history_consistency_weights,
    measurement_history_vector,
    noisy_truth_endpoint_structure,
    path_cell_distribution,
    select_temperature,
    smoothed_cell_distribution,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_arrays(*arrays: np.ndarray, prefix: str = "") -> str:
    digest = hashlib.sha256(prefix.encode("utf-8"))
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def source_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def configure(protocol: dict[str, object], execution: dict[str, object], seed: int) -> Config:
    generation = execution["trajectory_generation"]
    transformer = execution["transformer"]
    cfg = Config()
    cfg.seed = int(seed)
    cfg.gpu = False
    cfg.no_viz = True
    cfg.traj_steps = int(generation["traj_steps"])
    cfg.offline_stride = max(1, int(generation["offline_stride"]))
    cfg.window = int(generation["window"])
    cfg.maneuver_prob = float(generation["maneuver_prob"])
    cfg.force_std = float(generation["force_std_n"])
    cfg.torque_std = float(generation["torque_std_nm"])
    cfg.max_mach = float(generation["max_mach"])
    cfg.box_cross_scale = float(generation["box_cross_scale"])
    cfg.box_cross_min_m = float(generation["box_cross_min_m"])
    cfg.box_cross_max_m = float(generation["box_cross_max_m"])
    cfg.horizon_steps = tuple(
        int(round(float(value) / cfg.dt)) for value in protocol["horizons_seconds"]
    )
    cfg.epochs = int(transformer["epochs"])
    cfg.batch_size = int(transformer["batch_size"])
    cfg.d_model = int(transformer["d_model"])
    cfg.nhead = int(transformer["nhead"])
    cfg.nlayers = int(transformer["nlayers"])
    cfg.ff_mult = int(transformer["ff_mult"])
    cfg.dropout = float(transformer["dropout"])
    cfg.lr = float(transformer["learning_rate"])
    cfg.weight_decay = float(transformer["weight_decay"])
    cfg.label_smoothing = float(transformer["label_smoothing"])
    return cfg


def _event_subset(events: list[Phase3Event], event_ids: np.ndarray) -> list[Phase3Event]:
    by_id = {int(event.event_id): event for event in events}
    result: list[Phase3Event] = []
    for event_id in np.asarray(event_ids).reshape(-1):
        key = int(event_id)
        if key not in by_id:
            raise ValueError(f"unknown event ID {key}")
        result.append(by_id[key])
    return result


def _join_events(
    events: list[Phase3Event],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not events:
        raise ValueError("event split is empty")
    tokens = np.concatenate([event.tokens for event in events], axis=0)
    labels = np.concatenate([event.labels for event in events], axis=0)
    next_features = np.concatenate([event.next_features for event in events], axis=0)
    event_ids = np.concatenate(
        [np.full(len(event.tokens), event.event_id, dtype=np.int64) for event in events]
    )
    return tokens, labels, next_features, event_ids


def _normalize_training_arrays(
    x_train: np.ndarray,
    next_train: np.ndarray,
    *other_tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    scale = x_train.reshape(-1, x_train.shape[-1]).std(axis=0) + 1e-6
    next_mean = next_train.mean(axis=0)
    next_scale = next_train.std(axis=0) + 1e-6

    def norm_tokens(values: np.ndarray) -> np.ndarray:
        return ((values - mean[None, None, :]) / scale[None, None, :]).astype(np.float32)

    return (
        norm_tokens(x_train),
        ((next_train - next_mean) / next_scale).astype(np.float32),
        [norm_tokens(values) for values in other_tokens],
        mean,
        scale,
        next_mean,
        next_scale,
    )


def _transformer_probabilities(
    model,
    device: torch.device,
    normalized_tokens: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(normalized_tokens).float().to(device)
        logits, _ = model(tensor)
        temp = torch.from_numpy(np.asarray(temperatures, dtype=np.float32)).to(device).view(
            1, -1, 1
        )
        return F.softmax(logits / temp, dim=-1).cpu().numpy()


def _torch_model_digest(model) -> str:
    digest = hashlib.sha256(b"hti-phase3-transformer")
    for name, tensor in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _reference_box(cfg: Config, state: np.ndarray) -> VirtualBox | None:
    speed = float(np.linalg.norm(state[3:6]))
    if speed <= 1e-9:
        return None
    return VirtualBox(
        state[:3],
        state[3:6],
        np.array([0.0, 0.0, 1.0]),
        compute_box_L(cfg, speed),
        cfg.box_N,
        half_width=compute_box_cross_width(cfg, speed),
    )


def _cell_index(box: VirtualBox | None, point: np.ndarray, box_n: int) -> int | None:
    if box is None:
        return None
    cell = box.compute_exit_cell(np.asarray(point, dtype=float))
    if cell is None:
        return None
    return int(cell[0] * box_n + cell[1])


def _physics_path_bank(
    projectile: Projectile6DOF,
    cfg: Config,
    event: Phase3Event,
    frame: int,
    *,
    max_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = event.estimated_states[frame]
    u_current = event.controls[frame].copy() if frame < len(event.controls) else np.zeros(6)
    sigma_paths, sigma_weights = project_sigma_horizons(
        projectile,
        state,
        event.covariances[frame],
        u_current,
        cfg.dt,
        nominal_filter_env(),
        max_horizon,
    )
    maneuver_paths, _ = project_maneuver_hypotheses(
        projectile,
        state,
        u_current,
        cfg.dt,
        nominal_filter_env(),
        max_horizon,
        cfg.force_std,
    )
    mixture = float(np.clip(cfg.maneuver_prob, 0.0, 1.0))
    if len(maneuver_paths) == 0 or mixture <= 0.0:
        return sigma_paths, sigma_weights / sigma_weights.sum()
    combined_paths = np.concatenate([sigma_paths, maneuver_paths], axis=0)
    combined_weights = np.concatenate(
        [
            (1.0 - mixture) * sigma_weights,
            np.full(len(maneuver_paths), mixture / len(maneuver_paths), dtype=float),
        ]
    )
    combined_weights /= combined_weights.sum()
    return combined_paths, combined_weights


def _frozen_feature_split(
    events: list[Phase3Event],
    cfg: Config,
    execution: dict[str, object],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    classes = cfg.box_N * cfg.box_N
    horizons = list(cfg.horizon_steps)
    max_horizon = max(horizons)
    constant_cfg = execution["constant_velocity"]
    filter_cfg = execution["filter_direct"]
    physics_cfg = execution["physics_branch"]
    learned_cfg = execution["learned_only"]
    structural_cfg = execution["structural_branch"]
    topology_cfg = execution["topology_branch"]
    constant_smoothing = float(constant_cfg["probability_smoothing"])
    filter_smoothing = float(filter_cfg["probability_smoothing"])
    physics_smoothing = float(physics_cfg["one_hot_smoothing"])

    earth, atmosphere, aero = EarthModel(), Atmosphere(), AeroModel()
    projectile = Projectile6DOF(
        1000.0,
        np.diag([500.0, 800.0, 1000.0]),
        earth,
        atmosphere,
        aero,
    )

    probability_rows = {
        "constant_velocity": [],
        "filter_direct": [],
        "physics_only": [],
        "topology_prior": [],
    }
    measurement_rows: list[np.ndarray] = []
    structural_rows: list[np.ndarray] = []
    event_id_rows: list[int] = []

    for event in events:
        for frame in event.source_frames:
            source_frame = int(frame)
            state = event.estimated_states[source_frame]
            box = _reference_box(cfg, state)
            u_current = (
                event.controls[source_frame].copy()
                if source_frame < len(event.controls)
                else np.zeros(6, dtype=float)
            )
            direct_path = project_state_horizons(
                projectile,
                state,
                u_current,
                cfg.dt,
                nominal_filter_env(),
                max_horizon,
            )
            path_bank, path_weights = _physics_path_bank(
                projectile,
                cfg,
                event,
                source_frame,
                max_horizon=max_horizon,
            )
            measurement_rows.append(
                measurement_history_vector(
                    event,
                    source_frame,
                    history_points=int(learned_cfg["history_points"]),
                    history_stride_frames=int(learned_cfg["history_stride_frames"]),
                )
            )
            structural_vector, structural_travel_direction = noisy_truth_endpoint_structure(
                event,
                source_frame,
                seed=seed,
                dt=cfg.dt,
                history_points=int(structural_cfg["history_points"]),
                body_length=float(structural_cfg["body_length_m"]),
                endpoint_noise_std=float(structural_cfg["endpoint_noise_std_m"]),
            )
            structural_rows.append(structural_vector)
            event_id_rows.append(int(event.event_id))

            constant_h: list[np.ndarray] = []
            filter_h: list[np.ndarray] = []
            physics_h: list[np.ndarray] = []
            topology_h: list[np.ndarray] = []
            for horizon in horizons:
                seconds = float(horizon * cfg.dt)
                constant_point = state[:3] + state[3:6] * seconds
                constant_h.append(
                    smoothed_cell_distribution(
                        _cell_index(box, constant_point, cfg.box_N),
                        classes=classes,
                        smoothing=constant_smoothing,
                    )
                )
                filter_h.append(
                    smoothed_cell_distribution(
                        _cell_index(box, direct_path[horizon, :3], cfg.box_N),
                        classes=classes,
                        smoothing=filter_smoothing,
                    )
                )
                cells = np.asarray(
                    [
                        -1
                        if (cell := _cell_index(box, path[horizon, :3], cfg.box_N)) is None
                        else cell
                        for path in path_bank
                    ],
                    dtype=int,
                )
                physics_h.append(
                    path_cell_distribution(
                        path_weights,
                        cells,
                        classes=classes,
                        smoothing=physics_smoothing,
                    )
                )
                topology_weights = history_consistency_weights(
                    path_weights,
                    path_bank[:, : horizon + 1, :3],
                    origin=state[:3],
                    travel_direction=structural_travel_direction,
                    gamma=float(topology_cfg["history_angle_gamma"]),
                )
                topology_h.append(
                    path_cell_distribution(
                        topology_weights,
                        cells,
                        classes=classes,
                        smoothing=0.0,
                    )
                )
            probability_rows["constant_velocity"].append(np.stack(constant_h))
            probability_rows["filter_direct"].append(np.stack(filter_h))
            probability_rows["physics_only"].append(np.stack(physics_h))
            probability_rows["topology_prior"].append(np.stack(topology_h))

    return (
        {name: np.stack(rows) for name, rows in probability_rows.items()},
        np.stack(measurement_rows),
        np.stack(structural_rows),
        np.asarray(event_id_rows, dtype=np.int64),
    )


def _fit_ridge_branch(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    classes: int,
    ridge_lambda: float,
    temperature_candidates: np.ndarray,
    digest_prefix: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], str]:
    horizons = train_labels.shape[1]
    validation_probabilities = np.empty((len(validation_features), horizons, classes), dtype=float)
    test_probabilities = np.empty((len(test_features), horizons, classes), dtype=float)
    selections: list[dict[str, float]] = []
    digest = hashlib.sha256(digest_prefix.encode("utf-8"))
    for horizon in range(horizons):
        train_mask = train_labels[:, horizon] >= 0
        validation_mask = validation_labels[:, horizon] >= 0
        if not np.any(train_mask) or not np.any(validation_mask):
            raise RuntimeError(f"ridge branch horizon {horizon} lacks train/validation labels")
        classifier = RidgeProbabilisticClassifier.fit(
            train_features[train_mask],
            train_labels[train_mask, horizon],
            classes=classes,
            ridge_lambda=ridge_lambda,
        )
        temperature, validation_nll = select_temperature(
            classifier,
            validation_features[validation_mask],
            validation_labels[validation_mask, horizon],
            temperature_candidates,
        )
        validation_probabilities[:, horizon, :] = classifier.probabilities(
            validation_features, temperature=temperature
        )
        test_probabilities[:, horizon, :] = classifier.probabilities(
            test_features, temperature=temperature
        )
        for value in (classifier.mean, classifier.scale, classifier.coefficients):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        digest.update(np.asarray([temperature], dtype=float).tobytes())
        selections.append(
            {
                "horizon_index": int(horizon),
                "temperature": float(temperature),
                "validation_nll": float(validation_nll),
            }
        )
    return validation_probabilities, test_probabilities, selections, digest.hexdigest()


def _select_pool(
    first_validation: np.ndarray,
    second_validation: np.ndarray,
    validation_labels: np.ndarray,
    first_test: np.ndarray,
    second_test: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    validation_output = np.empty_like(first_validation)
    test_output = np.empty_like(first_test)
    weights: list[float] = []
    nlls: list[float] = []
    for horizon in range(validation_labels.shape[1]):
        mask = validation_labels[:, horizon] >= 0
        selection = select_structural_weight(
            first_validation[mask, horizon, :],
            second_validation[mask, horizon, :],
            validation_labels[mask, horizon],
            candidates=candidates,
        )
        validation_output[:, horizon, :] = log_linear_pool(
            first_validation[:, horizon, :],
            second_validation[:, horizon, :],
            structural_weight=selection.structural_weight,
        )
        test_output[:, horizon, :] = log_linear_pool(
            first_test[:, horizon, :],
            second_test[:, horizon, :],
            structural_weight=selection.structural_weight,
        )
        weights.append(float(selection.structural_weight))
        nlls.append(float(selection.validation_nll))
    return validation_output, test_output, weights, nlls


def _apply_topology_cube(
    probabilities: np.ndarray,
    topology_prior: np.ndarray,
    *,
    strength: float,
    floor: float,
) -> np.ndarray:
    if probabilities.shape != topology_prior.shape:
        raise ValueError("topology prior must match probability cube")
    result = np.empty_like(probabilities, dtype=float)
    for horizon in range(probabilities.shape[1]):
        result[:, horizon, :] = apply_cell_prior(
            probabilities[:, horizon, :],
            np.maximum(topology_prior[:, horizon, :], float(floor)),
            strength=float(strength),
        )
    return result


def _conformal_parameters(
    variants: dict[str, np.ndarray],
    validation_labels: np.ndarray,
    validation_event_ids: np.ndarray,
    *,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    qhats: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for name, cube in variants.items():
        values: list[float] = []
        for horizon in range(validation_labels.shape[1]):
            mask = validation_labels[:, horizon] >= 0
            values.append(
                conformal_quantile(
                    cube[mask, horizon, :],
                    validation_labels[mask, horizon],
                    alpha=alpha,
                )
            )
        qhat = np.asarray(values, dtype=float)
        qhats[name] = qhat
        digests[name] = sha256_arrays(
            cube,
            validation_labels,
            validation_event_ids,
            qhat,
            prefix=f"conformal:{name}:alpha={alpha}",
        )
    return qhats, digests


def execute_seed(
    *,
    seed: int,
    protocol: dict[str, object],
    execution: dict[str, object],
    protocol_path: Path,
    execution_path: Path,
    output: Path,
    selection_output: Path,
) -> dict[str, object]:
    frozen_seeds = [int(value) for value in protocol["seeds"]]
    if int(seed) not in frozen_seeds:
        raise ValueError(f"seed must be one of the frozen Phase 3 seeds {frozen_seeds}")
    cfg = configure(protocol, execution, int(seed))
    seed_everything(int(seed))
    generation_cfg = execution["trajectory_generation"]
    events, generation = generate_phase3_events(
        cfg,
        event_target=int(generation_cfg["target_event_groups"]),
        seed=int(seed),
    )
    if generation.samples < int(generation_cfg["minimum_windows"]):
        raise RuntimeError(
            f"generated {generation.samples} windows, below frozen minimum "
            f"{generation_cfg['minimum_windows']}"
        )

    split_cfg = protocol["split_policy"]
    all_event_ids = np.asarray([event.event_id for event in events], dtype=np.int64)
    splits = split_event_ids(
        all_event_ids,
        seed=int(seed),
        train_fraction=float(split_cfg["train_fraction"]),
        validation_fraction=float(split_cfg["validation_fraction"]),
    )
    train_events = _event_subset(events, splits["train"])
    validation_events = _event_subset(events, splits["validation"])
    test_events = _event_subset(events, splits["test"])
    x_train, y_train, nf_train, train_sample_events = _join_events(train_events)
    x_validation, y_validation, nf_validation, validation_sample_events = _join_events(
        validation_events
    )
    x_test, y_test, _, test_sample_events = _join_events(test_events)

    (
        x_train_n,
        nf_train_n,
        normalized_other,
        token_mean,
        token_scale,
        next_mean,
        next_scale,
    ) = _normalize_training_arrays(x_train, nf_train, x_validation, x_test)
    x_validation_n, x_test_n = normalized_other
    nf_validation_n = ((nf_validation - next_mean) / next_scale).astype(np.float32)

    device = torch.device("cpu")
    model, training = train_model(
        cfg,
        device,
        x_train_n,
        y_train,
        nf_train_n,
        x_validation_n,
        y_validation,
        nf_validation_n,
        {
            "phase3_seed": float(seed),
            "trajectory_groups": float(len(events)),
            "split_by_trajectory": 1.0,
        },
    )
    transformer_temperatures = fit_model_temperatures(
        cfg, model, device, x_validation_n, y_validation
    )
    transformer_validation = _transformer_probabilities(
        model, device, x_validation_n, transformer_temperatures
    )
    transformer_test = _transformer_probabilities(
        model, device, x_test_n, transformer_temperatures
    )
    transformer_digest = _torch_model_digest(model)

    train_handcrafted, train_measurements, train_structural, train_handcrafted_events = (
        _frozen_feature_split(train_events, cfg, execution, seed=int(seed))
    )
    validation_handcrafted, validation_measurements, validation_structural, validation_handcrafted_events = (
        _frozen_feature_split(validation_events, cfg, execution, seed=int(seed))
    )
    test_handcrafted, test_measurements, test_structural, test_handcrafted_events = (
        _frozen_feature_split(test_events, cfg, execution, seed=int(seed))
    )
    for expected, actual, name in (
        (train_sample_events, train_handcrafted_events, "train"),
        (validation_sample_events, validation_handcrafted_events, "validation"),
        (test_sample_events, test_handcrafted_events, "test"),
    ):
        if not np.array_equal(expected, actual):
            raise RuntimeError(f"{name} handcrafted sample order differs from core token order")

    classes = cfg.box_N * cfg.box_N
    learned_cfg = execution["learned_only"]
    learned_validation, learned_test, learned_selection, learned_digest = _fit_ridge_branch(
        train_measurements,
        y_train,
        validation_measurements,
        y_validation,
        test_measurements,
        classes=classes,
        ridge_lambda=float(learned_cfg["ridge_lambda"]),
        temperature_candidates=np.asarray(learned_cfg["temperature_candidates"], dtype=float),
        digest_prefix=f"measurement-only:{seed}",
    )
    structural_cfg = execution["structural_branch"]
    structural_validation, structural_test, structural_selection, structural_digest = _fit_ridge_branch(
        train_structural,
        y_train,
        validation_structural,
        y_validation,
        test_structural,
        classes=classes,
        ridge_lambda=float(structural_cfg["ridge_lambda"]),
        temperature_candidates=np.asarray(structural_cfg["temperature_candidates"], dtype=float),
        digest_prefix=f"structural:{seed}",
    )

    candidates = np.asarray(protocol["fusion_policy"]["candidate_structural_weights"], dtype=float)
    core_validation, core_test, core_physics_weights, core_validation_nll = _select_pool(
        transformer_validation,
        validation_handcrafted["physics_only"],
        y_validation,
        transformer_test,
        test_handcrafted["physics_only"],
        candidates,
    )
    core_plus_structural_validation, core_plus_structural_test, structural_weights, structural_validation_nll = _select_pool(
        core_validation,
        structural_validation,
        y_validation,
        core_test,
        structural_test,
        candidates,
    )

    topology_cfg = execution["topology_branch"]
    topology_strength = float(topology_cfg["cell_prior_strength"])
    topology_floor = float(topology_cfg["cell_prior_floor"])
    core_plus_topology_validation = _apply_topology_cube(
        core_validation,
        validation_handcrafted["topology_prior"],
        strength=topology_strength,
        floor=topology_floor,
    )
    core_plus_topology_test = _apply_topology_cube(
        core_test,
        test_handcrafted["topology_prior"],
        strength=topology_strength,
        floor=topology_floor,
    )
    combined_validation = _apply_topology_cube(
        core_plus_structural_validation,
        validation_handcrafted["topology_prior"],
        strength=topology_strength,
        floor=topology_floor,
    )
    combined_test = _apply_topology_cube(
        core_plus_structural_test,
        test_handcrafted["topology_prior"],
        strength=topology_strength,
        floor=topology_floor,
    )

    validation_variants = {
        "constant_velocity": validation_handcrafted["constant_velocity"],
        "filter_direct": validation_handcrafted["filter_direct"],
        "physics_only": validation_handcrafted["physics_only"],
        "learned_only": learned_validation,
        "core_hti": core_validation,
        "core_plus_structural": core_plus_structural_validation,
        "core_plus_topology": core_plus_topology_validation,
        "hti_08_combined": combined_validation,
    }
    test_variants = {
        "constant_velocity": test_handcrafted["constant_velocity"],
        "filter_direct": test_handcrafted["filter_direct"],
        "physics_only": test_handcrafted["physics_only"],
        "learned_only": learned_test,
        "core_hti": core_test,
        "core_plus_structural": core_plus_structural_test,
        "core_plus_topology": core_plus_topology_test,
        "hti_08_combined": combined_test,
    }
    if set(validation_variants) != {str(value) for value in protocol["variants"]}:
        raise RuntimeError("runner variants do not match the frozen protocol")

    conformal_cfg = execution["conformal"]
    conformal_qhat, conformal_digests = _conformal_parameters(
        validation_variants,
        y_validation,
        validation_sample_events,
        alpha=float(conformal_cfg["alpha"]),
    )

    selection_record: dict[str, object] = {
        "seed": int(seed),
        "selection_data": "validation_only",
        "core_transformer_temperatures": [float(value) for value in transformer_temperatures],
        "core_physics_weights": core_physics_weights,
        "core_validation_nll": core_validation_nll,
        "learned_only": learned_selection,
        "structural_branch": structural_selection,
        "core_structural_weights": structural_weights,
        "core_structural_validation_nll": structural_validation_nll,
        "conformal_alpha": float(conformal_cfg["alpha"]),
        "conformal_qhat": {
            name: [float(value) for value in values] for name, values in conformal_qhat.items()
        },
        "model_sha256": {
            "core_transformer": transformer_digest,
            "learned_only": learned_digest,
            "structural_branch": structural_digest,
        },
        "training": training,
    }
    selection_output.write_text(json.dumps(selection_record, indent=2) + "\n", encoding="utf-8")
    validation_selection_sha256 = sha256_file(selection_output)

    cell_ids = np.asarray(
        [f"front-cell-{row}-{col}" for row in range(cfg.box_N) for col in range(cfg.box_N)]
    )
    cell_partition_sha256 = hashlib.sha256(
        json.dumps(
            [str(value) for value in cell_ids], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    topology_definition_sha256 = sha256_json(
        {
            "kind": topology_cfg["kind"],
            "history_direction_source": topology_cfg["history_direction_source"],
            "path_distance": topology_cfg["path_distance"],
            "path_reweighting": "hti.phase3_models.history_consistency_weights",
            "cell_prior_application": "hti.fusion.apply_cell_prior",
        }
    )
    topology_coefficients_sha256 = sha256_json(topology_cfg)

    joint_valid = np.all(y_test >= 0, axis=1)
    if not np.any(joint_valid):
        raise RuntimeError("test split has no rows valid at every frozen horizon")
    final_labels = y_test[joint_valid]
    final_event_ids = test_sample_events[joint_valid]
    if set(np.unique(final_event_ids).tolist()) != set(np.asarray(splits["test"]).tolist()):
        raise RuntimeError("joint-valid final rows do not represent every frozen test event")

    suppression_delta = float(topology_cfg["true_support_suppression_delta"])
    suppression = np.zeros_like(final_labels, dtype=np.uint8)
    for horizon in range(final_labels.shape[1]):
        labels_h = final_labels[:, horizon]
        core_true = core_test[joint_valid, horizon, :][np.arange(len(labels_h)), labels_h]
        topology_true = core_plus_topology_test[joint_valid, horizon, :][
            np.arange(len(labels_h)), labels_h
        ]
        suppression[:, horizon] = (
            topology_true < core_true - suppression_delta
        ).astype(np.uint8)

    protocol_sha256 = sha256_file(protocol_path)
    execution_sha256 = sha256_file(execution_path)
    bundle: dict[str, np.ndarray] = {
        "labels": final_labels.astype(np.int64),
        "event_ids": final_event_ids.astype(np.int64),
        "seed": np.array([int(seed)], dtype=np.int64),
        "train_event_ids": np.asarray(splits["train"], dtype=np.int64),
        "validation_event_ids": np.asarray(splits["validation"], dtype=np.int64),
        "test_event_ids": np.asarray(splits["test"], dtype=np.int64),
        "orientation_source": np.array([str(structural_cfg["orientation_source"])]),
        "fusion_selection_data": np.array(["validation_only"]),
        "core_physics_weights": np.asarray(core_physics_weights, dtype=float),
        "fusion_structural_weights": np.asarray(structural_weights, dtype=float),
        "validation_selection_sha256": np.array([validation_selection_sha256]),
        "execution_config_sha256": np.array([execution_sha256]),
        "topology_definition_sha256": np.array([topology_definition_sha256]),
        "topology_coefficients_sha256": np.array([topology_coefficients_sha256]),
        "cell_partition_sha256": np.array([cell_partition_sha256]),
        "model_sha256__learned_only": np.array([learned_digest]),
        "model_sha256__core_hti": np.array([transformer_digest]),
        "model_sha256__structural_branch": np.array([structural_digest]),
        "core_transformer_probabilities": transformer_test[joint_valid],
        "structural_branch_probabilities": structural_test[joint_valid],
        "topology_cell_prior": test_handcrafted["topology_prior"][joint_valid],
        "topology_true_path_suppressed": suppression,
        "protocol_sha256": np.array([protocol_sha256]),
        "source_commit": np.array([source_commit()]),
        "generated_event_count": np.array([generation.generated_events], dtype=np.int64),
        "generated_window_count": np.array([generation.samples], dtype=np.int64),
        "token_mean": token_mean,
        "token_scale": token_scale,
    }
    for name in protocol["variants"]:
        variant = str(name)
        bundle[f"probs__{variant}"] = test_variants[variant][joint_valid]
        bundle[f"cell_ids__{variant}"] = cell_ids
        bundle[f"conformal_qhat__{variant}"] = conformal_qhat[variant]
        bundle[f"conformal_calibration_sha256__{variant}"] = np.array(
            [conformal_digests[variant]]
        )
    np.savez_compressed(output, **bundle)
    return {
        "output": str(output),
        "selection_output": str(selection_output),
        "seed": int(seed),
        "source_commit": source_commit(),
        "protocol_sha256": protocol_sha256,
        "execution_config_sha256": execution_sha256,
        "events": int(generation.generated_events),
        "windows": int(generation.samples),
        "test_joint_valid_rows": int(np.count_nonzero(joint_valid)),
        "model_sha256": selection_record["model_sha256"],
    }
