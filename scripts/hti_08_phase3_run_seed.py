#!/usr/bin/env python3
"""Execute one frozen HTI 0.8 Phase 3 synthetic ablation seed.

Training and parameter selection use only complete-event train and validation
splits. The resulting immutable final-test probability bundle is evaluated by a
separate script. Final-test metrics never select model, fusion, topology,
temperature, or conformal parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alien_exit_cell_predictor_v6_3 import (  # noqa: E402
    AeroModel,
    Atmosphere,
    Config,
    EarthModel,
    Projectile6DOF,
    VirtualBox,
    compute_box_L,
    compute_box_cross_width,
    fit_model_temperatures,
    nominal_filter_env,
    project_sigma_horizons,
    q_to_R,
    seed_everything,
    train_model,
)
from hti.assurance import conformal_quantile  # noqa: E402
from hti.fusion import (  # noqa: E402
    apply_cell_prior,
    log_linear_pool,
    select_structural_weight,
)
from hti.phase3 import Phase3Event, generate_phase3_events, split_event_ids  # noqa: E402
from hti.phase3_models import (  # noqa: E402
    RidgeProbabilisticClassifier,
    history_consistency_weights,
    path_cell_distribution,
    select_temperature,
    smoothed_cell_distribution,
    structural_feature_vector,
)
from hti.topological_entropy import (  # noqa: E402
    StructuralMotionFeatures,
    endpoint_observations_from_centerline,
    estimate_structural_motion,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_arrays(*arrays: np.ndarray, prefix: str = "") -> str:
    digest = hashlib.sha256(prefix.encode("utf-8"))
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _source_commit() -> str:
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


def _configure(
    protocol: dict[str, object], execution: dict[str, object], seed: int
) -> Config:
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
    result = []
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


def _learned_probabilities(
    model,
    device: torch.device,
    normalized_tokens: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(normalized_tokens).float().to(device)
        logits, _ = model(tensor)
        temp = torch.from_numpy(np.asarray(temperatures, dtype=np.float32)).to(device).view(1, -1, 1)
        return F.softmax(logits / temp, dim=-1).cpu().numpy()


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


def _structural_motion_at(
    event: Phase3Event,
    frame: int,
    *,
    dt: float,
    history_points: int,
    body_length: float,
) -> StructuralMotionFeatures:
    start = max(0, int(frame) - int(history_points) + 1)
    indices = np.arange(start, int(frame) + 1, dtype=int)
    if len(indices) < 3:
        raise ValueError("structural history contains fewer than three frames")
    states = event.estimated_states[indices]
    centers = states[:, :3]
    directions = np.asarray([q_to_R(state[6:10])[:, 0] for state in states])
    times = indices.astype(float) * float(dt)
    observations = endpoint_observations_from_centerline(times, centers, directions, body_length)
    return estimate_structural_motion(observations)


def _structural_matrix(
    events: list[Phase3Event], cfg: Config, execution: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    structural_cfg = execution["structural_branch"]
    history_points = int(structural_cfg["history_points"])
    body_length = float(structural_cfg["body_length_m"])
    rows: list[np.ndarray] = []
    event_rows: list[int] = []
    for event in events:
        for frame in event.source_frames:
            motion = _structural_motion_at(
                event,
                int(frame),
                dt=cfg.dt,
                history_points=history_points,
                body_length=body_length,
            )
            rows.append(structural_feature_vector(motion))
            event_rows.append(int(event.event_id))
    return np.stack(rows), np.asarray(event_rows, dtype=np.int64)


def _handcrafted_split(
    events: list[Phase3Event],
    cfg: Config,
    execution: dict[str, object],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    classes = cfg.box_N * cfg.box_N
    horizons = list(cfg.horizon_steps)
    max_horizon = max(horizons)
    physics_cfg = execution["physics_branch"]
    structural_cfg = execution["structural_branch"]
    topology_cfg = execution["topology_branch"]
    smoothing = float(physics_cfg["one_hot_smoothing"])
    gamma = float(topology_cfg["history_angle_gamma"])
    history_points = int(structural_cfg["history_points"])
    body_length = float(structural_cfg["body_length_m"])

    earth, atmosphere, aero = EarthModel(), Atmosphere(), AeroModel()
    projectile = Projectile6DOF(
        1000.0,
        np.diag([500.0, 800.0, 1000.0]),
        earth,
        atmosphere,
        aero,
    )
    projection_env = nominal_filter_env()

    constant_rows: list[np.ndarray] = []
    filter_rows: list[np.ndarray] = []
    physics_rows: list[np.ndarray] = []
    topology_rows: list[np.ndarray] = []
    structural_rows: list[np.ndarray] = []
    event_id_rows: list[int] = []

    for event in events:
        for frame in event.source_frames:
            frame_index = int(frame)
            state = event.estimated_states[frame_index]
            box = _reference_box(cfg, state)
            measurement = event.measurements[frame_index]
            u_current = (
                event.controls[frame_index].copy()
                if frame_index < len(event.controls)
                else np.zeros(6, dtype=float)
            )
            sigma_paths, sigma_weights = project_sigma_horizons(
                projectile,
                state,
                event.covariances[frame_index],
                u_current,
                cfg.dt,
                projection_env,
                max_horizon,
            )
            motion = _structural_motion_at(
                event,
                frame_index,
                dt=cfg.dt,
                history_points=history_points,
                body_length=body_length,
            )
            structural_rows.append(structural_feature_vector(motion))
            event_id_rows.append(int(event.event_id))

            constant_h = []
            filter_h = []
            physics_h = []
            topology_h = []
            for horizon in horizons:
                seconds = float(horizon * cfg.dt)
                constant_point = measurement[:3] + measurement[3:6] * seconds
                filter_point = state[:3] + state[3:6] * seconds
                constant_h.append(
                    smoothed_cell_distribution(
                        _cell_index(box, constant_point, cfg.box_N),
                        classes=classes,
                        smoothing=smoothing,
                    )
                )
                filter_h.append(
                    smoothed_cell_distribution(
                        _cell_index(box, filter_point, cfg.box_N),
                        classes=classes,
                        smoothing=smoothing,
                    )
                )
                cells = np.asarray(
                    [
                        -1
                        if (cell := _cell_index(box, path[horizon, :3], cfg.box_N)) is None
                        else cell
                        for path in sigma_paths
                    ],
                    dtype=int,
                )
                physics_h.append(
                    path_cell_distribution(
                        sigma_weights,
                        cells,
                        classes=classes,
                        smoothing=smoothing,
                    )
                )
                topology_weights = history_consistency_weights(
                    sigma_weights,
                    sigma_paths[:, : horizon + 1, :3],
                    origin=state[:3],
                    travel_direction=motion.travel_direction,
                    gamma=gamma,
                )
                topology_h.append(
                    path_cell_distribution(
                        topology_weights,
                        cells,
                        classes=classes,
                        smoothing=smoothing,
                    )
                )
            constant_rows.append(np.stack(constant_h))
            filter_rows.append(np.stack(filter_h))
            physics_rows.append(np.stack(physics_h))
            topology_rows.append(np.stack(topology_h))

    return (
        {
            "constant_velocity": np.stack(constant_rows),
            "filter_direct": np.stack(filter_rows),
            "physics_only": np.stack(physics_rows),
            "topology_prior": np.stack(topology_rows),
        },
        np.stack(structural_rows),
        np.asarray(event_id_rows, dtype=np.int64),
    )


def _classifier_sha256(classifier: RidgeProbabilisticClassifier) -> str:
    return _sha256_arrays(
        classifier.mean,
        classifier.scale,
        classifier.coefficients,
        prefix=f"ridge:{classifier.classes}:{classifier.ridge_lambda}",
    )


def _fit_structural_branch(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    classes: int,
    ridge_lambda: float,
    temperature_candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], str]:
    horizons = train_labels.shape[1]
    validation_probabilities = np.empty((len(validation_features), horizons, classes), dtype=float)
    test_probabilities = np.empty((len(test_features), horizons, classes), dtype=float)
    selections: list[dict[str, object]] = []
    model_digests = []
    for horizon in range(horizons):
        train_mask = train_labels[:, horizon] >= 0
        validation_mask = validation_labels[:, horizon] >= 0
        classifier = RidgeProbabilisticClassifier.fit(
            train_features[train_mask],
            train_labels[train_mask, horizon],
            classes=classes,
            ridge_lambda=ridge_lambda,
        )
        model_digest = _classifier_sha256(classifier)
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
        model_digests.append(model_digest)
        selections.append(
            {
                "horizon_index": int(horizon),
                "temperature": float(temperature),
                "validation_nll": float(validation_nll),
                "classifier_sha256": model_digest,
            }
        )
    return (
        validation_probabilities,
        test_probabilities,
        selections,
        _sha256_json(model_digests),
    )


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
    weights = []
    nlls = []
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
    output = np.empty_like(probabilities)
    for horizon in range(probabilities.shape[1]):
        output[:, horizon, :] = apply_cell_prior(
            probabilities[:, horizon, :],
            np.maximum(topology_prior[:, horizon, :], float(floor)),
            strength=float(strength),
        )
    return output


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
        values = []
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
        digests[name] = _sha256_arrays(
            cube,
            validation_labels,
            validation_event_ids,
            qhat,
            prefix=f"conformal:{name}:alpha={alpha}",
        )
    return qhats, digests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/hti_08_ablation_frozen.json")
    )
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=Path("configs/hti_08_phase3_execution_frozen.json"),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selection-out", type=Path)
    parser.add_argument("--model-out", type=Path)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    execution = json.loads(args.execution_config.read_text(encoding="utf-8"))
    seed = int(args.seed)
    frozen_seeds = [int(value) for value in protocol["seeds"]]
    if seed not in frozen_seeds:
        raise SystemExit(f"seed must be one of the frozen Phase 3 seeds {frozen_seeds}")

    output = args.out or Path(f"hti08_ablation_{seed}.npz")
    selection_output = args.selection_out or Path(f"hti08_selection_{seed}.json")
    model_output = args.model_out or Path(f"hti08_learned_model_{seed}.pt")
    cfg = _configure(protocol, execution, seed)
    seed_everything(seed)

    generation_cfg = execution["trajectory_generation"]
    events, generation = generate_phase3_events(
        cfg,
        event_target=int(generation_cfg["target_event_groups"]),
        seed=seed,
    )
    split_cfg = protocol["split_policy"]
    all_event_ids = np.asarray([event.event_id for event in events], dtype=np.int64)
    splits = split_event_ids(
        all_event_ids,
        seed=seed,
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
    train_stats = {
        "phase3_seed": float(seed),
        "trajectory_groups": float(len(events)),
        "split_by_trajectory": 1.0,
    }
    model, training = train_model(
        cfg,
        device,
        x_train_n,
        y_train,
        nf_train_n,
        x_validation_n,
        y_validation,
        nf_validation_n,
        train_stats,
    )
    learned_temperatures = fit_model_temperatures(
        cfg, model, device, x_validation_n, y_validation
    )
    learned_validation = _learned_probabilities(
        model, device, x_validation_n, learned_temperatures
    )
    learned_test = _learned_probabilities(model, device, x_test_n, learned_temperatures)
    torch.save(model.state_dict(), model_output)
    learned_model_sha256 = _sha256_file(model_output)

    train_structural, train_structural_events = _structural_matrix(train_events, cfg, execution)
    validation_handcrafted, validation_structural, validation_handcrafted_events = _handcrafted_split(
        validation_events, cfg, execution
    )
    test_handcrafted, test_structural, test_handcrafted_events = _handcrafted_split(
        test_events, cfg, execution
    )
    if not np.array_equal(train_structural_events, train_sample_events):
        raise RuntimeError("train structural sample order differs from Transformer order")
    if not np.array_equal(validation_handcrafted_events, validation_sample_events):
        raise RuntimeError("validation handcrafted sample order differs from Transformer order")
    if not np.array_equal(test_handcrafted_events, test_sample_events):
        raise RuntimeError("test handcrafted sample order differs from Transformer order")

    structural_cfg = execution["structural_branch"]
    (
        structural_validation,
        structural_test,
        structural_selection,
        structural_model_sha256,
    ) = _fit_structural_branch(
        train_structural,
        y_train,
        validation_structural,
        y_validation,
        test_structural,
        classes=cfg.box_N * cfg.box_N,
        ridge_lambda=float(structural_cfg["ridge_lambda"]),
        temperature_candidates=np.asarray(structural_cfg["temperature_candidates"], dtype=float),
    )
    candidates = np.asarray(protocol["fusion_policy"]["candidate_structural_weights"], dtype=float)

    core_validation, core_test, core_physics_weights, core_validation_nll = _select_pool(
        learned_validation,
        validation_handcrafted["physics_only"],
        y_validation,
        learned_test,
        test_handcrafted["physics_only"],
        candidates,
    )
    (
        structural_core_validation,
        structural_core_test,
        structural_weights,
        structural_validation_nll,
    ) = _select_pool(
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
    topology_core_validation = _apply_topology_cube(
        core_validation,
        validation_handcrafted["topology_prior"],
        strength=topology_strength,
        floor=topology_floor,
    )
    topology_core_test = _apply_topology_cube(
        core_test,
        test_handcrafted["topology_prior"],
        strength=topology_strength,
        floor=topology_floor,
    )
    combined_validation = _apply_topology_cube(
        structural_core_validation,
        validation_handcrafted["topology_prior"],
        strength=topology_strength,
        floor=topology_floor,
    )
    combined_test = _apply_topology_cube(
        structural_core_test,
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
        "core_plus_structural": structural_core_validation,
        "core_plus_topology": topology_core_validation,
        "hti_08_combined": combined_validation,
    }
    test_variants = {
        "constant_velocity": test_handcrafted["constant_velocity"],
        "filter_direct": test_handcrafted["filter_direct"],
        "physics_only": test_handcrafted["physics_only"],
        "learned_only": learned_test,
        "core_hti": core_test,
        "core_plus_structural": structural_core_test,
        "core_plus_topology": topology_core_test,
        "hti_08_combined": combined_test,
    }

    conformal_cfg = execution["conformal"]
    conformal_qhat, conformal_digests = _conformal_parameters(
        validation_variants,
        y_validation,
        validation_sample_events,
        alpha=float(conformal_cfg["alpha"]),
    )
    core_model_sha256 = _sha256_json(
        {
            "learned_model_sha256": learned_model_sha256,
            "physics_definition": "positive-weight UKF sigma paths; current control then zero-mean control",
            "core_physics_weights": core_physics_weights,
        }
    )

    selection_record = {
        "seed": seed,
        "selection_data": "validation_only",
        "core_physics_weights": core_physics_weights,
        "core_validation_nll": core_validation_nll,
        "structural_branch": structural_selection,
        "core_structural_weights": structural_weights,
        "core_structural_validation_nll": structural_validation_nll,
        "learned_temperatures": [float(value) for value in learned_temperatures],
        "learned_model_sha256": learned_model_sha256,
        "core_model_sha256": core_model_sha256,
        "structural_model_sha256": structural_model_sha256,
        "conformal_alpha": float(conformal_cfg["alpha"]),
        "conformal_qhat": {
            name: [float(value) for value in values] for name, values in conformal_qhat.items()
        },
        "training": training,
    }
    selection_output.write_text(json.dumps(selection_record, indent=2) + "\n", encoding="utf-8")
    validation_selection_sha256 = _sha256_file(selection_output)

    cell_ids = np.asarray(
        [f"front-cell-{row}-{col}" for row in range(cfg.box_N) for col in range(cfg.box_N)]
    )
    cell_partition_sha256 = hashlib.sha256(
        json.dumps(
            [str(value) for value in cell_ids], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    topology_definition_sha256 = _sha256_json(
        {
            "kind": topology_cfg["kind"],
            "path_reweighting": "hti.phase3_models.history_consistency_weights",
            "cell_prior_application": "hti.fusion.apply_cell_prior",
        }
    )
    topology_coefficients_sha256 = _sha256_json(topology_cfg)

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
        topology_true = topology_core_test[joint_valid, horizon, :][
            np.arange(len(labels_h)), labels_h
        ]
        suppression[:, horizon] = (
            topology_true < core_true - suppression_delta
        ).astype(np.uint8)

    bundle: dict[str, np.ndarray] = {
        "labels": final_labels.astype(np.int64),
        "event_ids": final_event_ids.astype(np.int64),
        "seed": np.array([seed], dtype=np.int64),
        "train_event_ids": np.asarray(splits["train"], dtype=np.int64),
        "validation_event_ids": np.asarray(splits["validation"], dtype=np.int64),
        "test_event_ids": np.asarray(splits["test"], dtype=np.int64),
        "orientation_source": np.array(["ukf_attitude_state_proxy"]),
        "fusion_selection_data": np.array(["validation_only"]),
        "fusion_structural_weights": np.asarray(structural_weights, dtype=float),
        "core_physics_weights": np.asarray(core_physics_weights, dtype=float),
        "validation_selection_sha256": np.array([validation_selection_sha256]),
        "execution_config_sha256": np.array([_sha256_file(args.execution_config)]),
        "topology_definition_sha256": np.array([topology_definition_sha256]),
        "topology_coefficients_sha256": np.array([topology_coefficients_sha256]),
        "cell_partition_sha256": np.array([cell_partition_sha256]),
        "model_sha256__learned_only": np.array([learned_model_sha256]),
        "model_sha256__core_hti": np.array([core_model_sha256]),
        "model_sha256__structural_branch": np.array([structural_model_sha256]),
        "structural_branch_probabilities": structural_test[joint_valid],
        "topology_cell_prior": test_handcrafted["topology_prior"][joint_valid],
        "topology_true_path_suppressed": suppression,
        "protocol_sha256": np.array([_sha256_file(args.protocol)]),
        "source_commit": np.array([_source_commit()]),
        "generated_event_count": np.array([generation.generated_events], dtype=np.int64),
        "generated_window_count": np.array([generation.samples], dtype=np.int64),
        "token_mean": token_mean,
        "token_scale": token_scale,
    }
    for name in protocol["variants"]:
        probabilities = test_variants[str(name)][joint_valid]
        bundle[f"probs__{name}"] = probabilities
        bundle[f"cell_ids__{name}"] = cell_ids
        bundle[f"conformal_qhat__{name}"] = conformal_qhat[str(name)]
        bundle[f"conformal_calibration_sha256__{name}"] = np.array(
            [conformal_digests[str(name)]]
        )

    np.savez_compressed(output, **bundle)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
