"""Low-capacity Phase 3 model helpers for the frozen HTI 0.8 ablation study.

These utilities support internal synthetic evaluation. They deliberately keep
structural evidence and topology/history reweighting simple, deterministic, and
separable from the high-capacity core model so their contribution can be
measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fusion import multiclass_nll
from .topological_entropy import StructuralMotionFeatures

MODE_ORDER = (
    "straight",
    "coordinated_turn",
    "alternating_zigzag",
    "drift",
    "deforming",
)


def _feature_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError("features must have shape (samples, features)")
    if not np.isfinite(array).all():
        raise ValueError("features must be finite")
    return array


def _labels(values: np.ndarray, *, classes: int) -> np.ndarray:
    labels = np.asarray(values, dtype=int).reshape(-1)
    if classes < 2:
        raise ValueError("classes must be at least two")
    if (labels < 0).any() or (labels >= classes).any():
        raise ValueError("labels are out of range")
    return labels


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(logits, dtype=float) / float(temperature)
    if values.ndim != 2:
        raise ValueError("logits must have shape (samples, classes)")
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


@dataclass(frozen=True)
class RidgeProbabilisticClassifier:
    """Deterministic ridge classifier used only for structural-feature ablation."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    classes: int
    ridge_lambda: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        classes: int,
        ridge_lambda: float,
    ) -> "RidgeProbabilisticClassifier":
        x = _feature_matrix(features)
        y = _labels(labels, classes=classes)
        if len(x) != len(y):
            raise ValueError("features and labels must contain the same number of samples")
        if not np.isfinite(ridge_lambda) or ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be finite and positive")

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        normalized = (x - mean) / scale
        design = np.column_stack([normalized, np.ones(len(normalized), dtype=float)])
        target = np.eye(classes, dtype=float)[y]
        gram = design.T @ design
        regularizer = np.eye(gram.shape[0], dtype=float) * float(ridge_lambda)
        regularizer[-1, -1] = 0.0
        coefficients = np.linalg.solve(gram + regularizer, design.T @ target)
        return cls(
            mean=mean,
            scale=scale,
            coefficients=coefficients,
            classes=int(classes),
            ridge_lambda=float(ridge_lambda),
        )

    def logits(self, features: np.ndarray) -> np.ndarray:
        x = _feature_matrix(features)
        if x.shape[1] != len(self.mean):
            raise ValueError("feature dimension does not match fitted classifier")
        normalized = (x - self.mean) / self.scale
        design = np.column_stack([normalized, np.ones(len(normalized), dtype=float)])
        return design @ self.coefficients

    def probabilities(self, features: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
        return _softmax(self.logits(features), temperature=temperature)


def select_temperature(
    classifier: RidgeProbabilisticClassifier,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    candidates: np.ndarray,
) -> tuple[float, float]:
    """Select structural temperature from validation NLL only."""

    values = np.unique(np.asarray(candidates, dtype=float).reshape(-1))
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("temperature candidates must be finite and positive")
    x = _feature_matrix(validation_features)
    y = _labels(validation_labels, classes=classifier.classes)
    if len(y) != len(x):
        raise ValueError("validation features and labels must match")

    scored: list[tuple[float, float, float]] = []
    for temperature in values:
        probabilities = classifier.probabilities(x, temperature=float(temperature))
        nll = multiclass_nll(probabilities, y)
        scored.append((nll, abs(float(np.log(temperature))), float(temperature)))
    best_nll, _, best_temperature = min(scored)
    return best_temperature, float(best_nll)


def structural_feature_vector(features: StructuralMotionFeatures) -> np.ndarray:
    """Convert source-derived structural descriptors into a fixed vector."""

    modes = features.mode_probabilities
    return np.asarray(
        [
            features.speed,
            features.slip_angle_rad,
            features.turn_rate_rad_s,
            features.curvature_inv_m,
            features.swept_area,
            features.zigzag_score,
            features.zigzag_bias,
            features.body_length_rate,
            *(float(modes.get(name, 0.0)) for name in MODE_ORDER),
        ],
        dtype=float,
    )


def smoothed_cell_distribution(
    cell: int | None,
    *,
    classes: int,
    smoothing: float,
) -> np.ndarray:
    """Return a finite categorical baseline, or uniform mass if unresolved."""

    if classes < 2:
        raise ValueError("classes must be at least two")
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must lie in [0, 1)")
    if cell is None or int(cell) < 0 or int(cell) >= classes:
        return np.full(classes, 1.0 / classes, dtype=float)
    result = np.full(classes, smoothing / classes, dtype=float)
    result[int(cell)] += 1.0 - smoothing
    return result / result.sum()


def history_consistency_weights(
    base_weights: np.ndarray,
    paths: np.ndarray,
    *,
    origin: np.ndarray,
    travel_direction: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Reweight paths by final-displacement agreement with recent travel direction."""

    weights = np.asarray(base_weights, dtype=float).reshape(-1)
    data = np.asarray(paths, dtype=float)
    start = np.asarray(origin, dtype=float).reshape(-1)
    direction = np.asarray(travel_direction, dtype=float).reshape(-1)
    if data.ndim != 3 or data.shape[0] != len(weights) or data.shape[-1] != len(start):
        raise ValueError("paths must have shape (paths, time, dimensions)")
    if len(direction) != len(start):
        raise ValueError("travel direction must match path dimensions")
    if not np.isfinite(data).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("paths and weights must be finite and weights non-negative")
    if weights.sum() <= 0:
        raise ValueError("base weights must have positive mass")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and non-negative")

    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("travel direction must be non-zero")
    direction = direction / direction_norm
    displacement = data[:, -1, :] - start[None, :]
    norms = np.linalg.norm(displacement, axis=1)
    valid = norms > 1e-12
    cosine = np.ones(len(weights), dtype=float)
    cosine[valid] = np.clip(
        (displacement[valid] / norms[valid, None]) @ direction,
        -1.0,
        1.0,
    )
    disagreement = np.arccos(cosine) / np.pi
    factors = np.exp(-float(gamma) * disagreement)
    combined = weights * factors
    if combined.sum() <= 0:
        raise ValueError("history consistency removed all path support")
    return combined / combined.sum()


def path_cell_distribution(
    path_weights: np.ndarray,
    cells: np.ndarray,
    *,
    classes: int,
    smoothing: float,
) -> np.ndarray:
    """Accumulate path support into cells; unresolved paths contribute uniform mass."""

    weights = np.asarray(path_weights, dtype=float).reshape(-1)
    cell_ids = np.asarray(cells, dtype=int).reshape(-1)
    if len(weights) == 0 or len(weights) != len(cell_ids):
        raise ValueError("path weights and cells must be same-size non-empty arrays")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("path weights must be finite, non-negative, and positive-mass")
    if classes < 2:
        raise ValueError("classes must be at least two")
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must lie in [0, 1)")

    weights = weights / weights.sum()
    result = np.zeros(classes, dtype=float)
    unresolved = 0.0
    for weight, cell in zip(weights, cell_ids, strict=True):
        if 0 <= int(cell) < classes:
            result[int(cell)] += float(weight)
        else:
            unresolved += float(weight)
    if unresolved:
        result += unresolved / classes
    if smoothing:
        result = (1.0 - smoothing) * result + smoothing / classes
    if result.sum() <= 0:
        return np.full(classes, 1.0 / classes, dtype=float)
    return result / result.sum()
