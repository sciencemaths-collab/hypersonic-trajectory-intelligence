"""Independent uncertainty and abstention utilities.

This module does not improve trajectory targeting performance. It sits after a
probabilistic classifier and answers a narrower assurance question: is the
reported class distribution sufficiently concentrated to use as a research
result, or should the system abstain and surface uncertainty instead?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AssuranceDecision:
    """Decision returned by :func:`assess_prediction`."""

    accept: bool
    prediction: int
    confidence: float
    normalized_entropy: float
    margin: float
    reason: str


def _probability_matrix(probabilities: np.ndarray) -> tuple[np.ndarray, bool]:
    raw = np.asarray(probabilities, dtype=float)
    was_vector = raw.ndim == 1
    p = raw[None, :] if was_vector else raw.copy()
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must have shape (classes,) or (samples, classes)")
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite")
    if (p < 0).any():
        raise ValueError("probabilities cannot be negative")
    totals = p.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("each probability row must have positive mass")
    return p / totals, was_vector


def normalized_entropy(probabilities: np.ndarray) -> float | np.ndarray:
    """Return Shannon entropy normalized to ``[0, 1]`` by class count."""

    p, was_vector = _probability_matrix(probabilities)
    eps = np.finfo(float).tiny
    entropy = -(p * np.log(np.clip(p, eps, 1.0))).sum(axis=1)
    entropy /= np.log(float(p.shape[1]))
    if was_vector:
        return float(entropy[0])
    return entropy


def confidence_margin(probabilities: np.ndarray) -> float | np.ndarray:
    """Return top-1 minus top-2 probability mass."""

    p, was_vector = _probability_matrix(probabilities)
    ordered = np.sort(p, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    if was_vector:
        return float(margin[0])
    return margin


def assess_prediction(
    probabilities: np.ndarray,
    *,
    min_confidence: float = 0.35,
    max_normalized_entropy: float = 0.85,
    min_margin: float = 0.02,
) -> AssuranceDecision:
    """Apply transparent research-use abstention criteria to one distribution.

    Thresholds are deliberately configuration inputs rather than learned from
    test data. Projects should freeze them before final evaluation.
    """

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if not 0.0 <= max_normalized_entropy <= 1.0:
        raise ValueError("max_normalized_entropy must be in [0, 1]")
    if not 0.0 <= min_margin <= 1.0:
        raise ValueError("min_margin must be in [0, 1]")

    p, was_vector = _probability_matrix(probabilities)
    if not was_vector:
        if p.shape[0] != 1:
            raise ValueError("assess_prediction accepts one probability vector")
        p = p[:1]

    vector = p[0]
    prediction = int(np.argmax(vector))
    confidence = float(vector[prediction])
    entropy = float(normalized_entropy(vector))
    margin = float(confidence_margin(vector))

    failures: list[str] = []
    if confidence < min_confidence:
        failures.append("low confidence")
    if entropy > max_normalized_entropy:
        failures.append("high predictive entropy")
    if margin < min_margin:
        failures.append("small top-class margin")

    return AssuranceDecision(
        accept=not failures,
        prediction=prediction,
        confidence=confidence,
        normalized_entropy=entropy,
        margin=margin,
        reason="accepted" if not failures else "; ".join(failures),
    )


def conformal_quantile(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    *,
    alpha: float = 0.10,
) -> float:
    """Fit a split-conformal threshold using the ``1 - p_true`` score.

    The calibration set must be frozen independently of the final test set.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    p, _ = _probability_matrix(calibration_probabilities)
    y = np.asarray(calibration_labels, dtype=int).reshape(-1)
    if len(y) != len(p):
        raise ValueError("calibration labels must match probability rows")
    if len(y) < 2:
        raise ValueError("at least two calibration examples are required")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("calibration labels are out of range")

    scores = 1.0 - p[np.arange(len(y)), y]
    rank = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(scores))
    return float(np.sort(scores)[rank - 1])


def conformal_prediction_set(probabilities: np.ndarray, qhat: float) -> list[int]:
    """Return a non-empty split-conformal class set for one prediction."""

    if not 0.0 <= qhat <= 1.0:
        raise ValueError("qhat must be in [0, 1]")
    p, was_vector = _probability_matrix(probabilities)
    if not was_vector and p.shape[0] != 1:
        raise ValueError("conformal_prediction_set accepts one probability vector")
    vector = p[0]
    threshold = 1.0 - qhat
    selected = np.flatnonzero(vector >= threshold).astype(int).tolist()
    if not selected:
        selected = [int(np.argmax(vector))]
    return selected


def empirical_set_coverage(
    probabilities: np.ndarray,
    labels: np.ndarray,
    qhat: float,
) -> float:
    """Measure empirical coverage of conformal prediction sets."""

    p, _ = _probability_matrix(probabilities)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if len(y) != len(p):
        raise ValueError("labels must match probability rows")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("labels are out of range")
    threshold = 1.0 - float(qhat)
    covered = p[np.arange(len(y)), y] >= threshold
    return float(np.mean(covered))
