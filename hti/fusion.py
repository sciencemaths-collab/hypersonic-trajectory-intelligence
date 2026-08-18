"""Validation-only probability fusion utilities for HTI 0.8 research.

The core predictor, structural/topology branch, and assurance layer remain
separable. Fusion weights must be selected on validation data only and frozen
before final-test evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class FusionSelection:
    structural_weight: float
    validation_nll: float
    candidate_count: int


def _probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must have shape (samples, classes) or (classes,)")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    totals = p.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("every probability row must have positive mass")
    return p / totals


def log_linear_pool(
    core_probabilities: np.ndarray,
    structural_probabilities: np.ndarray,
    *,
    structural_weight: float,
) -> np.ndarray:
    """Fuse two probability sources with a normalized logarithmic opinion pool."""

    if not 0.0 <= structural_weight <= 1.0:
        raise ValueError("structural_weight must be in [0, 1]")
    core = _probability_matrix(core_probabilities)
    structural = _probability_matrix(structural_probabilities)
    if core.shape != structural.shape:
        raise ValueError("core and structural probabilities must have matching shape")

    logp = (1.0 - structural_weight) * np.log(np.clip(core, EPS, 1.0))
    logp += structural_weight * np.log(np.clip(structural, EPS, 1.0))
    logp -= logp.max(axis=1, keepdims=True)
    pooled = np.exp(logp)
    pooled /= pooled.sum(axis=1, keepdims=True)
    return pooled


def apply_cell_prior(
    probabilities: np.ndarray,
    prior: np.ndarray,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply a non-negative cell-level feasibility prior and renormalize."""

    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and non-negative")
    p = _probability_matrix(probabilities)
    r = np.asarray(prior, dtype=float)
    if r.ndim == 1:
        r = r[None, :]
    if r.shape[1:] != p.shape[1:] or r.shape[0] not in (1, p.shape[0]):
        raise ValueError("prior must have shape (classes,) or match probabilities")
    if not np.isfinite(r).all() or (r < 0).any():
        raise ValueError("prior must be finite and non-negative")
    if r.shape[0] == 1:
        r = np.repeat(r, p.shape[0], axis=0)
    adjusted = p * np.power(r, strength)
    totals = adjusted.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("cell prior removed all probability mass")
    return adjusted / totals


def multiclass_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = _probability_matrix(probabilities)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if len(y) != len(p):
        raise ValueError("labels must match probability rows")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("labels are out of range")
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))))


def select_structural_weight(
    core_validation_probabilities: np.ndarray,
    structural_validation_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    *,
    candidates: np.ndarray | None = None,
) -> FusionSelection:
    """Select a fusion weight using validation NLL only.

    Ties favor the smaller structural weight so an added branch must earn its
    contribution rather than receiving it by default.
    """

    grid = np.linspace(0.0, 1.0, 21) if candidates is None else np.asarray(candidates, dtype=float)
    grid = np.unique(grid.reshape(-1))
    if grid.size == 0 or not np.isfinite(grid).all() or (grid < 0).any() or (grid > 1).any():
        raise ValueError("candidate weights must be finite values in [0, 1]")

    scored: list[tuple[float, float]] = []
    for weight in grid:
        pooled = log_linear_pool(
            core_validation_probabilities,
            structural_validation_probabilities,
            structural_weight=float(weight),
        )
        scored.append((multiclass_nll(pooled, validation_labels), float(weight)))
    best_nll, best_weight = min(scored, key=lambda item: (item[0], item[1]))
    return FusionSelection(
        structural_weight=best_weight,
        validation_nll=best_nll,
        candidate_count=int(grid.size),
    )
