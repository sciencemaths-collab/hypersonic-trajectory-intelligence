"""Assurance utilities for Hypersonic Trajectory Intelligence research outputs."""

from .assurance import (
    AssuranceDecision,
    assess_prediction,
    conformal_prediction_set,
    conformal_quantile,
    confidence_margin,
    empirical_set_coverage,
    normalized_entropy,
)

__all__ = [
    "AssuranceDecision",
    "assess_prediction",
    "conformal_prediction_set",
    "conformal_quantile",
    "confidence_margin",
    "empirical_set_coverage",
    "normalized_entropy",
]
