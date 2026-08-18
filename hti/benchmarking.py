"""Event-level benchmark metrics for HTI 0.8 research evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .assurance import conformal_prediction_set, normalized_entropy
from .topological_entropy import credible_cell_set

EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class BenchmarkMetrics:
    samples: int
    classes: int
    class_coverage: float
    accuracy_top1: float
    accuracy_top3: float
    accuracy_top5: float
    nll: float
    brier: float
    ece: float
    credible_coverage: float
    credible_mean_size: float
    credible_median_size: float
    mean_confidence: float
    mean_entropy_concentration: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _validate(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if p.ndim != 2 or p.shape[1] < 2 or len(p) != len(y):
        raise ValueError("probabilities must have shape (samples, classes) and match labels")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    totals = p.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("every probability row must have positive mass")
    p = p / totals
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("labels are out of range")
    return p, y


def assert_disjoint_event_splits(
    train_event_ids: np.ndarray,
    validation_event_ids: np.ndarray,
    test_event_ids: np.ndarray,
) -> None:
    """Fail when any complete event identity appears in more than one split."""

    train = set(np.asarray(train_event_ids).reshape(-1).tolist())
    validation = set(np.asarray(validation_event_ids).reshape(-1).tolist())
    test = set(np.asarray(test_event_ids).reshape(-1).tolist())
    if train & validation or train & test or validation & test:
        raise ValueError("event-level split leakage detected")


def topk_accuracy(probabilities: np.ndarray, labels: np.ndarray, k: int) -> float:
    p, y = _validate(probabilities, labels)
    if k < 1:
        raise ValueError("k must be positive")
    k = min(int(k), p.shape[1])
    top = np.argpartition(-p, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(np.any(top == y[:, None], axis=1)))


def reliability_bins(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 15
) -> list[dict[str, float | int | None]]:
    """Return top-label reliability-bin counts and empirical accuracy."""

    p, y = _validate(probabilities, labels)
    if bins < 2:
        raise ValueError("bins must be at least 2")
    pred = np.argmax(p, axis=1)
    confidence = np.max(p, axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    output: list[dict[str, float | int | None]] = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        upper = confidence <= hi if index == bins - 1 else confidence < hi
        mask = (confidence >= lo) & upper
        count = int(np.count_nonzero(mask))
        output.append(
            {
                "bin": int(index),
                "lower": float(lo),
                "upper": float(hi),
                "count": count,
                "mean_confidence": float(confidence[mask].mean()) if count else None,
                "accuracy": float(correct[mask].mean()) if count else None,
            }
        )
    return output


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 15
) -> float:
    p, y = _validate(probabilities, labels)
    records = reliability_bins(p, y, bins=bins)
    ece = 0.0
    for record in records:
        count = int(record["count"])
        if count:
            ece += (count / len(y)) * abs(
                float(record["accuracy"]) - float(record["mean_confidence"])
            )
    return float(ece)


def credible_region_stats(
    probabilities: np.ndarray, labels: np.ndarray, *, level: float = 0.95
) -> tuple[float, float, float]:
    p, y = _validate(probabilities, labels)
    if not 0.0 < level <= 1.0:
        raise ValueError("level must be in (0, 1]")
    covered = []
    sizes = []
    for row, label in zip(p, y, strict=True):
        selected = credible_cell_set(row, level)
        covered.append(int(label) in selected)
        sizes.append(len(selected))
    return float(np.mean(covered)), float(np.mean(sizes)), float(np.median(sizes))


def conformal_set_stats(
    probabilities: np.ndarray,
    labels: np.ndarray,
    qhat: float,
) -> dict[str, float]:
    """Evaluate a pre-calibrated split-conformal threshold on test predictions.

    ``qhat`` must be fitted outside the final test set. This function only
    evaluates its resulting sets and performs no calibration or tuning.
    """

    p, y = _validate(probabilities, labels)
    if not np.isfinite(qhat) or not 0.0 <= float(qhat) <= 1.0:
        raise ValueError("qhat must be a finite value in [0, 1]")
    covered: list[bool] = []
    sizes: list[int] = []
    for row, label in zip(p, y, strict=True):
        selected = conformal_prediction_set(row, float(qhat))
        covered.append(int(label) in selected)
        sizes.append(len(selected))
    return {
        "qhat": float(qhat),
        "coverage": float(np.mean(covered)),
        "mean_size": float(np.mean(sizes)),
        "median_size": float(np.median(sizes)),
    }


def evaluate_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    ece_bins: int = 15,
    credible_level: float = 0.95,
) -> BenchmarkMetrics:
    p, y = _validate(probabilities, labels)
    one_hot = np.eye(p.shape[1], dtype=float)[y]
    true_probability = p[np.arange(len(y)), y]
    credible_coverage, credible_mean_size, credible_median_size = credible_region_stats(
        p, y, level=credible_level
    )
    entropy = np.asarray(normalized_entropy(p), dtype=float)
    return BenchmarkMetrics(
        samples=int(len(y)),
        classes=int(p.shape[1]),
        class_coverage=float(len(np.unique(y)) / p.shape[1]),
        accuracy_top1=topk_accuracy(p, y, 1),
        accuracy_top3=topk_accuracy(p, y, 3),
        accuracy_top5=topk_accuracy(p, y, 5),
        nll=float(-np.mean(np.log(np.clip(true_probability, EPS, 1.0)))),
        brier=float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        ece=expected_calibration_error(p, y, bins=ece_bins),
        credible_coverage=credible_coverage,
        credible_mean_size=credible_mean_size,
        credible_median_size=credible_median_size,
        mean_confidence=float(np.mean(np.max(p, axis=1))),
        mean_entropy_concentration=float(np.mean(1.0 - entropy)),
    )


def selective_risk_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    coverages: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
) -> list[dict[str, float]]:
    """Report top-1 error after retaining the most concentrated forecasts."""

    p, y = _validate(probabilities, labels)
    if not coverages or any(not 0.0 < value <= 1.0 for value in coverages):
        raise ValueError("coverages must contain values in (0, 1]")
    concentration = 1.0 - np.asarray(normalized_entropy(p), dtype=float)
    order = np.argsort(concentration)[::-1]
    pred = np.argmax(p, axis=1)
    output = []
    for coverage in coverages:
        count = max(1, int(np.ceil(float(coverage) * len(y))))
        selected = order[:count]
        risk = float(np.mean(pred[selected] != y[selected]))
        output.append(
            {
                "coverage": float(count / len(y)),
                "selective_risk": risk,
                "mean_concentration": float(np.mean(concentration[selected])),
            }
        )
    return output


def event_bootstrap_delta(
    candidate_values: np.ndarray,
    baseline_values: np.ndarray,
    event_ids: np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 20260818,
    lower_is_better: bool = True,
) -> dict[str, float]:
    """Paired complete-event bootstrap interval for a per-sample metric contribution.

    Returned ``delta`` is candidate minus baseline. Negative is favorable when
    ``lower_is_better`` is true; positive is favorable otherwise. Complete
    events are sampled with replacement, retaining all windows in each sampled
    event, so the reported estimand remains sample-weighted while accounting for
    within-event dependence.
    """

    candidate = np.asarray(candidate_values, dtype=float).reshape(-1)
    baseline = np.asarray(baseline_values, dtype=float).reshape(-1)
    events = np.asarray(event_ids).reshape(-1)
    if candidate.shape != baseline.shape or candidate.shape != events.shape or candidate.size == 0:
        raise ValueError("candidate, baseline, and event_ids must be same-size non-empty vectors")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("bootstrap metric values must be finite")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")

    unique_events = np.unique(events)
    if unique_events.size < 2:
        raise ValueError("at least two independent events are required")
    indices = {event: np.flatnonzero(events == event) for event in unique_events}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.choice(unique_events, size=len(unique_events), replace=True)
        draw = np.concatenate([indices[event] for event in sampled])
        deltas[iteration] = float(np.mean(candidate[draw] - baseline[draw]))

    observed = float(np.mean(candidate - baseline))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    favorable = observed < 0.0 if lower_is_better else observed > 0.0
    interval_supports = hi < 0.0 if lower_is_better else lo > 0.0
    return {
        "delta": observed,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "favorable_direction": float(1.0 if favorable else 0.0),
        "interval_supports_direction": float(1.0 if interval_supports else 0.0),
    }


def nll_contributions(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    p, y = _validate(probabilities, labels)
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))
