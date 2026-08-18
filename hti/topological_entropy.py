"""Structural-motion, topology, entropy, and backward inference for HTI 0.8.

This layer is intentionally independent of the core 6DOF predictor so it can
be tested, ablated, or disabled without changing the physics estimator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .assurance import normalized_entropy

EPS = 1e-12


@dataclass(frozen=True)
class EndpointObservation:
    time: float
    nose: tuple[float, ...]
    rear: tuple[float, ...]


@dataclass(frozen=True)
class StructuralMotionFeatures:
    center: np.ndarray
    body_direction: np.ndarray
    body_length: float
    velocity: np.ndarray
    travel_direction: np.ndarray
    speed: float
    slip_angle_rad: float
    turn_rate_rad_s: float
    curvature_inv_m: float
    swept_area: float
    zigzag_score: float
    zigzag_bias: float
    body_length_rate: float
    mode_probabilities: dict[str, float]


@dataclass(frozen=True)
class DistributionSummary:
    probabilities: np.ndarray
    predicted_cell: int
    predicted_probability: float
    normalized_entropy: float
    entropy_concentration: float
    credible_cells: tuple[int, ...]
    credible_mass: float


@dataclass(frozen=True)
class BackwardExplanation:
    terminal_cell: int
    conditional_path_weights: np.ndarray
    mode_support: dict[str, float] | None
    expected_states: np.ndarray | None


@dataclass(frozen=True)
class SphericalKeepOutZone:
    """Non-sensitive 3D safety/reachability exclusion volume."""

    center: tuple[float, float, float]
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return value / norm


def _points(
    observations: Sequence[EndpointObservation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(observations) < 3:
        raise ValueError("at least three time-ordered endpoint observations are required")
    times = np.asarray([item.time for item in observations], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("observation times must be finite and strictly increasing")
    noses = np.asarray([item.nose for item in observations], dtype=float)
    rears = np.asarray([item.rear for item in observations], dtype=float)
    if noses.shape != rears.shape or noses.ndim != 2 or noses.shape[1] not in (2, 3):
        raise ValueError("nose and rear coordinates must have matching 2D or 3D shape")
    if not np.isfinite(noses).all() or not np.isfinite(rears).all():
        raise ValueError("endpoint coordinates must be finite")
    return times, noses, rears


def endpoint_geometry(
    observations: Sequence[EndpointObservation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centers, full body-axis vectors, and body lengths."""

    _, noses, rears = _points(observations)
    centers = (noses + rears) / 2.0
    axes = noses - rears
    lengths = np.linalg.norm(axes, axis=1)
    if np.any(lengths <= EPS):
        raise ValueError("nose and rear must be distinct at every observation")
    return centers, axes, lengths


def recover_endpoints(center: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert the center/axis transform."""

    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if center.shape != axis.shape or center.ndim != 1:
        raise ValueError("center and axis must be same-shape vectors")
    return center + 0.5 * axis, center - 0.5 * axis


def endpoint_observations_from_centerline(
    times: np.ndarray,
    centers: np.ndarray,
    body_directions: np.ndarray,
    body_lengths: np.ndarray | float,
) -> list[EndpointObservation]:
    """Bridge an existing state trace into the two-point representation.

    The caller remains responsible for the physical meaning of body_directions.
    A velocity-derived direction is a proxy, not an independently observed body
    orientation, and must be reported as such.
    """

    t = np.asarray(times, dtype=float).reshape(-1)
    c = np.asarray(centers, dtype=float)
    d = np.asarray(body_directions, dtype=float)
    if c.ndim != 2 or c.shape[1] not in (2, 3) or d.shape != c.shape or len(t) != len(c):
        raise ValueError("times, centers, and body_directions must describe one 2D/3D trace")
    if len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("times must contain at least three finite increasing values")
    if not np.isfinite(c).all() or not np.isfinite(d).all():
        raise ValueError("centerline inputs must be finite")
    if np.isscalar(body_lengths):
        lengths = np.full(len(t), float(body_lengths), dtype=float)
    else:
        lengths = np.asarray(body_lengths, dtype=float).reshape(-1)
    if len(lengths) != len(t) or not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("body_lengths must be positive and match the trace length")

    result: list[EndpointObservation] = []
    for time, center, direction, length in zip(t, c, d, lengths, strict=True):
        nose, rear = recover_endpoints(center, _unit(direction) * float(length))
        result.append(EndpointObservation(float(time), tuple(nose), tuple(rear)))
    return result


def circular_smooth(angles: np.ndarray, smoothing: float = 0.40) -> float:
    """Exponentially smooth wrapped angles on the unit circle."""

    values = np.asarray(angles, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("angles must be a non-empty finite sequence")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must be in (0, 1]")
    phasor = np.exp(1j * values[0])
    for angle in values[1:]:
        phasor = (1.0 - smoothing) * phasor + smoothing * np.exp(1j * angle)
        phasor = np.exp(1j * angle) if abs(phasor) <= EPS else phasor / abs(phasor)
    return float(np.angle(phasor))


def _signed_turn_rates(velocities: np.ndarray, dt: np.ndarray) -> np.ndarray:
    if len(velocities) < 2:
        return np.empty(0, dtype=float)
    directions = np.asarray([_unit(value) for value in velocities])
    if velocities.shape[1] == 2:
        headings = np.unwrap(np.arctan2(directions[:, 1], directions[:, 0]))
        return np.diff(headings) / dt[1:]

    cross = np.cross(directions[:-1], directions[1:])
    cross_norm = np.linalg.norm(cross, axis=1)
    dots = np.sum(directions[:-1] * directions[1:], axis=1)
    angles = np.arctan2(cross_norm, np.clip(dots, -1.0, 1.0))
    dominant = cross.sum(axis=0)
    if np.linalg.norm(dominant) <= EPS:
        nonzero = np.flatnonzero(cross_norm > EPS)
        dominant = cross[nonzero[0]] if nonzero.size else np.array([0.0, 0.0, 1.0])
    dominant = _unit(dominant)
    signs = np.sign(cross @ dominant)
    signs[signs == 0] = 1.0
    return signs * angles / dt[1:]


def _swept_area(centers: np.ndarray, turn_rates: np.ndarray) -> float:
    relative = centers - centers[0]
    if centers.shape[1] == 2:
        x, y = relative[:, 0], relative[:, 1]
        return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    area_vector = 0.5 * np.sum(np.cross(relative[:-1], relative[1:]), axis=0)
    magnitude = float(np.linalg.norm(area_vector))
    if magnitude <= EPS:
        return 0.0
    sign = np.sign(float(np.sum(turn_rates))) if turn_rates.size else 1.0
    return float((sign if sign != 0 else 1.0) * magnitude)


def infer_motion_modes(
    *,
    speed: float,
    turn_rate: float,
    slip_angle: float,
    zigzag_score: float,
    body_length: float,
    body_length_rate: float,
) -> dict[str, float]:
    """Transparent explanatory mode evidence, not operational intent probability."""

    turn_scale = np.deg2rad(8.0)
    turn_strength = min(abs(turn_rate) / max(turn_scale, EPS), 4.0)
    zig_strength = min(zigzag_score / max(turn_scale, EPS), 4.0)
    slip_strength = min(slip_angle / np.deg2rad(25.0), 4.0)
    deform_strength = min(abs(body_length_rate) / max(body_length, EPS) / 0.05, 4.0)
    raw = {
        "straight": np.exp(-turn_strength - zig_strength - 0.5 * slip_strength),
        "coordinated_turn": (1.0 - np.exp(-turn_strength)) * np.exp(-zig_strength),
        "alternating_zigzag": 1.0 - np.exp(-zig_strength),
        "drift": 1.0 - np.exp(-slip_strength),
        "deforming": 1.0 - np.exp(-deform_strength),
    }
    if speed <= EPS:
        raw["straight"] += 0.25
    total = float(sum(raw.values()))
    return {name: float(value / total) for name, value in raw.items()}


def estimate_structural_motion(
    observations: Sequence[EndpointObservation], *, smoothing: float = 0.40
) -> StructuralMotionFeatures:
    """Estimate slip, turn structure, swept area, deformation, and mode evidence."""

    times, _, _ = _points(observations)
    centers, axes, lengths = endpoint_geometry(observations)
    dt = np.diff(times)
    velocities = np.diff(centers, axis=0) / dt[:, None]
    speeds = np.linalg.norm(velocities, axis=1)
    if np.any(speeds <= EPS):
        raise ValueError("consecutive centers must move for structural motion estimation")
    directions = np.asarray([_unit(value) for value in velocities])

    if centers.shape[1] == 2:
        smoothed = circular_smooth(np.arctan2(directions[:, 1], directions[:, 0]), smoothing)
        travel_direction = np.array([np.cos(smoothed), np.sin(smoothed)])
    else:
        travel_direction = directions[0].copy()
        for direction in directions[1:]:
            travel_direction = _unit((1.0 - smoothing) * travel_direction + smoothing * direction)

    body_direction = _unit(axes[-1])
    slip = float(np.arccos(np.clip(body_direction @ travel_direction, -1.0, 1.0)))
    turn_rates = _signed_turn_rates(velocities, dt)
    turn_rate = float(np.median(turn_rates)) if turn_rates.size else 0.0
    speed = float(np.mean(speeds[-min(4, len(speeds)) :]))
    if turn_rates.size > 1:
        flips = np.sign(turn_rates[1:]) != np.sign(turn_rates[:-1])
        zigzag_score = float(np.mean(np.where(flips, np.abs(turn_rates[1:]), 0.0)))
        zigzag_bias = float(np.mean(flips) * turn_rates[-1])
    else:
        zigzag_score = 0.0
        zigzag_bias = 0.0
    length_rate = float((lengths[-1] - lengths[-2]) / dt[-1])
    modes = infer_motion_modes(
        speed=speed,
        turn_rate=turn_rate,
        slip_angle=slip,
        zigzag_score=zigzag_score,
        body_length=float(lengths[-1]),
        body_length_rate=length_rate,
    )
    return StructuralMotionFeatures(
        center=centers[-1].copy(),
        body_direction=body_direction,
        body_length=float(lengths[-1]),
        velocity=velocities[-1].copy(),
        travel_direction=travel_direction,
        speed=speed,
        slip_angle_rad=slip,
        turn_rate_rad_s=turn_rate,
        curvature_inv_m=float(turn_rate / max(speed, EPS)),
        swept_area=_swept_area(centers, turn_rates),
        zigzag_score=zigzag_score,
        zigzag_bias=zigzag_bias,
        body_length_rate=length_rate,
        mode_probabilities=modes,
    )


def topology_weight(
    *,
    distance_penalty: float = 0.0,
    barrier_count: float = 0.0,
    history_distance: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Compute exp[-alpha*d-beta*N-gamma*D] in (0, 1]."""

    values = np.asarray(
        [distance_penalty, barrier_count, history_distance, alpha, beta, gamma], dtype=float
    )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("topology penalties and coefficients must be finite and non-negative")
    return float(
        np.exp(-(alpha * distance_penalty + beta * barrier_count + gamma * history_distance))
    )


def spherical_zone_penalties(
    paths: np.ndarray, zones: Sequence[SphericalKeepOutZone]
) -> np.ndarray:
    """Count configured safety-zone intersections for each 3D candidate path."""

    data = np.asarray(paths, dtype=float)
    if data.ndim != 3 or data.shape[-1] != 3 or not np.isfinite(data).all():
        raise ValueError("paths must be finite with shape (paths, time, 3)")
    counts = np.zeros(data.shape[0], dtype=float)
    for zone in zones:
        center = np.asarray(zone.center, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("zone centers must be finite 3D points")
        inside = np.linalg.norm(data - center[None, None, :], axis=-1) <= zone.radius
        counts += np.any(inside, axis=1)
    return counts


def reweight_paths(base_weights: np.ndarray, topology_weights: np.ndarray) -> np.ndarray:
    """Multiply path support by topology factors and renormalize."""

    base = np.asarray(base_weights, dtype=float).reshape(-1)
    topo = np.asarray(topology_weights, dtype=float).reshape(-1)
    if base.shape != topo.shape or base.size == 0:
        raise ValueError("base_weights and topology_weights must be same-size non-empty vectors")
    if not np.isfinite(base).all() or not np.isfinite(topo).all() or (base < 0).any() or (topo < 0).any():
        raise ValueError("path weights must be finite and non-negative")
    combined = base * topo
    total = float(combined.sum())
    if total <= EPS:
        raise ValueError("topology weighting removed all path support")
    return combined / total


def terminal_cell_probabilities(
    path_weights: np.ndarray,
    terminal_cells: np.ndarray,
    *,
    num_cells: int,
    cell_prior: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate a normalized terminal distribution from one cell per path."""

    weights = np.asarray(path_weights, dtype=float).reshape(-1)
    cells = np.asarray(terminal_cells, dtype=int).reshape(-1)
    if weights.size == 0 or weights.shape != cells.shape or num_cells < 2:
        raise ValueError("invalid path/cell configuration")
    if not np.isfinite(weights).all() or (weights < 0).any() or (cells < 0).any() or (cells >= num_cells).any():
        raise ValueError("terminal path data contain invalid values")
    probabilities = np.bincount(cells, weights=weights, minlength=num_cells).astype(float)
    if probabilities.sum() <= EPS:
        raise ValueError("terminal path weights have zero total mass")
    probabilities /= probabilities.sum()
    if cell_prior is not None:
        prior = np.asarray(cell_prior, dtype=float).reshape(-1)
        if prior.shape != probabilities.shape or not np.isfinite(prior).all() or (prior < 0).any():
            raise ValueError("cell_prior must be finite, non-negative, and match num_cells")
        probabilities *= prior
        if probabilities.sum() <= EPS:
            raise ValueError("cell prior removed all terminal probability mass")
        probabilities /= probabilities.sum()
    return probabilities


def occupancy_probabilities(path_weights: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """Return per-cell path occupancy probability; entries need not sum to one."""

    weights = np.asarray(path_weights, dtype=float).reshape(-1)
    membership = np.asarray(occupancy, dtype=bool)
    if membership.ndim != 2 or membership.shape[0] != len(weights):
        raise ValueError("occupancy must have shape (paths, cells)")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= EPS:
        raise ValueError("path weights must be finite, non-negative, and positive-mass")
    return (weights / weights.sum()) @ membership.astype(float)


def credible_cell_set(probabilities: np.ndarray, level: float = 0.95) -> tuple[int, ...]:
    """Return the shortest descending-probability prefix reaching level mass."""

    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if p.size < 2 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= EPS:
        raise ValueError("probabilities must be finite, non-negative, and positive-mass")
    if not 0.0 < level <= 1.0:
        raise ValueError("level must lie in (0, 1]")
    p = p / p.sum()
    order = np.argsort(p)[::-1]
    cutoff = int(np.searchsorted(np.cumsum(p[order]), level, side="left")) + 1
    return tuple(int(index) for index in order[:cutoff])


def summarize_distribution(
    probabilities: np.ndarray, *, credible_level: float = 0.95
) -> DistributionSummary:
    """Report peak cell, normalized entropy, concentration, and credible set."""

    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if p.size < 2 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= EPS:
        raise ValueError("probabilities must be finite, non-negative, and positive-mass")
    p = p / p.sum()
    cell = int(np.argmax(p))
    credible = credible_cell_set(p, credible_level)
    entropy = float(normalized_entropy(p))
    return DistributionSummary(
        probabilities=p,
        predicted_cell=cell,
        predicted_probability=float(p[cell]),
        normalized_entropy=entropy,
        entropy_concentration=1.0 - entropy,
        credible_cells=credible,
        credible_mass=float(p[list(credible)].sum()),
    )


def condition_paths_on_terminal_cell(
    path_weights: np.ndarray, terminal_cells: np.ndarray, terminal_cell: int
) -> np.ndarray:
    """Condition stored forward path weights on one terminal cell."""

    weights = np.asarray(path_weights, dtype=float).reshape(-1)
    cells = np.asarray(terminal_cells, dtype=int).reshape(-1)
    if weights.shape != cells.shape or weights.size == 0:
        raise ValueError("path weights and terminal cells must match")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("path weights must be finite and non-negative")
    selected = weights * (cells == int(terminal_cell))
    if selected.sum() <= EPS:
        raise ValueError("selected terminal cell has zero forward support")
    return selected / selected.sum()


def explain_terminal_cell(
    path_weights: np.ndarray,
    terminal_cells: np.ndarray,
    terminal_cell: int,
    *,
    path_states: np.ndarray | None = None,
    path_modes: Sequence[str] | None = None,
) -> BackwardExplanation:
    """Condition the forward path bank for an interpretable terminal-cell explanation."""

    conditional = condition_paths_on_terminal_cell(path_weights, terminal_cells, terminal_cell)
    expected_states = None
    if path_states is not None:
        states = np.asarray(path_states, dtype=float)
        if states.shape[0] != len(conditional) or not np.isfinite(states).all():
            raise ValueError("path_states must be finite with one leading entry per path")
        expected_states = np.tensordot(conditional, states, axes=(0, 0))
    mode_support = None
    if path_modes is not None:
        if len(path_modes) != len(conditional):
            raise ValueError("path_modes must have one entry per path")
        support: dict[str, float] = {}
        for mode, weight in zip(path_modes, conditional, strict=True):
            support[str(mode)] = support.get(str(mode), 0.0) + float(weight)
        mode_support = dict(sorted(support.items(), key=lambda item: item[1], reverse=True))
    return BackwardExplanation(
        terminal_cell=int(terminal_cell),
        conditional_path_weights=conditional,
        mode_support=mode_support,
        expected_states=expected_states,
    )
