#!/usr/bin/env python3
"""Analyze endpoint geometry or an existing HTI online trace with HTI 0.8.

Preferred input is independently observed nose/rear data. Existing online trace
artifacts do not yet store attitude, so trace mode uses velocity direction only
as an explicit body-axis proxy and records that degraded provenance in output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hti.topological_entropy import (  # noqa: E402
    EndpointObservation,
    SphericalKeepOutZone,
    endpoint_observations_from_centerline,
    estimate_structural_motion,
    reweight_paths,
    spherical_zone_penalties,
    topology_weight,
)


def _parse_zone(value: str) -> SphericalKeepOutZone:
    try:
        x, y, z, radius = (float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("zone must be x,y,z,radius") from exc
    return SphericalKeepOutZone((x, y, z), radius)


def _load_endpoint_csv(path: Path) -> list[EndpointObservation]:
    observations: list[EndpointObservation] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"time", "nose_x", "nose_y", "rear_x", "rear_y"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain at least {sorted(required)}")
        is_3d = {"nose_z", "rear_z"}.issubset(reader.fieldnames)
        for row in reader:
            nose = (float(row["nose_x"]), float(row["nose_y"]))
            rear = (float(row["rear_x"]), float(row["rear_y"]))
            if is_3d:
                nose += (float(row["nose_z"]),)
                rear += (float(row["rear_z"]),)
            observations.append(EndpointObservation(float(row["time"]), nose, rear))
    return observations


def _trace_observations(trace, *, body_length: float, window: int) -> list[EndpointObservation]:
    centers = np.asarray(trace["pos_est"], dtype=float)
    velocities = np.asarray(trace["vel_est"], dtype=float)
    if centers.shape != velocities.shape or centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("online trace must contain matching pos_est and vel_est arrays")
    count = min(max(window, 3), len(centers))
    centers = centers[-count:]
    velocities = velocities[-count:]
    norms = np.linalg.norm(velocities, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("online trace contains zero velocity; proxy orientation is undefined")
    directions = velocities / norms[:, None]
    dt = float(np.asarray(trace["dt"]).reshape(-1)[0])
    times = np.arange(count, dtype=float) * dt
    return endpoint_observations_from_centerline(times, centers, directions, body_length)


def _feature_dict(features) -> dict[str, object]:
    return {
        "center": features.center.tolist(),
        "body_direction": features.body_direction.tolist(),
        "body_length": features.body_length,
        "velocity": features.velocity.tolist(),
        "travel_direction": features.travel_direction.tolist(),
        "speed": features.speed,
        "slip_angle_rad": features.slip_angle_rad,
        "turn_rate_rad_s": features.turn_rate_rad_s,
        "curvature_inv_m": features.curvature_inv_m,
        "swept_area": features.swept_area,
        "zigzag_score": features.zigzag_score,
        "zigzag_bias": features.zigzag_bias,
        "body_length_rate": features.body_length_rate,
        "mode_probabilities": features.mode_probabilities,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--endpoint-csv", type=Path)
    source.add_argument("--online-trace", type=Path)
    parser.add_argument("--body-length", type=float, default=20.0)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--keepout", action="append", type=_parse_zone, default=[])
    parser.add_argument("--topology-beta", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=Path("hti08_structural_analysis.json"))
    args = parser.parse_args()

    result: dict[str, object] = {"schema": "hti.structural-analysis.v0.8"}
    if args.endpoint_csv is not None:
        observations = _load_endpoint_csv(args.endpoint_csv)
        result["orientation_source"] = "observed_nose_rear"
        result["degraded_orientation_proxy"] = False
    else:
        if args.body_length <= 0:
            raise SystemExit("--body-length must be positive")
        with np.load(args.online_trace, allow_pickle=False) as trace:
            observations = _trace_observations(
                trace, body_length=args.body_length, window=args.window
            )
            result["orientation_source"] = "velocity_proxy"
            result["degraded_orientation_proxy"] = True
            if args.keepout and "physics_sigma_pos_pred" in trace:
                paths = np.asarray(trace["physics_sigma_pos_pred"], dtype=float)[-1]
                weights = np.asarray(trace["physics_sigma_weights"], dtype=float).reshape(-1)
                penalties = spherical_zone_penalties(paths, args.keepout)
                topology = np.asarray(
                    [
                        topology_weight(barrier_count=value, beta=args.topology_beta)
                        for value in penalties
                    ]
                )
                reweighted = reweight_paths(weights, topology)
                result["topology"] = {
                    "zone_count": len(args.keepout),
                    "paths_intersecting_zone": int(np.count_nonzero(penalties)),
                    "base_mass_intersecting_zone": float(weights[penalties > 0].sum()),
                    "reweighted_mass_intersecting_zone": float(
                        reweighted[penalties > 0].sum()
                    ),
                    "reweighted_path_weights": reweighted.tolist(),
                }

    result["structural_features"] = _feature_dict(estimate_structural_motion(observations))
    result["scientific_note"] = (
        "Entropy/mode concentration is not accuracy. Predictive claims require held-out "
        "calibration, ablation, baseline, and external-validation evidence."
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
