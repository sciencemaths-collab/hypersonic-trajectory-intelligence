import unittest

import numpy as np

from hti.topological_entropy import (
    EndpointObservation,
    SphericalKeepOutZone,
    circular_smooth,
    condition_paths_on_terminal_cell,
    credible_cell_set,
    endpoint_geometry,
    endpoint_observations_from_centerline,
    estimate_structural_motion,
    explain_terminal_cell,
    occupancy_probabilities,
    recover_endpoints,
    reweight_paths,
    spherical_zone_penalties,
    summarize_distribution,
    terminal_cell_probabilities,
    topology_weight,
)


class TopologicalEntropyTests(unittest.TestCase):
    def observations(self):
        return [
            EndpointObservation(0.0, (1.0, 0.2), (-1.0, 0.2)),
            EndpointObservation(1.0, (2.0, 0.8), (0.0, 0.8)),
            EndpointObservation(2.0, (2.7, 1.8), (0.7, 1.8)),
            EndpointObservation(3.0, (3.8, 2.6), (1.8, 2.6)),
        ]

    def test_endpoint_transform_round_trip(self):
        obs = self.observations()
        centers, axes, lengths = endpoint_geometry(obs)
        self.assertTrue((lengths > 0).all())
        nose, rear = recover_endpoints(centers[-1], axes[-1])
        np.testing.assert_allclose(nose, obs[-1].nose)
        np.testing.assert_allclose(rear, obs[-1].rear)

    def test_centerline_bridge_preserves_center_and_length(self):
        times = np.array([0.0, 1.0, 2.0])
        centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        directions = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        obs = endpoint_observations_from_centerline(times, centers, directions, 4.0)
        recovered_centers, _, lengths = endpoint_geometry(obs)
        np.testing.assert_allclose(recovered_centers, centers)
        np.testing.assert_allclose(lengths, np.full(3, 4.0))

    def test_circular_smoothing_is_wrap_invariant(self):
        angles = np.deg2rad(np.array([359.0, 1.0, 2.0]))
        shifted = angles.copy()
        shifted[1] += 2 * np.pi
        a = circular_smooth(angles)
        b = circular_smooth(shifted)
        self.assertAlmostEqual(float(np.angle(np.exp(1j * (a - b)))), 0.0, places=12)

    def test_structural_modes_are_normalized(self):
        features = estimate_structural_motion(self.observations())
        self.assertAlmostEqual(sum(features.mode_probabilities.values()), 1.0, places=12)
        self.assertGreater(features.speed, 0.0)
        self.assertGreaterEqual(features.slip_angle_rad, 0.0)

    def test_topology_weight_is_monotone(self):
        clean = topology_weight()
        penalized = topology_weight(barrier_count=1.0, beta=2.0)
        more_penalized = topology_weight(barrier_count=2.0, beta=2.0)
        self.assertEqual(clean, 1.0)
        self.assertGreater(penalized, more_penalized)
        self.assertGreater(more_penalized, 0.0)

    def test_topology_reweighting_transfers_support(self):
        base = np.array([0.5, 0.5])
        reweighted = reweight_paths(base, np.array([1.0, 0.1]))
        self.assertAlmostEqual(float(reweighted.sum()), 1.0)
        self.assertGreater(reweighted[0], reweighted[1])

    def test_terminal_distribution_normalizes(self):
        probs = terminal_cell_probabilities(
            np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 1, 2]), num_cells=4
        )
        self.assertAlmostEqual(float(probs.sum()), 1.0)
        self.assertEqual(int(np.argmax(probs)), 1)

    def test_cell_prior_is_renormalized(self):
        probs = terminal_cell_probabilities(
            np.array([1.0, 1.0]),
            np.array([0, 1]),
            num_cells=2,
            cell_prior=np.array([0.1, 1.0]),
        )
        self.assertAlmostEqual(float(probs.sum()), 1.0)
        self.assertGreater(probs[1], probs[0])

    def test_entropy_and_credible_set(self):
        summary = summarize_distribution(np.array([0.70, 0.20, 0.08, 0.02]), credible_level=0.90)
        self.assertEqual(summary.predicted_cell, 0)
        self.assertGreaterEqual(summary.credible_mass, 0.90)
        self.assertGreater(summary.entropy_concentration, 0.0)
        self.assertEqual(credible_cell_set(summary.probabilities, 0.90), summary.credible_cells)

    def test_occupancy_does_not_need_to_sum_to_one(self):
        occupancy = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
        probs = occupancy_probabilities(np.array([0.5, 0.5]), occupancy)
        np.testing.assert_allclose(probs, np.array([0.5, 1.0, 0.5]))
        self.assertGreater(float(probs.sum()), 1.0)

    def test_backward_conditioning_and_forward_consistency(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        cells = np.array([0, 1, 1, 2])
        conditional = condition_paths_on_terminal_cell(weights, cells, 1)
        self.assertAlmostEqual(float(conditional.sum()), 1.0)
        np.testing.assert_allclose(conditional, np.array([0.0, 0.4, 0.6, 0.0]))
        forward = terminal_cell_probabilities(weights, cells, num_cells=3)
        self.assertAlmostEqual(forward[1], weights[cells == 1].sum() / weights.sum())

    def test_backward_explanation_reports_mode_support(self):
        weights = np.array([0.2, 0.3, 0.5])
        cells = np.array([1, 1, 2])
        states = np.array([[[0.0], [1.0]], [[2.0], [3.0]], [[9.0], [9.0]]])
        explanation = explain_terminal_cell(
            weights,
            cells,
            1,
            path_states=states,
            path_modes=["straight", "turn", "turn"],
        )
        self.assertAlmostEqual(sum(explanation.mode_support.values()), 1.0)
        self.assertEqual(explanation.expected_states.shape, (2, 1))
        self.assertAlmostEqual(float(explanation.expected_states[0, 0]), 1.2)

    def test_spherical_keepout_counts_intersections(self):
        paths = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
            ]
        )
        zones = [SphericalKeepOutZone((1.0, 0.0, 0.0), 0.5)]
        counts = spherical_zone_penalties(paths, zones)
        np.testing.assert_array_equal(counts, np.array([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
