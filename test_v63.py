import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "alien_exit_cell_predictor_v6_3.py"


def load_engine():
    spec = importlib.util.spec_from_file_location("trajectory_v63", ENGINE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_engine()

    def test_atmosphere_is_positive_and_monotonic(self):
        atm = self.m.Atmosphere()
        heights = [0, 11e3, 20e3, 47e3, 86e3, 120e3, 200e3]
        rho = [atm.properties(h)[0] for h in heights]
        self.assertTrue(all(np.isfinite(r) and r > 0 for r in rho))
        self.assertTrue(all(a > b for a, b in zip(rho, rho[1:])))

    def test_ukf_transition_is_deterministic(self):
        cfg = self.m.Config()
        earth, atm, aero = self.m.EarthModel(), self.m.Atmosphere(), self.m.AeroModel()
        proj = self.m.Projectile6DOF(1000.0, np.diag([500.0, 800.0, 1000.0]), earth, atm, aero)
        x = self.m.init_state(cfg, earth)
        env = self.m.make_env(cfg)
        a = proj.step_rk2(x.copy(), np.zeros(6), cfg.dt, env)
        b = proj.step_rk2(x.copy(), np.zeros(6), cfg.dt, env)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)

    def test_missing_class_weights_are_finite(self):
        weights = self.m.compute_class_weights(np.array([[0, -1], [1, -1]]), 16)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights[2:] == 0).all())

    def test_spd_repair(self):
        repaired = self.m.enforce_spd(np.array([[1.0, 2.0], [2.0, 1.0]]))
        self.assertTrue((np.linalg.eigvalsh(repaired) > 0).all())

    def test_ui_has_no_future_estimate_leakage(self):
        text = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertNotIn('pos_est[[t + horizon]]', text)
        self.assertIn('pred_world = center + v * lookahead_s', text)
        self.assertIn('allow_pickle=False', text)

    def test_online_token_window_is_strictly_causal(self):
        feats = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
        ctrls = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
        original = self.m.build_causal_token_buffer(feats, ctrls, t=5, window=4)

        changed_future_feats = feats.copy()
        changed_future_ctrls = ctrls.copy()
        changed_future_feats[6:] = -9999
        changed_future_ctrls[6:] = 9999
        perturbed = self.m.build_causal_token_buffer(
            changed_future_feats, changed_future_ctrls, t=5, window=4)

        np.testing.assert_array_equal(original, perturbed)
        np.testing.assert_array_equal(original[0::2, :3], feats[2:6])
        np.testing.assert_array_equal(original[1::2, 3:], ctrls[2:6])

    def test_online_token_window_rejects_incomplete_history(self):
        feats = np.zeros((5, 3), dtype=np.float32)
        ctrls = np.zeros((5, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.m.build_causal_token_buffer(feats, ctrls, t=2, window=4)

    def test_calibration_error_distinguishes_good_and_bad_confidence(self):
        correct = np.array([1.0, 1.0, 0.0, 0.0])
        calibrated = self.m.expected_calibration_error(
            np.array([0.75, 0.75, 0.25, 0.25]), correct, bins=2)
        overconfident = self.m.expected_calibration_error(
            np.array([0.99, 0.99, 0.99, 0.99]), correct, bins=2)
        self.assertLess(calibrated, overconfident)

    def test_temperature_scaling_softens_overconfident_validation_logits(self):
        import torch
        logits = torch.tensor([[8.0, 0.0], [8.0, 0.0]])
        labels = torch.tensor([0, 1])
        temperature = self.m.fit_temperature_from_logits(logits, labels)
        self.assertGreater(temperature, 1.0)

    def test_default_horizons_cross_the_forward_face(self):
        cfg = self.m.Config()
        crossing_steps = cfg.box_L_scale / cfg.dt
        self.assertTrue(all(h > crossing_steps for h in cfg.horizon_steps))

    def test_ecef_rotation_terms_have_expected_direction(self):
        earth = self.m.EarthModel()
        pos = np.array([earth.radius, 0.0, 0.0])
        stationary = earth.rotating_frame_acceleration(np.zeros(3), pos)
        self.assertGreater(stationary[0], 0.0)
        self.assertAlmostEqual(stationary[1], 0.0)
        self.assertAlmostEqual(stationary[2], 0.0)

        eastward = earth.rotating_frame_acceleration(np.array([100.0, 0.0, 0.0]), np.zeros(3))
        self.assertLess(eastward[1], 0.0)

    def test_central_gravity_matches_inverse_square_law(self):
        earth = self.m.EarthModel()
        p1 = np.array([earth.radius, 0.0, 0.0])
        p2 = np.array([2.0 * earth.radius, 0.0, 0.0])
        g1 = np.linalg.norm(earth.gravity(p1))
        g2 = np.linalg.norm(earth.gravity(p2))
        self.assertAlmostEqual(g1 / g2, 4.0, places=10)
        self.assertAlmostEqual(g1, earth.mu / earth.radius**2, places=10)

    def test_filter_environment_contains_no_hidden_truth_jitter(self):
        env = self.m.nominal_filter_env()
        self.assertEqual(env["g_jit"], 0.0)
        self.assertEqual(env["rho_jit"], 0.0)
        self.assertEqual(env["T_jit"], 0.0)
        np.testing.assert_array_equal(env["wind_accel"], np.zeros(3))

    def test_physics_projection_is_finite_and_normalized(self):
        cfg = self.m.Config()
        earth, atm, aero = self.m.EarthModel(), self.m.Atmosphere(), self.m.AeroModel()
        proj = self.m.Projectile6DOF(1000.0, np.diag([500.0, 800.0, 1000.0]), earth, atm, aero)
        x = self.m.init_state(cfg, earth)
        states = self.m.project_state_horizons(
            proj, x, np.zeros(6), cfg.dt, self.m.nominal_filter_env(), 18)
        self.assertTrue(np.isfinite(states).all())
        np.testing.assert_allclose(np.linalg.norm(states[:, 6:10], axis=1), 1.0, atol=1e-10)

    def test_sigma_projection_is_finite_and_weighted(self):
        cfg = self.m.Config()
        earth, atm, aero = self.m.EarthModel(), self.m.Atmosphere(), self.m.AeroModel()
        proj = self.m.Projectile6DOF(1000.0, np.diag([500.0, 800.0, 1000.0]), earth, atm, aero)
        x = self.m.init_state(cfg, earth)
        paths, weights = self.m.project_sigma_horizons(
            proj, x, np.eye(13), np.zeros(6), cfg.dt,
            self.m.nominal_filter_env(), 4)
        self.assertEqual(paths.shape, (27, 5, 13))
        self.assertTrue(np.isfinite(paths).all())
        self.assertTrue((weights >= 0).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_maneuver_hypotheses_cover_axes_signs_and_times(self):
        cfg = self.m.Config()
        earth, atm, aero = self.m.EarthModel(), self.m.Atmosphere(), self.m.AeroModel()
        proj = self.m.Projectile6DOF(1000.0, np.diag([500.0, 800.0, 1000.0]), earth, atm, aero)
        x = self.m.init_state(cfg, earth)
        paths, events = self.m.project_maneuver_hypotheses(
            proj, x, np.zeros(6), cfg.dt, self.m.nominal_filter_env(),
            18, cfg.force_std)
        self.assertEqual(paths.shape, (18, 19, 13))
        self.assertEqual(set(events.tolist()), {2, 6, 12})
        self.assertTrue(np.isfinite(paths).all())

    def test_virtual_box_cell_centers_round_trip(self):
        box = self.m.VirtualBox(np.zeros(3), np.array([1.0, 0.0, 0.0]),
                                np.array([0.0, 0.0, 1.0]), 100.0, 4)
        for row in range(4):
            for col in range(4):
                u = -0.75 + 0.5 * row
                w = -0.75 + 0.5 * col
                point = np.array([200.0, 200.0 * u, 200.0 * w])
                self.assertEqual(box.compute_exit_cell(point), (row, col))

    def test_virtual_box_cross_width_is_independent_of_forward_distance(self):
        box = self.m.VirtualBox(np.zeros(3), np.array([1.0, 0.0, 0.0]),
                                np.array([0.0, 0.0, 1.0]), 100.0, 4,
                                half_width=20.0)
        self.assertEqual(box.compute_exit_cell(np.array([200.0, 30.0, 30.0])), (3, 3))
        self.assertEqual(box.compute_exit_cell(np.array([200.0, -30.0, -30.0])), (0, 0))


if __name__ == "__main__":
    unittest.main()
