import json
import unittest
from pathlib import Path

import numpy as np

from alien_exit_cell_predictor_v6_3 import (
    AeroModel,
    Atmosphere,
    Config,
    EarthModel,
    Projectile6DOF,
    init_state,
)
from hti.phase3 import Phase3Event
from hti.phase3_runner import _apply_topology_cube, _physics_path_bank, configure


class Phase3RunnerTests(unittest.TestCase):
    def protocol(self):
        return json.loads(Path("configs/hti_08_ablation_frozen.json").read_text(encoding="utf-8"))

    def execution(self):
        return json.loads(
            Path("configs/hti_08_phase3_execution_frozen.json").read_text(encoding="utf-8")
        )

    def test_configure_matches_frozen_execution(self):
        protocol = self.protocol()
        execution = self.execution()
        cfg = configure(protocol, execution, 101)
        generation = execution["trajectory_generation"]
        transformer = execution["transformer"]
        self.assertEqual(cfg.seed, 101)
        self.assertEqual(cfg.traj_steps, generation["traj_steps"])
        self.assertEqual(cfg.window, generation["window"])
        self.assertEqual(cfg.horizon_steps, (8, 10, 12))
        self.assertEqual(cfg.epochs, transformer["epochs"])
        self.assertAlmostEqual(cfg.maneuver_prob, generation["maneuver_prob"])
        self.assertAlmostEqual(cfg.force_std, generation["force_std_n"])

    def test_physics_path_bank_contains_sigma_and_maneuver_mass(self):
        cfg = Config()
        cfg.horizon_steps = (2, 3, 4)
        cfg.maneuver_prob = 0.14
        cfg.force_std = 25000.0
        earth, atmosphere, aero = EarthModel(), Atmosphere(), AeroModel()
        projectile = Projectile6DOF(
            1000.0,
            np.diag([500.0, 800.0, 1000.0]),
            earth,
            atmosphere,
            aero,
        )
        state = init_state(cfg, earth)
        frames = 6
        states = np.repeat(state[None, :], frames, axis=0)
        covariances = np.repeat((np.eye(13) * 0.1)[None, :, :], frames, axis=0)
        measurements = np.column_stack([states[:, :3], states[:, 3:6]])
        event = Phase3Event(
            event_id=101000000,
            tokens=np.zeros((1, 2, 2), dtype=np.float32),
            labels=np.zeros((1, 3), dtype=np.int64),
            next_features=np.zeros((1, 1), dtype=np.float32),
            source_frames=np.array([0], dtype=int),
            estimated_states=states,
            covariances=covariances,
            truth_states=states.copy(),
            measurements=measurements,
            controls=np.zeros((frames - 1, 6), dtype=float),
        )
        paths, weights = _physics_path_bank(
            projectile, cfg, event, 0, max_horizon=4
        )
        self.assertEqual(paths.shape[0], 27 + 18)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(weights[:27].sum()), 0.86, places=12)
        self.assertAlmostEqual(float(weights[27:].sum()), 0.14, places=12)

    def test_topology_application_uses_floor_and_strength(self):
        probabilities = np.array([[[0.5, 0.5]]], dtype=float)
        prior = np.array([[[1.0, 0.0]]], dtype=float)
        adjusted = _apply_topology_cube(
            probabilities, prior, strength=0.35, floor=0.001
        )
        self.assertAlmostEqual(float(adjusted.sum()), 1.0, places=12)
        self.assertGreater(adjusted[0, 0, 0], adjusted[0, 0, 1])
        self.assertGreater(adjusted[0, 0, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
