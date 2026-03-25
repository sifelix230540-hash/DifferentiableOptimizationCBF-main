import unittest

import numpy as np

from CBF_experiment.active.pybullet.welding_320_common import ExperimentConfig
from CBF_experiment.active.pybullet.welding_320_control import DynamicNominalReferenceMixer


class DynamicNominalReferenceMixerTests(unittest.TestCase):
    def test_disabled_mixer_keeps_nominal_reference(self):
        cfg = ExperimentConfig()
        cfg.use_dynamic_nominal_reference = False
        mixer = DynamicNominalReferenceMixer(cfg)

        nominal_refs = [
            np.array([0.1, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0]),
        ]

        mixed_positions, info = mixer.mix_positions(
            ee_pos=np.zeros(3),
            current_progress=0.0,
            nominal_positions=nominal_refs,
            signed_dist=0.01,
            obstacle_normal=np.array([1.0, 0.0, 0.0]),
        )

        self.assertTrue(all(np.allclose(a, b) for a, b in zip(mixed_positions, nominal_refs)))
        self.assertEqual(info["dynamic_nominal_weight"], 0.0)
        self.assertFalse(info["dynamic_nominal_stall_active"])

    def test_enabled_mixer_waits_until_stall_before_modifying_nominal(self):
        cfg = ExperimentConfig()
        cfg.use_dynamic_nominal_reference = True
        mixer = DynamicNominalReferenceMixer(cfg)

        nominal_refs = [
            np.array([0.1, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0]),
        ]

        for step in range(6):
            mixed_positions, info = mixer.mix_positions(
                ee_pos=np.array([0.02 * step, 0.0, 0.0]),
                current_progress=0.05 * step,
                nominal_positions=nominal_refs,
                signed_dist=0.03,
                obstacle_normal=np.array([1.0, 0.0, 0.0]),
            )

        self.assertTrue(all(np.allclose(a, b) for a, b in zip(mixed_positions, nominal_refs)))
        self.assertEqual(info["dynamic_nominal_weight"], 0.0)
        self.assertFalse(info["dynamic_nominal_stall_active"])

    def test_enabled_mixer_uses_executed_trajectory_direction_when_stalled(self):
        cfg = ExperimentConfig()
        cfg.use_dynamic_nominal_reference = True
        mixer = DynamicNominalReferenceMixer(cfg)

        nominal_refs = [
            np.array([0.1, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0]),
        ]

        for ee_pos in (
            np.array([0.00, 0.00, 0.0]),
            np.array([0.00, 0.02, 0.0]),
            np.array([0.00, 0.04, 0.0]),
            np.array([0.00, 0.06, 0.0]),
            np.array([0.00, 0.08, 0.0]),
            np.array([0.00, 0.10, 0.0]),
        ):
            mixed_positions, info = mixer.mix_positions(
                ee_pos=ee_pos,
                current_progress=0.01,
                nominal_positions=nominal_refs,
                signed_dist=0.03,
                obstacle_normal=np.array([1.0, 0.0, 0.0]),
            )

        self.assertTrue(info["dynamic_nominal_stall_active"])
        self.assertGreater(info["dynamic_nominal_weight"], 0.0)
        self.assertGreater(mixed_positions[0][1], nominal_refs[0][1])
        self.assertGreater(mixed_positions[0][0], nominal_refs[0][0])
        self.assertGreater(info["dynamic_reference_offset_norm"], 0.0)

    def test_enabled_mixer_triggers_for_small_but_nonzero_exec_motion(self):
        cfg = ExperimentConfig()
        cfg.use_dynamic_nominal_reference = True
        mixer = DynamicNominalReferenceMixer(cfg)

        nominal_refs = [
            np.array([0.19, 0.0, 0.0]),
            np.array([0.29, 0.0, 0.0]),
            np.array([0.39, 0.0, 0.0]),
        ]

        for ee_pos in (
            np.array([0.0000, 0.0000, 0.0]),
            np.array([0.0000, 0.00004, 0.0]),
            np.array([0.0000, 0.00008, 0.0]),
            np.array([0.0000, 0.00012, 0.0]),
            np.array([0.0000, 0.00016, 0.0]),
            np.array([0.0000, 0.00020, 0.0]),
        ):
            mixed_positions, info = mixer.mix_positions(
                ee_pos=ee_pos,
                current_progress=0.001,
                nominal_positions=nominal_refs,
                signed_dist=0.03,
                obstacle_normal=np.array([1.0, 0.0, 0.0]),
            )

        self.assertTrue(info["dynamic_nominal_stall_active"])
        self.assertGreater(info["dynamic_nominal_weight"], 0.0)
        self.assertGreater(info["dynamic_nominal_exec_motion"], 0.0)
        self.assertGreater(info["dynamic_reference_offset_norm"], 0.0)

    def test_enabled_mixer_caps_weight_and_keeps_nominal_pull(self):
        cfg = ExperimentConfig()
        cfg.use_dynamic_nominal_reference = True
        mixer = DynamicNominalReferenceMixer(cfg)

        nominal_refs = [
            np.array([0.1, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0]),
        ]

        for ee_pos in (
            np.array([0.00, 0.00, 0.0]),
            np.array([0.00, 0.02, 0.0]),
            np.array([0.00, 0.04, 0.0]),
            np.array([0.00, 0.06, 0.0]),
            np.array([0.00, 0.08, 0.0]),
            np.array([0.00, 0.10, 0.0]),
        ):
            mixed_positions, info = mixer.mix_positions(
                ee_pos=ee_pos,
                current_progress=0.0,
                nominal_positions=nominal_refs,
                signed_dist=0.03,
                obstacle_normal=np.array([1.0, 0.0, 0.0]),
            )

        self.assertTrue(info["dynamic_nominal_stall_active"])
        self.assertLessEqual(info["dynamic_nominal_weight"], 0.6)
        self.assertLess(mixed_positions[0][0], nominal_refs[0][0] + 0.08)

    def test_enabled_mixer_locks_escape_direction_during_stall(self):
        cfg = ExperimentConfig()
        cfg.use_dynamic_nominal_reference = True
        mixer = DynamicNominalReferenceMixer(cfg)

        nominal_refs = [
            np.array([0.1, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0]),
        ]

        for ee_pos in (
            np.array([0.00, 0.00, 0.0]),
            np.array([0.00, 0.02, 0.0]),
            np.array([0.00, 0.04, 0.0]),
            np.array([0.00, 0.06, 0.0]),
            np.array([0.00, 0.08, 0.0]),
            np.array([0.00, 0.10, 0.0]),
        ):
            mixed_positions, info = mixer.mix_positions(
                ee_pos=ee_pos,
                current_progress=0.0,
                nominal_positions=nominal_refs,
                signed_dist=0.03,
                obstacle_normal=np.array([1.0, 0.0, 0.0]),
            )
        first_offset_y = mixed_positions[0][1] - nominal_refs[0][1]

        for ee_pos in (
            np.array([0.02, 0.10, 0.0]),
            np.array([0.04, 0.08, 0.0]),
            np.array([0.06, 0.06, 0.0]),
            np.array([0.08, 0.04, 0.0]),
            np.array([0.10, 0.02, 0.0]),
            np.array([0.12, 0.00, 0.0]),
        ):
            mixed_positions, info = mixer.mix_positions(
                ee_pos=ee_pos,
                current_progress=0.0,
                nominal_positions=nominal_refs,
                signed_dist=0.03,
                obstacle_normal=np.array([1.0, 0.0, 0.0]),
            )

        self.assertTrue(info["dynamic_nominal_stall_active"])
        self.assertGreater(first_offset_y, 0.0)
        self.assertGreater(mixed_positions[0][1] - nominal_refs[0][1], 0.0)


class DummyRobot:
    total_dof = 1
    n_pris = 0
    n_revo = 1
    ee_link_index = 0

    def get_ee_jacobian(self, q, dq):
        return np.zeros((6, 1))

    def get_closest_point_to_obstacle(self, link_index, obs_body_id, max_dist=1.0):
        return np.zeros(3), 0.03, np.array([1.0, 0.0, 0.0])


class DummyObstacle:
    body_id = 1


class DummyTrajectory:
    progress_end = 1.0

    def sample_by_progress(self, progress):
        pos = np.array([progress, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        lin_vel = np.array([1.0, 0.0, 0.0])
        ang_vel = np.zeros(3)
        return pos, quat, lin_vel, ang_vel


class DynamicNominalControllerIntegrationTests(unittest.TestCase):
    def test_controller_reports_stall_aware_dynamic_nominal_weight(self):
        cfg = ExperimentConfig()
        cfg.use_dynamic_nominal_reference = True
        from CBF_experiment.active.pybullet.welding_320_control import MPCDCBFController

        mpc = MPCDCBFController(DummyRobot(), cfg, DummyTrajectory())
        mpc.config.mpc_replan_steps = 1
        mpc._build_cbf_data = lambda q, dq, obstacles: ([], [])

        info = {}
        for ee_pos in (
            np.array([0.00, 0.00, 0.0]),
            np.array([0.00, 0.02, 0.0]),
            np.array([0.00, 0.04, 0.0]),
            np.array([0.00, 0.06, 0.0]),
            np.array([0.00, 0.08, 0.0]),
            np.array([0.00, 0.10, 0.0]),
        ):
            _, info = mpc.solve(
                q=np.zeros(1),
                dq=np.zeros(1),
                ee_pos=ee_pos,
                ee_quat=np.array([0.0, 0.0, 0.0, 1.0]),
                ref_pos=np.array([0.1, 0.0, 0.0]),
                ref_quat=np.array([0.0, 0.0, 0.0, 1.0]),
                ref_lin_vel=np.array([0.1, 0.0, 0.0]),
                ref_ang_vel=np.zeros(3),
                obstacles=[DummyObstacle()],
                current_progress=0.01,
            )

        self.assertTrue(info["dynamic_nominal_stall_active"])
        self.assertGreater(info["dynamic_nominal_weight"], 0.0)


class DynamicNominalLoggingTests(unittest.TestCase):
    def test_step_log_includes_stall_aware_escape_fields(self):
        from CBF_experiment.active.pybullet.welding_320_experiment import format_step_status_line

        line = format_step_status_line(
            sim_step=120,
            seg_idx=1,
            progress_exec=1.234,
            progress_end=11.966,
            gantry_pos=np.array([12.0, -6.8, -0.1]),
            info={
                "tracking_error": 0.045,
                "orientation_error": np.deg2rad(12.0),
                "lag_error": 0.003,
                "contour_error": 0.067,
                "min_h": -0.0054,
                "status": "mpc_optimal",
                "dynamic_nominal_stall_active": True,
                "dynamic_nominal_weight": 0.63,
                "dynamic_nominal_progress_gain": 0.004,
                "dynamic_nominal_exec_motion": 0.082,
                "dynamic_reference_offset_norm": 0.057,
            },
        )

        self.assertIn("stall=Y", line)
        self.assertIn("w=0.63", line)
        self.assertIn("pg=4.0mm", line)
        self.assertIn("em=82.0mm", line)
        self.assertIn("off=57.0mm", line)


if __name__ == "__main__":
    unittest.main()
