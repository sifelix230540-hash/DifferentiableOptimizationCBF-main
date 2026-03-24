import importlib.util
import pathlib
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CBF_experiment" / "active" / "3_20_9axis_3d_mpc_cbf_experiment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cbf_exp_320", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyRobot:
    total_dof = 2
    n_pris = 0
    n_revo = 2


class MPCOrientationTests(unittest.TestCase):
    def test_build_qp_includes_orientation_cost(self):
        module = load_module()
        cfg = module.ExperimentConfig()
        cfg.N_mpc = 3
        cfg.mpc_control_weight = 0.0
        cfg.mpc_smooth_weight = 0.0
        cfg.mpc_tracking_weight = 0.0
        cfg.mpc_orientation_tracking_weight = 4.0
        controller = module.MPCDCBFController(DummyRobot(), cfg, trajectory=None)

        ee_pos = np.zeros(3)
        j_pos = np.zeros((3, 2))
        j_rot = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        ref_positions = [np.zeros(3) for _ in range(cfg.N_mpc)]
        ref_rotvecs = [np.array([0.3, -0.2, 0.0]) for _ in range(cfg.N_mpc)]

        h_mat, f_vec, _, _ = controller._build_qp(
            ee_pos, j_pos, j_rot, ref_positions, ref_rotvecs, [], []
        )

        self.assertGreater(np.linalg.norm(h_mat), 0.0)
        self.assertGreater(np.linalg.norm(f_vec), 0.0)


class SimplifiedModuleSurfaceTests(unittest.TestCase):
    def test_module_keeps_only_current_mainline_symbols(self):
        module = load_module()

        self.assertTrue(hasattr(module, "ExperimentConfig"))
        self.assertTrue(hasattr(module, "MPCDCBFController"))
        self.assertTrue(hasattr(module, "PathProgressTrajectory"))
        self.assertTrue(hasattr(module, "CartesianRRTNominalPlanner"))
        self.assertFalse(hasattr(module, "CBFQPController"))
        self.assertFalse(hasattr(module, "SphereObstacle"))
        self.assertFalse(hasattr(module, "PlateObstacle"))

    def test_experiment_config_hides_internal_or_unused_parameters(self):
        module = load_module()
        cfg = module.ExperimentConfig()

        self.assertFalse(hasattr(cfg, "workpiece_package_name"))
        self.assertFalse(hasattr(cfg, "workpiece_package_alias"))
        self.assertFalse(hasattr(cfg, "gravity"))
        self.assertFalse(hasattr(cfg, "reference_samples"))
        self.assertFalse(hasattr(cfg, "dq_nominal_gain"))
        self.assertTrue(hasattr(cfg, "ignore_all_collisions"))
        self.assertTrue(hasattr(cfg, "rrt_cartesian_margin"))

    def test_experiment_config_groups_fields_by_scene_trajectory_mpc_rrt(self):
        module = load_module()
        field_names = list(module.ExperimentConfig.__dataclass_fields__)

        self.assertLess(field_names.index("camera_distance"), field_names.index("approach_duration"))
        self.assertLess(field_names.index("approach_duration"), field_names.index("N_mpc"))
        self.assertLess(field_names.index("N_mpc"), field_names.index("rrt_max_iterations"))


class PathProgressTrajectoryTests(unittest.TestCase):
    def test_project_progress_is_monotonic_along_path(self):
        module = load_module()
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        base_traj = module.PiecewiseLineSlerpTrajectory([
            module.LineSlerpTrajectory([0.0, 0.0, 0.0], quat, [1.0, 0.0, 0.0], quat, 1.0, 0.1),
            module.LineSlerpTrajectory([1.0, 0.0, 0.0], quat, [1.0, 1.0, 0.0], quat, 1.0, 0.1),
        ])
        traj = module.PathProgressTrajectory(base_traj)

        probes = [
            np.array([0.1, 0.0, 0.0]),
            np.array([0.8, 0.0, 0.0]),
            np.array([1.0, 0.2, 0.0]),
            np.array([1.0, 0.9, 0.0]),
        ]
        progresses = [traj.project_progress(pos) for pos in probes]

        self.assertTrue(all(a <= b for a, b in zip(progresses, progresses[1:])))

    def test_sample_by_progress_is_continuous_across_segment_boundary(self):
        module = load_module()
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        base_traj = module.PiecewiseLineSlerpTrajectory([
            module.LineSlerpTrajectory([0.0, 0.0, 0.0], quat, [1.0, 0.0, 0.0], quat, 1.0, 0.1),
            module.LineSlerpTrajectory([1.0, 0.0, 0.0], quat, [1.0, 1.0, 0.0], quat, 1.0, 0.1),
        ])
        traj = module.PathProgressTrajectory(base_traj)
        boundary = traj.segment_end_progress[0]

        pos_before, quat_before, _, _ = traj.sample_by_progress(boundary - 1e-6)
        pos_after, quat_after, _, _ = traj.sample_by_progress(boundary + 1e-6)

        self.assertLess(np.linalg.norm(pos_after - pos_before), 1e-3)
        self.assertAlmostEqual(abs(float(np.dot(quat_before, quat_after))), 1.0, places=4)

    def test_project_progress_prefers_local_branch_when_start_end_overlap(self):
        module = load_module()
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        base_traj = module.PiecewiseLineSlerpTrajectory([
            module.LineSlerpTrajectory([0.0, 0.0, 0.0], quat, [1.0, 0.0, 0.0], quat, 1.0, 0.1),
            module.LineSlerpTrajectory([1.0, 0.0, 0.0], quat, [1.0, 1.0, 0.0], quat, 1.0, 0.1),
            module.LineSlerpTrajectory([1.0, 1.0, 0.0], quat, [0.0, 0.0, 0.0], quat, 1.0, 0.1),
        ])
        traj = module.PathProgressTrajectory(base_traj)

        start_pos = np.array([0.0, 0.0, 0.0])
        start_progress = traj.project_progress(start_pos, hint_progress=0.0)
        end_progress = traj.project_progress(start_pos, hint_progress=traj.progress_end)

        self.assertLess(start_progress, 1e-6)
        self.assertGreater(end_progress, traj.progress_end - 1e-3)


if __name__ == "__main__":
    unittest.main()
