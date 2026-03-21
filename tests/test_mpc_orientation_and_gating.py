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


class GatingThresholdTests(unittest.TestCase):
    def test_pose_threshold_requires_orientation_match(self):
        module = load_module()
        current_pos = np.array([1.0, 2.0, 3.0])
        goal_pos = current_pos + np.array([0.005, 0.0, 0.0])
        current_quat = np.array([0.0, 0.0, 0.0, 1.0])
        goal_quat = module.Rotation.from_euler("z", 40.0, degrees=True).as_quat()

        reached = module.pose_within_thresholds(
            current_pos,
            goal_pos,
            current_quat,
            goal_quat,
            pos_threshold=0.03,
            rot_threshold=0.10,
        )

        self.assertFalse(reached)


if __name__ == "__main__":
    unittest.main()
