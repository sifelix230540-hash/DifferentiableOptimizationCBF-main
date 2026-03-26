import unittest

import numpy as np

from CBF_experiment.active.pybullet.welding_320_common import ExperimentConfig
from CBF_experiment.active.pybullet.welding_320_control import MPCDCBFController


class _DummySurfaceRobot:
    total_dof = 1
    n_pris = 0
    n_revo = 1
    ee_link_index = 99
    cbf_link_indices = [3]
    welding_gun_links = [3]

    def get_closest_points_to_obstacle(self, link_index, obs_body_id, max_dist=1.0):
        return {
            "robot_link_index": link_index,
            "robot_link_name": "welding_gun_base",
            "obs_link_index": 1,
            "obs_link_name": "l3",
            "point_on_link": np.array([1.0, 0.0, 0.0]),
            "point_on_obstacle": np.array([0.5, 0.0, 0.0]),
            "normal_on_link": np.array([-1.0, 0.0, 0.0]),
            "normal_on_obstacle": np.array([1.0, 0.0, 0.0]),
            "signed_dist": 0.5,
            "euclidean_dist": 0.5,
        }

    def get_link_cbf_row_at_point(self, link_index, world_point, normal, q, dq):
        return np.array([1.0], dtype=float)

    def get_link_name(self, link_index):
        return "welding_gun_base"


class _DummyObstacle:
    body_id = 7
    cbf_link_indices = [1]


class _DummyTrajectory:
    progress_end = 1.0

    def sample_by_progress(self, progress):
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3)


class SurfaceCBFDataTests(unittest.TestCase):
    def test_build_cbf_data_keeps_both_surface_normals(self):
        cfg = ExperimentConfig()
        cfg.use_mesh_cbf = True
        controller = MPCDCBFController(_DummySurfaceRobot(), cfg, _DummyTrajectory())

        grad_rows, h_vals = controller._build_cbf_data(np.zeros(1), np.zeros(1), [_DummyObstacle()])

        self.assertEqual(len(grad_rows), 1)
        self.assertEqual(len(h_vals), 1)
        meta = controller._last_cbf_meta[0]
        self.assertTrue(np.allclose(meta["point_on_link"], [1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(meta["point_on_obstacle"], [0.5, 0.0, 0.0]))
        self.assertTrue(np.allclose(meta["normal_on_link"], [-1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(meta["normal_on_obstacle"], [1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
