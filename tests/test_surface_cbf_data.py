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


class _DummySDFRobot:
    total_dof = 1
    n_pris = 0
    n_revo = 1
    ee_link_index = 99
    cbf_link_indices = [3]
    welding_gun_links = [3]

    def get_link_origin(self, link_index):
        return np.array([0.25, 0.0, 0.0], dtype=float)

    def get_link_cbf_row(self, link_index, normal, q, dq):
        return np.array([2.0], dtype=float)

    def get_link_name(self, link_index):
        return "sdf_link"


class _DummySDFObstacle:
    body_id = -101
    cbf_link_indices = [3]
    is_sdf_obstacle = True

    def compute_distance(self, point):
        return 0.25, np.array([1.0, 0.0, 0.0], dtype=float)


class _DummySampledRobot:
    total_dof = 1
    n_pris = 0
    n_revo = 1
    ee_link_index = 99
    cbf_link_indices = [3]
    welding_gun_links = [3]

    def __init__(self):
        self.sampled_points_queried = []
        self.origin_queried = False

    def get_link_origin(self, link_index):
        self.origin_queried = True
        raise AssertionError("sample-based SDF CBF should not fall back to link origin")

    def transform_link_points_to_world(self, link_index, local_points, local_normals=None):
        pts = np.asarray(local_points, dtype=float).reshape(-1, 3) + np.array([1.0, 2.0, 3.0], dtype=float)
        normals = None if local_normals is None else np.asarray(local_normals, dtype=float).reshape(-1, 3)
        self.sampled_points_queried.extend(pts.tolist())
        return pts, normals

    def get_link_cbf_row_at_point(self, link_index, world_point, normal, q, dq):
        return np.array([float(world_point[0])], dtype=float)

    def get_link_name(self, link_index):
        return "sampled_link"


class _DummySampledSDFObstacle:
    body_id = -101
    cbf_link_indices = [3]
    is_sdf_obstacle = True

    def compute_distances(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        signed = np.array([0.3, 0.1, 0.2], dtype=float)[: pts.shape[0]]
        normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=float), (pts.shape[0], 1))
        return signed, normals


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

    def test_build_cbf_data_supports_sdf_obstacle_distance(self):
        cfg = ExperimentConfig()
        cfg.use_mesh_cbf = False
        cfg.use_sdf_cbf = True
        controller = MPCDCBFController(_DummySDFRobot(), cfg, _DummyTrajectory())

        grad_rows, h_vals = controller._build_cbf_data(np.zeros(1), np.zeros(1), [_DummySDFObstacle()])

        self.assertEqual(len(grad_rows), 1)
        self.assertEqual(len(h_vals), 1)
        self.assertAlmostEqual(h_vals[0], 0.25 - cfg.safety_margin, places=6)
        self.assertTrue(np.allclose(grad_rows[0], [2.0]))
        meta = controller._last_cbf_meta[0]
        self.assertFalse(meta["use_mesh"])
        self.assertTrue(np.allclose(meta["point_on_link"], [0.25, 0.0, 0.0]))
        self.assertTrue(np.allclose(meta["normal"], [1.0, 0.0, 0.0]))

    def test_dynamic_nominal_hint_supports_sdf_obstacle_distance(self):
        cfg = ExperimentConfig()
        cfg.use_sdf_cbf = True
        cfg.use_dynamic_nominal_reference = True
        controller = MPCDCBFController(_DummySDFRobot(), cfg, _DummyTrajectory())

        signed_dist, normal = controller._get_dynamic_nominal_hint([_DummySDFObstacle()])

        self.assertAlmostEqual(signed_dist, 0.25, places=6)
        self.assertTrue(np.allclose(normal, [1.0, 0.0, 0.0]))

    def test_build_cbf_data_uses_surface_sample_points_instead_of_link_origin(self):
        cfg = ExperimentConfig()
        cfg.use_sdf_cbf = True
        cfg.sdf_cbf_max_points_per_link = 2
        robot = _DummySampledRobot()
        controller = MPCDCBFController(
            robot,
            cfg,
            _DummyTrajectory(),
            surface_samples={
                3: {
                    "link_index": 3,
                    "link_name": "sampled_link",
                    "local_points": np.array([
                        [0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0],
                        [0.2, 0.0, 0.0],
                    ], dtype=float),
                    "local_normals": np.array([
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ], dtype=float),
                }
            },
        )

        grad_rows, h_vals = controller._build_cbf_data(np.zeros(1), np.zeros(1), [_DummySampledSDFObstacle()])

        self.assertFalse(robot.origin_queried)
        self.assertEqual(len(grad_rows), 2)
        self.assertEqual(len(h_vals), 2)
        self.assertAlmostEqual(h_vals[0], 0.1 - cfg.safety_margin, places=6)
        self.assertAlmostEqual(h_vals[1], 0.2 - cfg.safety_margin, places=6)
        self.assertTrue(np.allclose(grad_rows[0], [1.1]))
        self.assertTrue(np.allclose(grad_rows[1], [1.2]))
        meta = controller._last_cbf_meta[0]
        self.assertTrue(meta["used_surface_sample"])
        self.assertTrue(np.allclose(meta["sample_point_local"], [0.1, 0.0, 0.0]))
        self.assertTrue(np.allclose(meta["point_on_link"], [1.1, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
