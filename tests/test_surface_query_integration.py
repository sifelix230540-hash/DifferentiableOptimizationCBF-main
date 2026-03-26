import unittest

import numpy as np

from CBF_experiment.active.pybullet.welding_320_robot import JakaRobot


class _FakeSurfaceEngine:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def query_link_to_body(self, robot_body_id, robot_link_index, obstacle_body_id, obstacle_link_indices=None, max_dist=None):
        self.calls.append({
            "robot_body_id": robot_body_id,
            "robot_link_index": robot_link_index,
            "obstacle_body_id": obstacle_body_id,
            "obstacle_link_indices": obstacle_link_indices,
            "max_dist": max_dist,
        })
        return self.result


class SurfaceQueryIntegrationTests(unittest.TestCase):
    def test_robot_closest_point_prefers_surface_engine(self):
        robot = JakaRobot.__new__(JakaRobot)
        robot.body_id = 42
        robot._surface_engine = _FakeSurfaceEngine({
            "robot_link_index": 3,
            "robot_link_name": "welding_gun_base",
            "obs_link_index": 1,
            "obs_link_name": "l3",
            "point_on_link": np.array([1.0, 0.0, 0.0]),
            "point_on_obstacle": np.array([0.5, 0.0, 0.0]),
            "normal_on_link": np.array([-1.0, 0.0, 0.0]),
            "normal_on_obstacle": np.array([1.0, 0.0, 0.0]),
            "signed_dist": 0.5,
            "euclidean_dist": 0.5,
        })
        robot._surface_obstacle_links = {7: [0, 1]}

        result = JakaRobot.get_closest_points_to_obstacle(robot, 3, 7, max_dist=0.8)

        self.assertEqual(robot._surface_engine.calls[0]["robot_link_index"], 3)
        self.assertEqual(robot._surface_engine.calls[0]["obstacle_link_indices"], [0, 1])
        self.assertAlmostEqual(result["signed_dist"], 0.5)
        self.assertTrue(np.allclose(result["normal_on_obstacle"], [1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
