import unittest

import numpy as np
import pybullet as p

from CBF_experiment.active.pybullet.welding_320_geometry import (
    apply_contains_sign_to_distance,
    compute_surface_sample_count,
    compute_world_surface,
    find_closest_surface_pair_cpu,
    resolve_surface_sampling_params,
    resolve_surface_visual_max_points,
    sample_cloud_for_visualization,
)
from CBF_experiment.active.pybullet.welding_320_common import ExperimentConfig


class SurfaceGeometryTransformTests(unittest.TestCase):
    def test_compute_world_surface_applies_pose_to_points_and_normals(self):
        local_points = np.array([[1.0, 0.0, 0.0]], dtype=float)
        local_normals = np.array([[0.0, 1.0, 0.0]], dtype=float)
        world_quat = np.array(p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2.0]), dtype=float)

        world_points, world_normals = compute_world_surface(
            local_points,
            local_normals,
            world_pos=np.array([1.0, 2.0, 3.0], dtype=float),
            world_quat=world_quat,
        )

        self.assertTrue(np.allclose(world_points[0], [1.0, 3.0, 3.0], atol=1e-6))
        self.assertTrue(np.allclose(world_normals[0], [-1.0, 0.0, 0.0], atol=1e-6))


class SurfaceGeometryQueryTests(unittest.TestCase):
    def test_find_closest_surface_pair_cpu_returns_expected_points_normals_and_distance(self):
        robot_points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
        robot_normals = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
        obstacle_points = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        obstacle_normals = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)

        result = find_closest_surface_pair_cpu(
            robot_points,
            robot_normals,
            obstacle_points,
            obstacle_normals,
        )

        self.assertTrue(np.allclose(result["point_on_link"], [1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(result["point_on_obstacle"], [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(result["normal_on_link"], [-1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(result["normal_on_obstacle"], [1.0, 0.0, 0.0]))
        self.assertAlmostEqual(result["signed_dist"], 1.0, places=6)

    def test_find_closest_surface_pair_cpu_can_report_negative_signed_distance(self):
        robot_points = np.array([[0.2, 0.0, 0.0]], dtype=float)
        robot_normals = np.array([[-1.0, 0.0, 0.0]], dtype=float)
        obstacle_points = np.array([[0.5, 0.0, 0.0]], dtype=float)
        obstacle_normals = np.array([[1.0, 0.0, 0.0]], dtype=float)

        result = find_closest_surface_pair_cpu(
            robot_points,
            robot_normals,
            obstacle_points,
            obstacle_normals,
        )

        self.assertLess(result["signed_dist"], 0.0)
        self.assertAlmostEqual(result["euclidean_dist"], 0.3, places=6)

    def test_apply_contains_sign_to_distance_marks_outside_point_positive(self):
        import trimesh

        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        signed_dist = apply_contains_sign_to_distance(
            mesh=mesh,
            point_local=np.array([1.0, 0.0, 0.0], dtype=float),
            unsigned_dist=0.25,
            fallback_signed_dist=-0.25,
        )

        self.assertAlmostEqual(signed_dist, 0.25, places=6)

    def test_apply_contains_sign_to_distance_marks_inside_point_negative(self):
        import trimesh

        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        signed_dist = apply_contains_sign_to_distance(
            mesh=mesh,
            point_local=np.array([0.0, 0.0, 0.0], dtype=float),
            unsigned_dist=0.25,
            fallback_signed_dist=0.25,
        )

        self.assertAlmostEqual(signed_dist, -0.25, places=6)


class SurfaceVisualizationSamplingTests(unittest.TestCase):
    def test_sample_cloud_for_visualization_limits_point_count(self):
        points = np.array([[float(i), 0.0, 0.0] for i in range(10)], dtype=float)
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (10, 1))

        sampled_points, sampled_normals = sample_cloud_for_visualization(points, normals, max_points=4)

        self.assertEqual(sampled_points.shape, (4, 3))
        self.assertEqual(sampled_normals.shape, (4, 3))
        self.assertTrue(np.allclose(sampled_normals, [[0.0, 0.0, 1.0]] * 4))
        self.assertTrue(np.allclose(sampled_points[:, 0], [0.0, 3.0, 6.0, 9.0]))

    def test_obstacle_role_uses_higher_sampling_and_visual_density(self):
        cfg = ExperimentConfig()

        robot_density, robot_min_samples, robot_max_samples = resolve_surface_sampling_params(cfg, role="robot")
        obstacle_density, obstacle_min_samples, obstacle_max_samples = resolve_surface_sampling_params(cfg, role="obstacle")

        self.assertGreater(obstacle_density, robot_density)
        self.assertGreater(obstacle_min_samples, robot_min_samples)
        self.assertGreater(obstacle_max_samples, robot_max_samples)
        self.assertGreater(resolve_surface_visual_max_points(cfg, role="obstacle"), resolve_surface_visual_max_points(cfg, role="robot"))

    def test_robot_rear_six_role_uses_higher_sampling_and_visual_density_than_robot(self):
        cfg = ExperimentConfig()

        robot_density, robot_min_samples, robot_max_samples = resolve_surface_sampling_params(cfg, role="robot")
        rear_density, rear_min_samples, rear_max_samples = resolve_surface_sampling_params(cfg, role="robot_rear_six")

        self.assertGreater(rear_density, robot_density)
        self.assertGreater(rear_min_samples, robot_min_samples)
        self.assertGreater(rear_max_samples, robot_max_samples)
        self.assertGreater(resolve_surface_visual_max_points(cfg, role="robot_rear_six"), resolve_surface_visual_max_points(cfg, role="robot"))

    def test_compute_surface_sample_count_respects_density_and_limits(self):
        robot_count = compute_surface_sample_count(area=1.0, density=300.0, min_samples=96, max_samples=768)
        obstacle_count = compute_surface_sample_count(area=1.0, density=1200.0, min_samples=256, max_samples=4096)

        self.assertEqual(robot_count, 300)
        self.assertEqual(obstacle_count, 1200)


if __name__ == "__main__":
    unittest.main()
