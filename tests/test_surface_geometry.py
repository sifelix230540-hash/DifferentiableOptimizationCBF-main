import unittest

import numpy as np
import pybullet as p

from CBF_experiment.active.pybullet.welding_320_geometry import (
    apply_contains_sign_to_distance,
    compute_surface_sample_count,
    compute_world_surface,
    extract_local_surface_roi,
    find_closest_surface_pair_cpu,
    resolve_surface_sampling_params,
    resolve_surface_visual_max_points,
    sample_local_dense_surface,
    sample_cloud_for_visualization,
    SurfaceDistanceEngine,
    SurfaceLinkCloud,
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

    def test_obstacle_local_dense_role_uses_higher_sampling_and_visual_density_than_obstacle(self):
        cfg = ExperimentConfig()

        obstacle_density, obstacle_min_samples, obstacle_max_samples = resolve_surface_sampling_params(cfg, role="obstacle")
        local_density, local_min_samples, local_max_samples = resolve_surface_sampling_params(cfg, role="obstacle_local_dense")

        self.assertGreater(local_density, obstacle_density)
        self.assertGreater(local_min_samples, obstacle_min_samples)
        self.assertGreater(local_max_samples, obstacle_max_samples)
        self.assertGreater(
            resolve_surface_visual_max_points(cfg, role="obstacle_local_dense"),
            resolve_surface_visual_max_points(cfg, role="obstacle"),
        )

    def test_compute_surface_sample_count_respects_density_and_limits(self):
        robot_count = compute_surface_sample_count(area=1.0, density=300.0, min_samples=96, max_samples=768)
        obstacle_count = compute_surface_sample_count(area=1.0, density=1200.0, min_samples=256, max_samples=4096)

        self.assertEqual(robot_count, 300)
        self.assertEqual(obstacle_count, 1200)

    def test_extract_local_surface_roi_keeps_only_faces_near_query_center(self):
        import trimesh

        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        roi_mesh = extract_local_surface_roi(
            mesh,
            world_pos=np.zeros(3, dtype=float),
            world_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            center_world=np.array([0.49, 0.0, 0.0], dtype=float),
            radius=0.25,
        )

        self.assertIsNotNone(roi_mesh)
        self.assertTrue(np.all(roi_mesh.triangles_center[:, 0] > 0.0))

    def test_sample_local_dense_surface_focuses_samples_on_selected_roi(self):
        import trimesh

        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        points, normals = sample_local_dense_surface(
            mesh=mesh,
            world_pos=np.zeros(3, dtype=float),
            world_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            center_world=np.array([0.49, 0.0, 0.0], dtype=float),
            radius=0.25,
            density=400.0,
            min_samples=32,
            max_samples=128,
        )

        self.assertGreater(points.shape[0], 0)
        self.assertTrue(np.allclose(points[:, 0], 0.5, atol=1e-6))
        self.assertTrue(np.allclose(normals[:, 0], 1.0, atol=1e-6))


class SurfaceLocalDensePreferenceTests(unittest.TestCase):
    def test_visualization_prefers_local_dense_obstacle_cloud(self):
        cfg = ExperimentConfig()
        engine = SurfaceDistanceEngine(cfg)
        engine._body_clouds = {
            7: {
                0: SurfaceLinkCloud(
                    body_id=7,
                    link_index=0,
                    link_name="obstacle_link",
                    local_points=np.zeros((1, 3), dtype=float),
                    local_normals=np.zeros((1, 3), dtype=float),
                    role="obstacle",
                )
            }
        }
        engine._body_roles = {7: "obstacle"}
        engine._get_local_dense_world_cloud = lambda *args, **kwargs: {
            "body_id": 7,
            "link_index": 0,
            "link_name": "obstacle_link",
            "points": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            "normals": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            "local_mesh": None,
            "world_pos": np.zeros(3, dtype=float),
            "world_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        }
        engine._get_world_cloud = lambda body_id, link_index: {
            "body_id": int(body_id),
            "link_index": int(link_index),
            "link_name": "obstacle_link",
            "points": np.array([[9.0, 0.0, 0.0]], dtype=float),
            "normals": np.array([[1.0, 0.0, 0.0]], dtype=float),
            "local_mesh": None,
            "world_pos": np.zeros(3, dtype=float),
            "world_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        }

        clouds = engine.get_visualization_clouds(7, query_center_world=np.zeros(3, dtype=float))

        self.assertEqual(len(clouds), 1)
        self.assertTrue(np.allclose(clouds[0]["points"][0], [1.0, 0.0, 0.0]))

    def test_query_prefers_local_dense_obstacle_cloud(self):
        cfg = ExperimentConfig()
        engine = SurfaceDistanceEngine(cfg)
        engine._body_clouds = {
            1: {
                0: SurfaceLinkCloud(
                    body_id=1,
                    link_index=0,
                    link_name="robot_link",
                    local_points=np.zeros((1, 3), dtype=float),
                    local_normals=np.zeros((1, 3), dtype=float),
                    role="robot",
                )
            },
            7: {
                0: SurfaceLinkCloud(
                    body_id=7,
                    link_index=0,
                    link_name="obstacle_link",
                    local_points=np.zeros((1, 3), dtype=float),
                    local_normals=np.zeros((1, 3), dtype=float),
                    role="obstacle",
                )
            },
        }
        engine._body_roles = {1: "robot", 7: "obstacle"}
        engine._get_world_cloud = lambda body_id, link_index: {
            "body_id": int(body_id),
            "link_index": int(link_index),
            "link_name": "robot_link" if int(body_id) == 1 else "obstacle_link",
            "points": np.array([[0.0, 0.0, 0.0]], dtype=float) if int(body_id) == 1 else np.array([[5.0, 0.0, 0.0]], dtype=float),
            "normals": np.array([[1.0, 0.0, 0.0]], dtype=float),
            "local_mesh": None,
            "world_pos": np.zeros(3, dtype=float),
            "world_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            "device_points": None,
            "device_normals": None,
        }
        engine._get_local_dense_world_cloud = lambda *args, **kwargs: {
            "body_id": 7,
            "link_index": 0,
            "link_name": "obstacle_link",
            "points": np.array([[0.2, 0.0, 0.0]], dtype=float),
            "normals": np.array([[-1.0, 0.0, 0.0]], dtype=float),
            "local_mesh": None,
            "world_pos": np.zeros(3, dtype=float),
            "world_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            "device_points": None,
            "device_normals": None,
        }

        result = engine.query_link_to_body(
            robot_body_id=1,
            robot_link_index=0,
            obstacle_body_id=7,
            query_center_world=np.zeros(3, dtype=float),
        )

        self.assertIsNotNone(result)
        self.assertTrue(np.allclose(result["point_on_obstacle"], [0.2, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
