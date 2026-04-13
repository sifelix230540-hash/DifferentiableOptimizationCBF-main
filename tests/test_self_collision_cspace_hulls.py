import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CBF_experiment" / "active" / "pybullet" / "self_collision_cspace_hulls.py"


def load_module(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SelfCollisionCSpaceHullTests(unittest.TestCase):
    def test_normalize_joint_samples_maps_limits_to_unit_box(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hulls_normalize")
        samples = np.array([
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
        ], dtype=float)
        joint_limits = [(-1.0, 1.0), (0.0, 2.0)]

        normalized, lower, span = module.normalize_joint_samples(samples, joint_limits)

        np.testing.assert_allclose(lower, np.array([-1.0, 0.0], dtype=float))
        np.testing.assert_allclose(span, np.array([2.0, 2.0], dtype=float))
        np.testing.assert_allclose(normalized[0], np.array([0.0, 0.0], dtype=float))
        np.testing.assert_allclose(normalized[-1], np.array([1.0, 1.0], dtype=float))

    def test_cluster_samples_by_voxels_separates_disconnected_groups(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hulls_cluster")
        normalized = np.array([
            [0.10, 0.10],
            [0.12, 0.08],
            [0.78, 0.80],
            [0.82, 0.77],
        ], dtype=float)

        clusters = module.cluster_samples_by_voxels(normalized, voxel_size=0.10, min_cluster_size=1)

        cluster_sizes = sorted(len(cluster["sample_indices"]) for cluster in clusters)
        self.assertEqual(cluster_sizes, [2, 2])

    def test_fit_convex_hull_cluster_returns_halfspaces_that_classify_points(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hulls_fit")
        points = np.array([
            [0.10, 0.10],
            [0.30, 0.10],
            [0.30, 0.30],
            [0.10, 0.30],
            [0.20, 0.20],
        ], dtype=float)

        hull = module.fit_convex_hull_cluster(points)
        inside = module.max_halfspace_violation(np.array([[0.20, 0.20]], dtype=float), np.asarray(hull["equations"], dtype=float))
        outside = module.max_halfspace_violation(np.array([[0.45, 0.45]], dtype=float), np.asarray(hull["equations"], dtype=float))

        self.assertIn(hull["hull_type"], {"convex_hull", "aabb"})
        self.assertLessEqual(float(inside[0]), 1e-8)
        self.assertGreater(float(outside[0]), 1e-3)

    def test_build_collision_cspace_hulls_outputs_joint_space_equations(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hulls_build")
        samples = np.array([
            [-0.9, -0.9],
            [-0.7, -0.8],
            [-0.8, -0.6],
            [0.7, 0.7],
            [0.8, 0.9],
            [0.9, 0.8],
        ], dtype=float)
        joint_limits = [(-1.0, 1.0), (-1.0, 1.0)]

        payload = module.build_collision_cspace_hulls_from_samples(
            samples,
            joint_limits,
            voxel_size=0.15,
            min_cluster_size=2,
        )

        self.assertEqual(payload["dimension"], 2)
        self.assertEqual(len(payload["clusters"]), 2)
        for cluster in payload["clusters"]:
            self.assertIn("equations_joint", cluster)
            self.assertIn("equations_normalized", cluster)

    def test_progress_callback_receives_sampling_updates(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hulls_progress")
        events = []

        class _FakeRobot:
            def __init__(self):
                self.body_id = 1
                self.active_joints = [0, 1]
                self.revolute_joints = [0, 1]
                self.link_name_by_index = {0: "j1", 1: "j2"}
                self.q = np.zeros(2, dtype=float)

            def get_joint_state(self):
                return self.q.copy(), np.zeros_like(self.q)

            def set_joint_state(self, q, dq=None):
                self.q = np.asarray(q, dtype=float).copy()

        original_load_config = module.load_config
        original_robot = module.Robot
        original_resolve = module._resolve
        original_get_joint_info = module.p.getJointInfo
        original_connect = module.p.connect
        original_is_connected = module.p.isConnected
        original_disconnect = module.p.disconnect
        original_classifier = getattr(module, "classify_self_collision_sample", None)
        try:
            module.load_config = lambda _cfg_path=None: {}
            module.Robot = lambda cfg: _FakeRobot()
            module._resolve = lambda rel: rel
            module.p.getJointInfo = lambda body_id, joint_id: (None, None, None, None, None, None, None, None, -1.0, 1.0)
            module.p.connect = lambda mode: 0
            module.p.isConnected = lambda: False
            module.p.disconnect = lambda: None
            module.classify_self_collision_sample = lambda robot, **kwargs: {
                "is_collision": bool(robot.q[0] > 0.0),
                "min_distance": -0.01 if bool(robot.q[0] > 0.0) else 0.05,
                "active_pair": [0, 1] if bool(robot.q[0] > 0.0) else None,
            }

            payload = module.monte_carlo_self_collision_hulls(
                num_samples=6,
                seed=3,
                voxel_size=0.25,
                min_cluster_size=1,
                progress_callback=lambda info: events.append(dict(info)),
            )
        finally:
            module.load_config = original_load_config
            module.Robot = original_robot
            module._resolve = original_resolve
            module.p.getJointInfo = original_get_joint_info
            module.p.connect = original_connect
            module.p.isConnected = original_is_connected
            module.p.disconnect = original_disconnect
            if original_classifier is None:
                delattr(module, "classify_self_collision_sample")
            else:
                module.classify_self_collision_sample = original_classifier

        self.assertEqual(payload["num_samples"], 6)
        self.assertGreaterEqual(len(events), 3)
        self.assertEqual(events[0]["stage"], "sampling")
        self.assertEqual(events[-1]["stage"], "done")
        self.assertIn("collision_count", events[-1])

    def test_visualize_collision_cspace_hulls_writes_png(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hulls_visualize")
        payload = {
            "dimension": 2,
            "clusters": [
                {
                    "cluster_id": 0,
                    "num_samples": 3,
                    "sample_indices": [0, 1, 2],
                    "samples_normalized": [[0.1, 0.1], [0.2, 0.15], [0.15, 0.2]],
                    "aabb_normalized_min": [0.1, 0.1],
                    "aabb_normalized_max": [0.2, 0.2],
                },
                {
                    "cluster_id": 1,
                    "num_samples": 3,
                    "sample_indices": [3, 4, 5],
                    "samples_normalized": [[0.7, 0.75], [0.8, 0.8], [0.75, 0.7]],
                    "aabb_normalized_min": [0.7, 0.7],
                    "aabb_normalized_max": [0.8, 0.8],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = pathlib.Path(tmpdir) / "self_collision_hulls.png"
            returned = module.visualize_collision_cspace_hulls(payload, out_png)

            self.assertEqual(pathlib.Path(returned), out_png)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 0)

    def test_compute_pairwise_self_collision_distance_uses_only_selected_pairs(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hulls_pairs")

        class _DummyRobot:
            body_id = 17

        original_get_closest_points = module.p.getClosestPoints
        try:
            def _fake_get_closest_points(body_a, body_b, max_dist, linkIndexA, linkIndexB):
                pair = (int(linkIndexA), int(linkIndexB))
                if pair == (4, 7):
                    return [(None, None, None, int(linkIndexA), int(linkIndexB), None, None, None, -0.02)]
                if pair == (5, 8):
                    return [(None, None, None, int(linkIndexA), int(linkIndexB), None, None, None, 0.03)]
                return [(None, None, None, int(linkIndexA), int(linkIndexB), None, None, None, -0.50)]

            module.p.getClosestPoints = _fake_get_closest_points
            result = module.compute_pairwise_self_collision_distance(
                _DummyRobot(),
                monitored_pairs=[(4, 7), (5, 8)],
                query_distance=0.10,
            )
        finally:
            module.p.getClosestPoints = original_get_closest_points

        self.assertAlmostEqual(result["min_distance"], -0.02)
        self.assertEqual(tuple(result["active_pair"]), (4, 7))

    def test_collect_relevant_collision_samples_keeps_boundary_band(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hulls_boundary")
        q_samples = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
        ], dtype=float)
        metrics = [
            {"min_distance": -0.20, "is_collision": True, "active_pair": [4, 7]},
            {"min_distance": -0.01, "is_collision": True, "active_pair": [4, 7]},
            {"min_distance": 0.03, "is_collision": False, "active_pair": [5, 8]},
        ]

        selected = module.collect_relevant_collision_samples(
            q_samples,
            metrics,
            sample_mode="boundary_band",
            boundary_band=0.05,
        )

        self.assertEqual(selected.shape[0], 2)
        np.testing.assert_allclose(selected[0], q_samples[1])
        np.testing.assert_allclose(selected[1], q_samples[2])


if __name__ == "__main__":
    unittest.main()
