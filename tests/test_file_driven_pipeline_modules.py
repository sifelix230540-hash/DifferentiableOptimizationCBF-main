import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYBULLET_DIR = ROOT / "CBF_experiment" / "active" / "pybullet"
GEOMETRY_MODULE_PATH = PYBULLET_DIR / "geometry_module.py"
TRAJECTORY_MODULE_PATH = PYBULLET_DIR / "trajectory_module.py"
SIMULATION_MODULE_PATH = PYBULLET_DIR / "simulation_module.py"
CBF_QP_MODULE_PATH = PYBULLET_DIR / "cbf_qp_module.py"
CONFIGURATION_METRICS_MODULE_PATH = PYBULLET_DIR / "configuration_metrics.py"
PATH_PLANNING_MODULE_PATH = PYBULLET_DIR / "path_planning_module.py"
CONFIGURATION_QUALITY_EXPERIMENT_PATH = PYBULLET_DIR / "configuration_quality_experiment.py"
SUPER_CONFIG_PATH = PYBULLET_DIR / "Super_config.json"


def load_module(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GeometryModuleContractTests(unittest.TestCase):
    def test_query_sdf_transforms_points_and_gradients_back_to_pybullet(self):
        module = load_module(GEOMETRY_MODULE_PATH, "geometry_module_under_test")

        class _FakeField:
            def query_with_gradient(self, points, kind="o3d_sdf"):
                pts = np.asarray(points, dtype=float).reshape(-1, 3)
                vals = pts[:, 0] + 10.0
                grads = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=float), (pts.shape[0], 1))
                return vals, grads

        engine = module.GeometryEngine.__new__(module.GeometryEngine)
        engine.field = _FakeField()
        engine.kind = "o3d_sdf"
        engine._wp_pos = np.array([1.0, 2.0, 3.0], dtype=float)
        engine._r_inv = np.eye(3, dtype=float)
        engine._pb_to_sdf_linear = np.eye(3, dtype=float)

        distances, normals_pb = module.GeometryEngine.query_sdf(
            engine,
            np.array([[4.0, 2.0, 3.0]], dtype=float),
        )

        np.testing.assert_allclose(distances, np.array([13.0], dtype=float))
        np.testing.assert_allclose(normals_pb, np.array([[1.0, 0.0, 0.0]], dtype=float))

    def test_get_cbf_distances_flattens_link_samples(self):
        module = load_module(GEOMETRY_MODULE_PATH, "geometry_module_under_test_flatten")

        class _DummyRobot:
            def __init__(self):
                self.last_q = None

        engine = module.GeometryEngine.__new__(module.GeometryEngine)
        engine.enabled = True
        engine.sample_link_surfaces = lambda robot, q: {
            3: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            4: np.array([[2.0, 0.0, 0.0]], dtype=float),
        }
        engine.query_sdf = lambda points_pb: (
            np.array([0.2, 0.3, 0.4], dtype=float),
            np.array([[1.0, 0.0, 0.0]] * 3, dtype=float),
        )

        distances, normals, link_indices = module.GeometryEngine.get_cbf_distances(
            engine,
            _DummyRobot(),
            np.zeros(9, dtype=float),
        )

        np.testing.assert_allclose(distances, np.array([0.2, 0.3, 0.4], dtype=float))
        np.testing.assert_allclose(normals, np.array([[1.0, 0.0, 0.0]] * 3, dtype=float))
        self.assertEqual(link_indices, [3, 3, 4])

    def test_init_local_surface_samples_uses_collision_surface_sampling_not_aabb(self):
        module = load_module(GEOMETRY_MODULE_PATH, "geometry_module_surface_sampling")

        class _DummyRobot:
            body_id = 17

            def __init__(self):
                self.q = None

            def set_joint_state(self, q):
                self.q = np.asarray(q, dtype=float).copy()

        engine = module.GeometryEngine.__new__(module.GeometryEngine)
        engine.robot = _DummyRobot()
        engine.link_indices = [3]
        engine.sample_density = 60.0
        engine.min_samples = 24
        engine.max_samples = 48
        engine.enabled = True
        engine.surface_source = "collision_preferred"
        engine.surface_fallback_to_visual = True
        engine._local_surface_samples = {}
        engine._local_surface_meshes = {}

        original_get_aabb = module.p.getAABB
        original_get_collision_shape_data = module.p.getCollisionShapeData
        original_get_visual_shape_data = module.p.getVisualShapeData
        original_get_link_state = module.p.getLinkState
        original_get_matrix = module.p.getMatrixFromQuaternion
        try:
            module.p.getAABB = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not use AABB"))
            module.p.getCollisionShapeData = lambda body_id, link_index: [
                (body_id, link_index, module.p.GEOM_BOX, (1.0, 1.0, 1.0), "", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            ]
            module.p.getVisualShapeData = lambda body_id: []
            module.p.getLinkState = lambda body_id, link_index, computeForwardKinematics=True: (
                None, None, None, None, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
            )
            module.p.getMatrixFromQuaternion = lambda quat: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

            module.GeometryEngine._init_local_surface_samples(engine, engine.robot, np.zeros(9, dtype=float))
            samples = module.GeometryEngine.sample_link_surfaces(engine, engine.robot, np.zeros(9, dtype=float))
        finally:
            module.p.getAABB = original_get_aabb
            module.p.getCollisionShapeData = original_get_collision_shape_data
            module.p.getVisualShapeData = original_get_visual_shape_data
            module.p.getLinkState = original_get_link_state
            module.p.getMatrixFromQuaternion = original_get_matrix

        self.assertIn(3, samples)
        self.assertGreater(samples[3].shape[0], 0)
        self.assertTrue(np.all(np.any(np.isclose(np.abs(samples[3]), 0.5, atol=1e-5), axis=1)))

    def test_get_cbf_distances_includes_nonadjacent_self_collision_pairs(self):
        module = load_module(GEOMETRY_MODULE_PATH, "geometry_module_self_collision")

        class _DummyRobot:
            body_id = 17
            active_joints = [0, 1, 2, 3]

            def set_joint_state(self, q):
                return None

        engine = module.GeometryEngine.__new__(module.GeometryEngine)
        engine.enabled = True
        engine.self_collision_enabled = True
        engine.self_collision_min_index_gap = 2
        engine.self_collision_query_distance = 0.2
        engine.top_k_per_link = 0
        engine.last_query_points = np.zeros((0, 3), dtype=float)
        engine.last_query_meta = []
        engine._build_external_cbf_entries = lambda robot, q: []

        original_get_closest_points = module.p.getClosestPoints
        try:
            def _fake_closest_points(body_a, body_b, max_dist, linkIndexA, linkIndexB):
                if (int(linkIndexA), int(linkIndexB)) == (0, 2):
                    return [(None, None, None, None, None, [1.0, 0.0, 0.0], [0.7, 0.0, 0.0], [1.0, 0.0, 0.0], 0.3)]
                return []

            module.p.getClosestPoints = _fake_closest_points
            distances, normals, link_indices = module.GeometryEngine.get_cbf_distances(
                engine,
                _DummyRobot(),
                np.zeros(4, dtype=float),
            )
        finally:
            module.p.getClosestPoints = original_get_closest_points

        self.assertEqual(link_indices, [0])
        np.testing.assert_allclose(distances, np.array([0.3], dtype=float))
        np.testing.assert_allclose(normals, np.array([[1.0, 0.0, 0.0]], dtype=float))
        self.assertEqual(engine.last_query_meta[0]["kind"], "self_collision")
        self.assertEqual(engine.last_query_meta[0]["other_link_index"], 2)
        np.testing.assert_allclose(engine.last_query_meta[0]["point_on_other_link"], [0.7, 0.0, 0.0])


class ConfigurationMetricsContractTests(unittest.TestCase):
    def test_compute_manipulability_report_supports_motion_components(self):
        module = load_module(CONFIGURATION_METRICS_MODULE_PATH, "configuration_metrics_motion_components")

        jacobian = np.diag([3.0, 2.0, 1.0, 9.0, 8.0, 7.0]).astype(float)
        linear = module.compute_manipulability_report(jacobian, motion_component="linear")
        combined = module.compute_manipulability_report(jacobian, motion_component="combined")

        self.assertAlmostEqual(linear["manipulability"], 6.0)
        self.assertAlmostEqual(linear["min_singular_value"], 1.0)
        self.assertAlmostEqual(linear["inverse_condition"], 1.0 / 3.0)
        self.assertGreater(combined["manipulability"], linear["manipulability"])

    def test_compute_joint_limit_margin_penalizes_near_limit_configs(self):
        module = load_module(CONFIGURATION_METRICS_MODULE_PATH, "configuration_metrics_joint_limit")

        centered = module.compute_joint_limit_margin(
            np.array([0.0, 1.0], dtype=float),
            [(-1.0, 1.0), (0.0, 2.0)],
        )
        near_limit = module.compute_joint_limit_margin(
            np.array([1.0, 0.1], dtype=float),
            [(-1.0, 1.0), (0.0, 2.0)],
        )

        self.assertAlmostEqual(centered["min_margin"], 1.0)
        self.assertAlmostEqual(centered["mean_margin"], 1.0)
        self.assertAlmostEqual(near_limit["min_margin"], 0.0)
        self.assertLess(near_limit["mean_margin"], centered["mean_margin"])

    def test_rank_configuration_records_combines_metrics_and_compactness(self):
        module = load_module(CONFIGURATION_METRICS_MODULE_PATH, "configuration_metrics_ranking")

        ranked = module.rank_configuration_records([
            {
                "tag": "balanced",
                "aabb_max_dim": 0.40,
                "inverse_condition": 0.60,
                "self_collision_distance": 0.10,
                "joint_limit_margin": 0.70,
            },
            {
                "tag": "compact_but_singular",
                "aabb_max_dim": 0.20,
                "inverse_condition": 0.05,
                "self_collision_distance": 0.04,
                "joint_limit_margin": 0.10,
            },
        ])

        self.assertEqual(ranked[0]["tag"], "balanced")
        self.assertGreater(ranked[0]["selection_score"], ranked[1]["selection_score"])

    def test_summarize_clearance_entries_splits_self_and_environment(self):
        module = load_module(CONFIGURATION_METRICS_MODULE_PATH, "configuration_metrics_clearance")

        summary = module.summarize_clearance_entries([
            {"kind": "environment", "distance": 0.20},
            {"kind": "self_collision", "distance": 0.05},
            {"kind": "environment", "distance": 0.08},
        ])

        self.assertAlmostEqual(summary["min_distance"], 0.05)
        self.assertAlmostEqual(summary["environment_distance"], 0.08)
        self.assertAlmostEqual(summary["self_collision_distance"], 0.05)
        self.assertEqual(summary["environment_count"], 2)
        self.assertEqual(summary["self_collision_count"], 1)


class TrajectoryModuleContractTests(unittest.TestCase):
    def test_chomp_optimizer_improves_clearance_while_fixing_endpoints(self):
        module = load_module(TRAJECTORY_MODULE_PATH, "trajectory_module_chomp_under_test")

        class _FakeRobot:
            dof = 2

            def __init__(self):
                self.q = np.zeros(2, dtype=float)

            def set_joint_state(self, q, dq=None):
                self.q = np.asarray(q, dtype=float).copy()

        class _FakeGeometry:
            def get_cbf_distances(self, robot, q):
                q = np.asarray(q, dtype=float).reshape(-1)
                center = np.array([0.0, 0.0], dtype=float)
                delta = q[:2] - center
                dist = float(np.linalg.norm(delta))
                if dist < 1e-9:
                    normal = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    normal = np.array([delta[0] / dist, delta[1] / dist, 0.0], dtype=float)
                return np.array([dist], dtype=float), normal.reshape(1, 3), [0]

        q_refs = np.array([
            [-1.0, 0.0],
            [-0.5, 0.05],
            [0.0, 0.05],
            [0.5, 0.05],
            [1.0, 0.0],
        ], dtype=float)
        cfg = {
            "trajectory_planning": {
                "use_chomp_optimizer": True,
                "chomp_iters": 20,
                "chomp_step_size": 0.15,
                "chomp_smooth_weight": 0.2,
                "chomp_obstacle_weight": 2.0,
                "chomp_clearance_margin": 0.4,
                "chomp_fd_eps": 1e-3,
            }
        }

        q_opt, report = module._chomp_optimize_joint_refs(
            q_refs,
            robot=_FakeRobot(),
            geometry=_FakeGeometry(),
            cfg=cfg,
        )

        np.testing.assert_allclose(q_opt[0], q_refs[0])
        np.testing.assert_allclose(q_opt[-1], q_refs[-1])
        self.assertGreater(report["clearance_after"], report["clearance_before"])
        self.assertGreater(abs(float(q_opt[2, 1])), abs(float(q_refs[2, 1])))

    def test_run_writes_joint_trajectory_json(self):
        module = load_module(TRAJECTORY_MODULE_PATH, "trajectory_module_under_test")

        class _FakeRobot:
            n_pris = 3
            n_revo = 6
            dof = 9
            total_dof = 9

            def __init__(self, cfg):
                self.dt = float(cfg["simulation"]["dt"])
                self.q = np.zeros(9, dtype=float)

            def get_joint_state(self):
                return self.q.copy(), np.zeros_like(self.q)

            def set_joint_state(self, q, dq=None):
                self.q = np.asarray(q, dtype=float).copy()

            def get_ee_pose(self):
                pos = np.array([self.q[0], self.q[1], self.q[2]], dtype=float)
                quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
                return pos, quat

            def get_robobase_pose(self):
                return np.array([self.q[0], self.q[1], self.q[2]], dtype=float), np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

            def get_ee_jacobian(self, q, dq):
                jac = np.zeros((6, 9), dtype=float)
                jac[:3, :3] = np.eye(3, dtype=float)
                jac[3:, 3:6] = np.eye(3, dtype=float)
                return jac

        class _FakeWorkpiece:
            def __init__(self, cfg):
                self.cfg = cfg

        class _FakeGeometryEngine:
            def __init__(self, cfg, robot):
                self.cfg = cfg
                self.robot = robot
                self.last_query_meta = [
                    {"kind": "environment", "distance": 0.5},
                    {"kind": "self_collision", "distance": 0.25},
                ]

            def get_cbf_distances(self, robot, q):
                return (
                    np.full(4, 0.5, dtype=float),
                    np.tile(np.array([[1.0, 0.0, 0.0]], dtype=float), (4, 1)),
                    [3, 3, 4, 4],
                )

        def _fake_solver(robot, q, dq, *, dt=None, pos_ref=None, quat_ref=None, q_ref=None, geometry_engine=None, cfg=None):
            u = np.zeros(robot.dof, dtype=float)
            if q_ref is not None:
                q_ref = np.asarray(q_ref, dtype=float).reshape(-1)
                u[:] = np.clip(q_ref - np.asarray(q, dtype=float), -0.2, 0.2)
            else:
                pos_now, _ = robot.get_ee_pose()
                err = np.asarray(pos_ref, dtype=float) - np.asarray(pos_now, dtype=float)
                u[:3] = np.clip(err, -0.2, 0.2)
            return u, {
                "status": "cbf_optimal",
                "min_h": 0.5,
                "cbf_active": False,
                "clearance_summary": {
                    "min_distance": 0.25,
                    "environment_distance": 0.5,
                    "self_collision_distance": 0.25,
                },
            }

        original_robot = module.Robot
        original_workpiece = module.Workpiece
        original_geometry = module.GeometryEngine
        original_solver = module.solve_cbf_qp_step
        try:
            module.Robot = _FakeRobot
            module.Workpiece = _FakeWorkpiece
            module.GeometryEngine = _FakeGeometryEngine
            module.solve_cbf_qp_step = _fake_solver

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = pathlib.Path(tmpdir)
                cfg = json.loads(SUPER_CONFIG_PATH.read_text(encoding="utf-8"))
                cfg["trajectory_planning"] = {
                    "input_json": str(tmpdir_path / "plan_poses.json"),
                    "output_json": str(tmpdir_path / "joint_trajectory.json"),
                    "segment_durations": {"weld": 0.02},
                    "tracking_gain_pos": 1.0,
                    "tracking_gain_ori": 1.0,
                    "cbf_alpha": 1.0,
                }
                cfg["geometry"] = {
                    "sdf_npz": "dummy.npz",
                    "sdf_kind": "o3d_sdf",
                    "cbf_links": "auto",
                    "sample_density": 32.0,
                    "min_samples_per_link": 8,
                    "max_samples_per_link": 32,
                }
                cfg["configuration_quality"] = {
                    "track_during_trajectory": True,
                }
                cfg_path = tmpdir_path / "config.json"
                cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

                plan_poses = {
                    "segments": [
                        {
                            "name": "weld",
                            "motion_type": "焊接",
                            "n_points": 2,
                            "positions": [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]],
                            "quaternions": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                        }
                    ]
                }
                (tmpdir_path / "plan_poses.json").write_text(
                    json.dumps(plan_poses, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                payload = module.run(cfg_path=cfg_path)

                self.assertEqual(payload["n_segments"], 1)
                self.assertGreater(payload["n_steps"], 0)
                self.assertEqual(payload["steps"][0]["segment_name"], "weld")
                self.assertIn("configuration_quality", payload["steps"][0])
                self.assertIn("inverse_condition", payload["steps"][0]["configuration_quality"])
                self.assertIn("configuration_quality", payload["segments"][0])
                self.assertIn("min_inverse_condition", payload["segments"][0]["configuration_quality"])
                self.assertTrue((tmpdir_path / "joint_trajectory.json").exists())
        finally:
            module.Robot = original_robot
            module.Workpiece = original_workpiece
            module.GeometryEngine = original_geometry
            module.solve_cbf_qp_step = original_solver

    def test_run_can_disable_configuration_quality_logging(self):
        module = load_module(TRAJECTORY_MODULE_PATH, "trajectory_module_quality_toggle")

        class _FakeRobot:
            n_pris = 3
            n_revo = 6
            dof = 9
            total_dof = 9

            def __init__(self, cfg):
                self.q = np.zeros(9, dtype=float)

            def get_joint_state(self):
                return self.q.copy(), np.zeros_like(self.q)

            def set_joint_state(self, q, dq=None):
                self.q = np.asarray(q, dtype=float).copy()

            def get_ee_pose(self):
                return np.asarray(self.q[:3], dtype=float), np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)

            def get_robobase_pose(self):
                return np.asarray(self.q[:3], dtype=float), np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)

            def get_ee_jacobian(self, q, dq):
                jac = np.zeros((6, 9), dtype=float)
                jac[:3, :3] = np.eye(3, dtype=float)
                jac[3:, 3:6] = np.eye(3, dtype=float)
                return jac

        class _FakeWorkpiece:
            def __init__(self, cfg):
                self.cfg = cfg

        class _FakeGeometryEngine:
            def __init__(self, cfg, robot):
                self.last_query_meta = []

            def get_cbf_distances(self, robot, q):
                return np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float), []

        def _fake_solver(robot, q, dq, *, dt=None, pos_ref=None, quat_ref=None, q_ref=None, geometry_engine=None, cfg=None):
            return np.zeros(robot.dof, dtype=float), {"status": "cbf_optimal", "min_h": 0.0, "cbf_active": False}

        original_robot = module.Robot
        original_workpiece = module.Workpiece
        original_geometry = module.GeometryEngine
        original_solver = module.solve_cbf_qp_step
        original_quality = module._compute_configuration_quality
        try:
            module.Robot = _FakeRobot
            module.Workpiece = _FakeWorkpiece
            module.GeometryEngine = _FakeGeometryEngine
            module.solve_cbf_qp_step = _fake_solver
            module._compute_configuration_quality = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("quality logging should be disabled"))

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = pathlib.Path(tmpdir)
                cfg = json.loads(SUPER_CONFIG_PATH.read_text(encoding="utf-8"))
                cfg["trajectory_planning"] = {
                    "input_json": str(tmpdir_path / "plan_poses.json"),
                    "output_json": str(tmpdir_path / "joint_trajectory.json"),
                    "segment_durations": {"weld": 0.02},
                }
                cfg["geometry"] = {
                    "sdf_npz": "dummy.npz",
                    "sdf_kind": "o3d_sdf",
                }
                cfg["configuration_quality"] = {
                    "track_during_trajectory": False,
                }
                cfg_path = tmpdir_path / "config.json"
                cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
                (tmpdir_path / "plan_poses.json").write_text(
                    json.dumps({
                        "segments": [{
                            "name": "weld",
                            "motion_type": "焊接",
                            "n_points": 2,
                            "positions": [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]],
                            "quaternions": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                        }]
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                payload = module.run(cfg_path=cfg_path)

                self.assertNotIn("configuration_quality", payload["steps"][0])
                self.assertNotIn("configuration_quality", payload["segments"][0])
        finally:
            module.Robot = original_robot
            module.Workpiece = original_workpiece
            module.GeometryEngine = original_geometry
            module.solve_cbf_qp_step = original_solver
            module._compute_configuration_quality = original_quality


class PathPlanningQualityContractTests(unittest.TestCase):
    def test_rank_init_config_candidates_prefers_balanced_quality(self):
        module = load_module(PATH_PLANNING_MODULE_PATH, "path_planning_quality_ranking")

        ranked = module._rank_init_config_candidates([
            {
                "tag": "balanced",
                "aabb_max_dim": 0.40,
                "configuration_quality": {
                    "inverse_condition": 0.60,
                    "self_collision_distance": 0.12,
                    "joint_limit_margin": 0.70,
                },
            },
            {
                "tag": "compact_but_risky",
                "aabb_max_dim": 0.20,
                "configuration_quality": {
                    "inverse_condition": 0.04,
                    "self_collision_distance": 0.03,
                    "joint_limit_margin": 0.10,
                },
            },
        ])

        self.assertEqual(ranked[0]["tag"], "balanced")
        self.assertGreater(ranked[0]["selection_score"], ranked[1]["selection_score"])


class SimulationReplayContractTests(unittest.TestCase):
    def test_robot_point_jacobian_uses_world_point_offset(self):
        module = load_module(SIMULATION_MODULE_PATH, "simulation_module_point_jacobian")
        robot = module.Robot.__new__(module.Robot)
        robot.body_id = 17

        captured = {}
        original_get_link_state = module.p.getLinkState
        original_invert = module.p.invertTransform
        original_multiply = module.p.multiplyTransforms
        original_jacobian = module.p.calculateJacobian
        try:
            module.p.getLinkState = lambda body_id, link_index, computeForwardKinematics=True: (
                None, None, None, None, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]
            )
            module.p.invertTransform = lambda pos, quat: ([-1.0, -2.0, -3.0], [0.0, 0.0, 0.0, 1.0])
            module.p.multiplyTransforms = lambda inv_pos, inv_quat, world_point, ident_quat: (
                [world_point[0] - 1.0, world_point[1] - 2.0, world_point[2] - 3.0],
                [0.0, 0.0, 0.0, 1.0],
            )

            def _fake_calculate_jacobian(body_id, link_index, local_point, q, dq, zeros):
                captured["local_point"] = list(local_point)
                return [[1.0], [2.0], [3.0]], [[0.0], [0.0], [0.0]]

            module.p.calculateJacobian = _fake_calculate_jacobian

            jt = module.Robot.get_link_linear_jacobian_at_world_point(
                robot,
                3,
                np.array([1.25, 2.5, 3.75], dtype=float),
                np.array([0.0], dtype=float),
                np.array([0.0], dtype=float),
            )
        finally:
            module.p.getLinkState = original_get_link_state
            module.p.invertTransform = original_invert
            module.p.multiplyTransforms = original_multiply
            module.p.calculateJacobian = original_jacobian

        self.assertEqual(captured["local_point"], [0.25, 0.5, 0.75])
        np.testing.assert_allclose(jt, np.array([[1.0], [2.0], [3.0]], dtype=float))

    def test_replay_reads_joint_trajectory_and_applies_joint_states(self):
        module = load_module(SIMULATION_MODULE_PATH, "simulation_module_replay_under_test")

        captured = {"states": []}

        class _FakeScene:
            def __init__(self, cfg):
                self.cfg = cfg

            def enable_rendering(self):
                return None

            @staticmethod
            def draw_frame(pos, quat, length=0.10, width=3.0, replace_ids=None):
                return [1, 2, 3]

            @staticmethod
            def create_marker(radius, color, pos):
                return 1

        class _FakeRobot:
            def __init__(self, cfg):
                self.q = np.zeros(9, dtype=float)

            def set_joint_state(self, q, dq=None):
                captured["states"].append(np.asarray(q, dtype=float).copy())
                self.q = np.asarray(q, dtype=float).copy()

            def get_ee_pose(self):
                return np.asarray([0.0, 0.0, 0.0], dtype=float), np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)

        class _FakeWorkpiece:
            def __init__(self, cfg):
                self.cfg = cfg

            def get_frame_pose(self, link_name):
                return np.asarray([0.0, 0.0, 0.0], dtype=float), np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)

        original_scene = module.SimulationScene
        original_robot = module.Robot
        original_workpiece = module.Workpiece
        original_sleep = module.time.sleep
        try:
            module.SimulationScene = _FakeScene
            module.Robot = _FakeRobot
            module.Workpiece = _FakeWorkpiece
            module.time.sleep = lambda *_args, **_kwargs: None

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = pathlib.Path(tmpdir)
                cfg = json.loads(SUPER_CONFIG_PATH.read_text(encoding="utf-8"))
                cfg["trajectory_planning"] = {
                    "input_json": "unused.json",
                    "output_json": str(tmpdir_path / "joint_trajectory.json"),
                    "segment_durations": {"weld": 0.1},
                    "tracking_gain_pos": 1.0,
                    "tracking_gain_ori": 1.0,
                    "cbf_alpha": 1.0,
                }
                cfg["geometry"] = {
                    "sdf_npz": "dummy.npz",
                    "sdf_kind": "o3d_sdf",
                    "cbf_links": "auto",
                    "sample_density": 32.0,
                    "min_samples_per_link": 8,
                    "max_samples_per_link": 32,
                }
                cfg_path = tmpdir_path / "config.json"
                cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

                joint_payload = {
                    "dt": 0.01,
                    "n_steps": 2,
                    "steps": [
                        {"q": [0.0] * 9, "dq": [0.0] * 9, "segment_name": "weld"},
                        {"q": [0.1] * 9, "dq": [0.0] * 9, "segment_name": "weld"},
                    ],
                }
                joint_path = tmpdir_path / "joint_trajectory.json"
                joint_path.write_text(json.dumps(joint_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                module.replay(cfg_path=cfg_path, trajectory_json=joint_path, gui=False, realtime=False)

                self.assertEqual(len(captured["states"]), 2)
                np.testing.assert_allclose(captured["states"][-1], np.full(9, 0.1, dtype=float))
        finally:
            module.SimulationScene = original_scene
            module.Robot = original_robot
            module.Workpiece = original_workpiece
            module.time.sleep = original_sleep


class CbfQPModuleContractTests(unittest.TestCase):
    def test_build_cbf_rows_uses_point_jacobian_at_sample_points(self):
        module = load_module(CBF_QP_MODULE_PATH, "cbf_qp_module_under_test")

        class _DummyRobot:
            dof = 2

            def __init__(self):
                self.queried = []

            def get_link_linear_jacobian_at_world_point(self, link_index, world_point, q, dq):
                point = np.asarray(world_point, dtype=float).reshape(3)
                self.queried.append((int(link_index), point.copy()))
                return np.array([
                    [point[0], 0.0],
                    [0.0, point[1]],
                    [0.0, 0.0],
                ], dtype=float)

        class _DummyGeometry:
            def __init__(self):
                self.last_query_points = np.array([
                    [1.0, 2.0, 0.0],
                    [3.0, 4.0, 0.0],
                ], dtype=float)

            def get_cbf_distances(self, robot, q):
                return (
                    np.array([0.2, 0.3], dtype=float),
                    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                    [3, 4],
                )

        robot = _DummyRobot()
        geometry = _DummyGeometry()
        rows, rhs = module._build_cbf_rows(
            robot,
            np.zeros(2, dtype=float),
            geometry,
            {"control": {"safety_margin": 0.05}, "trajectory_planning": {"cbf_alpha": 2.0}},
        )

        self.assertEqual(len(robot.queried), 2)
        self.assertEqual(robot.queried[0][0], 3)
        self.assertEqual(robot.queried[1][0], 4)
        np.testing.assert_allclose(robot.queried[0][1], [1.0, 2.0, 0.0])
        np.testing.assert_allclose(robot.queried[1][1], [3.0, 4.0, 0.0])
        np.testing.assert_allclose(rows, np.array([[1.0, 0.0], [0.0, 4.0]], dtype=float))
        np.testing.assert_allclose(rhs, np.array([0.3, 0.5], dtype=float))

    def test_build_cbf_rows_uses_relative_jacobian_for_self_collision(self):
        module = load_module(CBF_QP_MODULE_PATH, "cbf_qp_module_self_under_test")

        class _DummyRobot:
            dof = 2

            def get_link_linear_jacobian_at_world_point(self, link_index, world_point, q, dq):
                if int(link_index) == 1:
                    return np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)
                return np.array([[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)

        class _DummyGeometry:
            def __init__(self):
                self.last_query_points = np.array([[1.0, 0.0, 0.0]], dtype=float)
                self.last_query_meta = [{
                    "kind": "self_collision",
                    "link_index": 1,
                    "other_link_index": 3,
                    "point_on_link": np.array([1.0, 0.0, 0.0], dtype=float),
                    "point_on_other_link": np.array([0.5, 0.0, 0.0], dtype=float),
                }]

            def get_cbf_distances(self, robot, q):
                return (
                    np.array([0.02], dtype=float),
                    np.array([[1.0, 0.0, 0.0]], dtype=float),
                    [1],
                )

        rows, rhs = module._build_cbf_rows(
            _DummyRobot(),
            np.zeros(2, dtype=float),
            _DummyGeometry(),
            {"control": {"safety_margin": 0.005, "self_collision_margin": 0.01}, "trajectory_planning": {"cbf_alpha": 2.0}},
        )

        np.testing.assert_allclose(rows, np.array([[1.5, 0.0]], dtype=float))
        np.testing.assert_allclose(rhs, np.array([0.02], dtype=float))

    def test_solve_cbf_qp_step_scales_joint_nominal_speed(self):
        module = load_module(CBF_QP_MODULE_PATH, "cbf_qp_module_speed_scale")

        class _DummyRobot:
            n_pris = 0
            n_revo = 2
            dof = 2

        u_cmd, info = module.solve_cbf_qp_step(
            _DummyRobot(),
            np.array([0.0, 0.0], dtype=float),
            np.array([0.0, 0.0], dtype=float),
            dt=0.5,
            q_ref=np.array([1.0, -2.0], dtype=float),
            geometry_engine=None,
            cfg={
                "robot": {"dq_limit": 10.0, "base_vel_limit": 10.0},
                "trajectory_planning": {"joint_nominal_scale": 0.25},
            },
        )

        np.testing.assert_allclose(u_cmd, np.array([0.5, -1.0], dtype=float), atol=1e-6)
        self.assertEqual(info["status"], "cbf_optimal")

    def test_solve_cbf_qp_step_reports_clearance_summary(self):
        module = load_module(CBF_QP_MODULE_PATH, "cbf_qp_module_clearance_summary")

        class _DummyRobot:
            n_pris = 0
            n_revo = 2
            dof = 2

        class _DummyGeometry:
            last_query_meta = [
                {"kind": "environment", "distance": 0.30},
                {"kind": "self_collision", "distance": 0.12},
            ]

            def get_cbf_distances(self, robot, q):
                return np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float), []

        _u_cmd, info = module.solve_cbf_qp_step(
            _DummyRobot(),
            np.array([0.0, 0.0], dtype=float),
            np.array([0.0, 0.0], dtype=float),
            dt=0.5,
            q_ref=np.array([0.0, 0.0], dtype=float),
            geometry_engine=_DummyGeometry(),
            cfg={
                "robot": {"dq_limit": 10.0, "base_vel_limit": 10.0},
                "trajectory_planning": {"joint_nominal_scale": 0.25},
            },
        )

        self.assertIn("clearance_summary", info)
        self.assertAlmostEqual(info["clearance_summary"]["environment_distance"], 0.30)
        self.assertAlmostEqual(info["clearance_summary"]["self_collision_distance"], 0.12)


class ConfigurationQualityExperimentContractTests(unittest.TestCase):
    def test_sample_joint_neighborhood_respects_limits(self):
        module = load_module(CONFIGURATION_QUALITY_EXPERIMENT_PATH, "configuration_quality_experiment_sampling")

        samples = module.sample_joint_neighborhood(
            np.array([0.0, 0.0], dtype=float),
            [(-1.0, 1.0), (-0.5, 0.5)],
            sigma=2.0,
            n_samples=32,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(samples.shape, (32, 2))
        self.assertTrue(np.all(samples[:, 0] >= -1.0))
        self.assertTrue(np.all(samples[:, 0] <= 1.0))
        self.assertTrue(np.all(samples[:, 1] >= -0.5))
        self.assertTrue(np.all(samples[:, 1] <= 0.5))


class SuperConfigContractTests(unittest.TestCase):
    def test_super_config_contains_geometry_and_trajectory_planning_sections(self):
        cfg = json.loads(SUPER_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertIn("geometry", cfg)
        self.assertIn("trajectory_planning", cfg)
        self.assertIn("configuration_quality", cfg)
        self.assertIn("sdf_npz", cfg["geometry"])
        self.assertIn("output_json", cfg["trajectory_planning"])
        self.assertIn("use_chomp_optimizer", cfg["trajectory_planning"])
        self.assertIn("selection_weights", cfg["configuration_quality"])


if __name__ == "__main__":
    unittest.main()
