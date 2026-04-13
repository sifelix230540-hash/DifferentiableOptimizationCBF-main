import importlib.util
import inspect
import json
import pathlib
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CBF_experiment" / "active" / "pybullet" / "3_20_9axis_3d_mpc_cbf_experiment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cbf_exp_320_pipeline", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _DummyRobot:
    total_dof = 8
    n_pris = 2
    n_revo = 6


class PipelineContractTests(unittest.TestCase):
    def test_module_only_exposes_main_as_top_level_function(self):
        module = load_module()
        function_names = sorted(
            name for name, value in vars(module).items()
            if inspect.isfunction(value) and getattr(value, "__module__", None) == module.__name__
        )
        self.assertEqual(function_names, ["main"])

    def test_module_exposes_pipeline_bundle_dataclasses(self):
        module = load_module()

        for name in (
            "InputSpec",
            "GeometryBundle",
            "PathPlanBundle",
            "TimedTrajectoryBundle",
            "SimulationResultBundle",
            "PipelineRuntime",
        ):
            self.assertTrue(hasattr(module, name), name)

    def test_build_path_plan_bundle_prefers_explicit_post_weld_and_return_segments(self):
        module = load_module()
        input_spec = module.InputSpec(
            weld_points=[
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([2.0, 0.0, 0.0], dtype=float),
            ],
            weld_quats=[
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ],
            initial_pose=(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            robot_urdf_path="robot.urdf",
            robot_mesh_paths=["robot.stl"],
            workpiece_urdf_path="workpiece.urdf",
            workpiece_mesh_paths=["workpiece.stl"],
        )
        payload = {
            "ee_bezier_path": [
                [0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0],
            ],
            "approach_line_path": [
                [0.6, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "approach_end_line_path": [
                [2.0, 0.0, 0.0],
                [2.4, 0.2, 0.0],
            ],
            "ee_path_return": [
                [2.4, 0.2, 0.0],
                [0.8, 0.1, 0.0],
                [0.0, 0.0, 0.0],
            ],
            "robot_q_at_goal": [0.0] * 8,
        }

        bundle = module.PipelineRuntime.build_path_plan_bundle(payload, input_spec)

        np.testing.assert_allclose(
            np.asarray(bundle.pre_weld_path_pb, dtype=float),
            np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        )
        np.testing.assert_allclose(
            np.asarray(bundle.weld_path_pb, dtype=float),
            np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
        )
        np.testing.assert_allclose(
            np.asarray(bundle.post_weld_path_pb, dtype=float),
            np.array([[2.0, 0.0, 0.0], [2.4, 0.2, 0.0]], dtype=float),
        )
        np.testing.assert_allclose(
            np.asarray(bundle.return_path_pb, dtype=float),
            np.array([[2.4, 0.2, 0.0], [0.8, 0.1, 0.0], [0.0, 0.0, 0.0]], dtype=float),
        )

    def test_build_path_plan_bundle_falls_back_to_reversed_segments_without_return_artifacts(self):
        module = load_module()
        input_spec = module.InputSpec(
            weld_points=[
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([2.0, 0.0, 0.0], dtype=float),
            ],
            weld_quats=[
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ],
            initial_pose=(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            robot_urdf_path="robot.urdf",
            robot_mesh_paths=["robot.stl"],
            workpiece_urdf_path="workpiece.urdf",
            workpiece_mesh_paths=["workpiece.stl"],
        )
        payload = {
            "ee_bezier_path": [
                [0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0],
            ],
            "approach_line_path": [
                [0.6, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "robot_q_at_goal": [0.0] * 8,
        }

        bundle = module.PipelineRuntime.build_path_plan_bundle(payload, input_spec)

        np.testing.assert_allclose(
            np.asarray(bundle.post_weld_path_pb, dtype=float),
            np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        )
        np.testing.assert_allclose(
            np.asarray(bundle.return_path_pb, dtype=float),
            np.array([[1.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
        )

    def test_build_timed_trajectory_bundle_returns_four_stage_progress_trajectory(self):
        module = load_module()
        cfg = module.ExperimentConfig()
        cfg.retreat_duration = 4.0
        input_spec = module.InputSpec(
            weld_points=[
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([2.0, 0.0, 0.0], dtype=float),
            ],
            weld_quats=[
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ],
            initial_pose=(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            robot_urdf_path="robot.urdf",
            robot_mesh_paths=["robot.stl"],
            workpiece_urdf_path="workpiece.urdf",
            workpiece_mesh_paths=["workpiece.stl"],
        )
        path_bundle = module.PathPlanBundle(
            init_config=np.zeros(8, dtype=float),
            via_point_pb=None,
            retreat_point_pb=np.array([0.6, 0.0, 0.0], dtype=float),
            pre_weld_path_pb=[np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
            weld_path_pb=[np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])],
            post_weld_path_pb=[np.array([2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
            return_path_pb=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
            robot_q_seeds={"goal": np.zeros(8, dtype=float)},
            raw_payload={},
        )

        timed = module.PipelineRuntime.build_timed_trajectory_bundle(cfg, input_spec, path_bundle)

        self.assertIsInstance(timed.trajectory, module.PathProgressTrajectory)
        self.assertEqual(len(timed.base_trajectory.segments), 4)
        self.assertEqual(
            [seg.planner_status for seg in timed.base_trajectory.segments],
            ["sdf_pre_weld", "weld_nominal", "sdf_post_weld", "sdf_return_home"],
        )
        self.assertEqual(timed.base_trajectory.segments[2].duration, 4.0)

    def test_create_tracking_controller_supports_dual_modes(self):
        module = load_module()
        cfg = module.ExperimentConfig()
        cfg.controller_mode = "cbf_qp"

        controller = module.PipelineRuntime.create_tracking_controller(_DummyRobot(), cfg, trajectory=None)

        self.assertEqual(type(controller).__name__, "CBFQPController")

    def test_pipeline_orchestrator_runs_modules_in_fixed_order(self):
        module = load_module()
        input_spec = module.InputSpec(
            weld_points=[
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([2.0, 0.0, 0.0], dtype=float),
            ],
            weld_quats=[
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ],
            initial_pose=(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            robot_urdf_path="robot.urdf",
            robot_mesh_paths=["robot.stl"],
            workpiece_urdf_path="workpiece.urdf",
            workpiece_mesh_paths=["workpiece.stl"],
        )
        geometry_bundle = module.GeometryBundle(
            aligned_sdf="sdf",
            occupancy_field="occ",
            pb_to_sdf="pb2sdf",
            sdf_to_pb="sdf2pb",
            robot_surface_samples={},
            alignment_report={},
        )
        path_bundle = module.PathPlanBundle(
            init_config=np.zeros(8, dtype=float),
            via_point_pb=None,
            retreat_point_pb=np.array([0.6, 0.0, 0.0], dtype=float),
            pre_weld_path_pb=[np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
            weld_path_pb=[np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])],
            post_weld_path_pb=[np.array([2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
            return_path_pb=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
            robot_q_seeds={},
            raw_payload={},
        )
        timed_bundle = module.TimedTrajectoryBundle(
            base_trajectory="base",
            trajectory="progress",
            segment_metadata=[],
            reference_pose_stream=[],
            reference_twist_stream=[],
            timing_report={},
        )
        sim_result = module.SimulationResultBundle(
            state_log=[],
            tracking_metrics={},
            safety_metrics={},
            render_artifacts={},
            optimizer_diagnostics={},
        )
        order = []

        def geometry_builder(cfg, spec):
            order.append("geometry")
            self.assertIs(spec, input_spec)
            return geometry_bundle

        def path_planner(cfg, spec, geom):
            order.append("path")
            self.assertIs(spec, input_spec)
            self.assertIs(geom, geometry_bundle)
            return path_bundle

        def trajectory_builder(cfg, spec, plan):
            order.append("trajectory")
            self.assertIs(plan, path_bundle)
            return timed_bundle

        def simulation_runner(cfg, spec, geom, plan, timed):
            order.append("simulation")
            self.assertIs(geom, geometry_bundle)
            self.assertIs(plan, path_bundle)
            self.assertIs(timed, timed_bundle)
            return sim_result

        orchestrator = module.PipelineOrchestrator(
            module.ExperimentConfig(),
            module.PipelineModules(
                geometry_builder=geometry_builder,
                path_planner=path_planner,
                trajectory_builder=trajectory_builder,
                simulation_runner=simulation_runner,
            ),
        )

        result = orchestrator.run(input_spec)

        self.assertIs(result, sim_result)
        self.assertEqual(order, ["geometry", "path", "trajectory", "simulation"])

    def test_default_path_planner_prefers_existing_artifact(self):
        module = load_module()
        cfg = module.ExperimentConfig()
        input_spec = module.InputSpec(
            weld_points=[np.array([1.0, 0.0, 0.0], dtype=float)],
            weld_quats=[np.array([0.0, 0.0, 0.0, 1.0], dtype=float)],
            initial_pose=(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            robot_urdf_path="robot.urdf",
            robot_mesh_paths=["robot.stl"],
            workpiece_urdf_path="workpiece.urdf",
            workpiece_mesh_paths=["workpiece.stl"],
        )
        geometry_bundle = module.GeometryBundle(
            aligned_sdf=None,
            occupancy_field=None,
            pb_to_sdf=None,
            sdf_to_pb=None,
            robot_surface_samples={},
            alignment_report={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plan_json = pathlib.Path(tmpdir) / "plan.json"
            plan_json.write_text(json.dumps({
                "ee_bezier_path": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                "approach_line_path": [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "robot_q_at_goal": [0.0] * 8,
            }), encoding="utf-8")
            cfg.sdf_plan_json_path = str(plan_json)

            called = {"fallback": False}

            def fail_fallback(_cfg):
                called["fallback"] = True
                raise AssertionError("fallback planner should not run")

            original = module.PipelineRuntime.run_sdf_pipeline_fallback
            module.PipelineRuntime.run_sdf_pipeline_fallback = classmethod(lambda cls, _cfg: fail_fallback(_cfg))
            try:
                bundle = module.PipelineRuntime.default_path_planner(cfg, input_spec, geometry_bundle)
            finally:
                module.PipelineRuntime.run_sdf_pipeline_fallback = original

        self.assertFalse(called["fallback"])
        self.assertEqual(len(bundle.pre_weld_path_pb), 3)

    def test_default_geometry_builder_exports_robot_surface_samples(self):
        module = load_module()
        cfg = module.ExperimentConfig()
        input_spec = module.InputSpec(
            weld_points=[np.array([1.0, 0.0, 0.0], dtype=float)],
            weld_quats=[np.array([0.0, 0.0, 0.0, 1.0], dtype=float)],
            initial_pose=(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            robot_urdf_path="robot.urdf",
            robot_mesh_paths=["robot.stl"],
            workpiece_urdf_path="workpiece.urdf",
            workpiece_mesh_paths=["workpiece.stl"],
        )

        class _DummyRunner:
            pb2sdf = staticmethod(lambda pts: np.asarray(pts, dtype=float))
            sdf2pb = staticmethod(lambda pts: np.asarray(pts, dtype=float))

            def load_field(self, _path):
                return object()

        class _DummySdfModule:
            class ExperimentParameters:
                DEFAULT_SDF_NPZ = "dummy.npz"
                ALIGN_OUTPUT_JSON = "dummy.json"

            ExperimentRunner = _DummyRunner

        class _DummyRobot:
            body_id = 1

            def __init__(self, config, scene=None):
                self.config = config

            def get_surface_local_samples(self, link_indices=None):
                return {
                    3: {
                        "link_index": 3,
                        "link_name": "sampled_link",
                        "local_points": np.array([[0.0, 0.0, 0.0]], dtype=float),
                        "local_normals": np.array([[1.0, 0.0, 0.0]], dtype=float),
                        "role": "robot",
                    }
                }

        original_module = module.PipelineRuntime.get_sdf_pipeline_module
        original_robot = module.JakaRobot
        module.PipelineRuntime.get_sdf_pipeline_module = classmethod(lambda cls: _DummySdfModule)
        module.JakaRobot = _DummyRobot
        try:
            bundle = module.PipelineRuntime.default_geometry_builder(cfg, input_spec)
        finally:
            module.PipelineRuntime.get_sdf_pipeline_module = original_module
            module.JakaRobot = original_robot

        self.assertIn(3, bundle.robot_surface_samples)
        self.assertTrue(np.allclose(bundle.robot_surface_samples[3]["local_points"], [[0.0, 0.0, 0.0]]))

    def test_default_simulation_runner_passes_injected_pipeline_objects(self):
        module = load_module()
        cfg = module.ExperimentConfig()
        cfg.controller_mode = "mpc_dcbf"
        input_spec = module.InputSpec(
            weld_points=[
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([2.0, 0.0, 0.0], dtype=float),
            ],
            weld_quats=[
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ],
            initial_pose=(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            robot_urdf_path="robot.urdf",
            robot_mesh_paths=["robot.stl"],
            workpiece_urdf_path="workpiece.urdf",
            workpiece_mesh_paths=["workpiece.stl"],
        )
        class _DummyField:
            def query_with_gradient(self, points, kind="auto"):
                return np.array([0.1], dtype=float), np.array([[1.0, 0.0, 0.0]], dtype=float)

        geometry_bundle = module.GeometryBundle(_DummyField(), None, lambda pts: np.asarray(pts, dtype=float), None, {}, {})
        path_bundle = module.PathPlanBundle(
            init_config=np.arange(8, dtype=float),
            via_point_pb=None,
            retreat_point_pb=np.array([0.6, 0.0, 0.0], dtype=float),
            pre_weld_path_pb=[np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
            weld_path_pb=[np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])],
            post_weld_path_pb=[np.array([2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
            return_path_pb=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
            robot_q_seeds={"goal": np.zeros(8, dtype=float)},
            raw_payload={},
        )
        timed_bundle = module.TimedTrajectoryBundle(
            base_trajectory="base",
            trajectory="progress",
            segment_metadata=[],
            reference_pose_stream=[],
            reference_twist_stream=[],
            timing_report={},
        )
        captured = {}

        class FakeExperiment:
            def __init__(self, config, **kwargs):
                captured["config"] = config
                captured["kwargs"] = kwargs

            def run(self):
                return "sim-result"

        module.AvoidanceExperiment = FakeExperiment

        result = module.PipelineRuntime.default_simulation_runner(cfg, input_spec, geometry_bundle, path_bundle, timed_bundle)

        self.assertEqual(result, "sim-result")
        self.assertIs(captured["kwargs"]["trajectory_override"], timed_bundle.trajectory)
        self.assertTrue(callable(captured["kwargs"]["controller_factory"]))
        np.testing.assert_allclose(captured["kwargs"]["joint_state_override"], path_bundle.init_config)
        self.assertEqual(len(captured["kwargs"]["obstacles_override"]), 1)
        self.assertEqual(type(captured["kwargs"]["obstacles_override"][0]).__name__, "SDFObstacleAdapter")
        self.assertTrue(cfg.use_sdf_cbf)
        self.assertFalse(cfg.use_mesh_cbf)
        self.assertFalse(cfg.obstacle_local_dense_enabled)

    def test_default_simulation_runner_resolves_auto_sdf_kind_before_building_obstacle(self):
        module = load_module()
        cfg = module.ExperimentConfig()
        cfg.controller_mode = "mpc_dcbf"
        cfg.sdf_cbf_kind = "auto"
        input_spec = module.InputSpec(
            weld_points=[
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([2.0, 0.0, 0.0], dtype=float),
            ],
            weld_quats=[
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ],
            initial_pose=(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            robot_urdf_path="robot.urdf",
            robot_mesh_paths=["robot.stl"],
            workpiece_urdf_path="workpiece.urdf",
            workpiece_mesh_paths=["workpiece.stl"],
        )

        class _DummyField:
            pass

        geometry_bundle = module.GeometryBundle(_DummyField(), None, lambda pts: np.asarray(pts, dtype=float), None, {}, {})
        path_bundle = module.PathPlanBundle(
            init_config=np.arange(8, dtype=float),
            via_point_pb=None,
            retreat_point_pb=np.array([0.6, 0.0, 0.0], dtype=float),
            pre_weld_path_pb=[np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
            weld_path_pb=[np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])],
            post_weld_path_pb=[np.array([2.0, 0.0, 0.0]), np.array([2.4, 0.2, 0.0])],
            return_path_pb=[np.array([2.4, 0.2, 0.0]), np.array([0.0, 0.0, 0.0])],
            robot_q_seeds={"goal": np.zeros(8, dtype=float)},
            raw_payload={},
        )
        timed_bundle = module.TimedTrajectoryBundle(
            base_trajectory="base",
            trajectory="progress",
            segment_metadata=[],
            reference_pose_stream=[],
            reference_twist_stream=[],
            timing_report={},
        )
        captured = {}

        class FakeExperiment:
            def __init__(self, config, **kwargs):
                captured["kwargs"] = kwargs

            def run(self):
                return "sim-result"

        class _DummyRunner:
            def resolve_kind(self, field, kind):
                self.last = (field, kind)
                return "o3d_sdf"

        class _DummySdfModule:
            class ExperimentRunner:
                def resolve_kind(self, field, kind):
                    return "o3d_sdf"

        original_exp = module.AvoidanceExperiment
        original_sdf_module = module.PipelineRuntime.get_sdf_pipeline_module
        module.AvoidanceExperiment = FakeExperiment
        module.PipelineRuntime.get_sdf_pipeline_module = classmethod(lambda cls: _DummySdfModule)
        try:
            result = module.PipelineRuntime.default_simulation_runner(
                cfg,
                input_spec,
                geometry_bundle,
                path_bundle,
                timed_bundle,
            )
        finally:
            module.AvoidanceExperiment = original_exp
            module.PipelineRuntime.get_sdf_pipeline_module = original_sdf_module

        self.assertEqual(result, "sim-result")
        self.assertEqual(captured["kwargs"]["obstacles_override"][0].kind, "o3d_sdf")

    def test_main_uses_default_orchestrator_entry(self):
        module = load_module()
        called = {}
        original = module.PipelineRuntime.run_default
        module.PipelineRuntime.run_default = classmethod(lambda cls, config=None: called.setdefault("ran", True) or "ok")
        try:
            module.main()
        finally:
            module.PipelineRuntime.run_default = original
        self.assertTrue(called.get("ran", False))

    def test_create_tracking_controller_passes_surface_samples_to_mpc(self):
        module = load_module()
        cfg = module.ExperimentConfig()
        cfg.controller_mode = "mpc_dcbf"
        captured = {}

        class _FakeController:
            def __init__(self, robot, config, trajectory, surface_samples=None):
                captured["surface_samples"] = surface_samples

        original = module.MPCDCBFController
        module.MPCDCBFController = _FakeController
        try:
            controller = module.PipelineRuntime.create_tracking_controller(
                _DummyRobot(),
                cfg,
                trajectory="traj",
                surface_samples={3: {"local_points": np.zeros((1, 3), dtype=float)}},
            )
        finally:
            module.MPCDCBFController = original

        self.assertIsNotNone(controller)
        self.assertIn(3, captured["surface_samples"])


if __name__ == "__main__":
    unittest.main()
