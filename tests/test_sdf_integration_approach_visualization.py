import unittest
from types import SimpleNamespace

import numpy as np

from CBF_experiment.active.pybullet.sdf_integration_experiments import ExperimentParameters, PlannerExperiment


class _FakeRunner:
    def query_field(self, field, points, kind="auto", safe_oob=True):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        vals = np.full(pts.shape[0], 0.2, dtype=float)
        if pts.shape[0] >= 5 and np.allclose(pts[:, 1], 0.0):
            vals[pts.shape[0] // 2] = -0.05
        return vals


class SmoothEePathsVisualizationTests(unittest.TestCase):
    def test_forces_straight_approach_line_and_keeps_c1_join(self):
        planner = PlannerExperiment(_FakeRunner(), settings=None)
        ee_path = np.array([
            [0.0, 2.0, 0.0],
            [0.5, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ], dtype=float)
        retreat_path = np.array([
            [0.0, 0.0, 0.0],
            [0.6, 0.6, 0.0],
            [1.2, 0.0, 0.0],
        ], dtype=float)

        result = planner._smooth_ee_paths(
            field=object(),
            kind="o3d_sdf",
            ee_path_sdf_arr=ee_path,
            retreat_path_sdf_arr=retreat_path,
            ee_clearance=0.05,
            approach_min_sdf=0.05,
            n_samples_per_seg=5,
        )

        expected_candidate = np.array([
            [1.2, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=float)

        np.testing.assert_allclose(result["approach_smoothed_sdf"], expected_candidate)
        np.testing.assert_allclose(result["approach_smoothed_d"], np.array([0.2, 0.2, -0.05, 0.2, 0.2]))
        np.testing.assert_allclose(
            result["approach_controls_sdf"],
            np.vstack([expected_candidate[0], expected_candidate[-1]]),
        )
        np.testing.assert_allclose(result["approach_line_sdf"], expected_candidate)
        np.testing.assert_allclose(result["approach_line_d"], np.array([0.2, 0.2, -0.05, 0.2, 0.2]))
        self.assertFalse(result["approach_line_feasible"])
        expected_end_tangent = (expected_candidate[-1] - expected_candidate[0]) / 3.0
        np.testing.assert_allclose(
            result["ee_controls_sdf"][-1] - result["ee_controls_sdf"][-2],
            expected_end_tangent,
        )

    def test_plan_escape_bundle_returns_retreat_and_return_artifacts(self):
        class _BundleRunner(_FakeRunner):
            @staticmethod
            def pb2sdf(points):
                return np.asarray(points, dtype=float)

            @staticmethod
            def sdf2pb(points):
                return np.asarray(points, dtype=float)

        class _FakePlanner(PlannerExperiment):
            def _gradient_backtrack_escape(self, *args, **kwargs):
                return (
                    [
                        np.array([1.0, 0.0, 0.0], dtype=float),
                        np.array([2.0, 0.0, 0.0], dtype=float),
                    ],
                    np.array([0.01, 0.10], dtype=float),
                    np.array([2.0, 0.0, 0.0], dtype=float),
                )

            def _rrt_star_plan(self, **kwargs):
                return [
                    np.array([3.0, 0.0, 0.0], dtype=float),
                    np.array([4.0, 0.0, 0.0], dtype=float),
                ]

            def _shortcut_smooth(self, path, **kwargs):
                return list(path)

            def _resample_path(self, path, spacing):
                return list(path)

            def _smooth_ee_paths(self, *args, **kwargs):
                return {
                    "ee_smoothed_sdf": np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
                    "ee_smoothed_d": np.array([0.2, 0.3], dtype=float),
                    "ee_controls_sdf": np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
                    "approach_smoothed_sdf": np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float),
                    "approach_smoothed_d": np.array([0.1, 0.2], dtype=float),
                    "approach_controls_sdf": np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float),
                    "approach_line_sdf": np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float),
                    "approach_line_d": np.array([0.1, 0.2], dtype=float),
                    "approach_line_feasible": True,
                }

        planner = _FakePlanner(_BundleRunner(), settings=None)
        args = SimpleNamespace(
            ee_target_sdf=ExperimentParameters.PLAN_EE_TARGET_SDF,
            ee_backtrack_init_step=ExperimentParameters.PLAN_EE_BACKTRACK_INIT_STEP,
            ee_backtrack_min_step=ExperimentParameters.PLAN_EE_BACKTRACK_MIN_STEP,
            ee_backtrack_shrink=ExperimentParameters.PLAN_EE_BACKTRACK_SHRINK,
            ee_backtrack_max_iters=ExperimentParameters.PLAN_EE_BACKTRACK_MAX_ITERS,
            ee_backtrack_curv_eps=ExperimentParameters.PLAN_EE_BACKTRACK_CURV_EPS,
            ee_backtrack_armijo_c1=ExperimentParameters.PLAN_EE_BACKTRACK_ARMIJO_C1,
            bound_margin=ExperimentParameters.PLAN_BOUND_MARGIN,
            auto_fix_endpoints=False,
            endpoint_fix_radius=ExperimentParameters.PLAN_ENDPOINT_FIX_RADIUS,
            endpoint_fix_step=ExperimentParameters.PLAN_ENDPOINT_FIX_STEP,
            ee_min_clearance=ExperimentParameters.PLAN_EE_MIN_CLEARANCE,
            ee_step_size=ExperimentParameters.PLAN_EE_STEP_SIZE,
            ee_near_radius=ExperimentParameters.PLAN_EE_NEAR_RADIUS,
            goal_sample_prob=ExperimentParameters.PLAN_GOAL_SAMPLE_PROB,
            ee_max_iter=ExperimentParameters.PLAN_EE_MAX_ITER,
            ee_edge_check_step=ExperimentParameters.PLAN_EE_EDGE_CHECK_STEP,
            ee_goal_tolerance=ExperimentParameters.PLAN_EE_GOAL_TOLERANCE,
            smooth_iters=ExperimentParameters.PLAN_SMOOTH_ITERS,
            ee_resample_spacing=ExperimentParameters.PLAN_EE_RESAMPLE_SPACING,
            ee_bezier_approach_min_sdf=ExperimentParameters.PLAN_EE_BEZIER_APPROACH_MIN_SDF,
            ee_bezier_samples_per_seg=ExperimentParameters.PLAN_EE_BEZIER_SAMPLES_PER_SEG,
        )

        result = planner._plan_escape_bundle(
            field=object(),
            kind="o3d_sdf",
            args=args,
            weld_point_pb=np.array([1.0, 0.0, 0.0], dtype=float),
            ee_start_pb=np.array([3.0, 0.0, 0.0], dtype=float),
            field_bbox_min=np.array([-10.0, -10.0, -10.0], dtype=float),
            field_bbox_max=np.array([10.0, 10.0, 10.0], dtype=float),
        )

        np.testing.assert_allclose(
            result["retreat_path_pb"],
            np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
        )
        np.testing.assert_allclose(result["retreat_goal_pb"], np.array([2.0, 0.0, 0.0], dtype=float))
        np.testing.assert_allclose(
            result["ee_path_pb"],
            np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
        )
        np.testing.assert_allclose(
            result["ee_smoothed_pb"],
            np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
        )
        np.testing.assert_allclose(
            result["approach_line_pb"],
            np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float),
        )
        self.assertTrue(result["approach_line_feasible"])


if __name__ == "__main__":
    unittest.main()
