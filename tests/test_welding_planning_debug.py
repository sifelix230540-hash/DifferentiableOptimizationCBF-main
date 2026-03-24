import pathlib
import subprocess
import sys
import tempfile
import unittest
import inspect

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBF_experiment.active.welding_320_common import ExperimentConfig
from CBF_experiment.active.welding_320_control import (
    CartesianRRTNominalPlanner,
    PlannerDiagnostics,
    JointSpaceNominalPlanner,
)
from CBF_experiment.active.welding_320_corridor import JointSpaceCorridor, PyBulletModelSnapshot
from CBF_experiment.active.welding_320_experiment import AvoidanceExperiment, build_pre_approach_pose
from CBF_experiment.active.welding_320_ik import MultiSeedIKSolver


class FakeIKRobot:
    def __init__(self):
        self.current_q = np.zeros(2)
        self.dof = 2

    def get_joint_state(self):
        return self.current_q.copy(), np.zeros_like(self.current_q)

    def set_joint_state(self, q, dq=None):
        self.current_q = np.array(q, dtype=float)

    def calculate_ik(self, target_pos, target_quat, rest_poses=None):
        rest = np.zeros(2) if rest_poses is None else np.array(rest_poses, dtype=float)
        return rest

    def get_active_joint_limits(self):
        return np.array([-1.0, -1.0]), np.array([1.0, 1.0])

    def evaluate_pose_candidate(self, q, target_pos, target_quat):
        q = np.array(q, dtype=float)
        return {
            "position_error": float(abs(q[0] - target_pos[0])),
            "orientation_error": float(abs(q[1])),
        }

    def is_state_collision_free(self, q, **_kwargs):
        q = np.array(q, dtype=float)
        return bool(q[0] <= 0.6)


class PlannerDiagnosticsTests(unittest.TestCase):
    def test_default_debug_scene_keeps_collisions_enabled(self):
        cfg = ExperimentConfig()

        self.assertFalse(cfg.ignore_all_collisions)
        self.assertLess(cfg.workpiece_position[1], 1.0)

    def test_avoidance_experiment_no_longer_disables_all_robot_workpiece_pairs(self):
        source = inspect.getsource(AvoidanceExperiment.__init__)

        self.assertNotIn("disable_collision_with(", source)

    def test_planner_diagnostics_accumulate_counts(self):
        diag = PlannerDiagnostics()
        diag.record_ik_attempt(0.01, success=True)
        diag.record_ik_attempt(0.08, success=False)
        diag.record_collision_check(is_collision=False)
        diag.record_collision_check(is_collision=True)

        self.assertEqual(diag.ik_calls, 2)
        self.assertEqual(diag.ik_failures, 1)
        self.assertEqual(diag.collision_checks, 2)
        self.assertEqual(diag.collision_failures, 1)
        self.assertAlmostEqual(diag.max_fk_error, 0.08)
        self.assertAlmostEqual(diag.mean_fk_error, 0.045)

    def test_seeded_cartesian_sampling_is_reproducible(self):
        cfg = ExperimentConfig(planner_seed=17)

        planner_a = object.__new__(CartesianRRTNominalPlanner)
        planner_a.config = cfg
        planner_a._rng = np.random.default_rng(cfg.planner_seed)

        planner_b = object.__new__(CartesianRRTNominalPlanner)
        planner_b.config = cfg
        planner_b._rng = np.random.default_rng(cfg.planner_seed)

        lower = np.array([-1.0, -2.0, -3.0])
        upper = np.array([1.0, 2.0, 3.0])

        sample_a = planner_a._sample_cartesian_point(lower, upper)
        sample_b = planner_b._sample_cartesian_point(lower, upper)
        np.testing.assert_allclose(sample_a, sample_b)


class MultiSeedIKSolverTests(unittest.TestCase):
    def test_solver_returns_sorted_unique_candidates(self):
        robot = FakeIKRobot()
        cfg = ExperimentConfig(ik_random_seeds=0, ik_max_candidates=3, ik_position_tolerance=0.5)
        solver = MultiSeedIKSolver(robot, cfg)

        candidates = solver.solve(
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            reference_q=np.array([0.0, 0.0]),
            extra_seed_qs=[
                np.array([0.4, 0.0]),
                np.array([0.4, 0.0]),
                np.array([0.9, 0.0]),
                np.array([0.1, 0.0]),
            ],
        )

        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(c.is_collision_free for c in candidates))
        self.assertLessEqual(candidates[0].position_error, candidates[-1].position_error)


class JointPlannerTests(unittest.TestCase):
    def test_joint_planner_prefers_nearest_collision_free_goal(self):
        planner = object.__new__(JointSpaceNominalPlanner)
        current_q = np.array([0.0, 0.0])
        goal_candidates = [
            np.array([1.2, 0.0]),
            np.array([0.3, 0.0]),
            np.array([0.8, 0.0]),
        ]
        is_free = lambda q: float(q[0]) < 1.0

        chosen = planner._select_goal_candidate(current_q, goal_candidates, is_free)
        np.testing.assert_allclose(chosen, np.array([0.3, 0.0]))


class PreApproachTests(unittest.TestCase):
    def test_build_pre_approach_pose_offsets_against_weld_direction(self):
        pos = np.array([1.0, 2.0, 3.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        pre_pos, pre_quat = build_pre_approach_pose(
            pos,
            quat,
            np.array([0.0, 0.0, -1.0]),
            retract_distance=0.2,
            lift_distance=0.1,
        )

        np.testing.assert_allclose(pre_pos, np.array([1.0, 2.0, 3.3]))
        np.testing.assert_allclose(pre_quat, quat)


class CorridorPrepTests(unittest.TestCase):
    def test_joint_space_corridor_contains_halfspace_points(self):
        corridor = JointSpaceCorridor(
            A=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
            b=np.array([1.0, 1.0, 1.0, 1.0]),
            source="unit-test",
        )

        self.assertTrue(corridor.contains(np.array([0.2, -0.5])))
        self.assertFalse(corridor.contains(np.array([1.2, 0.0])))

    def test_model_snapshot_reports_same_signature_for_same_inputs(self):
        snap_a = PyBulletModelSnapshot(
            robot_urdf="robot.urdf",
            workpiece_urdf="workpiece.urdf",
            active_joint_names=("j1", "j2"),
            active_joint_limits=((0.0, 1.0), (-1.0, 1.0)),
        )
        snap_b = PyBulletModelSnapshot(
            robot_urdf="robot.urdf",
            workpiece_urdf="workpiece.urdf",
            active_joint_names=("j1", "j2"),
            active_joint_limits=((0.0, 1.0), (-1.0, 1.0)),
        )

        self.assertEqual(snap_a.signature(), snap_b.signature())


class DirectScriptBootstrapTests(unittest.TestCase):
    def test_welding_experiment_script_can_be_loaded_outside_repo_root(self):
        script_path = ROOT / "CBF_experiment" / "active" / "welding_320_experiment.py"
        cmd = [
            sys.executable,
            "-c",
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout}\nstderr={result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
