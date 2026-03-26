import unittest

import numpy as np

from CBF_experiment.active.pybullet.welding_320_common import ExperimentConfig
from CBF_experiment.active.pybullet.welding_320_experiment import (
    build_linear_nominal_trajectory,
    build_nominal_trajectory,
)


class _FakePlanner:
    def __init__(self):
        self.calls = []
        self.last_plan_statuses = ["rrt", "rrt", "rrt"]

    def build_three_phase_trajectory(self, q_init, q_start, q_goal, initial_pose, start_ref, goal_ref):
        self.calls.append({
            "q_init": np.array(q_init, dtype=float),
            "q_start": np.array(q_start, dtype=float),
            "q_goal": np.array(q_goal, dtype=float),
            "initial_pose": initial_pose,
            "start_ref": start_ref,
            "goal_ref": goal_ref,
        })
        return "fake_rrt_trajectory"


class NominalPlannerSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ExperimentConfig()
        self.q_init = np.array([0.0, 0.0])
        self.q_start = np.array([1.0, 1.0])
        self.q_goal = np.array([2.0, 2.0])
        self.initial_pose = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.start_ref = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.goal_ref = (np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))

    def test_build_linear_nominal_trajectory_returns_three_linear_segments(self):
        trajectory = build_linear_nominal_trajectory(
            self.cfg,
            self.initial_pose,
            self.start_ref,
            self.goal_ref,
        )

        self.assertEqual(len(trajectory.segments), 3)
        self.assertEqual(
            [seg.planner_status for seg in trajectory.segments],
            ["linear_nominal", "linear_nominal", "linear_nominal"],
        )
        self.assertTrue(np.allclose(trajectory.segments[0].waypoints_pos[0], self.initial_pose[0]))
        self.assertTrue(np.allclose(trajectory.segments[0].waypoints_pos[-1], self.start_ref[0]))
        self.assertTrue(np.allclose(trajectory.segments[1].waypoints_pos[0], self.start_ref[0]))
        self.assertTrue(np.allclose(trajectory.segments[1].waypoints_pos[-1], self.goal_ref[0]))
        self.assertTrue(np.allclose(trajectory.segments[2].waypoints_pos[0], self.goal_ref[0]))
        self.assertTrue(np.allclose(trajectory.segments[2].waypoints_pos[-1], self.initial_pose[0]))

    def test_build_nominal_trajectory_uses_linear_path_when_rrt_disabled(self):
        self.cfg.use_rrt_nominal_planner = False
        planner = _FakePlanner()

        trajectory = build_nominal_trajectory(
            self.cfg,
            planner,
            self.q_init,
            self.q_start,
            self.q_goal,
            self.initial_pose,
            self.start_ref,
            self.goal_ref,
        )

        self.assertEqual(planner.calls, [])
        self.assertEqual(len(trajectory.segments), 3)
        self.assertTrue(all(seg.planner_status == "linear_nominal" for seg in trajectory.segments))

    def test_build_nominal_trajectory_delegates_to_rrt_planner_when_enabled(self):
        self.cfg.use_rrt_nominal_planner = True
        planner = _FakePlanner()

        trajectory = build_nominal_trajectory(
            self.cfg,
            planner,
            self.q_init,
            self.q_start,
            self.q_goal,
            self.initial_pose,
            self.start_ref,
            self.goal_ref,
        )

        self.assertEqual(trajectory, "fake_rrt_trajectory")
        self.assertEqual(len(planner.calls), 1)


if __name__ == "__main__":
    unittest.main()
