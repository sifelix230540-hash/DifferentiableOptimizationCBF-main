import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import VisibilityConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import FreeSample
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.visibility import build_visibility_graph


class FakeVisibilityOracle:
    def segment_is_collision_free(self, q_a, q_b, *, num_steps):
        midpoint = 0.5 * (np.asarray(q_a, dtype=float) + np.asarray(q_b, dtype=float))
        return bool(float(midpoint[0]) < 0.75)


class VisibilityGraphTests(unittest.TestCase):
    def test_build_visibility_graph_marks_blocked_edge(self):
        samples = [
            FreeSample(q=np.array([0.0, 0.0]), clearance=1.0, active_pair=None),
            FreeSample(q=np.array([0.5, 0.0]), clearance=1.0, active_pair=None),
            FreeSample(q=np.array([1.5, 0.0]), clearance=1.0, active_pair=None),
        ]
        graph = build_visibility_graph(
            samples,
            FakeVisibilityOracle(),
            VisibilityConfig(MAX_CANDIDATE_PAIRS=None, SEGMENT_INTERPOLATION_STEPS=4),
        )

        self.assertEqual(graph.num_candidate_pairs, 3)
        self.assertIn(1, graph.adjacency[0])
        self.assertNotIn(2, graph.adjacency[0])


if __name__ == "__main__":
    unittest.main()

