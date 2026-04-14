import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.clique_cover import greedy_clique_cover
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import CliqueCoverConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import VisibilityGraph


class CliqueCoverTests(unittest.TestCase):
    def test_greedy_clique_cover_returns_complete_subgraph(self):
        adjacency = (
            frozenset({1, 2}),
            frozenset({0, 2}),
            frozenset({0, 1, 3}),
            frozenset({2}),
        )
        graph = VisibilityGraph(
            vertices=np.zeros((4, 2), dtype=float),
            adjacency=adjacency,
            edges=((0, 1), (0, 2), (1, 2), (2, 3)),
            num_candidate_pairs=4,
            num_visible_edges=4,
        )
        cliques = greedy_clique_cover(graph, CliqueCoverConfig(MIN_CLIQUE_SIZE=2, MAX_CLIQUES=4))

        self.assertGreaterEqual(len(cliques), 1)
        first = set(cliques[0].vertex_indices)
        self.assertEqual(first, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()

