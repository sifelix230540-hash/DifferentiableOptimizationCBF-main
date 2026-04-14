import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.ellipsoids import summarize_cliques_with_ellipsoids
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import Clique, FreeSample


class EllipsoidTests(unittest.TestCase):
    def test_summarize_clique_returns_positive_definite_shape(self):
        samples = [
            FreeSample(q=np.array([0.0, 0.0]), clearance=1.0, active_pair=None),
            FreeSample(q=np.array([1.0, 0.0]), clearance=1.0, active_pair=None),
            FreeSample(q=np.array([0.0, 1.0]), clearance=1.0, active_pair=None),
        ]
        clique = Clique(vertex_indices=(0, 1, 2), score=3.0)
        ellipsoid = summarize_cliques_with_ellipsoids(samples, [clique])[0]

        self.assertEqual(ellipsoid.clique_size, 3)
        eigvals = np.linalg.eigvalsh(ellipsoid.C @ ellipsoid.C.T)
        self.assertTrue(np.all(eigvals > 0.0))
        np.testing.assert_allclose(ellipsoid.center, np.array([1.0 / 3.0, 1.0 / 3.0]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()

