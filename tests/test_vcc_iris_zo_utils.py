import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBF_experiment.active.pybullet.self_collision.vcc_iris.polytope_sampling import (
    is_inside_polytope,
    sample_polytope_hit_and_run,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.statistical_test import (
    required_trials,
    unadaptive_collision_test,
    union_bound_delta,
)


class IrisZoUtilsTests(unittest.TestCase):
    def test_hit_and_run_samples_stay_inside_box(self):
        A = np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=float,
        )
        b = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
        samples, final_state = sample_polytope_hit_and_run(
            A,
            b,
            np.zeros(2, dtype=float),
            num_samples=32,
            rng=np.random.default_rng(7),
            mixing_steps=4,
        )
        self.assertEqual(samples.shape, (32, 2))
        self.assertTrue(all(is_inside_polytope(A, b, q) for q in samples))
        self.assertTrue(is_inside_polytope(A, b, final_state))

    def test_statistical_thresholds_are_reasonable(self):
        delta_11 = union_bound_delta(total_delta=0.1, outer_iter=1, inner_iter=1)
        delta_22 = union_bound_delta(total_delta=0.1, outer_iter=2, inner_iter=2)
        self.assertGreater(delta_11, delta_22)

        trials = required_trials(epsilon=0.1, delta=delta_11, tau=0.5)
        report = unadaptive_collision_test(
            0,
            num_samples=trials,
            epsilon=0.1,
            total_delta=0.1,
            tau=0.5,
            outer_iter=1,
            inner_iter=1,
        )
        self.assertTrue(report["accept"])


if __name__ == "__main__":
    unittest.main()
