import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBF_experiment.active.pybullet.self_collision.vcc_iris.config import IrisZoConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.iris_zo import run_iris_zo
from CBF_experiment.active.pybullet.self_collision.vcc_iris.types import CliqueEllipsoid, RobotModelMetadata


class FakeIrisOracle:
    def __init__(self):
        self.metadata = RobotModelMetadata(
            revolute_ids=(0, 1),
            revolute_names=("q0", "q1"),
            joint_limits=((-1.0, 1.0), (-1.0, 1.0)),
            q_indices=(0, 1),
            q_base=np.zeros(2, dtype=float),
            dq_base=np.zeros(2, dtype=float),
            monitored_link_ids=(),
            monitored_link_names=(),
            monitored_pairs=(),
        )

    def is_self_collision(self, q):
        q = np.asarray(q, dtype=float)
        return bool(q[0] > 0.6 or q[1] > 0.6)

    def pair_distances_at(self, q):
        return [
            {"pair": (0, 1), "distance": 0.2, "is_collision": False},
        ]

    def query(self, q):
        q = np.asarray(q, dtype=float)
        return {
            "is_collision": bool(self.is_self_collision(q)),
            "min_clearance": float(0.6 - max(q[0], q[1])),
            "active_pair": (0, 1) if self.is_self_collision(q) else None,
            "contact_penetration_depth": None,
        }

    def first_collision_on_segment(self, q_free, q_target, *, num_steps, bisection_steps):
        return self.first_pair_collision_on_segment(
            q_free,
            q_target,
            (0, 1),
            num_steps=num_steps,
            bisection_steps=bisection_steps,
        )

    def first_pair_collision_on_segment(self, q_free, q_target, pair, *, num_steps, bisection_steps):
        q_free = np.asarray(q_free, dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        if self.is_self_collision(q_target):
            low = q_free.copy()
            high = q_target.copy()
            for _ in range(int(bisection_steps)):
                mid = 0.5 * (low + high)
                if self.is_self_collision(mid):
                    high = mid
                else:
                    low = mid
            delta = high - low
            normal = delta / max(np.linalg.norm(delta), 1e-12)
            return {
                "pair": tuple(pair),
                "q_free": low,
                "q_collision": high,
                "normal": normal,
                "offset": float(np.dot(normal, low)),
                "active_pair": None,
                "clearance": 0.0,
            }
        return None


class IrisZoTests(unittest.TestCase):
    def test_run_iris_zo_returns_feasible_region(self):
        ellipsoid = CliqueEllipsoid(
            vertex_indices=(0, 1, 2),
            center=np.array([0.0, 0.0], dtype=float),
            C=np.eye(2, dtype=float) * 0.2,
            clique_size=3,
        )
        region = run_iris_zo(
            ellipsoid,
            FakeIrisOracle(),
            IrisZoConfig(
                MAX_OUTER_ITERATIONS=3,
                MAX_INNER_ITERATIONS=4,
                NUM_PARTICLES=32,
                NUM_BISECTION_STEPS=8,
                MAX_NEW_FACES_PER_INNER_ITER=4,
            ),
            region_id=0,
        )

        self.assertGreaterEqual(len(region.iterations), 1)
        self.assertFalse(np.any(region.A @ region.center > region.b + 1e-8))
        self.assertGreater(float(region.log_det), -20.0)


if __name__ == "__main__":
    unittest.main()

