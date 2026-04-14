import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBF_experiment.active.pybullet.self_collision.vcc_iris.config import (
    CliqueCoverConfig,
    ExperimentConfig,
    IrisZoConfig,
    ReportingConfig,
    RobotQueryConfig,
    SamplingConfig,
    VisibilityConfig,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.pipeline import run_vcc_iris_pipeline
from CBF_experiment.active.pybullet.self_collision.vcc_iris.types import RobotModelMetadata


class FakePipelineOracle:
    def __init__(self, config):
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

    def close(self):
        return None

    def is_self_collision(self, q):
        q = np.asarray(q, dtype=float)
        return bool(np.linalg.norm(q) > 0.9)

    def query(self, q):
        q = np.asarray(q, dtype=float)
        is_collision = self.is_self_collision(q)
        clearance = 0.9 - float(np.linalg.norm(q))
        return {
            "is_collision": bool(is_collision),
            "min_clearance": float(clearance),
            "active_pair": None,
            "contact_penetration_depth": None,
        }

    def segment_is_collision_free(self, q_a, q_b, *, num_steps):
        for alpha in np.linspace(0.0, 1.0, int(num_steps) + 1):
            q = (1.0 - alpha) * np.asarray(q_a, dtype=float) + alpha * np.asarray(q_b, dtype=float)
            if self.is_self_collision(q):
                return False
        return True

    def pair_distances_at(self, q):
        return [{"pair": (0, 1), "distance": 0.1, "is_collision": False}]

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
        if not self.is_self_collision(q_target):
            return None
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
            "clearance": float(0.9 - np.linalg.norm(low)),
        }


class PipelineTests(unittest.TestCase):
    def test_pipeline_runs_with_fake_oracle_and_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(
                ROBOT=RobotQueryConfig(),
                SAMPLING=SamplingConfig(
                    NUM_SEED_SAMPLES=24,
                    BATCH_SIZE=32,
                    SAMPLE_CACHE_PATH=str(pathlib.Path(tmpdir) / "samples.json"),
                ),
                VISIBILITY=VisibilityConfig(MAX_CANDIDATE_PAIRS=None, SEGMENT_INTERPOLATION_STEPS=8),
                CLIQUE=CliqueCoverConfig(MIN_CLIQUE_SIZE=4, MAX_CLIQUES=4),
                IRIS_ZO=IrisZoConfig(
                    MAX_OUTER_ITERATIONS=2,
                    MAX_INNER_ITERATIONS=3,
                    NUM_PARTICLES=32,
                    NUM_BISECTION_STEPS=8,
                    MAX_NEW_FACES_PER_INNER_ITER=4,
                    MAX_REGIONS=3,
                ),
                REPORTING=ReportingConfig(
                    OUTPUT_DIR=tmpdir,
                    COVER_JSON=str(pathlib.Path(tmpdir) / "cover.json"),
                    EXPERIMENT_JSON=str(pathlib.Path(tmpdir) / "experiment.json"),
                ),
                CURVE_NUM_POINTS=20,
                CURVE_MAX_ATTEMPTS=4,
                COVERAGE_TARGET=0.2,
            )
            with mock.patch(
                "CBF_experiment.active.pybullet.self_collision.vcc_iris.pipeline.CoalSelfCollisionOracle",
                FakePipelineOracle,
            ):
                report = run_vcc_iris_pipeline(cfg)

            self.assertGreaterEqual(len(report.regions), 1)
            self.assertTrue(pathlib.Path(cfg.REPORTING.COVER_JSON).exists())
            self.assertTrue(pathlib.Path(cfg.REPORTING.EXPERIMENT_JSON).exists())


if __name__ == "__main__":
    unittest.main()

