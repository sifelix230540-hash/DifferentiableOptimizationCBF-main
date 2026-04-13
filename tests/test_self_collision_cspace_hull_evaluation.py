import importlib.util
import json
import pathlib
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CBF_experiment" / "active" / "pybullet" / "self_collision_cspace_hull_evaluation.py"


def load_module(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _square_hull_payload():
    equations = [
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, 0.0],
    ]
    return {
        "dimension": 2,
        "clusters": [
            {
                "cluster_id": 0,
                "equations_joint": equations,
                "equations_normalized": equations,
            }
        ],
    }


class SelfCollisionCSpaceHullEvaluationTests(unittest.TestCase):
    def test_classify_points_against_hulls_returns_inside_mask(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hull_eval_classify")
        hull_payload = _square_hull_payload()
        points = np.array([
            [0.2, 0.2],
            [1.2, 1.2],
            [0.5, 0.5],
        ], dtype=float)

        inside, violations = module.classify_points_against_hulls(points, hull_payload, space="joint")

        np.testing.assert_array_equal(inside, np.array([True, False, True]))
        self.assertLessEqual(float(violations[0]), 1e-8)
        self.assertGreater(float(violations[1]), 1e-3)

    def test_evaluate_hulls_computes_core_metrics_and_boundary_metrics(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hull_eval_metrics")
        hull_payload = _square_hull_payload()
        collision_samples = np.array([
            [0.2, 0.2],
            [0.8, 0.8],
            [1.2, 1.2],
        ], dtype=float)
        free_samples = np.array([
            [1.5, 1.5],
            [0.5, 0.5],
        ], dtype=float)
        collision_distances = np.array([-0.01, -0.20, -0.03], dtype=float)
        free_distances = np.array([0.20, 0.01], dtype=float)

        report = module.evaluate_hulls(
            hull_payload,
            collision_samples,
            free_samples,
            collision_distances=collision_distances,
            free_distances=free_distances,
            boundary_band=0.05,
            space="joint",
        )

        self.assertAlmostEqual(report["coverage"], 2.0 / 3.0)
        self.assertAlmostEqual(report["miss_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(report["false_positive_rate"], 0.5)
        self.assertEqual(report["boundary_collision_count"], 2)
        self.assertAlmostEqual(report["boundary_collision_coverage"], 0.5)
        self.assertEqual(report["boundary_free_count"], 1)
        self.assertAlmostEqual(report["boundary_free_false_positive_rate"], 1.0)

    def test_evaluate_hulls_from_files_reads_and_writes_json_report(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hull_eval_files")
        hull_payload = _square_hull_payload()
        sample_payload = {
            "collision_samples": [[0.2, 0.2], [1.2, 1.2]],
            "free_samples": [[1.5, 1.5], [0.5, 0.5]],
            "collision_distances": [-0.01, -0.02],
            "free_distances": [0.02, 0.10],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            hull_json = tmpdir_path / "hulls.json"
            samples_json = tmpdir_path / "samples.json"
            out_json = tmpdir_path / "report.json"
            hull_json.write_text(json.dumps(hull_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            samples_json.write_text(json.dumps(sample_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            report = module.evaluate_hulls_from_files(
                hull_json=hull_json,
                sample_json=samples_json,
                output_json=out_json,
                boundary_band=0.05,
                space="joint",
            )

            self.assertTrue(out_json.exists())
            saved = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertAlmostEqual(report["coverage"], saved["coverage"])
            self.assertIn("num_collision_samples", saved)
            self.assertIn("num_free_samples", saved)

    def test_chunked_min_violation_matches_dense_result(self):
        module = load_module(MODULE_PATH, "self_collision_cspace_hull_eval_chunked")
        points = np.array([
            [0.2, 0.2],
            [1.2, 1.2],
            [0.5, 0.5],
            [0.9, 0.1],
        ], dtype=float)
        equations = np.array([
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, -1.7],
            [-1.0, -1.0, 0.1],
        ], dtype=float)

        dense = module._max_violation_against_equations(points, equations)
        chunked = module._min_violation_against_equations_chunked(
            points,
            equations,
            point_batch_size=2,
            equation_batch_size=3,
        )

        np.testing.assert_allclose(chunked, dense)


if __name__ == "__main__":
    unittest.main()
