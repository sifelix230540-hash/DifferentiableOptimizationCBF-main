import importlib.util
import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CBF_experiment" / "active" / "pybullet" / "self_collision" / "safe_cover" / "iris_exact.py"


def load_module(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class IrisExactTests(unittest.TestCase):
    def test_tangent_halfspace_matches_unit_ball_contact_point(self):
        module = load_module(MODULE_PATH, "iris_exact_tangent_plane")
        C = np.eye(2, dtype=float)
        d = np.zeros(2, dtype=float)
        x_star = np.array([1.0, 0.0], dtype=float)

        a, b = module.tangent_halfspace_to_ellipsoid(C, d, x_star)

        self.assertAlmostEqual(float(np.dot(a, x_star)), float(b), places=6)
        self.assertLessEqual(float(np.dot(a, np.array([0.0, 0.0]))), float(b) + 1e-9)
        self.assertGreater(float(np.dot(a, np.array([1.2, 0.0]))), float(b) - 1e-9)

    def test_maximum_volume_inscribed_ellipsoid_finds_square_center(self):
        module = load_module(MODULE_PATH, "iris_exact_mvie")
        A = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ], dtype=float)
        b = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)

        result = module.maximum_volume_inscribed_ellipsoid(A, b)

        np.testing.assert_allclose(result["center"], np.array([0.5, 0.5]), atol=5e-3)
        eigvals = np.linalg.eigvalsh(np.asarray(result["C"], dtype=float))
        self.assertTrue(np.all(eigvals > 0.0))
        self.assertGreater(float(np.linalg.det(np.asarray(result["C"], dtype=float))), 0.20)

    def test_run_iris_exact_has_non_decreasing_log_det_history(self):
        module = load_module(MODULE_PATH, "iris_exact_full_loop")
        domain_A = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ], dtype=float)
        domain_b = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
        obstacles = [
            {
                "A_full": np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=float),
                "b_full": np.array([0.2, 0.0, 1.0, 0.0], dtype=float),
                "pair": [0, 1],
                "cluster_id": 0,
            },
            {
                "A_full": np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=float),
                "b_full": np.array([1.0, -0.8, 1.0, 0.0], dtype=float),
                "pair": [1, 2],
                "cluster_id": 0,
            },
        ]

        result = module.run_iris_exact(
            seed=np.array([0.5, 0.5], dtype=float),
            obstacles=obstacles,
            domain_A=domain_A,
            domain_b=domain_b,
            max_iters=4,
            convergence_tol=1e-5,
        )

        history = [float(item["log_det"]) for item in result["iterations"]]
        self.assertGreaterEqual(len(history), 1)
        self.assertTrue(all(history[idx + 1] + 1e-8 >= history[idx] for idx in range(len(history) - 1)))
        self.assertTrue(np.all(np.asarray(result["A"], dtype=float) @ np.asarray(result["center"], dtype=float) <= np.asarray(result["b"], dtype=float) + 1e-8))


if __name__ == "__main__":
    unittest.main()

