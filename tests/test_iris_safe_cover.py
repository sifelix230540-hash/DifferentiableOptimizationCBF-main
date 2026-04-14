import importlib.util
import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CBF_experiment" / "active" / "pybullet" / "self_collision" / "safe_cover" / "iris_safe_cover.py"


def load_module(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class IrisSafeCoverTests(unittest.TestCase):
    def test_build_random_safe_curve_stays_inside_region(self):
        module = load_module(MODULE_PATH, "iris_safe_cover_curve_inside")
        region = {
            "center_normalized": [0.5, 0.5],
            "radius_normalized": 0.2,
            "A_normalized": [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            "b_normalized": [1.0, 0.0, 1.0, 0.0],
        }

        curve = module.build_random_safe_curve(region, rng=np.random.default_rng(0), num_points=64)
        pts = np.asarray(curve["curve_normalized"], dtype=float)
        A = np.asarray(region["A_normalized"], dtype=float)
        b = np.asarray(region["b_normalized"], dtype=float)

        self.assertEqual(pts.shape, (64, 2))
        self.assertTrue(np.all(pts @ A.T <= b.reshape(1, -1) + 1e-9))

    def test_build_random_safe_curve_returns_inside_control_points(self):
        module = load_module(MODULE_PATH, "iris_safe_cover_curve_controls")
        region = {
            "center_normalized": [0.6, 0.4, 0.5],
            "radius_normalized": 0.15,
            "A_normalized": [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            "b_normalized": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }

        curve = module.build_random_safe_curve(region, rng=np.random.default_rng(1), num_points=40)
        controls = np.asarray(curve["control_points_normalized"], dtype=float)
        A = np.asarray(region["A_normalized"], dtype=float)
        b = np.asarray(region["b_normalized"], dtype=float)

        self.assertEqual(controls.shape, (4, 3))
        self.assertTrue(np.all(controls @ A.T <= b.reshape(1, -1) + 1e-9))

    def test_build_random_safe_curve_supports_ellipsoid_region(self):
        module = load_module(MODULE_PATH, "iris_safe_cover_curve_ellipsoid")
        region = {
            "center_normalized": [0.5, 0.5],
            "C_normalized": [
                [0.12, 0.0],
                [0.0, 0.08],
            ],
            "A_normalized": [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            "b_normalized": [1.0, 0.0, 1.0, 0.0],
        }

        curve = module.build_random_safe_curve(region, rng=np.random.default_rng(2), num_points=50)
        pts = np.asarray(curve["curve_normalized"], dtype=float)
        controls = np.asarray(curve["control_points_normalized"], dtype=float)
        A = np.asarray(region["A_normalized"], dtype=float)
        b = np.asarray(region["b_normalized"], dtype=float)

        self.assertEqual(pts.shape, (50, 2))
        self.assertTrue(np.all(controls @ A.T <= b.reshape(1, -1) + 1e-9))
        self.assertTrue(np.all(pts @ A.T <= b.reshape(1, -1) + 1e-9))


if __name__ == "__main__":
    unittest.main()

