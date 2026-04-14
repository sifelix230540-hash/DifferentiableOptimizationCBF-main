import importlib.util
import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CBF_experiment" / "active" / "pybullet" / "self_collision" / "safe_cover" / "pairwise_cspace_obstacles.py"


def load_module(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class PairwiseCSpaceObstacleTests(unittest.TestCase):
    def test_compute_pair_active_joint_indices_on_serial_chain(self):
        module = load_module(MODULE_PATH, "pairwise_cspace_obstacles_active_q")
        metadata = {
            "active_joint_ids": [0, 1, 2, 4, 5, 6, 7, 8, 9],
            "joint_parent_link": {
                0: -1,
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                9: 8,
                10: 9,
            },
        }

        self.assertEqual(module.compute_pair_active_joint_indices(metadata, 2, 5), [4, 5])
        self.assertEqual(module.compute_pair_active_joint_indices(metadata, 3, 5), [4, 5])
        self.assertEqual(module.compute_pair_active_joint_indices(metadata, 5, 10), [6, 7, 8, 9])

    def test_lift_halfspaces_to_full_space_embeds_selected_axes(self):
        module = load_module(MODULE_PATH, "pairwise_cspace_obstacles_lift")
        A_sub = np.array([[1.0, -2.0], [-1.5, 0.5]], dtype=float)
        b_sub = np.array([0.3, 0.7], dtype=float)

        A_full, b_full = module.lift_halfspaces_to_full_space(
            A_sub,
            b_sub,
            active_q_indices=[1, 4],
            full_dim=6,
        )

        self.assertEqual(A_full.shape, (2, 6))
        np.testing.assert_allclose(b_full, b_sub)
        np.testing.assert_allclose(A_full[:, [1, 4]], A_sub)
        np.testing.assert_allclose(A_full[:, [0, 2, 3, 5]], 0.0)


if __name__ == "__main__":
    unittest.main()

