import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CBF_experiment" / "active" / "pybullet" / "self_collision" / "self_collision_coal_validation.py"


def load_module(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class SelfCollisionCoalValidationTests(unittest.TestCase):
    def test_extract_contact_penetration_depth_returns_none_without_collision(self):
        module = load_module(MODULE_PATH, "self_collision_coal_validation_test_no_contact")

        class FakeResult:
            def isCollision(self):
                return False

            def numContacts(self):
                return 0

        self.assertIsNone(module._extract_contact_penetration_depth(FakeResult()))

    def test_extract_contact_penetration_depth_returns_deepest_contact(self):
        module = load_module(MODULE_PATH, "self_collision_coal_validation_test_depth")

        class FakeContact:
            def __init__(self, depth):
                self.penetration_depth = depth

        class FakeResult:
            def isCollision(self):
                return True

            def numContacts(self):
                return 3

            def getContact(self, idx):
                return [FakeContact(-0.01), FakeContact(-0.03), FakeContact(-0.02)][idx]

        self.assertAlmostEqual(module._extract_contact_penetration_depth(FakeResult()), -0.03)

    def test_coal_collision_decision_uses_collide_result_for_zero_distance(self):
        module = load_module(MODULE_PATH, "self_collision_coal_validation_test_zero")

        self.assertTrue(module._coal_pair_is_collision(0.0, True, penetration_thresh=-0.001))

    def test_coal_collision_decision_accepts_negative_distance(self):
        module = load_module(MODULE_PATH, "self_collision_coal_validation_test_negative")

        self.assertTrue(module._coal_pair_is_collision(-1e-4, False, penetration_thresh=-0.001))

    def test_coal_collision_decision_rejects_positive_distance_without_contact(self):
        module = load_module(MODULE_PATH, "self_collision_coal_validation_test_positive")

        self.assertFalse(module._coal_pair_is_collision(0.01, False, penetration_thresh=-0.001))


if __name__ == "__main__":
    unittest.main()
