import pathlib
import sys
import types
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CBF_experiment" / "history" / "3_11_2D_CFB_MCP_QP_deg2.py"
RUN_MARKER = "# =====================================================================\n# Run and Compare"


def load_partial_module():
    source = MODULE_PATH.read_text(encoding="utf-8")
    cutoff = source.index(RUN_MARKER)
    module = types.ModuleType("cbf_2d_second_order_partial")
    module.__file__ = str(MODULE_PATH)
    fake_cvxpy = types.SimpleNamespace()
    original_cvxpy = sys.modules.get("cvxpy")
    sys.modules["cvxpy"] = fake_cvxpy
    try:
        exec(compile(source[:cutoff], str(MODULE_PATH), "exec"), module.__dict__)
    finally:
        if original_cvxpy is None:
            sys.modules.pop("cvxpy", None)
        else:
            sys.modules["cvxpy"] = original_cvxpy
    return module


class SecondOrderReferenceTests(unittest.TestCase):
    def test_second_order_reference_bends_away_from_circle(self):
        module = load_partial_module()
        module.goal = np.array([3.0, 0.0], dtype=float)
        module.N_mpc = 8
        module.STATIC_OBS = [{"center": np.array([1.0, 0.0], dtype=float), "radius": 0.45, "velocity": np.zeros(2)}]
        module.DYN_OBS_DEFS = []
        module.n_obs = len(module.STATIC_OBS)

        x0 = np.array([0.0, 0.12, 0.0, 0.0], dtype=float)
        obs_cache = [[dict(o) for o in module.STATIC_OBS] for _ in range(module.N_mpc + 1)]
        u_seed = np.zeros((module.N_mpc, 2), dtype=float)

        x_track, u_track = module.build_second_order_reference(x0, obs_cache, u_seed)

        self.assertEqual(x_track.shape, (module.N_mpc + 1, 4))
        self.assertEqual(u_track.shape, (module.N_mpc, 2))
        self.assertGreater(x_track[1, 1], x0[1])
        self.assertGreater(u_track[0, 1], 0.0)

    def test_second_order_reference_keeps_controls_smooth(self):
        module = load_partial_module()
        module.goal = np.array([3.0, 0.0], dtype=float)
        module.N_mpc = 10
        module.STATIC_OBS = [{"center": np.array([1.1, 0.0], dtype=float), "radius": 0.5, "velocity": np.zeros(2)}]
        module.DYN_OBS_DEFS = []
        module.n_obs = len(module.STATIC_OBS)

        x0 = np.array([0.0, 0.16, 0.0, 0.0], dtype=float)
        obs_cache = [[dict(o) for o in module.STATIC_OBS] for _ in range(module.N_mpc + 1)]
        u_seed = np.zeros((module.N_mpc, 2), dtype=float)

        _, u_track = module.build_second_order_reference(x0, obs_cache, u_seed)
        delta_u = np.diff(u_track, axis=0)

        self.assertLess(np.max(np.linalg.norm(delta_u, axis=1)), 1.5)

    def test_mpc_accepts_optimal_inaccurate_status(self):
        module = load_partial_module()
        module.cp.OPTIMAL = "optimal"
        module.cp.OPTIMAL_INACCURATE = "optimal_inaccurate"

        self.assertTrue(module.mpc_status_acceptable(module.cp.OPTIMAL))
        self.assertTrue(module.mpc_status_acceptable(module.cp.OPTIMAL_INACCURATE))
        self.assertFalse(module.mpc_status_acceptable("infeasible"))

    def test_mpc_failure_falls_back_to_single_step_cbf(self):
        module = load_partial_module()
        module.goal = np.array([3.0, 0.0], dtype=float)
        module.STATIC_OBS = [{"center": np.array([1.0, 0.0], dtype=float), "radius": 0.45, "velocity": np.zeros(2)}]
        module.DYN_OBS_DEFS = []
        module.n_obs = len(module.STATIC_OBS)

        x = np.array([0.0, 0.12, 0.5, 0.0], dtype=float)
        u_nom = np.array([1.0, 0.0], dtype=float)
        u_fallback = module.compute_single_step_cbf_safe_control(x, u_nom, module.STATIC_OBS)

        self.assertGreater(u_fallback[1], 0.0)
        self.assertLessEqual(np.linalg.norm(u_fallback, ord=np.inf), module.u_max)


if __name__ == "__main__":
    unittest.main()
