from __future__ import annotations

import cvxpy as cp
import numpy as np

from .two_d_core import TwoDConfig


class CBFSafetyFilter2D:
    """Single-step CBF-QP safety filter for the 2D mass-point nominal action."""

    def __init__(
        self,
        cfg: TwoDConfig,
        n_obs: int,
        alpha_cbf: float = 4.0,
        slack_weight: float = 800.0,
    ):
        self.cfg = cfg
        self.n_obs = n_obs
        self.alpha_cbf = alpha_cbf
        self.slack_weight = slack_weight

        self.u_var = cp.Variable(2)
        self.slk = cp.Variable(n_obs, nonneg=True)
        self.u_des_p = cp.Parameter(2)
        self.gh_p = cp.Parameter((n_obs, 2))
        self.rhs_p = cp.Parameter(n_obs)

        constraints = [self.u_var >= -cfg.u_max, self.u_var <= cfg.u_max]
        for i in range(n_obs):
            constraints.append(self.gh_p[i] @ self.u_var + self.rhs_p[i] + self.slk[i] >= 0)

        objective = cp.Minimize(
            cp.sum_squares(self.u_var - self.u_des_p) + slack_weight * cp.sum(self.slk)
        )
        self.problem = cp.Problem(objective, constraints)

    def filter(self, x: np.ndarray, u_des: np.ndarray, obs_list: list[dict]) -> tuple[np.ndarray, bool]:
        gh_vals = np.zeros((self.n_obs, 2))
        rhs_vals = np.zeros(self.n_obs)
        pos = x[:2]
        vel = x[2:]

        for i, o in enumerate(obs_list):
            d_vec = pos - o["center"]
            nd = np.linalg.norm(d_vec)
            h = nd - (o["radius"] + self.cfg.safety_margin)
            gh = d_vec / nd if nd > 1e-9 else np.array([1.0, 0.0])
            h_dot = gh @ (vel - o["velocity"])
            gh_vals[i] = gh
            rhs_vals[i] = self.cfg.mass * (2 * self.alpha_cbf * h_dot + self.alpha_cbf**2 * h)

        self.u_des_p.value = u_des
        self.gh_p.value = gh_vals
        self.rhs_p.value = rhs_vals

        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            ok = self.problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and self.u_var.value is not None
            if ok:
                return np.asarray(self.u_var.value, dtype=float).reshape(2), True
        except Exception:
            pass
        return np.asarray(u_des, dtype=float).reshape(2), False
