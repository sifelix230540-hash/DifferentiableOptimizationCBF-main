"""为种子团拟合最小体积外包椭球 (MVEE)，作为 IRIS-ZO 初始椭球。"""
from __future__ import annotations

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import Clique, CliqueEllipsoid, FreeSample


def _mvee(points: np.ndarray, solver_preference: tuple[str, ...] = ("MOSEK", "CLARABEL", "SCS")) -> tuple[np.ndarray, np.ndarray]:
    """Minimum Volume Enclosing Ellipsoid (Löwner-John) via SDP [Boyd §8.4.1].

    Returns (center, C) where the ellipsoid is {x | ||C^{-1}(x - center)|| <= 1}.
    """
    import cvxpy as cp

    n_pts, dim = points.shape
    if n_pts <= 1:
        center = points[0] if n_pts == 1 else np.zeros(dim)
        return center, np.eye(dim, dtype=float) * 1e-2

    L = cp.Variable((dim, dim), PSD=True)
    d = cp.Variable(dim)
    constraints = [cp.norm(L @ (points[i] - d), 2) <= 1.0 for i in range(n_pts)]
    prob = cp.Problem(cp.Maximize(cp.log_det(L)), constraints)

    available = set(cp.installed_solvers())
    solved = False
    for solver_name in solver_preference:
        if solver_name not in available:
            continue
        try:
            prob.solve(solver=getattr(cp, solver_name), verbose=False)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                solved = True
                break
        except Exception:
            continue

    if solved and L.value is not None and d.value is not None:
        L_val = np.asarray(L.value, dtype=float)
        C_val = np.linalg.inv(L_val)
        return np.asarray(d.value, dtype=float), C_val

    return _ellipsoid_from_points_fallback(points)


def _ellipsoid_from_points_fallback(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Covariance-based fallback when MVEE solver is unavailable."""
    center = np.mean(points, axis=0)
    centered = points - center.reshape(1, -1)
    dim = points.shape[1]
    if points.shape[0] <= 1:
        return center, np.eye(dim, dtype=float) * 1e-2
    cov = (centered.T @ centered) / max(points.shape[0], 1)
    cov += np.eye(dim, dtype=float) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    base = eigvecs @ np.diag(np.sqrt(eigvals))
    inv_base = np.linalg.pinv(base)
    radii = [float(np.linalg.norm(inv_base @ (point - center))) for point in points]
    scale = max(1.05 * max(radii), 1e-3)
    return center, base * scale


def summarize_cliques_with_ellipsoids(samples: list[FreeSample], cliques: list[Clique], oracle=None) -> list[CliqueEllipsoid]:
    ellipsoids: list[CliqueEllipsoid] = []
    for clique in cliques:
        points = np.asarray([np.asarray(samples[idx].q, dtype=float) for idx in clique.vertex_indices], dtype=float)
        center, C = _mvee(points)
        if oracle is not None and oracle.is_self_collision(center):
            best_idx = max(clique.vertex_indices, key=lambda idx: float(samples[idx].clearance))
            center = np.asarray(samples[best_idx].q, dtype=float)
        ellipsoids.append(
            CliqueEllipsoid(
                vertex_indices=tuple(int(x) for x in clique.vertex_indices),
                center=np.asarray(center, dtype=float),
                C=np.asarray(C, dtype=float),
                clique_size=len(clique.vertex_indices),
            )
        )
    return ellipsoids

