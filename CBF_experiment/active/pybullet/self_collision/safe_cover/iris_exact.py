from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require_cvxpy():
    try:
        import cvxpy as cp  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - installation/runtime specific
        raise RuntimeError(
            "论文版 IRIS 需要 cvxpy 求解最大体积内接椭球，请先安装 cvxpy/scs。"
        ) from exc
    return cp


def tangent_halfspace_to_ellipsoid(C: np.ndarray, d: np.ndarray, x_star: np.ndarray) -> tuple[np.ndarray, float]:
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)
    x_star = np.asarray(x_star, dtype=float).reshape(-1)
    C_inv = np.linalg.inv(C)
    normal = 2.0 * C_inv @ C_inv.T @ (x_star - d)
    offset = float(np.dot(normal, x_star))
    return normal.astype(float), offset


def _initial_ellipsoid_matrix(seed: np.ndarray, domain_A: np.ndarray, domain_b: np.ndarray) -> np.ndarray:
    seed = np.asarray(seed, dtype=float).reshape(-1)
    slacks = np.asarray(domain_b, dtype=float).reshape(-1) - np.asarray(domain_A, dtype=float) @ seed
    norms = np.linalg.norm(np.asarray(domain_A, dtype=float), axis=1)
    clearances = slacks / np.maximum(norms, 1e-12)
    radius = max(1e-3, 0.25 * float(np.min(clearances)))
    return np.eye(seed.shape[0], dtype=float) * radius


def maximum_volume_inscribed_ellipsoid(
    A: np.ndarray,
    b: np.ndarray,
    *,
    solver_preference: tuple[str, ...] = ("MOSEK", "CLARABEL", "SCS"),
) -> dict:
    cp = _require_cvxpy()
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    dim = int(A.shape[1])

    C = cp.Variable((dim, dim), PSD=True)
    d = cp.Variable(dim)
    constraints = []
    for row, rhs in zip(A, b):
        a_i = np.asarray(row, dtype=float).reshape(-1)
        constraints.append(cp.norm(C @ a_i, 2) + a_i @ d <= float(rhs))
    problem = cp.Problem(cp.Maximize(cp.log_det(C)), constraints)

    available = set(cp.installed_solvers())
    last_error = None
    for solver_name in solver_preference:
        if solver_name not in available:
            continue
        try:
            problem.solve(solver=getattr(cp, solver_name), verbose=False)
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
            last_error = RuntimeError(f"{solver_name} returned status={problem.status}")
        except Exception as exc:  # pragma: no cover - solver availability dependent
            last_error = exc
    else:  # pragma: no cover - runtime dependent
        raise RuntimeError(f"MVIE 求解失败，最后错误: {last_error}")

    if C.value is None or d.value is None:
        raise RuntimeError(f"MVIE 求解未返回有效结果，status={problem.status}")
    return {
        "C": np.asarray(C.value, dtype=float),
        "center": np.asarray(d.value, dtype=float),
        "log_det": float(np.linalg.slogdet(np.asarray(C.value, dtype=float))[1]),
    }


def closest_point_on_hpolytope(
    obstacle_A: np.ndarray,
    obstacle_b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    *,
    solver_preference: tuple[str, ...] = ("OSQP", "SCS"),
) -> np.ndarray:
    cp = _require_cvxpy()
    obstacle_A = np.asarray(obstacle_A, dtype=float)
    obstacle_b = np.asarray(obstacle_b, dtype=float).reshape(-1)
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)
    dim = int(C.shape[0])
    T = np.linalg.inv(C)

    x = cp.Variable(dim)
    objective = cp.Minimize(cp.sum_squares(T @ (x - d)))
    constraints = [obstacle_A @ x <= obstacle_b]
    problem = cp.Problem(objective, constraints)

    available = set(cp.installed_solvers())
    last_error = None
    for solver_name in solver_preference:
        if solver_name not in available:
            continue
        try:
            problem.solve(solver=getattr(cp, solver_name), verbose=False)
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
            last_error = RuntimeError(f"{solver_name} returned status={problem.status}")
        except Exception as exc:  # pragma: no cover - solver availability dependent
            last_error = exc
    else:  # pragma: no cover - runtime dependent
        raise RuntimeError(f"最近点 QP 求解失败，最后错误: {last_error}")

    if x.value is None:
        raise RuntimeError(f"最近点 QP 未返回有效结果，status={problem.status}")
    return np.asarray(x.value, dtype=float).reshape(-1)


def is_obstacle_separated_by_halfspace(
    obstacle_A: np.ndarray,
    obstacle_b: np.ndarray,
    plane_a: np.ndarray,
    plane_b: float,
) -> bool:
    plane_a = np.asarray(plane_a, dtype=float).reshape(-1)
    result = linprog(
        c=plane_a,
        A_ub=np.asarray(obstacle_A, dtype=float),
        b_ub=np.asarray(obstacle_b, dtype=float).reshape(-1),
        bounds=[(None, None)] * int(plane_a.shape[0]),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"障碍物分离 LP 失败: {result.message}")
    return bool(float(result.fun) >= float(plane_b) - 1e-9)


def build_separating_hyperplanes(
    C: np.ndarray,
    d: np.ndarray,
    obstacles: list[dict],
    *,
    domain_A: np.ndarray,
    domain_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    domain_A = np.asarray(domain_A, dtype=float)
    domain_b = np.asarray(domain_b, dtype=float).reshape(-1)
    remaining = list(obstacles)
    selected = []
    A_rows = [row.copy() for row in domain_A]
    b_vals = [float(x) for x in domain_b]

    while remaining:
        scored = []
        for idx, obstacle in enumerate(remaining):
            x_star = closest_point_on_hpolytope(obstacle["A_full"], obstacle["b_full"], C, d)
            delta = np.linalg.solve(np.asarray(C, dtype=float), x_star - np.asarray(d, dtype=float))
            alpha = float(np.linalg.norm(delta))
            scored.append((alpha, idx, x_star))
        scored.sort(key=lambda item: item[0])
        _alpha, best_idx, x_star = scored[0]
        obstacle = remaining[best_idx]
        plane_a, plane_b = tangent_halfspace_to_ellipsoid(C, d, x_star)
        A_rows.append(plane_a)
        b_vals.append(float(plane_b))
        selected.append({
            "pair": obstacle.get("pair"),
            "cluster_id": obstacle.get("cluster_id"),
            "tangent_point": np.asarray(x_star, dtype=float).tolist(),
            "plane_a": np.asarray(plane_a, dtype=float).tolist(),
            "plane_b": float(plane_b),
        })

        new_remaining = []
        for candidate in remaining:
            if not is_obstacle_separated_by_halfspace(
                candidate["A_full"],
                candidate["b_full"],
                plane_a,
                plane_b,
            ):
                new_remaining.append(candidate)
        remaining = new_remaining

    return np.asarray(A_rows, dtype=float), np.asarray(b_vals, dtype=float), selected


def run_iris_exact(
    *,
    seed: np.ndarray,
    obstacles: list[dict],
    domain_A: np.ndarray,
    domain_b: np.ndarray,
    max_iters: int = 10,
    convergence_tol: float = 1e-3,
) -> dict:
    dim = int(np.asarray(seed, dtype=float).shape[0])
    C = _initial_ellipsoid_matrix(seed, domain_A, domain_b)
    d = np.asarray(seed, dtype=float).reshape(-1)
    iter_history = []
    initial_log_det = float(np.linalg.slogdet(C)[1])
    last_log_det = initial_log_det
    final_A = np.asarray(domain_A, dtype=float)
    final_b = np.asarray(domain_b, dtype=float).reshape(-1)
    final_planes = []

    for _ in range(int(max_iters)):
        candidate_A, candidate_b, candidate_planes = build_separating_hyperplanes(
            C,
            d,
            obstacles,
            domain_A=domain_A,
            domain_b=domain_b,
        )
        mvie = maximum_volume_inscribed_ellipsoid(candidate_A, candidate_b)
        new_C = np.asarray(mvie["C"], dtype=float)
        new_d = np.asarray(mvie["center"], dtype=float)
        log_det = float(mvie["log_det"])
        if log_det + 1e-9 < last_log_det:
            break
        C = new_C
        d = new_d
        final_A = candidate_A
        final_b = candidate_b
        final_planes = candidate_planes
        iter_history.append({
            "center": d.tolist(),
            "C": C.tolist(),
            "log_det": log_det,
            "num_planes": int(final_A.shape[0]),
            "num_selected_obstacles": int(len(final_planes)),
        })
        rel_growth = (log_det - last_log_det) / max(abs(last_log_det), 1e-9)
        if rel_growth < float(convergence_tol):
            last_log_det = log_det
            break
        last_log_det = log_det

    return {
        "A": np.asarray(final_A, dtype=float),
        "b": np.asarray(final_b, dtype=float),
        "C": np.asarray(C, dtype=float),
        "center": np.asarray(d, dtype=float),
        "planes": final_planes,
        "iterations": iter_history,
        "log_det": float(last_log_det if iter_history else math.log(max(np.linalg.det(C), 1e-12))),
    }

