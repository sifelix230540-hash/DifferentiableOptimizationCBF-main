from __future__ import annotations

import math

import numpy as np

from CBF_experiment.active.pybullet.self_collision.safe_cover.iris_exact import maximum_volume_inscribed_ellipsoid
from CBF_experiment.active.pybullet.self_collision.vcc_iris.config import IrisNpConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.types import CliqueEllipsoid, IrisRegion


def _normalized_rows(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(A, axis=1)
    keep = norms > 1e-12
    A = A[keep]
    b = b[keep]
    norms = norms[keep]
    return A / norms.reshape(-1, 1), b / norms


def _ray_box_intersection(center: np.ndarray, direction: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    t_values = []
    for axis in range(center.shape[0]):
        if abs(float(direction[axis])) <= 1e-12:
            continue
        bound = upper[axis] if float(direction[axis]) > 0.0 else lower[axis]
        t = (float(bound) - float(center[axis])) / float(direction[axis])
        if t > 0.0:
            t_values.append(float(t))
    return min(t_values) if t_values else 0.0


def _ray_polytope_intersection(center: np.ndarray, direction: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    max_t = float("inf")
    for row, rhs in zip(np.asarray(A, dtype=float), np.asarray(b, dtype=float).reshape(-1)):
        denom = float(np.dot(row, direction))
        if denom <= 1e-12:
            continue
        slack = float(rhs) - float(np.dot(row, center))
        t = slack / denom
        if t > 0.0:
            max_t = min(max_t, float(t))
    return 0.0 if not np.isfinite(max_t) else max_t


def _direction_set(C: np.ndarray, rng: np.random.Generator, num_rays: int) -> list[np.ndarray]:
    dim = C.shape[0]
    directions = []
    for axis in range(dim):
        column = np.asarray(C[:, axis], dtype=float).reshape(-1)
        norm = float(np.linalg.norm(column))
        if norm > 1e-12:
            directions.append(column / norm)
            directions.append(-column / norm)
    while len(directions) < int(num_rays):
        sample = np.asarray(C, dtype=float) @ rng.normal(size=dim)
        norm = float(np.linalg.norm(sample))
        if norm <= 1e-12:
            continue
        directions.append(sample / norm)
    return directions[: int(num_rays)]


def _metric_matrix(C: np.ndarray) -> np.ndarray:
    C_inv = np.linalg.pinv(np.asarray(C, dtype=float))
    return C_inv.T @ C_inv


def _pair_order(center: np.ndarray, oracle) -> list[tuple[int, int]]:
    reports = oracle.pair_distances_at(center)
    reports.sort(key=lambda item: float(item["distance"]))
    return [tuple(item["pair"]) for item in reports]


def _search_counterexample_for_pair(
    center: np.ndarray,
    C: np.ndarray,
    oracle,
    pair: tuple[int, int],
    A: np.ndarray,
    b: np.ndarray,
    cfg: IrisNpConfig,
    rng: np.random.Generator,
) -> dict | None:
    directions = _direction_set(C, rng, int(cfg.NUM_PAIR_CANDIDATE_DIRECTIONS))
    metric_matrix = _metric_matrix(C)
    best_hit = None
    best_cost = float("inf")
    for direction in directions:
        t_max = _ray_polytope_intersection(center, direction, A, b)
        if t_max <= 1e-8:
            continue
        q_target = center + direction * max(t_max - float(cfg.RAY_BOUNDARY_MARGIN), 0.0)
        hit = oracle.first_pair_collision_on_segment(
            center,
            q_target,
            pair,
            num_steps=max(int(cfg.BISECTION_STEPS), 4),
            bisection_steps=int(cfg.BISECTION_STEPS),
        )
        if hit is None:
            continue
        q_star = np.asarray(hit["q_collision"], dtype=float)
        delta = q_star - np.asarray(center, dtype=float)
        cost = float(delta.T @ metric_matrix @ delta)
        if cost < best_cost:
            best_cost = cost
            best_hit = hit
    return best_hit


def _append_separating_plane(
    rows: list[np.ndarray],
    rhs: list[float],
    center: np.ndarray,
    C: np.ndarray,
    counterexample: dict,
    cfg: IrisNpConfig,
) -> dict | None:
    q_star = np.asarray(counterexample["q_collision"], dtype=float).reshape(-1)
    metric_matrix = _metric_matrix(C)
    normal = metric_matrix @ (q_star - np.asarray(center, dtype=float))
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    normal = normal / norm
    offset = float(np.dot(normal, q_star) - float(cfg.CONFIGURATION_SPACE_MARGIN))
    rows.append(normal)
    rhs.append(offset)
    return {
        "pair": list(counterexample.get("pair", [])),
        "normal": normal.tolist(),
        "offset": float(offset),
        "active_pair": list(counterexample["active_pair"]) if counterexample["active_pair"] else None,
        "clearance": float(counterexample["clearance"]),
        "q_collision": np.asarray(counterexample["q_collision"], dtype=float).tolist(),
    }


def run_iris_np(
    ellipsoid: CliqueEllipsoid,
    oracle,
    cfg: IrisNpConfig,
    *,
    region_id: int,
) -> IrisRegion:
    lower = np.asarray([float(lo) for lo, _ in oracle.metadata.joint_limits], dtype=float)
    upper = np.asarray([float(hi) for _, hi in oracle.metadata.joint_limits], dtype=float)
    rng = np.random.default_rng(int(cfg.RNG_SEED) + int(region_id))
    center = np.asarray(ellipsoid.center, dtype=float).reshape(-1)
    C = np.asarray(ellipsoid.C, dtype=float)
    last_log_det = float(np.linalg.slogdet(C)[1])
    box_rows = []
    box_rhs = []
    for axis in range(center.shape[0]):
        upper_row = np.zeros(center.shape[0], dtype=float)
        upper_row[axis] = 1.0
        box_rows.append(upper_row)
        box_rhs.append(float(upper[axis]))
        lower_row = np.zeros(center.shape[0], dtype=float)
        lower_row[axis] = -1.0
        box_rows.append(lower_row)
        box_rhs.append(-float(lower[axis]))
    final_A, final_b = _normalized_rows(np.asarray(box_rows, dtype=float), np.asarray(box_rhs, dtype=float))
    iter_history = []

    for iteration in range(int(cfg.MAX_ITERATIONS)):
        working_rows = [row.copy() for row in np.asarray(final_A, dtype=float)]
        working_rhs = [float(x) for x in np.asarray(final_b, dtype=float)]
        plane_reports = []
        pair_order = _pair_order(center, oracle)
        for pair in pair_order:
            failures = 0
            while failures < int(cfg.MAX_CONSECUTIVE_INFEASIBLE_SAMPLES):
                hit = _search_counterexample_for_pair(
                    center,
                    C,
                    oracle,
                    pair,
                    np.asarray(working_rows, dtype=float),
                    np.asarray(working_rhs, dtype=float),
                    cfg,
                    rng,
                )
                if hit is None:
                    failures += 1
                    continue
                plane_report = _append_separating_plane(working_rows, working_rhs, center, C, hit, cfg)
                if plane_report is None:
                    failures += 1
                    continue
                plane_reports.append(plane_report)
                failures = 0
        A, b = _normalized_rows(np.asarray(working_rows, dtype=float), np.asarray(working_rhs, dtype=float))
        mvie = maximum_volume_inscribed_ellipsoid(A, b)
        new_center = np.asarray(mvie["center"], dtype=float)
        new_C = np.asarray(mvie["C"], dtype=float)
        new_log_det = float(mvie["log_det"])
        if oracle.is_self_collision(new_center):
            break
        if new_log_det + 1e-9 < last_log_det:
            break
        center = new_center
        C = new_C
        final_A = A
        final_b = b
        iter_history.append({
            "iteration": int(iteration),
            "center": center.tolist(),
            "log_det": float(new_log_det),
            "num_planes": int(A.shape[0]),
            "pair_order": [list(pair) for pair in pair_order],
            "plane_reports": plane_reports,
        })
        rel_growth = (new_log_det - last_log_det) / max(abs(last_log_det), 1e-9)
        last_log_det = new_log_det
        if rel_growth < float(cfg.CONVERGENCE_TOL):
            break

    return IrisRegion(
        region_id=int(region_id),
        source_clique_indices=tuple(int(x) for x in ellipsoid.vertex_indices),
        A=np.asarray(final_A, dtype=float),
        b=np.asarray(final_b, dtype=float),
        center=np.asarray(center, dtype=float),
        C=np.asarray(C, dtype=float),
        log_det=float(last_log_det if iter_history else math.log(max(np.linalg.det(C), 1e-12))),
        iterations=tuple(iter_history),
    )

