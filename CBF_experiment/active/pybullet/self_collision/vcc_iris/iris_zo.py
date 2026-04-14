from __future__ import annotations

import math

import numpy as np

from CBF_experiment.active.pybullet.self_collision.safe_cover.iris_exact import maximum_volume_inscribed_ellipsoid
from CBF_experiment.active.pybullet.self_collision.vcc_iris.config import IrisZoConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.polytope_sampling import (
    is_inside_polytope,
    sample_polytope_hit_and_run,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.statistical_test import (
    required_trials,
    unadaptive_collision_test,
    union_bound_delta,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.progress import ProgressBar, stage_print
from CBF_experiment.active.pybullet.self_collision.vcc_iris.types import CliqueEllipsoid, IrisRegion


def _normalized_rows(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    norms = np.linalg.norm(A, axis=1)
    keep = norms > 1e-12
    A = A[keep]
    b = b[keep]
    norms = norms[keep]
    return A / norms.reshape(-1, 1), b / norms


def _metric_matrix(C: np.ndarray) -> np.ndarray:
    C_inv = np.linalg.pinv(np.asarray(C, dtype=float))
    return C_inv.T @ C_inv


def _joint_box_halfspaces(oracle) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([float(lo) for lo, _ in oracle.metadata.joint_limits], dtype=float)
    upper = np.asarray([float(hi) for _, hi in oracle.metadata.joint_limits], dtype=float)
    rows = []
    rhs = []
    for axis in range(lower.shape[0]):
        upper_row = np.zeros(lower.shape[0], dtype=float)
        upper_row[axis] = 1.0
        rows.append(upper_row)
        rhs.append(float(upper[axis]))
        lower_row = np.zeros(lower.shape[0], dtype=float)
        lower_row[axis] = -1.0
        rows.append(lower_row)
        rhs.append(-float(lower[axis]))
    return _normalized_rows(np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float))


def _initial_metric(ellipsoid: CliqueEllipsoid, cfg: IrisZoConfig) -> np.ndarray:
    C = np.asarray(ellipsoid.C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        return np.eye(np.asarray(ellipsoid.center, dtype=float).size, dtype=float) * float(cfg.STARTING_BALL_RADIUS)
    sign, log_det = np.linalg.slogdet(C)
    if sign <= 0.0 or not np.isfinite(log_det):
        return np.eye(C.shape[0], dtype=float) * float(cfg.STARTING_BALL_RADIUS)
    return C


def _append_tangent_plane(
    rows: list[np.ndarray],
    rhs: list[float],
    *,
    center: np.ndarray,
    C: np.ndarray,
    q_collision: np.ndarray,
    margin: float,
) -> dict | None:
    metric = _metric_matrix(C)
    q_collision = np.asarray(q_collision, dtype=float).reshape(-1)
    center = np.asarray(center, dtype=float).reshape(-1)
    normal = metric @ (q_collision - center)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    normal = normal / norm
    offset = float(np.dot(normal, q_collision) - float(margin))
    if float(np.dot(normal, center)) > offset + 1e-9:
        return None
    rows.append(normal)
    rhs.append(offset)
    return {
        "normal": normal.tolist(),
        "offset": float(offset),
        "q_collision": q_collision.tolist(),
    }


def run_iris_zo(
    ellipsoid: CliqueEllipsoid,
    oracle,
    cfg: IrisZoConfig,
    *,
    region_id: int,
) -> IrisRegion:
    center = np.asarray(ellipsoid.center, dtype=float).reshape(-1)
    if oracle.is_self_collision(center):
        raise ValueError("IRIS-ZO seed must be collision-free.")
    A_box, b_box = _joint_box_halfspaces(oracle)
    if not is_inside_polytope(A_box, b_box, center):
        raise ValueError("IRIS-ZO seed must lie inside the joint box.")

    rng = np.random.default_rng(int(cfg.RNG_SEED) + int(region_id))
    C = _initial_metric(ellipsoid, cfg)
    sampler_state = center.copy()
    sign, log_det = np.linalg.slogdet(C)
    last_log_det = float(log_det if sign > 0.0 else math.log(max(float(cfg.STARTING_BALL_RADIUS), 1e-12)))
    final_A = np.asarray(A_box, dtype=float).copy()
    final_b = np.asarray(b_box, dtype=float).copy()
    iter_history: list[dict] = []
    Np = int(cfg.NUM_PARTICLES)
    Nb = int(cfg.NUM_BISECTION_STEPS)
    Nf = int(cfg.MAX_NEW_FACES_PER_INNER_ITER)
    margin = float(cfg.STEPBACK_MARGIN)
    has_fast_bisect = hasattr(oracle, "first_collision_on_segment_fast")

    for outer_iter in range(1, int(cfg.MAX_OUTER_ITERATIONS) + 1):
        w_A = final_A.copy()
        w_b = final_b.copy()
        inner_reports: list[dict] = []
        metric_matrix = _metric_matrix(C)

        for inner_iter in range(1, int(cfg.MAX_INNER_ITERATIONS) + 1):
            if not is_inside_polytope(w_A, w_b, sampler_state):
                sampler_state = center.copy()
            delta_ik = union_bound_delta(
                total_delta=float(cfg.DELTA),
                outer_iter=outer_iter,
                inner_iter=inner_iter,
            )
            sample_budget = max(
                Np,
                required_trials(
                    epsilon=float(cfg.EPSILON),
                    delta=delta_ik,
                    tau=float(cfg.TAU),
                ),
            )
            sample_points, sampler_state = sample_polytope_hit_and_run(
                w_A, w_b, sampler_state,
                num_samples=sample_budget,
                rng=rng,
                mixing_steps=int(cfg.HIT_AND_RUN_MIXING_STEPS),
            )

            collision_costs: list[tuple[float, np.ndarray]] = []
            num_collisions = 0
            pb_tag = f"R{region_id} O{outer_iter}/{cfg.MAX_OUTER_ITERATIONS} I{inner_iter}/{cfg.MAX_INNER_ITERATIONS}"
            pb = ProgressBar(len(sample_points), prefix=f"[iris-zo {pb_tag}]")
            for q in sample_points:
                if oracle.is_self_collision(q):
                    num_collisions += 1
                    d = q - center
                    cost = float(d @ metric_matrix @ d)
                    collision_costs.append((cost, q))
                pb.update(suffix=f"col={num_collisions}")
            pb.close(suffix=f"col={num_collisions}")

            test_report = unadaptive_collision_test(
                num_collisions,
                num_samples=len(sample_points),
                epsilon=float(cfg.EPSILON),
                total_delta=float(cfg.DELTA),
                tau=float(cfg.TAU),
                outer_iter=outer_iter,
                inner_iter=inner_iter,
            )

            plane_reports: list[dict] = []
            if not test_report["accept"]:
                collision_costs.sort(key=lambda t: t[0])
                candidates_to_bisect = collision_costs[:Np]

                boundary_items: list[tuple[float, np.ndarray]] = []
                for _cost, q_col in candidates_to_bisect:
                    if has_fast_bisect:
                        q_boundary = oracle.first_collision_on_segment_fast(
                            center, q_col, bisection_steps=Nb,
                        )
                    else:
                        res = oracle.first_collision_on_segment(
                            center, q_col,
                            num_steps=max(Nb, 4),
                            bisection_steps=Nb,
                        )
                        q_boundary = np.asarray(res["q_collision"], dtype=float) if res is not None else None
                    if q_boundary is None:
                        continue
                    if not is_inside_polytope(w_A, w_b, q_boundary, tol=1e-7):
                        continue
                    d = q_boundary - center
                    boundary_items.append((float(d @ metric_matrix @ d), q_boundary))

                boundary_items.sort(key=lambda t: t[0])
                rows_list = list(w_A)
                rhs_list = list(w_b)
                for b_cost, q_collision in boundary_items:
                    if len(plane_reports) >= Nf:
                        break
                    cur_A = np.asarray(rows_list, dtype=float)
                    cur_b = np.asarray(rhs_list, dtype=float)
                    if not is_inside_polytope(cur_A, cur_b, q_collision, tol=1e-7):
                        continue
                    plane_report = _append_tangent_plane(
                        rows_list, rhs_list,
                        center=center, C=C,
                        q_collision=q_collision, margin=margin,
                    )
                    if plane_report is not None:
                        plane_report["cost"] = b_cost
                        plane_reports.append(plane_report)
                w_A = np.asarray(rows_list, dtype=float)
                w_b = np.asarray(rhs_list, dtype=float)

            inner_reports.append({
                "inner_iteration": int(inner_iter),
                "num_samples": len(sample_points),
                "num_collisions": num_collisions,
                "statistical_test": test_report,
                "num_planes_added": len(plane_reports),
                "plane_reports": plane_reports,
            })
            if test_report["accept"] or not plane_reports:
                break

        stage_print(f"  R{region_id} outer={outer_iter}: 求解 MVIE ({w_A.shape[0]} 半平面) ...")
        try:
            candidate_A, candidate_b = _normalized_rows(w_A, w_b)
            mvie = maximum_volume_inscribed_ellipsoid(candidate_A, candidate_b)
        except Exception as exc:
            iter_history.append({
                "outer_iteration": int(outer_iter),
                "center": center.tolist(),
                "log_det": float(last_log_det),
                "num_planes": int(final_A.shape[0]),
                "inner_reports": inner_reports,
                "solver_error": str(exc),
            })
            break

        new_center = np.asarray(mvie["center"], dtype=float)
        new_C = np.asarray(mvie["C"], dtype=float)
        new_log_det = float(mvie["log_det"])
        if oracle.is_self_collision(new_center):
            break
        if new_log_det + 1e-9 < last_log_det:
            break

        center = new_center
        C = new_C
        sampler_state = center.copy()
        final_A = candidate_A
        final_b = candidate_b
        iter_history.append({
            "outer_iteration": int(outer_iter),
            "center": center.tolist(),
            "log_det": float(new_log_det),
            "num_planes": int(final_A.shape[0]),
            "inner_reports": inner_reports,
        })
        rel_growth = (new_log_det - last_log_det) / max(abs(last_log_det), 1e-9)
        last_log_det = new_log_det
        metric_matrix = _metric_matrix(C)
        stage_print(f"  R{region_id} outer={outer_iter}: log_det={new_log_det:+.4f}  planes={final_A.shape[0]}  growth={rel_growth:.6f}")
        if rel_growth < float(cfg.CONVERGENCE_TOL):
            break

    return IrisRegion(
        region_id=int(region_id),
        source_clique_indices=tuple(int(x) for x in ellipsoid.vertex_indices),
        A=final_A,
        b=final_b,
        center=center,
        C=C,
        log_det=float(last_log_det),
        iterations=tuple(iter_history),
    )
