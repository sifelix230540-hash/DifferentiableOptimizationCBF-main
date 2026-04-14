from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.optimize import linprog

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import (  # noqa: E402
    Robot,
    SimulationScene,
    load_config,
    _resolve,
)
from CBF_experiment.active.pybullet.self_collision.self_collision_backend_coal import (  # noqa: E402
    build_coal_link_models,
    classify_self_collision_sample,
)
from CBF_experiment.active.pybullet.self_collision.self_collision_cspace_hulls import (  # noqa: E402
    build_monitored_link_pairs,
    extract_revolute_metadata,
    extract_self_collision_monitor_metadata,
    normalize_joint_samples,
    sample_revolute_configurations,
)
from CBF_experiment.active.pybullet.self_collision.safe_cover.iris_exact import (  # noqa: E402
    run_iris_exact,
)
from CBF_experiment.active.pybullet.self_collision.safe_cover.pairwise_cspace_obstacles import (  # noqa: E402
    PairwiseObstacleConfig,
    build_pairwise_cspace_obstacles,
)


@dataclass(frozen=True)
class IrisSafeCoverConfig:
    CFG_PATH: str | None = None
    NUM_SAMPLES: int = 100000
    SEED: int = 11
    MIN_INDEX_GAP: int = 2
    PENETRATION_THRESH: float = -0.001
    MAX_REGIONS: int = 24
    MAX_IRIS_ITERS: int = 8
    COVERAGE_TARGET: float = 0.95
    MIN_RADIUS: float = 0.015
    INIT_RADIUS_SCALE: float = 0.90
    GROWTH_TOL: float = 1e-3
    SEED_CLEARANCE_MARGIN: float = 0.0
    PAIRWISE_SAMPLE_SCALE: float = 1.0
    OUTPUT_JSON: str = str(Path(_resolve("artifacts/sdf_exp/self_collision_iris_safe_cover.json")))
    EXPERIMENT_OUTPUT_JSON: str = str(Path(_resolve("artifacts/sdf_exp/self_collision_iris_safe_cover_experiment.json")))
    PAIRWISE_OUTPUT_JSON: str = str(Path(_resolve("artifacts/sdf_exp/pairwise_cspace_obstacles.json")))
    CURVE_NUM_POINTS: int = 90
    CURVE_MAX_ATTEMPTS: int = 24
    CURVE_KEEP_BEST_IF_COLLIDING: bool = True
    PLAYBACK_GUI: bool = False
    CAMERA_DISTANCE: float = 1.45
    CAMERA_YAW: float = -220.0
    CAMERA_PITCH: float = -28.0
    SLEEP_DT: float = 1.0 / 60.0


def _unit_box_halfspaces(dim: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    rhs = []
    for axis in range(dim):
        upper = np.zeros(dim, dtype=float)
        upper[axis] = 1.0
        rows.append(upper)
        rhs.append(1.0)
        lower = np.zeros(dim, dtype=float)
        lower[axis] = -1.0
        rows.append(lower)
        rhs.append(0.0)
    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def _point_in_polytope(point: np.ndarray, A: np.ndarray, b: np.ndarray, *, tol: float = 1e-9) -> bool:
    return bool(np.all(A @ point <= b + float(tol)))


def _points_in_polytope(points: np.ndarray, A: np.ndarray, b: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.all(points @ A.T <= b.reshape(1, -1) + float(tol), axis=1)


def _joint_box_halfspaces(joint_limits: list[list[float]] | list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    rhs = []
    for axis, (lo, hi) in enumerate(joint_limits):
        upper = np.zeros(len(joint_limits), dtype=float)
        upper[axis] = 1.0
        rows.append(upper)
        rhs.append(float(hi))
        lower = np.zeros(len(joint_limits), dtype=float)
        lower[axis] = -1.0
        rows.append(lower)
        rhs.append(-float(lo))
    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def _normalize_ellipsoid_to_unit_box(
    center_joint: np.ndarray,
    C_joint: np.ndarray,
    lower: np.ndarray,
    span: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.diag(1.0 / span.reshape(-1))
    center_normalized = (np.asarray(center_joint, dtype=float) - lower) / span
    C_normalized = scale @ np.asarray(C_joint, dtype=float)
    return center_normalized, C_normalized


def _flatten_pairwise_obstacles(pairwise_payload: dict) -> list[dict]:
    obstacles = []
    for pair_payload in pairwise_payload.get("pairs", []):
        for obstacle in pair_payload.get("obstacles", []):
            obstacles.append({
                "pair": list(pair_payload.get("pair", [])),
                "pair_names": list(pair_payload.get("pair_names", [])),
                "cluster_id": int(obstacle.get("cluster_id", 0)),
                "subspace_dim": int(pair_payload.get("subspace_dim", 0)),
                "active_q_indices": list(obstacle.get("active_q_indices", [])),
                "A_full": np.asarray(obstacle["A_full"], dtype=float),
                "b_full": np.asarray(obstacle["b_full"], dtype=float),
            })
    return obstacles


def _normalize_plane_rows(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(A, axis=1)
    norms = np.maximum(norms, 1e-12)
    return A / norms.reshape(-1, 1), norms


def _max_inscribed_ball(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    dim = A.shape[1]
    A_unit, norms = _normalize_plane_rows(A)
    A_ub = np.hstack([A_unit, np.ones((A.shape[0], 1), dtype=float)])
    b_ub = b / norms
    c = np.zeros(dim + 1, dtype=float)
    c[-1] = -1.0
    bounds = [(None, None)] * dim + [(0.0, None)]
    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"linprog failed while computing inscribed ball: {result.message}")
    center = np.asarray(result.x[:dim], dtype=float)
    radius = float(result.x[-1])
    return center, radius


def _sort_collision_points_by_distance(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=int)
    distances = np.linalg.norm(points - center.reshape(1, -1), axis=1)
    return np.argsort(distances)


def _initial_radius(seed: np.ndarray, collision_points: np.ndarray, *, scale: float) -> float:
    box_clearance = float(np.min(np.minimum(seed, 1.0 - seed)))
    collision_clearance = float("inf")
    if collision_points.size:
        collision_clearance = float(np.min(np.linalg.norm(collision_points - seed.reshape(1, -1), axis=1)))
    base_clearance = min(box_clearance, collision_clearance)
    if not np.isfinite(base_clearance):
        base_clearance = box_clearance
    return max(float(scale) * max(base_clearance, 0.0), 1e-6)


def _tangent_plane_for_ball(center: np.ndarray, radius: float, obstacle_point: np.ndarray) -> tuple[np.ndarray, float]:
    delta = np.asarray(obstacle_point, dtype=float) - np.asarray(center, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-12:
        raise ValueError("Obstacle point coincides with ball center; seed is not safely initialized.")
    normal = delta / norm
    offset = float(np.dot(normal, center) + radius)
    return normal, offset


def _build_region_from_seed(
    seed: np.ndarray,
    collision_points: np.ndarray,
    *,
    max_iters: int,
    min_radius: float,
    growth_tol: float,
    init_radius_scale: float,
) -> dict | None:
    dim = int(seed.shape[0])
    box_A, box_b = _unit_box_halfspaces(dim)
    center = np.asarray(seed, dtype=float).copy()
    radius = _initial_radius(center, collision_points, scale=float(init_radius_scale))
    if radius < float(min_radius):
        return None

    A = box_A.copy()
    b = box_b.copy()
    iter_history = []
    for _ in range(int(max_iters)):
        inside_mask = _points_in_polytope(collision_points, A, b, tol=1e-9)
        remaining = collision_points[inside_mask]
        order = _sort_collision_points_by_distance(remaining, center)
        remaining = remaining[order]

        while remaining.shape[0] > 0:
            obstacle_point = remaining[0]
            plane_a, plane_b = _tangent_plane_for_ball(center, radius, obstacle_point)
            A = np.vstack([A, plane_a.reshape(1, -1)])
            b = np.append(b, plane_b)
            still_inside = _points_in_polytope(remaining, plane_a.reshape(1, -1), np.asarray([plane_b]), tol=1e-9)
            remaining = remaining[still_inside]

        new_center, new_radius = _max_inscribed_ball(A, b)
        iter_history.append({
            "center": new_center.tolist(),
            "radius": float(new_radius),
            "num_planes": int(A.shape[0]),
        })
        if new_radius < float(min_radius):
            return None
        if new_radius <= radius * (1.0 + float(growth_tol)):
            center = new_center
            radius = new_radius
            break
        center = new_center
        radius = new_radius

    return {
        "center_normalized": center.tolist(),
        "radius_normalized": float(radius),
        "A_normalized": A.tolist(),
        "b_normalized": b.tolist(),
        "iterations": iter_history,
    }


def _convert_region_to_joint_space(region: dict, lower: np.ndarray, span: np.ndarray) -> dict:
    center_n = np.asarray(region["center_normalized"], dtype=float)
    A_n = np.asarray(region["A_normalized"], dtype=float)
    b_n = np.asarray(region["b_normalized"], dtype=float)
    center_q = center_n * span + lower
    A_q = A_n / span.reshape(1, -1)
    b_q = b_n + np.sum(A_n * (lower / span).reshape(1, -1), axis=1)
    return {
        **region,
        "center_joint": center_q.tolist(),
        "A_joint": A_q.tolist(),
        "b_joint": b_q.tolist(),
    }


def _sample_points_in_ball(
    center: np.ndarray,
    radius: float,
    *,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dim = int(center.shape[0])
    dirs = rng.normal(size=(int(num_points), dim))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    dirs = dirs / norms
    scales = rng.random(int(num_points)).reshape(-1, 1) ** (1.0 / max(dim, 1))
    return center.reshape(1, -1) + float(radius) * dirs * scales


def _sample_points_in_ellipsoid(
    center: np.ndarray,
    C: np.ndarray,
    *,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    unit = _sample_points_in_ball(
        np.zeros(np.asarray(center, dtype=float).shape[0], dtype=float),
        1.0,
        num_points=int(num_points),
        rng=rng,
    )
    return np.asarray(center, dtype=float).reshape(1, -1) + unit @ np.asarray(C, dtype=float).T


def _bezier_curve(control_points: np.ndarray, *, num_points: int) -> np.ndarray:
    cps = np.asarray(control_points, dtype=float).reshape(4, -1)
    t = np.linspace(0.0, 1.0, int(num_points), dtype=float).reshape(-1, 1)
    omt = 1.0 - t
    return (
        (omt ** 3) * cps[0].reshape(1, -1)
        + 3.0 * (omt ** 2) * t * cps[1].reshape(1, -1)
        + 3.0 * omt * (t ** 2) * cps[2].reshape(1, -1)
        + (t ** 3) * cps[3].reshape(1, -1)
    )


def build_random_safe_curve(
    region: dict,
    *,
    rng: np.random.Generator,
    num_points: int,
) -> dict:
    center = np.asarray(region["center_normalized"], dtype=float).reshape(-1)
    A = np.asarray(region["A_normalized"], dtype=float)
    b = np.asarray(region["b_normalized"], dtype=float)

    control_points = None
    curve = None
    for _ in range(64):
        if "C_normalized" in region:
            control_points = _sample_points_in_ellipsoid(
                center,
                np.asarray(region["C_normalized"], dtype=float) * 0.85,
                num_points=4,
                rng=rng,
            )
        else:
            radius = float(region["radius_normalized"])
            control_points = _sample_points_in_ball(
                center,
                max(0.7 * radius, 1e-6),
                num_points=4,
                rng=rng,
            )
        if not np.all(control_points @ A.T <= b.reshape(1, -1) + 1e-9):
            continue
        curve = _bezier_curve(control_points, num_points=int(num_points))
        if np.all(curve @ A.T <= b.reshape(1, -1) + 1e-9):
            break
    if control_points is None or curve is None:
        raise RuntimeError("Failed to sample a Bezier curve inside the selected IRIS region.")
    return {
        "control_points_normalized": control_points.tolist(),
        "curve_normalized": curve.tolist(),
    }


def _compose_full_q(q_base: np.ndarray, q_indices: list[int], revolute_q: np.ndarray) -> np.ndarray:
    q_full = np.asarray(q_base, dtype=float).copy()
    q_full[np.asarray(q_indices, dtype=int)] = np.asarray(revolute_q, dtype=float).reshape(-1)
    return q_full


def _pick_camera_target(robot) -> list[float]:
    ee_pos, _ = robot.get_ee_pose()
    base_pos, _ = robot.get_robobase_pose()
    target = 0.65 * np.asarray(ee_pos, dtype=float) + 0.35 * np.asarray(base_pos, dtype=float)
    target[2] = max(float(target[2]), 0.35)
    return target.tolist()


def _monitor_name_map(ids: list[int], names: list[str]) -> dict[int, str]:
    return {int(i): str(name) for i, name in zip(ids, names)}


def _pair_to_text(pair, name_map: dict[int, str]) -> str:
    if not pair:
        return "none"
    a, b = [int(x) for x in pair]
    return f"[{a}, {b}] ({name_map.get(a, a)} <-> {name_map.get(b, b)})"


def _evaluate_curve_with_robot(curve_joint: np.ndarray, cfg: IrisSafeCoverConfig) -> tuple[list[dict], dict]:
    created_connection = False
    if not p.isConnected():
        p.connect(p.DIRECT)
        created_connection = True
    try:
        robot = Robot(load_config(cfg.CFG_PATH))
        q_base, dq_base = robot.get_joint_state()
        revolute_ids, _revolute_names, _joint_limits, q_indices = extract_revolute_metadata(robot)
        monitored_link_ids, monitored_link_names = extract_self_collision_monitor_metadata(robot)
        monitored_pairs = build_monitored_link_pairs(monitored_link_ids, min_index_gap=int(cfg.MIN_INDEX_GAP))
        link_models = build_coal_link_models(robot, monitored_link_ids)
        name_map = _monitor_name_map(monitored_link_ids, monitored_link_names)

        step_reports = []
        min_distance = float("inf")
        any_collision = False
        min_step = None
        for step_idx, rq in enumerate(np.asarray(curve_joint, dtype=float), start=1):
            q_full = _compose_full_q(q_base, q_indices, rq)
            robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
            metric = classify_self_collision_sample(
                robot,
                monitored_pairs=monitored_pairs,
                link_models=link_models,
                penetration_thresh=float(cfg.PENETRATION_THRESH),
            )
            pair = metric.get("active_pair")
            report = {
                "step": int(step_idx),
                "min_distance": float(metric["min_distance"]),
                "is_collision": bool(metric["is_collision"]),
                "active_pair": pair,
                "active_pair_text": _pair_to_text(pair, name_map),
                "contact_penetration_depth": metric.get("contact_penetration_depth"),
            }
            step_reports.append(report)
            if float(metric["min_distance"]) < min_distance:
                min_distance = float(metric["min_distance"])
                min_step = report
            any_collision = any_collision or bool(metric["is_collision"])
        robot.set_joint_state(q_base, dq_base)
        summary = {
            "num_steps": int(len(step_reports)),
            "min_distance": float(min_distance),
            "any_collision": bool(any_collision),
            "min_step": min_step,
            "monitored_link_indices": [int(i) for i in monitored_link_ids],
            "monitored_link_names": [str(n) for n in monitored_link_names],
        }
        return step_reports, summary
    finally:
        if created_connection and p.isConnected():
            p.disconnect()


def playback_safe_curve_gui(report: dict, cfg: IrisSafeCoverConfig) -> dict:
    robot_cfg = load_config(cfg.CFG_PATH)
    scene = SimulationScene(robot_cfg)
    scene.enable_rendering()
    robot = Robot(robot_cfg)
    q_base, dq_base = robot.get_joint_state()
    revolute_ids, _revolute_names, _joint_limits, q_indices = extract_revolute_metadata(robot)
    monitored_link_ids = [int(x) for x in report["curve_summary"]["monitored_link_indices"]]
    monitored_link_names = [str(x) for x in report["curve_summary"]["monitored_link_names"]]
    name_map = _monitor_name_map(monitored_link_ids, monitored_link_names)

    status_ids = [-1] * 8
    trace_color = [0.10, 0.45, 0.95]
    prev_ee = None
    try:
        for step_idx, rq in enumerate(np.asarray(report["curve_joint"], dtype=float), start=1):
            q_full = _compose_full_q(q_base, q_indices, rq)
            robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
            ee_pos, ee_quat = robot.get_ee_pose()
            if prev_ee is not None:
                p.addUserDebugLine(prev_ee.tolist(), ee_pos.tolist(), trace_color, lineWidth=2.5)
            prev_ee = ee_pos.copy()
            p.resetDebugVisualizerCamera(
                cameraDistance=float(cfg.CAMERA_DISTANCE),
                cameraYaw=float(cfg.CAMERA_YAW),
                cameraPitch=float(cfg.CAMERA_PITCH),
                cameraTargetPosition=_pick_camera_target(robot),
            )
            step_report = report["curve_steps"][step_idx - 1]
            region_summary = report.get("selected_region_summary", {})
            base_pos, _ = robot.get_robobase_pose()
            anchor = np.asarray(base_pos, dtype=float) + np.array([0.0, -0.38, 0.92], dtype=float)
            lines = [
                f"region={report['selected_region_id']} step={step_idx}/{len(report['curve_joint'])}",
                f"logdet={float(report.get('selected_region_log_det', 0.0)):+.4f} planes={int(region_summary.get('num_planes', 0))}",
                f"selected_obstacles={int(region_summary.get('num_selected_obstacles', 0))} new_cover={int(region_summary.get('newly_covered_free_samples', 0))}",
                f"curve_min_distance={float(report['curve_summary']['min_distance']):+.6f}",
                f"step_distance={float(step_report['min_distance']):+.6f} collision={bool(step_report['is_collision'])}",
                f"pair={_pair_to_text(step_report.get('active_pair'), name_map)}",
                f"contact_depth={step_report.get('contact_penetration_depth')}",
                "curve playback: this trajectory is sampled inside one convex IRIS region",
            ]
            for ti, line in enumerate(lines):
                pos = (anchor + np.array([0.0, 0.0, -0.07 * ti], dtype=float)).tolist()
                status_ids[ti] = p.addUserDebugText(
                    line,
                    pos,
                    textColorRGB=[0.08, 0.08, 0.08],
                    textSize=1.15,
                    replaceItemUniqueId=status_ids[ti],
                )
            for lid in monitored_link_ids:
                color = [0.82, 0.82, 0.82, 1.0]
                active_pair = step_report.get("active_pair") or []
                if int(lid) in [int(x) for x in active_pair]:
                    color = [0.95, 0.2, 0.1, 1.0] if bool(step_report["is_collision"]) else [0.1, 0.8, 0.3, 1.0]
                p.changeVisualShape(int(robot.body_id), int(lid), rgbaColor=color)
            time.sleep(float(cfg.SLEEP_DT))
        while p.isConnected():
            time.sleep(1.0 / 30.0)
    finally:
        robot.set_joint_state(q_base, dq_base)
        if p.isConnected():
            p.disconnect()
    return report


def _collect_labeled_samples(cfg: IrisSafeCoverConfig) -> dict:
    rng = np.random.default_rng(int(cfg.SEED))
    created_connection = False
    if not p.isConnected():
        p.connect(p.DIRECT)
        created_connection = True
    try:
        robot = Robot(load_config(cfg.CFG_PATH))
        q_base, dq_base = robot.get_joint_state()
        revolute_ids, revolute_names, joint_limits, q_indices = extract_revolute_metadata(robot)
        monitored_link_ids, monitored_link_names = extract_self_collision_monitor_metadata(robot)
        monitored_pairs = build_monitored_link_pairs(monitored_link_ids, min_index_gap=int(cfg.MIN_INDEX_GAP))
        link_models = build_coal_link_models(robot, monitored_link_ids)
        sampled_q = sample_revolute_configurations(
            q_base,
            q_indices,
            joint_limits,
            num_samples=int(cfg.NUM_SAMPLES),
            rng=rng,
        )

        free_points = []
        free_clearances = []
        collision_points = []
        collision_clearances = []
        for idx, q in enumerate(sampled_q, start=1):
            robot.set_joint_state(q, dq=np.zeros_like(q))
            metric = classify_self_collision_sample(
                robot,
                monitored_pairs=monitored_pairs,
                link_models=link_models,
                penetration_thresh=float(cfg.PENETRATION_THRESH),
            )
            rq = np.asarray(q, dtype=float)[q_indices]
            if metric["is_collision"]:
                collision_points.append(rq.tolist())
                collision_clearances.append(float(metric["min_distance"]))
            else:
                free_points.append(rq.tolist())
                free_clearances.append(float(metric["min_distance"]))
            if idx % 500 == 0 or idx == int(cfg.NUM_SAMPLES):
                print(
                    f"\r[iris-cover] sampling {idx}/{cfg.NUM_SAMPLES} "
                    f"free={len(free_points)} collision={len(collision_points)}",
                    end="",
                    flush=True,
                )
        print()
        robot.set_joint_state(q_base, dq_base)
        return {
            "revolute_ids": [int(j) for j in revolute_ids],
            "revolute_names": [str(name) for name in revolute_names],
            "monitored_link_ids": [int(j) for j in monitored_link_ids],
            "monitored_link_names": [str(name) for name in monitored_link_names],
            "joint_limits": [[float(lo), float(hi)] for lo, hi in joint_limits],
            "free_points": np.asarray(free_points, dtype=float).reshape(-1, len(revolute_ids)),
            "free_clearances": np.asarray(free_clearances, dtype=float).reshape(-1),
            "collision_points": np.asarray(collision_points, dtype=float).reshape(-1, len(revolute_ids)),
            "collision_clearances": np.asarray(collision_clearances, dtype=float).reshape(-1),
        }
    finally:
        if created_connection and p.isConnected():
            p.disconnect()


def run_iris_safe_cover(cfg: IrisSafeCoverConfig = IrisSafeCoverConfig()) -> dict:
    sample_data = _collect_labeled_samples(cfg)
    free_points = np.asarray(sample_data["free_points"], dtype=float)
    free_clearances = np.asarray(sample_data["free_clearances"], dtype=float)
    joint_limits = sample_data["joint_limits"]

    if free_points.shape[0] == 0:
        raise ValueError("No free samples collected; cannot build safe cover.")
    free_normalized, lower, span = normalize_joint_samples(free_points, joint_limits)
    pairwise_payload = build_pairwise_cspace_obstacles(
        PairwiseObstacleConfig(
            CFG_PATH=cfg.CFG_PATH,
            SEED=cfg.SEED,
            MIN_INDEX_GAP=cfg.MIN_INDEX_GAP,
            PENETRATION_THRESH=cfg.PENETRATION_THRESH,
            SAMPLE_SCALE=cfg.PAIRWISE_SAMPLE_SCALE,
            OUTPUT_JSON=cfg.PAIRWISE_OUTPUT_JSON,
        )
    )
    obstacles = _flatten_pairwise_obstacles(pairwise_payload)
    if not obstacles:
        raise ValueError("Pairwise obstacle precomputation returned no convex obstacles.")
    domain_A, domain_b = _joint_box_halfspaces(joint_limits)

    covered = np.zeros(free_normalized.shape[0], dtype=bool)
    regions = []
    for region_idx in range(int(cfg.MAX_REGIONS)):
        uncovered_idx = np.where(~covered)[0]
        if uncovered_idx.size == 0:
            break
        uncovered_clearance = free_clearances[uncovered_idx]
        candidate_order = uncovered_idx[np.argsort(-uncovered_clearance)]
        seed = None
        for idx in candidate_order:
            candidate = free_points[int(idx)]
            candidate_n = free_normalized[int(idx)]
            if np.min(np.minimum(candidate_n, 1.0 - candidate_n)) <= float(cfg.SEED_CLEARANCE_MARGIN):
                continue
            seed = candidate
            seed_index = int(idx)
            break
        if seed is None:
            break

        iris_result = run_iris_exact(
            seed=seed,
            obstacles=obstacles,
            domain_A=domain_A,
            domain_b=domain_b,
            max_iters=int(cfg.MAX_IRIS_ITERS),
            convergence_tol=float(cfg.GROWTH_TOL),
        )
        A_joint = np.asarray(iris_result["A"], dtype=float)
        b_joint = np.asarray(iris_result["b"], dtype=float)
        covers = _points_in_polytope(free_points, A_joint, b_joint, tol=1e-9)
        newly_covered = covers & (~covered)
        if int(np.sum(newly_covered)) == 0:
            covered[seed_index] = True
            continue

        covered |= covers
        center_joint = np.asarray(iris_result["center"], dtype=float)
        C_joint = np.asarray(iris_result["C"], dtype=float)
        center_normalized, C_normalized = _normalize_ellipsoid_to_unit_box(center_joint, C_joint, lower, span)
        region = {
            "region_id": int(region_idx),
            "seed_free_index": int(seed_index),
            "seed_joint": free_points[seed_index].tolist(),
            "seed_normalized": free_normalized[seed_index].tolist(),
            "seed_clearance": float(free_clearances[seed_index]),
            "center_joint": center_joint.tolist(),
            "center_normalized": center_normalized.tolist(),
            "C_joint": C_joint.tolist(),
            "C_normalized": C_normalized.tolist(),
            "A_joint": A_joint.tolist(),
            "b_joint": b_joint.tolist(),
            "A_normalized": (A_joint * span.reshape(1, -1)).tolist(),
            "b_normalized": (b_joint - np.sum(A_joint * lower.reshape(1, -1), axis=1)).tolist(),
            "log_det": float(iris_result["log_det"]),
            "iterations": iris_result["iterations"],
            "selected_planes": iris_result["planes"],
            "num_covered_free_samples": int(np.sum(covers)),
            "newly_covered_free_samples": int(np.sum(newly_covered)),
            "coverage_ratio_after_region": float(np.mean(covered)),
        }
        regions.append(region)
        print(
            f"[iris-cover] region={region_idx + 1} "
            f"logdet={region['log_det']:+.4f} "
            f"new={region['newly_covered_free_samples']} "
            f"coverage={region['coverage_ratio_after_region']:.4f}"
        )
        if float(np.mean(covered)) >= float(cfg.COVERAGE_TARGET):
            break

    output = {
        "method": "paper_style_iris_safe_cover",
        "notes": [
            "6R free seeds are still selected from coal-labeled Monte Carlo samples.",
            "Obstacle input is now precomputed pairwise collision subspace polytopes lifted to full 6R.",
            "IRIS main loop uses nearest-obstacle separating hyperplanes and an MVIE E-step solved with cvxpy.",
            "The 6R joint-limit box is folded into the domain polytope for each region.",
        ],
        "num_samples": int(cfg.NUM_SAMPLES),
        "num_free_samples": int(free_points.shape[0]),
        "coverage_ratio": float(np.mean(covered)),
        "uncovered_free_samples": int(np.sum(~covered)),
        "joint_indices": sample_data["revolute_ids"],
        "joint_names": sample_data["revolute_names"],
        "monitored_link_indices": sample_data["monitored_link_ids"],
        "monitored_link_names": sample_data["monitored_link_names"],
        "joint_limits": joint_limits,
        "pairwise_obstacle_json": str(Path(cfg.PAIRWISE_OUTPUT_JSON)),
        "num_pairwise_obstacles": int(len(obstacles)),
        "regions": regions,
    }
    output_path = Path(cfg.OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[iris-cover] saved -> {output_path}")
    return output


def run_iris_safe_cover_experiment(cfg: IrisSafeCoverConfig = IrisSafeCoverConfig()) -> dict:
    rng = np.random.default_rng(int(cfg.SEED))
    cover = run_iris_safe_cover(cfg)
    joint_limits = cover["joint_limits"]
    lower = np.asarray([float(lo) for lo, _ in joint_limits], dtype=float)
    upper = np.asarray([float(hi) for _, hi in joint_limits], dtype=float)
    span = upper - lower

    regions = sorted(
        list(cover.get("regions", [])),
        key=lambda region: (
            -int(region.get("newly_covered_free_samples", 0)),
            -float(region.get("log_det", -1e18)),
        ),
    )
    if not regions:
        raise ValueError("IRIS safe cover produced no usable regions.")

    best_report = None
    for region in regions:
        for attempt in range(int(cfg.CURVE_MAX_ATTEMPTS)):
            curve_payload = build_random_safe_curve(
                region,
                rng=rng,
                num_points=int(cfg.CURVE_NUM_POINTS),
            )
            curve_normalized = np.asarray(curve_payload["curve_normalized"], dtype=float)
            curve_joint = curve_normalized * span.reshape(1, -1) + lower.reshape(1, -1)
            step_reports, curve_summary = _evaluate_curve_with_robot(curve_joint, cfg)
            report = {
                "cover_json": str(Path(cfg.OUTPUT_JSON)),
                "pairwise_obstacle_json": str(Path(cfg.PAIRWISE_OUTPUT_JSON)),
                "selected_region_id": int(region["region_id"]),
                "selected_region_log_det": float(region["log_det"]),
                "selected_region_summary": {
                    "num_planes": int(len(region.get("A_joint", []))),
                    "num_selected_obstacles": int(len(region.get("selected_planes", []))),
                    "newly_covered_free_samples": int(region.get("newly_covered_free_samples", 0)),
                },
                "curve_attempt": int(attempt + 1),
                "control_points_normalized": curve_payload["control_points_normalized"],
                "curve_normalized": curve_payload["curve_normalized"],
                "curve_joint": curve_joint.tolist(),
                "curve_steps": step_reports,
                "curve_summary": curve_summary,
            }
            if best_report is None or float(curve_summary["min_distance"]) > float(best_report["curve_summary"]["min_distance"]):
                best_report = report
            if not bool(curve_summary["any_collision"]):
                best_report = report
                break
        if best_report is not None and not bool(best_report["curve_summary"]["any_collision"]):
            break

    if best_report is None:
        raise RuntimeError("Failed to generate any curve candidate inside the safe cover.")
    if bool(best_report["curve_summary"]["any_collision"]) and not bool(cfg.CURVE_KEEP_BEST_IF_COLLIDING):
        raise RuntimeError("No collision-free curve was found inside the sampled safe regions.")

    output_path = Path(cfg.EXPERIMENT_OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[iris-cover-exp] "
        f"region={best_report['selected_region_id']} "
        f"curve_min_distance={float(best_report['curve_summary']['min_distance']):+.6f} "
        f"collision={bool(best_report['curve_summary']['any_collision'])}"
    )
    print(f"[iris-cover-exp] saved -> {output_path}")
    if bool(cfg.PLAYBACK_GUI):
        playback_safe_curve_gui(best_report, cfg)
    return best_report


if __name__ == "__main__":
    run_iris_safe_cover_experiment()

