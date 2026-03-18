from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass
class DynamicObstacleDef:
    center_fn: Callable[[float], Array]
    velocity_fn: Callable[[float], Array]
    radius: float
    color: str
    name: str


@dataclass
class TwoDConfig:
    dt: float = 0.02
    max_steps: int = 1500
    goal_tol: float = 0.15
    safety_margin: float = 0.2
    mass: float = 2.0
    start: Array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0], dtype=float))
    goal: Array = field(default_factory=lambda: np.array([5.0, 0.0], dtype=float))
    u_max: float = 8.0
    kp: float = 1.2
    kd: float = 1.5
    plan_bounds: tuple[float, float, float, float] = (-0.5, 5.8, -1.8, 1.8)
    rrt_step_len: float = 0.35
    rrt_goal_sample_rate: float = 0.18
    rrt_search_radius: float = 0.7
    rrt_max_iter: int = 1200
    rrt_clearance: float = 0.08
    path_point_spacing: float = 0.10
    path_lookahead: int = 10


def build_default_problem() -> tuple[TwoDConfig, list[dict], list[DynamicObstacleDef]]:
    cfg = TwoDConfig()

    static_obs = [
        {"center": np.array([2.5, 0.0], dtype=float), "radius": 0.55, "velocity": np.zeros(2, dtype=float)},
    ]

    def dyn1_center(t: float) -> Array:
        return np.array([2.5, max(1.3 - 1.2 * t, -1.5)], dtype=float)

    def dyn1_velocity(t: float) -> Array:
        return np.array([0.0, -1.2], dtype=float) if 1.3 - 1.2 * t > -1.5 else np.zeros(2, dtype=float)

    def dyn2_center(t: float) -> Array:
        if t < 1.0:
            return np.array([3.5, 1.3], dtype=float)
        y = 1.3 - 0.8 * (t - 1.0)
        return np.array([3.5, max(y, -1.5)], dtype=float)

    def dyn2_velocity(t: float) -> Array:
        if t < 1.0:
            return np.zeros(2, dtype=float)
        return np.array([0.0, -0.8], dtype=float) if 1.3 - 0.8 * (t - 1.0) > -1.5 else np.zeros(2, dtype=float)

    def dyn3_center(t: float) -> Array:
        if t < 2.0:
            return np.array([4.2, -1.3], dtype=float)
        y = -1.3 + 0.9 * (t - 2.0)
        return np.array([4.2, min(y, 1.5)], dtype=float)

    def dyn3_velocity(t: float) -> Array:
        if t < 3.0:
            return np.zeros(2, dtype=float)
        return np.array([0.0, 0.9], dtype=float) if -1.3 + 0.9 * (t - 3.0) < 1.5 else np.zeros(2, dtype=float)

    dir_norm = np.sqrt(1.0 + 0.36)
    dx4, dy4 = 1.0 / dir_norm, -0.6 / dir_norm

    def dyn4_center(t: float) -> Array:
        if t < 0.5:
            return np.array([1.6, 1.0], dtype=float)
        prog = min((t - 0.5) * 1.1, 3.5)
        return np.array([1.6 + prog * dx4, 1.0 + prog * dy4], dtype=float)

    def dyn4_velocity(t: float) -> Array:
        if t < 0.5:
            return np.zeros(2, dtype=float)
        if (t - 0.5) * 1.1 >= 3.5:
            return np.zeros(2, dtype=float)
        return np.array([1.1 * dx4, 1.1 * dy4], dtype=float)

    dyn_defs = [
        DynamicObstacleDef(dyn1_center, dyn1_velocity, 0.22, "orange", "DynObs1"),
        DynamicObstacleDef(dyn2_center, dyn2_velocity, 0.20, "gold", "DynObs2"),
        DynamicObstacleDef(dyn3_center, dyn3_velocity, 0.20, "darkorange", "DynObs3"),
        DynamicObstacleDef(dyn4_center, dyn4_velocity, 0.20, "purple", "DynObs4"),
    ]
    return cfg, static_obs, dyn_defs


def get_obstacles_at(t: float, static_obs: list[dict], dyn_defs: list[DynamicObstacleDef]) -> list[dict]:
    obs = [
        {"center": o["center"].copy(), "radius": o["radius"], "velocity": o["velocity"].copy()}
        for o in static_obs
    ]
    for d in dyn_defs:
        obs.append({"center": d.center_fn(t), "radius": d.radius, "velocity": d.velocity_fn(t)})
    return obs


def step_dynamics(x: Array, u: Array, cfg: TwoDConfig) -> Array:
    pos = x[:2]
    vel = x[2:]
    acc = u / cfg.mass
    x_next = np.empty_like(x, dtype=float)
    x_next[:2] = pos + cfg.dt * vel + 0.5 * cfg.dt**2 * acc
    x_next[2:] = vel + cfg.dt * acc
    return x_next


def nominal_pd_control(
    x: Array,
    cfg: TwoDConfig,
    target: Array | None = None,
    u_bias: Array | None = None,
) -> Array:
    pos = x[:2]
    vel = x[2:]
    goal = cfg.goal if target is None else target
    u_nom = cfg.kp * (goal - pos) - cfg.kd * vel
    if u_bias is not None:
        u_nom = u_nom + u_bias
    return np.clip(u_nom, -cfg.u_max, cfg.u_max)


def build_track_reference(
    x0: Array,
    cfg: TwoDConfig,
    horizon: int,
    nominal_fn: Callable[[Array], Array],
) -> Array:
    x_track = np.zeros((horizon + 1, x0.shape[0]), dtype=float)
    x_track[0] = x0
    for k in range(horizon):
        x_track[k + 1] = step_dynamics(x_track[k], nominal_fn(x_track[k]), cfg)
    return x_track


def get_min_h(x: Array, obs_list: list[dict], cfg: TwoDConfig) -> float:
    pos = x[:2]
    min_h = float("inf")
    for o in obs_list:
        min_h = min(min_h, np.linalg.norm(pos - o["center"]) - (o["radius"] + cfg.safety_margin))
    return min_h


def reward_terms(
    x_next: Array,
    u: Array,
    goal: Array,
    obs_list: list[dict],
    cfg: TwoDConfig,
    x_prev: Array | None = None,
) -> dict[str, float]:
    curr_dist = float(np.linalg.norm(x_next[:2] - goal))
    pos_err = min(curr_dist, 8.0)

    progress = 0.0
    if x_prev is not None:
        prev_dist = float(np.linalg.norm(x_prev[:2] - goal))
        progress = prev_dist - curr_dist

    vel_pen = float(np.linalg.norm(x_next[2:]))
    control_pen = float(np.linalg.norm(u))
    min_h = get_min_h(x_next, obs_list, cfg)
    collision = float(min_h < 0.0)

    prox_threshold = 0.5
    proximity_pen = 0.0
    if min_h < prox_threshold:
        proximity_pen = (prox_threshold - max(min_h, -0.5)) / prox_threshold

    xmin, xmax, ymin, ymax = cfg.plan_bounds
    oob_margin = 2.0
    out_of_bounds = float(
        x_next[0] < xmin - oob_margin
        or x_next[0] > xmax + oob_margin
        or x_next[1] < ymin - oob_margin
        or x_next[1] > ymax + oob_margin
    )

    goal_proximity = 1.0 / (1.0 + curr_dist)

    reward = (
        10.0 * progress
        + 0.5 * goal_proximity
        - 0.02 * vel_pen
        - 0.01 * control_pen
        - 2.0 * proximity_pen
        - 10.0 * collision
        - 3.0 * out_of_bounds
        - 0.005
    )

    return {
        "reward": reward,
        "pos_err": curr_dist,
        "vel_pen": vel_pen,
        "control_pen": control_pen,
        "min_h": min_h,
        "collision": collision,
        "out_of_bounds": out_of_bounds,
        "progress": progress,
        "goal_proximity": goal_proximity,
    }


def _in_bounds(p: Array, cfg: TwoDConfig) -> bool:
    xmin, xmax, ymin, ymax = cfg.plan_bounds
    return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax


def _point_collision_free(p: Array, static_obs: list[dict], cfg: TwoDConfig) -> bool:
    for o in static_obs:
        if np.linalg.norm(p - o["center"]) <= o["radius"] + cfg.safety_margin + cfg.rrt_clearance:
            return False
    return _in_bounds(p, cfg)


def _segment_collision_free(p0: Array, p1: Array, static_obs: list[dict], cfg: TwoDConfig) -> bool:
    seg = p1 - p0
    seg_len = np.linalg.norm(seg)
    n_chk = max(2, int(np.ceil(seg_len / 0.05)))
    for i in range(n_chk + 1):
        pt = p0 + (i / n_chk) * seg
        if not _point_collision_free(pt, static_obs, cfg):
            return False
    return True


def _extract_path(nodes: list[Array], parents: list[int], goal_idx: int) -> Array:
    path = []
    idx = goal_idx
    while idx >= 0:
        path.append(nodes[idx])
        idx = parents[idx]
    path.reverse()
    return np.array(path)


def _shortcut_path(path: Array, static_obs: list[dict], cfg: TwoDConfig) -> Array:
    if len(path) <= 2:
        return path
    short = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _segment_collision_free(path[i], path[j], static_obs, cfg):
                break
            j -= 1
        short.append(path[j])
        i = j
    return np.array(short)


def _densify_path(path: Array, spacing: float) -> Array:
    pts = [path[0]]
    for i in range(len(path) - 1):
        seg = path[i + 1] - path[i]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-9:
            continue
        n_seg = max(1, int(np.ceil(seg_len / spacing)))
        for j in range(1, n_seg + 1):
            pts.append(path[i] + (j / n_seg) * seg)
    return np.array(pts)


def fallback_detour_path(start_pos: Array, goal_pos: Array, static_obs: list[dict], cfg: TwoDConfig) -> Array:
    obs = static_obs[0]
    offset_y = obs["radius"] + cfg.safety_margin + 0.45
    waypoint = np.array([obs["center"][0], offset_y], dtype=float)
    return np.array([start_pos, waypoint, goal_pos], dtype=float)


def plan_static_rrt_star(
    start_pos: Array,
    goal_pos: Array,
    static_obs: list[dict],
    cfg: TwoDConfig,
    seed: int = 7,
) -> Array:
    rng = np.random.default_rng(seed)
    start_pos = np.asarray(start_pos, dtype=float)
    goal_pos = np.asarray(goal_pos, dtype=float)

    if _segment_collision_free(start_pos, goal_pos, static_obs, cfg):
        return _densify_path(np.array([start_pos, goal_pos]), cfg.path_point_spacing)

    nodes = [start_pos]
    parents = [-1]
    costs = [0.0]
    goal_indices: list[int] = []
    xmin, xmax, ymin, ymax = cfg.plan_bounds

    for _ in range(cfg.rrt_max_iter):
        if rng.random() < cfg.rrt_goal_sample_rate:
            sample = goal_pos
        else:
            sample = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=float)

        dists = np.array([np.linalg.norm(n - sample) for n in nodes])
        nearest_idx = int(np.argmin(dists))
        nearest = nodes[nearest_idx]
        direction = sample - nearest
        dist = np.linalg.norm(direction)
        if dist < 1e-9:
            continue

        new_pos = nearest + direction / dist * min(cfg.rrt_step_len, dist)
        if not _point_collision_free(new_pos, static_obs, cfg):
            continue
        if not _segment_collision_free(nearest, new_pos, static_obs, cfg):
            continue

        near_indices = [i for i, n in enumerate(nodes) if np.linalg.norm(n - new_pos) <= cfg.rrt_search_radius]
        best_parent = nearest_idx
        best_cost = costs[nearest_idx] + np.linalg.norm(new_pos - nearest)

        for i in near_indices:
            cand_cost = costs[i] + np.linalg.norm(nodes[i] - new_pos)
            if cand_cost < best_cost and _segment_collision_free(nodes[i], new_pos, static_obs, cfg):
                best_parent = i
                best_cost = cand_cost

        nodes.append(new_pos)
        parents.append(best_parent)
        costs.append(best_cost)
        new_idx = len(nodes) - 1

        for i in near_indices:
            rewired_cost = best_cost + np.linalg.norm(nodes[i] - new_pos)
            if rewired_cost + 1e-9 < costs[i] and _segment_collision_free(new_pos, nodes[i], static_obs, cfg):
                parents[i] = new_idx
                costs[i] = rewired_cost

        if np.linalg.norm(new_pos - goal_pos) <= cfg.rrt_step_len and _segment_collision_free(new_pos, goal_pos, static_obs, cfg):
            nodes.append(goal_pos)
            parents.append(new_idx)
            costs.append(best_cost + np.linalg.norm(goal_pos - new_pos))
            goal_indices.append(len(nodes) - 1)

    if goal_indices:
        best_goal_idx = min(goal_indices, key=lambda i: costs[i])
        raw_path = _extract_path(nodes, parents, best_goal_idx)
    else:
        raw_path = fallback_detour_path(start_pos, goal_pos, static_obs, cfg)

    return _densify_path(_shortcut_path(raw_path, static_obs, cfg), cfg.path_point_spacing)


def get_path_target(path: Array, pos: Array, lookahead: int) -> Array:
    dists = np.linalg.norm(path - pos, axis=1)
    nearest_idx = int(np.argmin(dists))
    target_idx = min(nearest_idx + lookahead, len(path) - 1)
    return path[target_idx]
