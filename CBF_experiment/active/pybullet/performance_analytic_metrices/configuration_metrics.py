from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pybullet as p


DEFAULT_SELECTION_WEIGHTS = {
    "aabb_max_dim": 0.25,
    "inverse_condition": 0.20,
    "min_singular_value": 0.15,
    "self_collision_distance": 0.15,
    "environment_distance": 0.15,
    "joint_limit_margin": 0.10,
}


def select_motion_rows(jacobian, motion_component: str | Sequence[int] = "combined") -> np.ndarray:
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2:
        raise ValueError("Jacobian must be a 2D array.")
    if jac.shape[0] == 0:
        return jac.copy()

    if isinstance(motion_component, str):
        mode = motion_component.strip().lower()
        if mode in {"combined", "all"}:
            return jac.copy()
        if mode == "linear":
            return jac[: min(3, jac.shape[0]), :].copy()
        if mode == "angular":
            start = min(3, jac.shape[0])
            return jac[start:, :].copy()
        raise ValueError(f"Unsupported motion component: {motion_component}")

    mask = np.asarray(list(motion_component), dtype=bool).reshape(-1)
    if mask.shape[0] != jac.shape[0]:
        raise ValueError("Custom motion component mask must match Jacobian rows.")
    return jac[mask, :].copy()


def compute_manipulability_report(
    jacobian,
    motion_component: str | Sequence[int] = "combined",
    eps: float = 1e-9,
) -> dict:
    jac = select_motion_rows(jacobian, motion_component=motion_component)
    if jac.size == 0:
        return {
            "manipulability": 0.0,
            "min_singular_value": 0.0,
            "inverse_condition": 0.0,
            "singular_values": [],
        }
    singular_values = np.linalg.svd(jac, compute_uv=False)
    if singular_values.size == 0:
        return {
            "manipulability": 0.0,
            "min_singular_value": 0.0,
            "inverse_condition": 0.0,
            "singular_values": [],
        }
    positive_sv = singular_values[singular_values > eps]
    manipulability = float(np.prod(positive_sv)) if positive_sv.size else 0.0
    sigma_max = float(np.max(singular_values))
    sigma_min = float(np.min(singular_values))
    inverse_condition = 0.0 if sigma_max <= eps else float(sigma_min / sigma_max)
    return {
        "manipulability": manipulability,
        "min_singular_value": sigma_min,
        "inverse_condition": inverse_condition,
        "singular_values": singular_values.astype(float).tolist(),
    }


def compute_joint_limit_margin(q, joint_limits: Iterable[Sequence[float]]) -> dict:
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    limits = list(joint_limits)
    if len(limits) != q_arr.shape[0]:
        raise ValueError("Joint limit list must match the joint configuration length.")
    margins = []
    for qi, (lo, hi) in zip(q_arr, limits):
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            margins.append(1.0)
            continue
        span = max(hi - lo, 1e-9)
        normalized = 2.0 * min(float(qi) - lo, hi - float(qi)) / span
        margins.append(float(np.clip(normalized, 0.0, 1.0)))
    margins_arr = np.asarray(margins, dtype=float)
    return {
        "per_joint": margins_arr.tolist(),
        "min_margin": float(np.min(margins_arr)) if margins_arr.size else 1.0,
        "mean_margin": float(np.mean(margins_arr)) if margins_arr.size else 1.0,
    }


def summarize_clearance_entries(entries: Iterable[dict] | None) -> dict:
    env_distances: list[float] = []
    self_distances: list[float] = []
    for entry in list(entries or []):
        distance = float(entry.get("distance", np.inf))
        kind = str(entry.get("kind", "environment"))
        if kind == "self_collision":
            self_distances.append(distance)
        else:
            env_distances.append(distance)

    def _min_or_inf(values: list[float]) -> float:
        return float(np.min(np.asarray(values, dtype=float))) if values else float("inf")

    env_min = _min_or_inf(env_distances)
    self_min = _min_or_inf(self_distances)
    return {
        "min_distance": float(min(env_min, self_min)),
        "environment_distance": env_min,
        "self_collision_distance": self_min,
        "environment_count": int(len(env_distances)),
        "self_collision_count": int(len(self_distances)),
    }


def extract_revolute_joint_limits(robot) -> list[tuple[float, float]]:
    limits: list[tuple[float, float]] = []
    for joint_index in getattr(robot, "revolute_joints", []):
        info = p.getJointInfo(int(robot.body_id), int(joint_index))
        limits.append((float(info[8]), float(info[9])))
    return limits


def compute_self_collision_clearance(
    robot,
    *,
    link_indices: Sequence[int] | None = None,
    min_index_gap: int = 2,
    query_distance: float = 0.12,
) -> float:
    if query_distance <= 0.0:
        return float("inf")
    links = [int(link) for link in (link_indices or getattr(robot, "revolute_joints", []))]
    best = float("inf")
    for i, link_a in enumerate(links):
        for j in range(i + 1, len(links)):
            if (j - i) < max(int(min_index_gap), 1):
                continue
            contacts = p.getClosestPoints(
                int(robot.body_id),
                int(robot.body_id),
                float(query_distance),
                linkIndexA=int(link_a),
                linkIndexB=int(links[j]),
            )
            if not contacts:
                continue
            best = min(best, float(min(float(contact[8]) for contact in contacts)))
    return best


def compute_environment_clearance(
    robot,
    obstacle_body_id: int,
    link_indices: Sequence[int],
    *,
    max_distance: float = 0.2,
) -> float:
    best = float("inf")
    for link_index in link_indices:
        contacts = p.getClosestPoints(
            int(robot.body_id),
            int(obstacle_body_id),
            float(max_distance),
            linkIndexA=int(link_index),
        )
        if not contacts:
            continue
        best = min(best, float(min(float(contact[8]) for contact in contacts)))
    return best


def evaluate_configuration_quality(
    robot,
    q,
    *,
    dq=None,
    motion_component: str | Sequence[int] = "linear",
    clearance_summary: dict | None = None,
    joint_limits: Iterable[Sequence[float]] | None = None,
) -> dict:
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    dq_arr = np.zeros_like(q_arr) if dq is None else np.asarray(dq, dtype=float).reshape(-1)
    jacobian = np.asarray(robot.get_ee_jacobian(q_arr, dq_arr), dtype=float)
    revolute_cols = int(getattr(robot, "n_revo", jacobian.shape[1]))
    if jacobian.shape[1] > revolute_cols:
        jacobian = jacobian[:, jacobian.shape[1] - revolute_cols :]
    manip = compute_manipulability_report(jacobian, motion_component=motion_component)

    if joint_limits is None:
        joint_limits = extract_revolute_joint_limits(robot)
    joint_limits = list(joint_limits)
    revolute_q = q_arr[-len(joint_limits) :] if joint_limits else np.zeros(0, dtype=float)
    joint_margin = compute_joint_limit_margin(revolute_q, joint_limits) if joint_limits else {
        "per_joint": [],
        "min_margin": 1.0,
        "mean_margin": 1.0,
    }
    clearance = dict(clearance_summary or summarize_clearance_entries([]))
    return {
        **manip,
        "joint_limit_margin": float(joint_margin["min_margin"]),
        "joint_limit_margin_mean": float(joint_margin["mean_margin"]),
        "joint_limit_per_joint": list(joint_margin["per_joint"]),
        "self_collision_distance": float(clearance.get("self_collision_distance", float("inf"))),
        "environment_distance": float(clearance.get("environment_distance", float("inf"))),
        "clearance_distance": float(clearance.get("min_distance", float("inf"))),
    }


def _lookup_metric(record: dict, key: str):
    if key in record:
        return record[key]
    quality = record.get("configuration_quality")
    if isinstance(quality, dict) and key in quality:
        return quality[key]
    return None


def _normalize_metric_values(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, 0.5, dtype=float)
    clipped = arr.copy()
    clipped[np.isposinf(clipped)] = np.max(finite)
    clipped[np.isneginf(clipped)] = np.min(finite)
    clipped[np.isnan(clipped)] = float(np.mean(finite))
    lo = float(np.min(clipped))
    hi = float(np.max(clipped))
    if hi - lo <= 1e-9:
        norm = np.full_like(clipped, 0.5, dtype=float)
    else:
        norm = (clipped - lo) / (hi - lo)
    return norm if higher_is_better else (1.0 - norm)


def rank_configuration_records(
    records: Sequence[dict],
    *,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    annotated = [dict(record) for record in records]
    if not annotated:
        return []
    weights = dict(DEFAULT_SELECTION_WEIGHTS if not weights else weights)
    normalized_metrics: dict[str, np.ndarray] = {}
    for key in weights:
        values = np.array([
            np.nan if _lookup_metric(record, key) is None else _lookup_metric(record, key)
            for record in annotated
        ], dtype=float)
        normalized_metrics[key] = _normalize_metric_values(values, higher_is_better=(key != "aabb_max_dim"))

    for idx, record in enumerate(annotated):
        score_components = {
            key: float(weights[key] * normalized_metrics[key][idx])
            for key in weights
        }
        record["selection_score"] = float(sum(score_components.values()))
        record["selection_score_components"] = score_components
    return sorted(annotated, key=lambda item: float(item.get("selection_score", -np.inf)), reverse=True)
