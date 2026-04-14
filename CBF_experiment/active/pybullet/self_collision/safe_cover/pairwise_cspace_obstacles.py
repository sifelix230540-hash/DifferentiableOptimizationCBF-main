from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import (  # noqa: E402
    Robot,
    load_config,
    _resolve,
)
from CBF_experiment.active.pybullet.self_collision.self_collision_backend_coal import (  # noqa: E402
    build_coal_link_models,
    compute_pairwise_self_collision_distance,
)
from CBF_experiment.active.pybullet.self_collision.self_collision_cspace_hulls import (  # noqa: E402
    build_monitored_link_pairs,
    cluster_samples_by_voxels,
    convert_equations_to_joint_space,
    extract_revolute_metadata,
    extract_self_collision_monitor_metadata,
    fit_convex_hull_cluster,
    normalize_joint_samples,
)


@dataclass(frozen=True)
class PairwiseObstacleConfig:
    CFG_PATH: str | None = None
    SEED: int = 11
    MIN_INDEX_GAP: int = 2
    PENETRATION_THRESH: float = -0.001
    SAMPLE_SCALE: float = 1.0
    OUTPUT_JSON: str = str(Path(_resolve("artifacts/sdf_exp/pairwise_cspace_obstacles.json")))


DEFAULT_NUM_SAMPLES_BY_DIM = {
    1: 384,
    2: 1536,
    3: 3072,
    4: 6144,
    5: 9216,
    6: 12000,
}

DEFAULT_VOXEL_SIZE_BY_DIM = {
    1: 0.025,
    2: 0.04,
    3: 0.06,
    4: 0.08,
    5: 0.10,
    6: 0.12,
}

DEFAULT_MIN_CLUSTER_SIZE_BY_DIM = {
    1: 4,
    2: 8,
    3: 12,
    4: 16,
    5: 20,
    6: 24,
}


def build_robot_cspace_metadata(robot) -> dict:
    active_joint_ids = [int(j) for j in getattr(robot, "active_joints", [])]
    joint_parent_link = {}
    for joint_id in range(int(getattr(robot, "num_joints", 0))):
        info = p.getJointInfo(int(robot.body_id), int(joint_id))
        joint_parent_link[int(joint_id)] = int(info[16])
    active_q_index_by_joint = {int(j): idx for idx, j in enumerate(active_joint_ids)}
    return {
        "active_joint_ids": active_joint_ids,
        "joint_parent_link": joint_parent_link,
        "active_q_index_by_joint": active_q_index_by_joint,
    }


def _ancestor_path_to_root(joint_parent_link: dict[int, int], link_index: int) -> list[int]:
    path = []
    current = int(link_index)
    while current >= 0:
        path.append(int(current))
        current = int(joint_parent_link.get(int(current), -1))
    path.append(-1)
    return path


def compute_pair_active_joint_indices(metadata: dict, link_a: int, link_b: int) -> list[int]:
    parent = metadata["joint_parent_link"]
    active_joint_ids = set(int(j) for j in metadata["active_joint_ids"])
    path_a = _ancestor_path_to_root(parent, int(link_a))
    path_b = _ancestor_path_to_root(parent, int(link_b))
    ancestors_a = set(path_a)

    lca = -1
    for node in path_b:
        if int(node) in ancestors_a:
            lca = int(node)
            break

    path_joint_ids = set()
    current = int(link_a)
    while current != lca and current >= 0:
        path_joint_ids.add(int(current))
        current = int(parent.get(int(current), -1))
    current = int(link_b)
    while current != lca and current >= 0:
        path_joint_ids.add(int(current))
        current = int(parent.get(int(current), -1))

    return sorted(int(j) for j in path_joint_ids if int(j) in active_joint_ids)


def lift_halfspaces_to_full_space(
    A_sub: np.ndarray,
    b_sub: np.ndarray,
    *,
    active_q_indices: list[int],
    full_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    A_sub = np.asarray(A_sub, dtype=float)
    b_sub = np.asarray(b_sub, dtype=float).reshape(-1)
    A_full = np.zeros((A_sub.shape[0], int(full_dim)), dtype=float)
    A_full[:, np.asarray(active_q_indices, dtype=int)] = A_sub
    return A_full, b_sub.copy()


def _sample_active_subspace(
    q_ref_revolute: np.ndarray,
    *,
    active_q_indices: list[int],
    active_joint_limits: list[tuple[float, float]],
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.repeat(np.asarray(q_ref_revolute, dtype=float).reshape(1, -1), int(num_samples), axis=0)
    for row in range(out.shape[0]):
        for axis_idx, q_idx in enumerate(active_q_indices):
            lo, hi = active_joint_limits[axis_idx]
            if hi > lo:
                out[row, int(q_idx)] = float(rng.uniform(lo, hi))
    return out


def _pair_result(robot, link_models, pair: tuple[int, int], *, penetration_thresh: float) -> dict:
    return compute_pairwise_self_collision_distance(
        robot,
        link_models=link_models,
        monitored_pairs=[(int(pair[0]), int(pair[1]))],
        penetration_thresh=float(penetration_thresh),
    )


def _build_pair_obstacles_from_samples(
    collision_samples_subspace: np.ndarray,
    *,
    active_joint_limits: list[tuple[float, float]],
    active_q_indices: list[int],
    full_dim: int,
    voxel_size: float,
    min_cluster_size: int,
) -> list[dict]:
    if collision_samples_subspace.size == 0:
        return []
    normalized, lower, span = normalize_joint_samples(collision_samples_subspace, active_joint_limits)
    clusters = cluster_samples_by_voxels(
        normalized,
        voxel_size=float(voxel_size),
        min_cluster_size=int(min_cluster_size),
    )
    if not clusters and collision_samples_subspace.shape[0] > 0:
        clusters = [{"sample_indices": list(range(collision_samples_subspace.shape[0]))}]

    obstacles = []
    for cluster_id, cluster in enumerate(clusters):
        sample_indices = cluster["sample_indices"]
        pts_norm = normalized[sample_indices]
        hull = fit_convex_hull_cluster(pts_norm)
        eq_norm = np.asarray(hull["equations"], dtype=float)
        eq_joint = convert_equations_to_joint_space(eq_norm, lower, span)
        A_sub = np.asarray(eq_joint[:, :-1], dtype=float)
        b_sub = -np.asarray(eq_joint[:, -1], dtype=float)
        A_full, b_full = lift_halfspaces_to_full_space(
            A_sub,
            b_sub,
            active_q_indices=active_q_indices,
            full_dim=int(full_dim),
        )
        obstacles.append({
            "cluster_id": int(cluster_id),
            "num_collision_samples": int(len(sample_indices)),
            "hull_type": str(hull["hull_type"]),
            "active_q_indices": [int(i) for i in active_q_indices],
            "A_subspace": A_sub.tolist(),
            "b_subspace": b_sub.tolist(),
            "A_full": A_full.tolist(),
            "b_full": b_full.tolist(),
            "samples_subspace": np.asarray(collision_samples_subspace[sample_indices], dtype=float).tolist(),
        })
    return obstacles


def build_pairwise_cspace_obstacles(cfg: PairwiseObstacleConfig = PairwiseObstacleConfig()) -> dict:
    rng = np.random.default_rng(int(cfg.SEED))
    created_connection = False
    if not p.isConnected():
        p.connect(p.DIRECT)
        created_connection = True
    try:
        robot = Robot(load_config(cfg.CFG_PATH))
        metadata = build_robot_cspace_metadata(robot)
        q_base, dq_base = robot.get_joint_state()
        revolute_ids, revolute_names, joint_limits, q_indices = extract_revolute_metadata(robot)
        monitored_link_ids, monitored_link_names = extract_self_collision_monitor_metadata(robot)
        monitored_pairs = build_monitored_link_pairs(monitored_link_ids, min_index_gap=int(cfg.MIN_INDEX_GAP))
        link_models = build_coal_link_models(robot, monitored_link_ids)
        q_ref_revolute = np.asarray(q_base, dtype=float)[np.asarray(q_indices, dtype=int)]
        revolute_limit_by_joint = {
            int(joint_id): joint_limits[idx]
            for idx, joint_id in enumerate(revolute_ids)
        }
        revolute_q_index_by_joint = {
            int(joint_id): int(idx)
            for idx, joint_id in enumerate(revolute_ids)
        }

        pair_payloads = []
        full_dim = int(len(revolute_ids))
        for pair in monitored_pairs:
            active_joint_ids = compute_pair_active_joint_indices(metadata, int(pair[0]), int(pair[1]))
            active_joint_ids = [j for j in active_joint_ids if j in revolute_q_index_by_joint]
            active_q_indices = [int(revolute_q_index_by_joint[j]) for j in active_joint_ids]
            if not active_joint_ids or not active_q_indices:
                continue

            dim = int(len(active_joint_ids))
            active_joint_limits = [revolute_limit_by_joint[int(j)] for j in active_joint_ids]
            base_num_samples = int(DEFAULT_NUM_SAMPLES_BY_DIM.get(dim, DEFAULT_NUM_SAMPLES_BY_DIM[max(DEFAULT_NUM_SAMPLES_BY_DIM)]))
            num_samples = max(64, int(round(base_num_samples * float(cfg.SAMPLE_SCALE))))
            sampled_revolute = _sample_active_subspace(
                q_ref_revolute,
                active_q_indices=active_q_indices,
                active_joint_limits=active_joint_limits,
                num_samples=num_samples,
                rng=rng,
            )

            collision_samples_subspace = []
            collision_clearances = []
            for sample_revolute in sampled_revolute:
                q_full = np.asarray(q_base, dtype=float).copy()
                q_full[np.asarray(q_indices, dtype=int)] = np.asarray(sample_revolute, dtype=float)
                robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
                metric = _pair_result(
                    robot,
                    link_models,
                    pair,
                    penetration_thresh=float(cfg.PENETRATION_THRESH),
                )
                if bool(metric["is_collision"]):
                    collision_samples_subspace.append(np.asarray(sample_revolute, dtype=float)[active_q_indices].tolist())
                    collision_clearances.append(float(metric["min_distance"]))

            collision_samples_subspace = np.asarray(collision_samples_subspace, dtype=float).reshape(-1, dim)
            voxel_size = float(DEFAULT_VOXEL_SIZE_BY_DIM.get(dim, 0.12))
            min_cluster_size = int(DEFAULT_MIN_CLUSTER_SIZE_BY_DIM.get(dim, 24))
            obstacles = _build_pair_obstacles_from_samples(
                collision_samples_subspace,
                active_joint_limits=active_joint_limits,
                active_q_indices=active_q_indices,
                full_dim=full_dim,
                voxel_size=voxel_size,
                min_cluster_size=min_cluster_size,
            )
            pair_payloads.append({
                "pair": [int(pair[0]), int(pair[1])],
                "pair_names": [
                    str(monitored_link_names[monitored_link_ids.index(int(pair[0]))]),
                    str(monitored_link_names[monitored_link_ids.index(int(pair[1]))]),
                ],
                "active_joint_ids": [int(j) for j in active_joint_ids],
                "active_joint_names": [str(robot.link_name_by_index.get(int(j), f"joint_{int(j)}")) for j in active_joint_ids],
                "active_q_indices": [int(i) for i in active_q_indices],
                "subspace_dim": dim,
                "num_samples": int(num_samples),
                "num_collision_samples": int(collision_samples_subspace.shape[0]),
                "mean_collision_distance": (
                    float(np.mean(collision_clearances)) if collision_clearances else None
                ),
                "obstacles": obstacles,
            })

        robot.set_joint_state(q_base, dq_base)
        payload = {
            "method": "pairwise_subspace_collision_obstacles",
            "joint_indices": [int(j) for j in revolute_ids],
            "joint_names": [str(name) for name in revolute_names],
            "monitored_link_indices": [int(j) for j in monitored_link_ids],
            "monitored_link_names": [str(name) for name in monitored_link_names],
            "pairs": pair_payloads,
        }
        output_path = Path(cfg.OUTPUT_JSON)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    finally:
        if created_connection and p.isConnected():
            p.disconnect()

