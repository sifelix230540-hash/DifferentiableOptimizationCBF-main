from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pybullet as p
from scipy.spatial import ConvexHull, QhullError


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import Robot, load_config, _resolve  # noqa: E402


def extract_revolute_metadata(robot) -> tuple[list[int], list[str], list[tuple[float, float]], list[int]]:
    active_index = {joint_id: idx for idx, joint_id in enumerate(getattr(robot, "active_joints", []))}
    revolute_ids = [int(j) for j in getattr(robot, "revolute_joints", [])]
    revolute_names = [str(getattr(robot, "link_name_by_index", {}).get(j, f"joint_{j}")) for j in revolute_ids]
    joint_limits: list[tuple[float, float]] = []
    q_indices: list[int] = []
    for joint_id in revolute_ids:
        info = p.getJointInfo(int(robot.body_id), int(joint_id))
        joint_limits.append((float(info[8]), float(info[9])))
        if joint_id not in active_index:
            raise KeyError(f"Joint {joint_id} not found in active joint list.")
        q_indices.append(int(active_index[joint_id]))
    return revolute_ids, revolute_names, joint_limits, q_indices


def normalize_joint_samples(samples, joint_limits):
    pts = np.asarray(samples, dtype=float).reshape(-1, len(joint_limits))
    lower = np.asarray([float(lo) for lo, _hi in joint_limits], dtype=float)
    upper = np.asarray([float(hi) for _lo, hi in joint_limits], dtype=float)
    span = np.maximum(upper - lower, 1e-9)
    normalized = (pts - lower.reshape(1, -1)) / span.reshape(1, -1)
    return normalized, lower, span


def denormalize_joint_samples(normalized_samples, lower, span):
    pts = np.asarray(normalized_samples, dtype=float)
    return pts * np.asarray(span, dtype=float).reshape(1, -1) + np.asarray(lower, dtype=float).reshape(1, -1)


def sample_revolute_configurations(
    q_base,
    q_indices,
    joint_limits,
    *,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    q_base = np.asarray(q_base, dtype=float).reshape(-1)
    out = np.repeat(q_base.reshape(1, -1), int(num_samples), axis=0)
    for row in range(out.shape[0]):
        for local_idx, q_idx in enumerate(q_indices):
            lo, hi = joint_limits[local_idx]
            if hi > lo:
                out[row, int(q_idx)] = float(rng.uniform(lo, hi))
            else:
                out[row, int(q_idx)] = float(q_base[int(q_idx)])
    return out


def has_self_collision(robot, *, penetration_thresh: float = -0.001, min_index_gap: int = 2) -> bool:
    p.performCollisionDetection()
    contacts = p.getContactPoints(bodyA=int(robot.body_id), bodyB=int(robot.body_id))
    for contact in contacts:
        link_a = int(contact[3])
        link_b = int(contact[4])
        distance = float(contact[8])
        if link_a == link_b:
            continue
        if abs(link_a - link_b) < max(int(min_index_gap), 1):
            continue
        if distance < float(penetration_thresh):
            return True
    return False


def build_monitored_link_pairs(
    link_indices,
    *,
    min_index_gap: int = 2,
) -> list[tuple[int, int]]:
    links = [int(link) for link in link_indices]
    pairs: list[tuple[int, int]] = []
    for i, link_a in enumerate(links):
        for j in range(i + 1, len(links)):
            if (j - i) < max(int(min_index_gap), 1):
                continue
            pairs.append((int(link_a), int(links[j])))
    return pairs


def compute_pairwise_self_collision_distance(
    robot,
    *,
    monitored_pairs,
    link_models: dict | None = None,
    penetration_thresh: float = -0.001,
    query_distance: float = 0.12,
) -> dict:
    if link_models is not None:
        from CBF_experiment.active.pybullet.self_collision.self_collision_backend_coal import (
            compute_pairwise_self_collision_distance as _coal_impl,
        )
        return _coal_impl(
            robot,
            link_models=link_models,
            monitored_pairs=monitored_pairs,
            penetration_thresh=float(penetration_thresh),
        )
    best = float("inf")
    active_pair = None
    for link_a, link_b in monitored_pairs:
        contacts = p.getClosestPoints(
            int(robot.body_id),
            int(robot.body_id),
            float(query_distance),
            linkIndexA=int(link_a),
            linkIndexB=int(link_b),
        )
        if not contacts:
            continue
        pair_min = min(float(contact[8]) for contact in contacts)
        if pair_min < best:
            best = float(pair_min)
            active_pair = [int(link_a), int(link_b)]
    return {
        "min_distance": float(best),
        "active_pair": active_pair,
        "is_collision": bool(np.isfinite(best) and best < 0.0),
    }


def classify_self_collision_sample(
    robot,
    *,
    monitored_pairs,
    link_models: dict | None = None,
    penetration_thresh: float = -0.001,
    query_distance: float = 0.12,
) -> dict:
    result = compute_pairwise_self_collision_distance(
        robot,
        monitored_pairs=monitored_pairs,
        link_models=link_models,
        penetration_thresh=float(penetration_thresh),
        query_distance=float(query_distance),
    )
    if link_models is None:
        result["is_collision"] = bool(
            np.isfinite(result["min_distance"]) and result["min_distance"] < float(penetration_thresh)
        )
    return result


def max_halfspace_violation(points, equations):
    pts = np.asarray(points, dtype=float).reshape(-1, np.asarray(equations, dtype=float).shape[1] - 1)
    eq = np.asarray(equations, dtype=float)
    lhs = pts @ eq[:, :-1].T + eq[:, -1].reshape(1, -1)
    return np.max(lhs, axis=1)


def _build_aabb_equations(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    lower = np.min(pts, axis=0)
    upper = np.max(pts, axis=0)
    dim = pts.shape[1]
    equations = []
    for axis in range(dim):
        row_upper = np.zeros(dim + 1, dtype=float)
        row_upper[axis] = 1.0
        row_upper[-1] = -float(upper[axis])
        equations.append(row_upper)

        row_lower = np.zeros(dim + 1, dtype=float)
        row_lower[axis] = -1.0
        row_lower[-1] = float(lower[axis])
        equations.append(row_lower)
    return np.asarray(equations, dtype=float)


def fit_convex_hull_cluster(points_normalized, qhull_options: str = "QJ") -> dict:
    pts = np.asarray(points_normalized, dtype=float).reshape(-1, np.asarray(points_normalized, dtype=float).shape[-1])
    dim = pts.shape[1]
    if pts.shape[0] == 0:
        raise ValueError("Cluster points must not be empty.")
    rank = int(np.linalg.matrix_rank(pts - np.mean(pts, axis=0, keepdims=True))) if pts.shape[0] > 1 else 0
    if pts.shape[0] >= dim + 1 and rank >= dim:
        try:
            hull = ConvexHull(pts, qhull_options=qhull_options)
            return {
                "hull_type": "convex_hull",
                "equations": np.asarray(hull.equations, dtype=float).tolist(),
                "vertex_indices": np.asarray(hull.vertices, dtype=int).tolist(),
                "num_points": int(pts.shape[0]),
                "rank": rank,
            }
        except QhullError:
            pass
    equations = _build_aabb_equations(pts)
    return {
        "hull_type": "aabb",
        "equations": equations.tolist(),
        "vertex_indices": [],
        "num_points": int(pts.shape[0]),
        "rank": rank,
    }


def convert_equations_to_joint_space(equations_normalized, lower, span):
    eq = np.asarray(equations_normalized, dtype=float)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    span = np.asarray(span, dtype=float).reshape(-1)
    out = np.zeros_like(eq, dtype=float)
    out[:, :-1] = eq[:, :-1] / span.reshape(1, -1)
    out[:, -1] = eq[:, -1] - np.sum(eq[:, :-1] * (lower / span).reshape(1, -1), axis=1)
    return out


def cluster_samples_by_voxels(normalized_samples, *, voxel_size: float = 0.08, min_cluster_size: int = 16) -> list[dict]:
    samples = np.asarray(normalized_samples, dtype=float)
    if samples.size == 0:
        return []
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive.")
    voxel_map: dict[tuple[int, ...], list[int]] = defaultdict(list)
    indices = np.floor(samples / float(voxel_size)).astype(int)
    for sample_idx, cell in enumerate(indices):
        voxel_map[tuple(int(x) for x in cell.tolist())].append(int(sample_idx))

    neighbor_offsets = [offset for offset in product((-1, 0, 1), repeat=samples.shape[1])]
    visited: set[tuple[int, ...]] = set()
    clusters: list[dict] = []
    for root in voxel_map:
        if root in visited:
            continue
        queue = deque([root])
        visited.add(root)
        cells: list[tuple[int, ...]] = []
        sample_ids: list[int] = []
        while queue:
            cell = queue.popleft()
            cells.append(cell)
            sample_ids.extend(voxel_map[cell])
            for offset in neighbor_offsets:
                neighbor = tuple(cell[axis] + int(offset[axis]) for axis in range(len(cell)))
                if neighbor in voxel_map and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(sample_ids) < int(min_cluster_size):
            continue
        clusters.append({
            "voxel_cells": cells,
            "sample_indices": sorted(sample_ids),
        })
    return clusters


def build_collision_cspace_hulls_from_samples(
    collision_samples,
    joint_limits,
    *,
    voxel_size: float = 0.08,
    min_cluster_size: int = 16,
) -> dict:
    samples = np.asarray(collision_samples, dtype=float).reshape(-1, len(joint_limits))
    normalized, lower, span = normalize_joint_samples(samples, joint_limits)
    clusters = cluster_samples_by_voxels(
        normalized,
        voxel_size=float(voxel_size),
        min_cluster_size=int(min_cluster_size),
    )
    payload_clusters = []
    for cluster_id, cluster in enumerate(clusters):
        sample_indices = cluster["sample_indices"]
        cluster_points_norm = normalized[sample_indices]
        hull = fit_convex_hull_cluster(cluster_points_norm)
        equations_normalized = np.asarray(hull["equations"], dtype=float)
        equations_joint = convert_equations_to_joint_space(equations_normalized, lower, span)
        cluster_points_joint = samples[sample_indices]
        payload_clusters.append({
            "cluster_id": int(cluster_id),
            "num_samples": int(len(sample_indices)),
            "hull_type": str(hull["hull_type"]),
            "rank": int(hull["rank"]),
            "sample_indices": [int(i) for i in sample_indices],
            "samples_normalized": cluster_points_norm.astype(float).tolist(),
            "samples_joint": cluster_points_joint.astype(float).tolist(),
            "equations_normalized": equations_normalized.tolist(),
            "equations_joint": equations_joint.tolist(),
            "aabb_normalized_min": np.min(cluster_points_norm, axis=0).astype(float).tolist(),
            "aabb_normalized_max": np.max(cluster_points_norm, axis=0).astype(float).tolist(),
            "aabb_joint_min": np.min(cluster_points_joint, axis=0).astype(float).tolist(),
            "aabb_joint_max": np.max(cluster_points_joint, axis=0).astype(float).tolist(),
        })
    return {
        "dimension": int(samples.shape[1]),
        "num_collision_samples": int(samples.shape[0]),
        "voxel_size": float(voxel_size),
        "min_cluster_size": int(min_cluster_size),
        "normalization_lower": np.asarray(lower, dtype=float).tolist(),
        "normalization_span": np.asarray(span, dtype=float).tolist(),
        "clusters": payload_clusters,
    }


def collect_relevant_collision_samples(
    samples,
    metrics,
    *,
    sample_mode: str = "collision_only",
    boundary_band: float = 0.02,
) -> np.ndarray:
    pts = np.asarray(samples, dtype=float)
    selected: list[np.ndarray] = []
    mode = str(sample_mode).strip().lower()
    for idx, metric in enumerate(metrics):
        min_distance = float(metric.get("min_distance", float("inf")))
        is_collision = bool(metric.get("is_collision", False))
        if mode == "collision_only":
            keep = is_collision
        elif mode == "boundary_band":
            keep = abs(min_distance) <= float(boundary_band)
        elif mode == "collision_and_boundary":
            keep = is_collision or abs(min_distance) <= float(boundary_band)
        else:
            raise ValueError(f"Unsupported sample mode: {sample_mode}")
        if keep:
            selected.append(np.asarray(pts[idx], dtype=float))
    if not selected:
        return np.zeros((0, pts.shape[1]), dtype=float)
    return np.vstack(selected)


def _default_progress_callback(info: dict) -> None:
    stage = str(info.get("stage", "sampling"))
    current = int(info.get("current", 0))
    total = max(int(info.get("total", 1)), 1)
    frac = current / total
    width = 32
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    suffix = ""
    if "collision_count" in info:
        suffix = f" collision={int(info['collision_count'])}"
    print(f"\r[self-cspace] {stage} |{bar}| {100.0 * frac:5.1f}% [{current}/{total}]{suffix}", end="", flush=True)
    if stage == "done":
        print()


def _project_points_for_visualization(points: np.ndarray) -> tuple[np.ndarray, list[str], str]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[1] <= 2:
        padded = pts if pts.shape[1] == 2 else np.column_stack([pts[:, 0], np.zeros(pts.shape[0], dtype=float)])
        return padded, ["dim1", "dim2"], "direct"
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    return proj, ["pc1", "pc2"], "pca"


def visualize_collision_cspace_hulls(payload: dict, output_png: str | Path) -> Path:
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clusters = list(payload.get("clusters", []))
    fig = plt.figure(figsize=(8.5, 6.0))
    ax = fig.add_subplot(111)
    if not clusters:
        ax.text(0.5, 0.5, "no collision clusters", ha="center", va="center")
        ax.set_axis_off()
    else:
        all_points = []
        cluster_arrays = []
        for cluster in clusters:
            pts = np.asarray(cluster.get("samples_normalized", []), dtype=float).reshape(-1, int(payload.get("dimension", 0)))
            if pts.size == 0:
                continue
            cluster_arrays.append((cluster, pts))
            all_points.append(pts)
        if all_points:
            stacked = np.vstack(all_points)
            projected_all, axis_labels, projection_type = _project_points_for_visualization(stacked)
            offset = 0
            cmap = plt.get_cmap("tab10")
            for idx, (cluster, pts) in enumerate(cluster_arrays):
                count = pts.shape[0]
                proj = projected_all[offset: offset + count]
                offset += count
                color = cmap(idx % 10)
                ax.scatter(proj[:, 0], proj[:, 1], s=18, alpha=0.75, color=color, label=f"cluster {cluster['cluster_id']}")
                min_box = np.asarray(cluster.get("aabb_normalized_min", []), dtype=float).reshape(-1)
                max_box = np.asarray(cluster.get("aabb_normalized_max", []), dtype=float).reshape(-1)
                if min_box.size >= 2 and projection_type == "direct":
                    rect_x = [min_box[0], max_box[0], max_box[0], min_box[0], min_box[0]]
                    rect_y = [min_box[1], min_box[1], max_box[1], max_box[1], min_box[1]]
                    ax.plot(rect_x, rect_y, "--", color=color, linewidth=1.2)
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_title(f"Self-collision C-space clusters ({projection_type})")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "clusters without sample points", ha="center", va="center")
            ax.set_axis_off()
    fig.tight_layout()
    with open(output_path, "wb") as fp:
        fig.savefig(fp, dpi=140, format="png")
    plt.close(fig)
    return output_path


def monte_carlo_self_collision_hulls(
    *,
    cfg_path: str | Path | None = None,
    num_samples: int = 20000,
    seed: int = 7,
    voxel_size: float = 0.08,
    min_cluster_size: int = 24,
    penetration_thresh: float = -0.001,
    min_index_gap: int = 2,
    query_distance: float = 0.12,
    sample_mode: str = "collision_only",
    boundary_band: float = 0.02,
    output_json: str | Path | None = None,
    output_png: str | Path | None = None,
    progress_callback=None,
    progress_every: int = 250,
) -> dict:
    cfg = load_config(cfg_path)
    rng = np.random.default_rng(int(seed))
    created_connection = False
    if not p.isConnected():
        p.connect(p.DIRECT)
        created_connection = True
    try:
        robot = Robot(cfg)
        q_base, dq_base = robot.get_joint_state()
        revolute_ids, revolute_names, joint_limits, q_indices = extract_revolute_metadata(robot)
        monitored_pairs = build_monitored_link_pairs(revolute_ids, min_index_gap=int(min_index_gap))
        from CBF_experiment.active.pybullet.self_collision.self_collision_backend_coal import (
            build_coal_link_models,
        )
        link_models = build_coal_link_models(robot, revolute_ids)
        sampled_q = sample_revolute_configurations(
            q_base,
            q_indices,
            joint_limits,
            num_samples=int(num_samples),
            rng=rng,
        )
        sample_metrics: list[dict] = []
        callback = progress_callback if progress_callback is not None else _default_progress_callback
        callback({"stage": "sampling", "current": 0, "total": int(num_samples), "collision_count": 0})
        for sample_idx, q in enumerate(sampled_q, start=1):
            robot.set_joint_state(q, dq=np.zeros_like(q))
            metric = classify_self_collision_sample(
                robot,
                monitored_pairs=monitored_pairs,
                link_models=link_models,
                penetration_thresh=float(penetration_thresh),
            )
            sample_metrics.append(metric)
            if (
                sample_idx == int(num_samples)
                or int(progress_every) <= 1
                or sample_idx % int(progress_every) == 0
            ):
                callback({
                    "stage": "sampling",
                    "current": int(sample_idx),
                    "total": int(num_samples),
                    "collision_count": int(sum(1 for item in sample_metrics if item["is_collision"])),
                })
        revolute_samples = np.asarray(sampled_q[:, q_indices], dtype=float)
        collision_revolute_samples = collect_relevant_collision_samples(
            revolute_samples,
            sample_metrics,
            sample_mode=str(sample_mode),
            boundary_band=float(boundary_band),
        )
        collision_metrics = [metric for metric in sample_metrics if (
            (str(sample_mode).strip().lower() == "collision_only" and metric["is_collision"])
            or (str(sample_mode).strip().lower() == "boundary_band" and abs(float(metric["min_distance"])) <= float(boundary_band))
            or (str(sample_mode).strip().lower() == "collision_and_boundary" and (metric["is_collision"] or abs(float(metric["min_distance"])) <= float(boundary_band)))
        )]
        if collision_revolute_samples.size == 0:
            payload = {
                "dimension": int(len(joint_limits)),
                "joint_indices": [int(j) for j in revolute_ids],
                "joint_names": revolute_names,
                "joint_limits": [[float(lo), float(hi)] for lo, hi in joint_limits],
                "num_samples": int(num_samples),
                "num_collision_samples": 0,
                "collision_ratio": 0.0,
                "sample_mode": str(sample_mode),
                "clusters": [],
            }
        else:
            hull_payload = build_collision_cspace_hulls_from_samples(
                np.asarray(collision_revolute_samples, dtype=float),
                joint_limits,
                voxel_size=float(voxel_size),
                min_cluster_size=int(min_cluster_size),
            )
            payload = {
                **hull_payload,
                "joint_indices": [int(j) for j in revolute_ids],
                "joint_names": revolute_names,
                "joint_limits": [[float(lo), float(hi)] for lo, hi in joint_limits],
                "num_samples": int(num_samples),
                "num_collision_samples_total": int(sum(1 for item in sample_metrics if item["is_collision"])),
                "collision_ratio": float(sum(1 for item in sample_metrics if item["is_collision"]) / max(int(num_samples), 1)),
                "penetration_thresh": float(penetration_thresh),
                "min_index_gap": int(min_index_gap),
                "query_distance": float(query_distance),
                "sample_mode": str(sample_mode),
                "boundary_band": float(boundary_band),
                "monitored_pairs": [[int(a), int(b)] for a, b in monitored_pairs],
                "selected_metrics_preview": collision_metrics[: min(32, len(collision_metrics))],
            }

        if output_json is None:
            output_json = _resolve("artifacts/sdf_exp/self_collision_cspace_hulls.json")
        if output_png is None:
            output_png = _resolve("artifacts/sdf_exp/self_collision_cspace_hulls.png")
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        image_path = visualize_collision_cspace_hulls(payload, output_png)
        payload["visualization_png"] = str(image_path)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        callback({
            "stage": "done",
            "current": int(num_samples),
            "total": int(num_samples),
            "collision_count": int(payload["num_collision_samples"]),
            "cluster_count": int(len(payload.get("clusters", []))),
        })
        print(f"[self-cspace] samples={payload['num_samples']} selected={payload['num_collision_samples']} total_collision={payload.get('num_collision_samples_total', payload['num_collision_samples'])}")
        print(f"[self-cspace] hull clusters={len(payload.get('clusters', []))}")
        print(f"[self-cspace] saved -> {output_path}")
        print(f"[self-cspace] figure -> {image_path}")
        robot.set_joint_state(q_base, dq_base)
        return payload
    finally:
        if created_connection and p.isConnected():
            p.disconnect()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo offline self-collision C-space hull builder for six-axis joints.")
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--min-cluster-size", type=int, default=24)
    parser.add_argument("--penetration-thresh", type=float, default=-0.001)
    parser.add_argument("--min-index-gap", type=int, default=2)
    parser.add_argument("--query-distance", type=float, default=0.12)
    parser.add_argument("--sample-mode", type=str, default="collision_only", choices=["collision_only", "boundary_band", "collision_and_boundary"])
    parser.add_argument("--boundary-band", type=float, default=0.02)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-png", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = _build_arg_parser().parse_args(argv)
    return monte_carlo_self_collision_hulls(
        cfg_path=args.cfg_path,
        num_samples=args.num_samples,
        seed=args.seed,
        voxel_size=args.voxel_size,
        min_cluster_size=args.min_cluster_size,
        penetration_thresh=args.penetration_thresh,
        min_index_gap=args.min_index_gap,
        query_distance=args.query_distance,
        sample_mode=args.sample_mode,
        boundary_band=args.boundary_band,
        output_json=args.output_json,
        output_png=args.output_png,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
