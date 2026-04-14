from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import coal  # noqa: E402

from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import Robot, load_config, _resolve  # noqa: E402
from CBF_experiment.active.pybullet.self_collision.self_collision_cspace_hulls import (  # noqa: E402
    build_monitored_link_pairs,
    classify_self_collision_sample,
    extract_revolute_metadata,
    extract_self_collision_monitor_metadata,
)


@dataclass(frozen=True)
class CoalValidationParameters:
    CFG_PATH: str | None = None
    SAMPLE_JSON: str = str(Path(_resolve("artifacts/sdf_exp/self_collision_eval_samples.json")))
    OUTPUT_JSON: str = str(Path(_resolve("artifacts/sdf_exp/self_collision_coal_validation.json")))
    MAX_COLLISION_SAMPLES: int = 32
    MAX_FREE_SAMPLES: int = 32
    MIN_INDEX_GAP: int = 2
    QUERY_DISTANCE: float = 0.12
    PENETRATION_THRESH: float = -0.001


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _decode_path(path_value) -> str:
    if isinstance(path_value, bytes):
        return path_value.decode("utf-8", errors="replace")
    return str(path_value)


def _compose_full_q(q_base: np.ndarray, q_indices: list[int], revolute_q) -> np.ndarray:
    q_full = np.asarray(q_base, dtype=float).copy()
    q_full[np.asarray(q_indices, dtype=int)] = np.asarray(revolute_q, dtype=float).reshape(-1)
    return q_full


def _build_transform(world_pos, world_quat) -> coal.Transform3s:
    tf = coal.Transform3s()
    rot = np.asarray(p.getMatrixFromQuaternion(world_quat), dtype=float).reshape(3, 3)
    tf.setRotation(rot)
    tf.setTranslation(np.asarray(world_pos, dtype=float).reshape(3))
    return tf


def _get_world_collision_transform(robot, link_index: int, local_pos, local_quat) -> coal.Transform3s:
    link_state = p.getLinkState(int(robot.body_id), int(link_index), computeForwardKinematics=True)
    world_pos, world_quat = p.multiplyTransforms(
        link_state[0],
        link_state[1],
        np.asarray(local_pos, dtype=float).reshape(3).tolist(),
        np.asarray(local_quat, dtype=float).reshape(4).tolist(),
    )
    return _build_transform(world_pos, world_quat)


def _coal_pair_is_collision(distance: float, collide_flag: bool, *, penetration_thresh: float) -> bool:
    if bool(collide_flag):
        return True
    if not np.isfinite(float(distance)):
        return False
    return bool(float(distance) <= 0.0 or float(distance) < float(penetration_thresh))


def _extract_contact_penetration_depth(collision_result) -> float | None:
    if not bool(collision_result.isCollision()):
        return None
    num_contacts = int(collision_result.numContacts())
    if num_contacts <= 0:
        return None
    depths = [float(collision_result.getContact(i).penetration_depth) for i in range(num_contacts)]
    if not depths:
        return None
    return float(min(depths))


def build_coal_link_models(robot, link_indices) -> dict[int, dict]:
    loader = coal.MeshLoader()
    models: dict[int, dict] = {}
    for link_index in link_indices:
        collision_data = p.getCollisionShapeData(int(robot.body_id), int(link_index))
        if not collision_data:
            raise ValueError(f"Link {link_index} has no collision shape data.")
        shape = collision_data[0]
        mesh_path = _decode_path(shape[4])
        mesh_scale = np.asarray(shape[3], dtype=float).reshape(3)
        if np.allclose(mesh_scale, np.ones(3, dtype=float)):
            mesh = loader.load(mesh_path)
        else:
            mesh = loader.load(mesh_path, mesh_scale)
        models[int(link_index)] = {
            "geometry": mesh,
            "mesh_path": mesh_path,
            "mesh_scale": mesh_scale.tolist(),
            "local_pos": np.asarray(shape[5], dtype=float).reshape(3),
            "local_quat": np.asarray(shape[6], dtype=float).reshape(4),
        }
    return models


def compute_pairwise_self_collision_distance_coal(
    robot,
    *,
    link_models: dict[int, dict],
    monitored_pairs,
    penetration_thresh: float,
) -> dict:
    best = float("inf")
    active_pair = None
    best_contact_penetration = float("inf")
    contact_active_pair = None
    any_collision = False
    pair_reports = []
    distance_request = coal.DistanceRequest()
    distance_request.enable_signed_distance = True
    collision_request = coal.CollisionRequest()
    collision_request.enable_contact = True
    collision_request.num_max_contacts = 8
    for link_a, link_b in monitored_pairs:
        model_a = link_models[int(link_a)]
        model_b = link_models[int(link_b)]
        tf_a = _get_world_collision_transform(robot, int(link_a), model_a["local_pos"], model_a["local_quat"])
        tf_b = _get_world_collision_transform(robot, int(link_b), model_b["local_pos"], model_b["local_quat"])
        distance_result = coal.DistanceResult()
        collision_result = coal.CollisionResult()
        coal.distance(model_a["geometry"], tf_a, model_b["geometry"], tf_b, distance_request, distance_result)
        coal.collide(model_a["geometry"], tf_a, model_b["geometry"], tf_b, collision_request, collision_result)
        distance = float(distance_result.min_distance)
        collide_flag = bool(collision_result.isCollision())
        contact_penetration_depth = _extract_contact_penetration_depth(collision_result)
        pair_is_collision = _coal_pair_is_collision(
            distance,
            collide_flag,
            penetration_thresh=float(penetration_thresh),
        )
        pair_reports.append({
            "pair": [int(link_a), int(link_b)],
            "distance": distance,
            "collide_flag": collide_flag,
            "contact_penetration_depth": contact_penetration_depth,
            "is_collision": pair_is_collision,
        })
        any_collision = any_collision or pair_is_collision
        if distance < best:
            best = distance
            active_pair = [int(link_a), int(link_b)]
        if contact_penetration_depth is not None and contact_penetration_depth < best_contact_penetration:
            best_contact_penetration = contact_penetration_depth
            contact_active_pair = [int(link_a), int(link_b)]
    return {
        "min_distance": float(best),
        "active_pair": active_pair,
        "contact_penetration_depth": (
            float(best_contact_penetration) if contact_active_pair is not None else None
        ),
        "contact_active_pair": contact_active_pair,
        "is_collision": bool(any_collision),
        "pair_reports": pair_reports,
    }


def _select_samples(sample_payload: dict, max_collision_samples: int, max_free_samples: int) -> list[dict]:
    collision_samples = np.asarray(sample_payload.get("collision_samples", []), dtype=float)
    free_samples = np.asarray(sample_payload.get("free_samples", []), dtype=float)
    collision_distances = np.asarray(sample_payload.get("collision_distances", []), dtype=float).reshape(-1)
    free_distances = np.asarray(sample_payload.get("free_distances", []), dtype=float).reshape(-1)
    if collision_samples.ndim == 1 and collision_samples.size:
        collision_samples = collision_samples.reshape(1, -1)
    if free_samples.ndim == 1 and free_samples.size:
        free_samples = free_samples.reshape(1, -1)
    selected = []
    for idx in range(min(int(max_collision_samples), int(collision_samples.shape[0]))):
        selected.append({
            "label": "collision",
            "sample_index": int(idx),
            "revolute_q": collision_samples[idx].tolist(),
            "stored_min_distance": float(collision_distances[idx]),
        })
    for idx in range(min(int(max_free_samples), int(free_samples.shape[0]))):
        selected.append({
            "label": "free",
            "sample_index": int(idx),
            "revolute_q": free_samples[idx].tolist(),
            "stored_min_distance": float(free_distances[idx]),
        })
    return selected


def run_coal_validation(params=CoalValidationParameters) -> dict:
    sample_payload = load_json(params.SAMPLE_JSON)
    selected_samples = _select_samples(
        sample_payload,
        max_collision_samples=int(params.MAX_COLLISION_SAMPLES),
        max_free_samples=int(params.MAX_FREE_SAMPLES),
    )
    if not selected_samples:
        raise ValueError("No validation samples were selected.")

    cfg = load_config(params.CFG_PATH)
    created_connection = False
    if not p.isConnected():
        p.connect(p.DIRECT)
        created_connection = True

    try:
        robot = Robot(cfg)
        q_base, dq_base = robot.get_joint_state()
        revolute_ids, revolute_names, _joint_limits, q_indices = extract_revolute_metadata(robot)
        monitored_link_ids, monitored_link_names = extract_self_collision_monitor_metadata(robot)
        monitored_pairs = build_monitored_link_pairs(monitored_link_ids, min_index_gap=int(params.MIN_INDEX_GAP))
        link_models = build_coal_link_models(robot, monitored_link_ids)

        sample_reports = []
        for sample in selected_samples:
            q_full = _compose_full_q(q_base, q_indices, sample["revolute_q"])
            robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
            pybullet_metric = classify_self_collision_sample(
                robot,
                monitored_pairs=monitored_pairs,
                query_distance=float(params.QUERY_DISTANCE),
                penetration_thresh=float(params.PENETRATION_THRESH),
            )
            coal_metric = compute_pairwise_self_collision_distance_coal(
                robot,
                link_models=link_models,
                monitored_pairs=monitored_pairs,
                penetration_thresh=float(params.PENETRATION_THRESH),
            )
            sample_reports.append({
                "label": str(sample["label"]),
                "sample_index": int(sample["sample_index"]),
                "revolute_q": np.asarray(sample["revolute_q"], dtype=float).tolist(),
                "stored_min_distance": float(sample["stored_min_distance"]),
                "pybullet_min_distance": float(pybullet_metric["min_distance"]),
                "pybullet_is_collision": bool(pybullet_metric["is_collision"]),
                "pybullet_active_pair": pybullet_metric.get("active_pair"),
                "coal_min_distance": float(coal_metric["min_distance"]),
                "coal_is_collision": bool(coal_metric["is_collision"]),
                "coal_active_pair": coal_metric.get("active_pair"),
                "coal_contact_penetration_depth": coal_metric.get("contact_penetration_depth"),
                "coal_contact_active_pair": coal_metric.get("contact_active_pair"),
                "coal_pair_reports": coal_metric.get("pair_reports", []),
                "distance_delta_coal_minus_pybullet": float(coal_metric["min_distance"] - pybullet_metric["min_distance"]),
                "collision_agree": bool(bool(pybullet_metric["is_collision"]) == bool(coal_metric["is_collision"])),
            })
        robot.set_joint_state(q_base, dq_base)
    finally:
        if created_connection and p.isConnected():
            p.disconnect()

    total = len(sample_reports)
    agree_count = int(sum(1 for item in sample_reports if item["collision_agree"]))
    disagreement_reports = [item for item in sample_reports if not item["collision_agree"]]
    report = {
        "sample_json": str(Path(params.SAMPLE_JSON)),
        "num_samples": int(total),
        "num_collision_samples_checked": int(sum(1 for item in sample_reports if item["label"] == "collision")),
        "num_free_samples_checked": int(sum(1 for item in sample_reports if item["label"] == "free")),
        "agreement_rate": float(agree_count / max(total, 1)),
        "disagreement_count": int(len(disagreement_reports)),
        "mean_abs_distance_delta": float(np.mean([abs(item["distance_delta_coal_minus_pybullet"]) for item in sample_reports])),
        "joint_indices": [int(j) for j in revolute_ids],
        "joint_names": [str(name) for name in revolute_names],
        "monitored_link_indices": [int(j) for j in monitored_link_ids],
        "monitored_link_names": [str(name) for name in monitored_link_names],
        "monitored_pairs": [[int(a), int(b)] for a, b in monitored_pairs],
        "samples": sample_reports,
        "disagreements": disagreement_reports,
    }
    output_path = Path(params.OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[self-cspace-coal] "
        f"checked={report['num_samples']} "
        f"agreement_rate={report['agreement_rate']:.4f} "
        f"disagreements={report['disagreement_count']}"
    )
    print(f"[self-cspace-coal] saved -> {output_path}")
    return report


if __name__ == "__main__":
    run_coal_validation(CoalValidationParameters)
