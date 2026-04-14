"""Unified coal-based self-collision detection backend.

All self-collision experiment scripts should import from here
instead of calling PyBullet getClosestPoints / getContactPoints directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pybullet as p

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import coal  # noqa: E402


def _decode_path(path_value) -> str:
    if isinstance(path_value, bytes):
        return path_value.decode("utf-8", errors="replace")
    return str(path_value)


def _build_transform(world_pos, world_quat) -> coal.Transform3s:
    tf = coal.Transform3s()
    rot = np.asarray(p.getMatrixFromQuaternion(world_quat), dtype=float).reshape(3, 3)
    tf.setRotation(rot)
    tf.setTranslation(np.asarray(world_pos, dtype=float).reshape(3))
    return tf


def get_world_collision_transform(robot, link_index: int, local_pos, local_quat) -> coal.Transform3s:
    link_state = p.getLinkState(int(robot.body_id), int(link_index), computeForwardKinematics=True)
    world_pos, world_quat = p.multiplyTransforms(
        link_state[0],
        link_state[1],
        np.asarray(local_pos, dtype=float).reshape(3).tolist(),
        np.asarray(local_quat, dtype=float).reshape(4).tolist(),
    )
    return _build_transform(world_pos, world_quat)


def coal_pair_is_collision(distance: float, collide_flag: bool, *, penetration_thresh: float) -> bool:
    if bool(collide_flag):
        return True
    if not np.isfinite(float(distance)):
        return False
    return bool(float(distance) <= 0.0 or float(distance) < float(penetration_thresh))


def extract_contact_penetration_depth(collision_result) -> float | None:
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


def compute_pairwise_self_collision_distance(
    robot,
    *,
    link_models: dict[int, dict],
    monitored_pairs,
    penetration_thresh: float = -0.001,
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
        tf_a = get_world_collision_transform(robot, int(link_a), model_a["local_pos"], model_a["local_quat"])
        tf_b = get_world_collision_transform(robot, int(link_b), model_b["local_pos"], model_b["local_quat"])
        distance_result = coal.DistanceResult()
        collision_result = coal.CollisionResult()
        coal.distance(model_a["geometry"], tf_a, model_b["geometry"], tf_b, distance_request, distance_result)
        coal.collide(model_a["geometry"], tf_a, model_b["geometry"], tf_b, collision_request, collision_result)
        distance = float(distance_result.min_distance)
        collide_flag = bool(collision_result.isCollision())
        contact_depth = extract_contact_penetration_depth(collision_result)
        pair_is_collision = coal_pair_is_collision(
            distance, collide_flag, penetration_thresh=float(penetration_thresh),
        )
        pair_reports.append({
            "pair": [int(link_a), int(link_b)],
            "distance": distance,
            "collide_flag": collide_flag,
            "contact_penetration_depth": contact_depth,
            "is_collision": pair_is_collision,
        })
        any_collision = any_collision or pair_is_collision
        if distance < best:
            best = distance
            active_pair = [int(link_a), int(link_b)]
        if contact_depth is not None and contact_depth < best_contact_penetration:
            best_contact_penetration = contact_depth
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


def is_any_pair_collision_fast(
    robot,
    *,
    link_models: dict[int, dict],
    monitored_pairs,
) -> bool:
    """Collide-only check with early exit. ~2x faster than full distance query."""
    req = coal.CollisionRequest()
    for link_a, link_b in monitored_pairs:
        model_a = link_models[int(link_a)]
        model_b = link_models[int(link_b)]
        tf_a = get_world_collision_transform(robot, int(link_a), model_a["local_pos"], model_a["local_quat"])
        tf_b = get_world_collision_transform(robot, int(link_b), model_b["local_pos"], model_b["local_quat"])
        res = coal.CollisionResult()
        coal.collide(model_a["geometry"], tf_a, model_b["geometry"], tf_b, req, res)
        if res.isCollision():
            return True
    return False


def classify_self_collision_sample(
    robot,
    *,
    link_models: dict[int, dict],
    monitored_pairs,
    penetration_thresh: float = -0.001,
) -> dict:
    return compute_pairwise_self_collision_distance(
        robot,
        link_models=link_models,
        monitored_pairs=monitored_pairs,
        penetration_thresh=float(penetration_thresh),
    )
