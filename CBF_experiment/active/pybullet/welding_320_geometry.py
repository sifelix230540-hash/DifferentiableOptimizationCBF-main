from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pybullet as p


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 1e-9 else np.zeros_like(arr)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms > 1e-9, norms, 1.0)
    normalized = arr / safe
    normalized[norms[:, 0] <= 1e-9] = 0.0
    return normalized


def sample_cloud_for_visualization(
    points: np.ndarray,
    normals: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    points_arr = np.asarray(points, dtype=float).reshape(-1, 3)
    normals_arr = np.asarray(normals, dtype=float).reshape(-1, 3)
    if points_arr.shape[0] == 0:
        return points_arr.copy(), normals_arr.copy()
    limit = max(int(max_points), 1)
    if points_arr.shape[0] <= limit:
        return points_arr.copy(), normals_arr.copy()
    indices = np.linspace(0, points_arr.shape[0] - 1, num=limit, dtype=int)
    return points_arr[indices], normals_arr[indices]


def compute_surface_sample_count(area: float, density: float, min_samples: int, max_samples: int) -> int:
    area_val = float(area)
    density_val = float(density)
    min_val = int(min_samples)
    max_val = int(max_samples)
    count_by_density = int(area_val * density_val) if area_val > 0.0 else min_val
    return max(min_val, min(max_val, count_by_density))


def resolve_surface_sampling_params(config, role: str) -> tuple[float, int, int]:
    role_name = str(role or "default").lower()
    if role_name == "robot_rear_six":
        return (
            float(getattr(config, "robot_rear_six_surface_target_density", getattr(config, "robot_surface_target_density", getattr(config, "surface_target_density", 400.0)))),
            int(getattr(config, "robot_rear_six_surface_min_samples", getattr(config, "robot_surface_min_samples", getattr(config, "surface_min_samples", 64)))),
            int(getattr(config, "robot_rear_six_surface_max_samples", getattr(config, "robot_surface_max_samples", getattr(config, "surface_max_samples", 1024)))),
        )
    if role_name == "obstacle":
        return (
            float(getattr(config, "obstacle_surface_target_density", getattr(config, "surface_target_density", 400.0))),
            int(getattr(config, "obstacle_surface_min_samples", getattr(config, "surface_min_samples", 64))),
            int(getattr(config, "obstacle_surface_max_samples", getattr(config, "surface_max_samples", 1024))),
        )
    if role_name == "robot":
        return (
            float(getattr(config, "robot_surface_target_density", getattr(config, "surface_target_density", 400.0))),
            int(getattr(config, "robot_surface_min_samples", getattr(config, "surface_min_samples", 64))),
            int(getattr(config, "robot_surface_max_samples", getattr(config, "surface_max_samples", 1024))),
        )
    return (
        float(getattr(config, "surface_target_density", 400.0)),
        int(getattr(config, "surface_min_samples", 64)),
        int(getattr(config, "surface_max_samples", 1024)),
    )


def resolve_surface_visual_max_points(config, role: str) -> int:
    role_name = str(role or "default").lower()
    if role_name == "robot_rear_six":
        return int(
            getattr(
                config,
                "robot_rear_six_visual_max_points_per_link",
                getattr(config, "robot_surface_visual_max_points_per_link", getattr(config, "surface_visual_max_points_per_link", 48)),
            )
        )
    if role_name == "obstacle":
        return int(
            getattr(
                config,
                "obstacle_surface_visual_max_points_per_link",
                getattr(config, "surface_visual_max_points_per_link", 48),
            )
        )
    if role_name == "robot":
        return int(
            getattr(
                config,
                "robot_surface_visual_max_points_per_link",
                getattr(config, "surface_visual_max_points_per_link", 48),
            )
        )
    return int(getattr(config, "surface_visual_max_points_per_link", 48))


def apply_contains_sign_to_distance(
    mesh,
    point_local: np.ndarray,
    unsigned_dist: float,
    fallback_signed_dist: float,
) -> float:
    if mesh is None:
        return float(fallback_signed_dist)
    try:
        inside = bool(mesh.contains(np.asarray(point_local, dtype=float).reshape(1, 3))[0])
    except Exception:
        return float(fallback_signed_dist)
    magnitude = abs(float(unsigned_dist))
    return -magnitude if inside else magnitude


def compute_world_surface(
    local_points: np.ndarray,
    local_normals: np.ndarray,
    world_pos: np.ndarray,
    world_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rot = np.array(p.getMatrixFromQuaternion(np.asarray(world_quat, dtype=float).tolist()), dtype=float).reshape(3, 3)
    pts = (rot @ np.asarray(local_points, dtype=float).T).T + np.asarray(world_pos, dtype=float)
    normals = (rot @ np.asarray(local_normals, dtype=float).T).T
    return pts, _normalize_rows(normals)


def _compose_surface_result(
    robot_points: np.ndarray,
    robot_normals: np.ndarray,
    obstacle_points: np.ndarray,
    obstacle_normals: np.ndarray,
    idx_robot: int,
    idx_obstacle: int,
) -> dict:
    point_on_link = np.asarray(robot_points[idx_robot], dtype=float)
    point_on_obstacle = np.asarray(obstacle_points[idx_obstacle], dtype=float)
    normal_on_link = _normalize_rows(np.asarray(robot_normals[idx_robot], dtype=float))
    normal_on_obstacle = _normalize_rows(np.asarray(obstacle_normals[idx_obstacle], dtype=float))
    separation = point_on_link - point_on_obstacle
    euclidean_dist = float(np.linalg.norm(separation))
    signed_from_obstacle = float(np.dot(separation, normal_on_obstacle))
    signed_from_link = float(np.dot(-separation, normal_on_link))
    signed_dist = 0.5 * (signed_from_obstacle + signed_from_link)
    return {
        "point_on_link": point_on_link,
        "point_on_obstacle": point_on_obstacle,
        "normal_on_link": normal_on_link,
        "normal_on_obstacle": normal_on_obstacle,
        "signed_dist": float(signed_dist),
        "euclidean_dist": euclidean_dist,
        "robot_point_index": int(idx_robot),
        "obstacle_point_index": int(idx_obstacle),
    }


def find_closest_surface_pair_cpu(
    robot_points: np.ndarray,
    robot_normals: np.ndarray,
    obstacle_points: np.ndarray,
    obstacle_normals: np.ndarray,
) -> dict:
    robot_points = np.asarray(robot_points, dtype=float)
    obstacle_points = np.asarray(obstacle_points, dtype=float)
    if robot_points.size == 0 or obstacle_points.size == 0:
        raise ValueError("最近点查询需要非空点云。")
    diff = robot_points[:, None, :] - obstacle_points[None, :, :]
    dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
    flat_idx = int(np.argmin(dist_sq))
    idx_robot, idx_obstacle = np.unravel_index(flat_idx, dist_sq.shape)
    return _compose_surface_result(
        robot_points,
        np.asarray(robot_normals, dtype=float),
        obstacle_points,
        np.asarray(obstacle_normals, dtype=float),
        idx_robot,
        idx_obstacle,
    )


def _import_torch():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import trimesh  # type: ignore[reportMissingImports]  # noqa: F401
    except Exception:
        pass
    import torch

    return torch


def find_closest_surface_pair_torch(
    robot_points,
    robot_normals,
    obstacle_points,
    obstacle_normals,
    chunk_size: int = 2048,
) -> dict:
    torch = _import_torch()
    if robot_points.numel() == 0 or obstacle_points.numel() == 0:
        raise ValueError("最近点查询需要非空点云。")
    best_dist = None
    best_row = 0
    best_col = 0
    for start in range(0, robot_points.shape[0], max(int(chunk_size), 1)):
        stop = min(start + max(int(chunk_size), 1), robot_points.shape[0])
        dist = torch.cdist(robot_points[start:stop], obstacle_points)
        row_min, row_idx = torch.min(dist, dim=1)
        chunk_best, chunk_row = torch.min(row_min, dim=0)
        if best_dist is None or float(chunk_best) < best_dist:
            best_dist = float(chunk_best)
            best_row = int(start + int(chunk_row))
            best_col = int(row_idx[int(chunk_row)])
    return _compose_surface_result(
        robot_points[best_row].detach().cpu().numpy()[None, :],
        robot_normals[best_row].detach().cpu().numpy()[None, :],
        obstacle_points[best_col].detach().cpu().numpy()[None, :],
        obstacle_normals[best_col].detach().cpu().numpy()[None, :],
        0,
        0,
    )


@dataclass
class SurfaceLinkCloud:
    body_id: int
    link_index: int
    link_name: str
    local_points: np.ndarray
    local_normals: np.ndarray
    role: str = "default"
    local_mesh: object | None = None
    device_points: object | None = None
    device_normals: object | None = None


class SurfaceDistanceEngine:
    def __init__(self, config):
        self.config = config
        self._body_clouds: dict[int, dict[int, SurfaceLinkCloud]] = {}
        self._body_roles: dict[int, str] = {}
        self._world_cache: dict[tuple[int, int], dict] = {}
        self._torch = None
        self._gpu_enabled = False
        if getattr(config, "surface_prefer_gpu", False):
            try:
                torch = _import_torch()
                if torch.cuda.is_available():
                    self._torch = torch
                    self._gpu_enabled = True
            except Exception:
                self._torch = None
                self._gpu_enabled = False

    @property
    def gpu_enabled(self) -> bool:
        return self._gpu_enabled

    def clear_world_cache(self):
        self._world_cache.clear()

    @staticmethod
    def _get_body_link_pose(body_id: int, link_index: int):
        if int(link_index) < 0:
            return p.getBasePositionAndOrientation(body_id)
        state = p.getLinkState(body_id, int(link_index), computeForwardKinematics=True)
        return state[0], state[1]

    @staticmethod
    def _get_body_link_name(body_id: int, link_index: int) -> str:
        if int(link_index) < 0:
            return "base_link"
        return p.getJointInfo(body_id, int(link_index))[12].decode()

    @staticmethod
    def _import_trimesh():
        import trimesh  # type: ignore[reportMissingImports]

        return trimesh

    def _load_shape_mesh(self, shape_data):
        trimesh = self._import_trimesh()
        geom_type = int(shape_data[2])
        dims = tuple(float(v) for v in shape_data[3])
        filename = shape_data[4]
        if isinstance(filename, (bytes, bytearray)):
            filename = filename.decode("utf-8")
        if geom_type == p.GEOM_MESH:
            return trimesh.load(str(filename), force="mesh")
        if geom_type == p.GEOM_BOX:
            return trimesh.creation.box(extents=dims[:3])
        if geom_type == p.GEOM_SPHERE:
            return trimesh.creation.icosphere(radius=dims[0], subdivisions=2)
        if geom_type == p.GEOM_CYLINDER:
            return trimesh.creation.cylinder(radius=dims[1], height=dims[0], sections=24)
        if geom_type == p.GEOM_CAPSULE:
            return trimesh.creation.capsule(radius=dims[1], height=dims[0], count=[8, 16])
        raise ValueError(f"不支持的碰撞几何类型: {geom_type}")

    def _build_shape_local_mesh(self, shape_data):
        mesh = self._load_shape_mesh(shape_data).copy()
        local_pos = np.asarray(shape_data[5], dtype=float)
        local_quat = np.asarray(shape_data[6], dtype=float)
        rot = np.array(p.getMatrixFromQuaternion(local_quat.tolist()), dtype=float).reshape(3, 3)
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rot
        transform[:3, 3] = local_pos
        mesh.apply_transform(transform)
        return mesh

    def _sample_shape_surface(self, shape_data, role: str = "default") -> tuple[np.ndarray, np.ndarray]:
        trimesh = self._import_trimesh()
        mesh = self._load_shape_mesh(shape_data)
        mesh = mesh.copy()
        if hasattr(mesh, "remove_unreferenced_vertices"):
            mesh.remove_unreferenced_vertices()
        if hasattr(mesh, "fix_normals"):
            mesh.fix_normals()

        area = float(getattr(mesh, "area", 0.0))
        density, min_samples, max_samples = resolve_surface_sampling_params(self.config, role=role)
        count = compute_surface_sample_count(area, density, min_samples, max_samples)

        points_local, face_indices = trimesh.sample.sample_surface(mesh, count)
        normals_local = mesh.face_normals[face_indices]

        local_pos = np.asarray(shape_data[5], dtype=float)
        local_quat = np.asarray(shape_data[6], dtype=float)
        rot = np.array(p.getMatrixFromQuaternion(local_quat.tolist()), dtype=float).reshape(3, 3)
        points = (rot @ points_local.T).T + local_pos
        normals = _normalize_rows((rot @ normals_local.T).T)
        return np.asarray(points, dtype=float), np.asarray(normals, dtype=float)

    def register_body(
        self,
        body_id: int,
        link_indices: list[int] | None = None,
        role: str = "default",
        link_role_map: dict[int, str] | None = None,
    ):
        selected = None if link_indices is None else {int(li) for li in link_indices}
        body_clouds: dict[int, SurfaceLinkCloud] = {}
        role_name = str(role or "default").lower()
        for link_index in range(-1, p.getNumJoints(body_id)):
            if selected is not None and link_index not in selected:
                continue
            current_role = str((link_role_map or {}).get(int(link_index), role_name)).lower()
            shape_datas = p.getCollisionShapeData(body_id, link_index)
            if not shape_datas:
                continue
            shape_points = []
            shape_normals = []
            shape_meshes = []
            for shape_data in shape_datas:
                pts, normals = self._sample_shape_surface(shape_data, role=current_role)
                shape_points.append(pts)
                shape_normals.append(normals)
                try:
                    shape_meshes.append(self._build_shape_local_mesh(shape_data))
                except Exception:
                    pass
            if not shape_points:
                continue
            local_points = np.vstack(shape_points)
            local_normals = _normalize_rows(np.vstack(shape_normals))
            local_mesh = None
            if shape_meshes:
                trimesh = self._import_trimesh()
                try:
                    local_mesh = trimesh.util.concatenate(shape_meshes)
                except Exception:
                    local_mesh = shape_meshes[0]
            cloud = SurfaceLinkCloud(
                body_id=int(body_id),
                link_index=int(link_index),
                link_name=self._get_body_link_name(body_id, link_index),
                local_points=local_points,
                local_normals=local_normals,
                role=current_role,
                local_mesh=local_mesh,
            )
            if self._gpu_enabled:
                cloud.device_points = self._torch.as_tensor(local_points, dtype=self._torch.float32, device="cuda")
                cloud.device_normals = self._torch.as_tensor(local_normals, dtype=self._torch.float32, device="cuda")
            body_clouds[int(link_index)] = cloud
        self._body_clouds[int(body_id)] = body_clouds
        self._body_roles[int(body_id)] = role_name
        self.clear_world_cache()

    def _get_world_cloud(self, body_id: int, link_index: int) -> dict | None:
        body_clouds = self._body_clouds.get(int(body_id), {})
        cloud = body_clouds.get(int(link_index))
        if cloud is None:
            return None
        world_pos, world_quat = self._get_body_link_pose(body_id, link_index)
        cache_key = (
            int(body_id),
            int(link_index),
            tuple(np.asarray(world_pos, dtype=float).round(9)),
            tuple(np.asarray(world_quat, dtype=float).round(9)),
        )
        cached = self._world_cache.get(cache_key)
        if cached is not None:
            return cached
        world_points, world_normals = compute_world_surface(
            cloud.local_points,
            cloud.local_normals,
            np.asarray(world_pos, dtype=float),
            np.asarray(world_quat, dtype=float),
        )
        payload = {
            "body_id": int(body_id),
            "link_index": int(link_index),
            "link_name": cloud.link_name,
            "points": world_points,
            "normals": world_normals,
            "local_mesh": cloud.local_mesh,
            "world_pos": np.asarray(world_pos, dtype=float),
            "world_quat": np.asarray(world_quat, dtype=float),
        }
        if self._gpu_enabled:
            payload["device_points"] = self._torch.as_tensor(world_points, dtype=self._torch.float32, device="cuda")
            payload["device_normals"] = self._torch.as_tensor(world_normals, dtype=self._torch.float32, device="cuda")
        self._world_cache[cache_key] = payload
        return payload

    def get_visualization_clouds(
        self,
        body_id: int,
        link_indices: list[int] | None = None,
        max_points_per_link: int | None = None,
    ) -> list[dict]:
        available = self._body_clouds.get(int(body_id), {})
        candidate_links = (
            [int(li) for li in link_indices if int(li) in available]
            if link_indices is not None
            else list(available.keys())
        )
        clouds = []
        for link_index in candidate_links:
            world_cloud = self._get_world_cloud(int(body_id), int(link_index))
            if world_cloud is None:
                continue
            point_limit = (
                int(max_points_per_link)
                if max_points_per_link is not None
                else resolve_surface_visual_max_points(self.config, role=getattr(available.get(int(link_index)), "role", self._body_roles.get(int(body_id), "default")))
            )
            points, normals = sample_cloud_for_visualization(
                world_cloud["points"],
                world_cloud["normals"],
                max_points=point_limit,
            )
            if points.shape[0] == 0:
                continue
            clouds.append({
                "body_id": int(body_id),
                "link_index": int(link_index),
                "link_name": str(world_cloud["link_name"]),
                "points": points,
                "normals": normals,
            })
        return clouds

    @staticmethod
    def _aabb_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
        min_a = points_a.min(axis=0)
        max_a = points_a.max(axis=0)
        min_b = points_b.min(axis=0)
        max_b = points_b.max(axis=0)
        delta = np.maximum(0.0, np.maximum(min_a - max_b, min_b - max_a))
        return float(np.linalg.norm(delta))

    @staticmethod
    def _transform_world_point_to_local(point_world: np.ndarray, world_pos: np.ndarray, world_quat: np.ndarray) -> np.ndarray:
        inv_pos, inv_quat = p.invertTransform(
            np.asarray(world_pos, dtype=float).tolist(),
            np.asarray(world_quat, dtype=float).tolist(),
        )
        point_local, _ = p.multiplyTransforms(inv_pos, inv_quat, np.asarray(point_world, dtype=float).tolist(), [0, 0, 0, 1])
        return np.asarray(point_local, dtype=float)

    def query_link_to_body(
        self,
        robot_body_id: int,
        robot_link_index: int,
        obstacle_body_id: int,
        obstacle_link_indices: list[int] | None = None,
        max_dist: float | None = None,
    ) -> dict | None:
        robot_cloud = self._get_world_cloud(robot_body_id, robot_link_index)
        if robot_cloud is None:
            return None
        available_obs = self._body_clouds.get(int(obstacle_body_id), {})
        candidate_links = (
            [int(li) for li in obstacle_link_indices if int(li) in available_obs]
            if obstacle_link_indices is not None
            else list(available_obs.keys())
        )
        best = None
        for obs_link_index in candidate_links:
            obs_cloud = self._get_world_cloud(obstacle_body_id, obs_link_index)
            if obs_cloud is None:
                continue
            if max_dist is not None and self._aabb_distance(robot_cloud["points"], obs_cloud["points"]) > float(max_dist):
                continue
            if self._gpu_enabled:
                pair = find_closest_surface_pair_torch(
                    robot_cloud["device_points"],
                    robot_cloud["device_normals"],
                    obs_cloud["device_points"],
                    obs_cloud["device_normals"],
                    chunk_size=int(getattr(self.config, "surface_gpu_chunk_size", 2048)),
                )
            else:
                pair = find_closest_surface_pair_cpu(
                    robot_cloud["points"],
                    robot_cloud["normals"],
                    obs_cloud["points"],
                    obs_cloud["normals"],
                )
            point_on_obstacle_local = self._transform_world_point_to_local(
                pair["point_on_obstacle"],
                robot_cloud["world_pos"],
                robot_cloud["world_quat"],
            )
            pair["signed_dist"] = apply_contains_sign_to_distance(
                mesh=robot_cloud.get("local_mesh"),
                point_local=point_on_obstacle_local,
                unsigned_dist=pair["euclidean_dist"],
                fallback_signed_dist=pair["signed_dist"],
            )
            pair.update({
                "robot_link_index": int(robot_link_index),
                "robot_link_name": str(robot_cloud["link_name"]),
                "obs_link_index": int(obs_link_index),
                "obs_link_name": str(obs_cloud["link_name"]),
            })
            if best is None or float(pair["signed_dist"]) < float(best["signed_dist"]):
                best = pair
        return best
