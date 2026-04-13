"""Geometry/SDF module for file-driven pipeline."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.simulation_module import Robot, load_config, _resolve  # noqa: E402


def _normalize_rows(rows: np.ndarray) -> np.ndarray:
    arr = np.asarray(rows, dtype=float).reshape(-1, 3)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms > 1e-9, norms, 1.0)
    out = arr / safe
    out[norms[:, 0] <= 1e-9] = np.array([1.0, 0.0, 0.0], dtype=float)
    return out


class GeometryEngine:
    """Load SDF and batch-query it against sampled robot link points."""

    def __init__(self, cfg: dict | object, robot: Robot):
        if hasattr(cfg, "_cfg"):
            cfg = cfg._cfg
        self.cfg = cfg
        self.robot = robot
        self.geometry_cfg = dict(cfg.get("geometry", {}))
        self.enabled = bool(self.geometry_cfg.get("enabled", True))
        self.kind = str(self.geometry_cfg.get("sdf_kind", "o3d_sdf"))
        self.sample_density = float(self.geometry_cfg.get("sample_density", 300.0))
        self.min_samples = int(self.geometry_cfg.get("min_samples_per_link", 96))
        self.max_samples = int(self.geometry_cfg.get("max_samples_per_link", 768))
        self.top_k_per_link = int(self.geometry_cfg.get("cbf_top_k_per_link", 3))
        self.surface_source = str(self.geometry_cfg.get("surface_source", "collision_preferred")).lower()
        self.surface_fallback_to_visual = bool(self.geometry_cfg.get("surface_fallback_to_visual", True))
        self.self_collision_enabled = bool(self.geometry_cfg.get("self_collision_enabled", True))
        self.self_collision_min_index_gap = int(self.geometry_cfg.get("self_collision_min_index_gap", 2))
        self.self_collision_query_distance = float(self.geometry_cfg.get("self_collision_query_distance", 0.12))
        self.link_indices = self._resolve_link_indices()
        self.last_query_points = np.zeros((0, 3), dtype=float)
        self.last_query_meta: list[dict] = []
        self._local_surface_samples: dict[int, np.ndarray] = {}
        self._local_surface_meshes: dict[int, object] = {}
        self._surface_sample_sources: dict[int, str] = {}

        if not self.enabled:
            self.field = None
            self._wp_pos = np.zeros(3, dtype=float)
            self._r_inv = np.eye(3, dtype=float)
            self._pb_to_sdf_linear = np.eye(3, dtype=float)
            return

        sdf_npz = self.geometry_cfg.get("sdf_npz")
        if not sdf_npz:
            raise ValueError("geometry.sdf_npz is required")

        udf_module = self._load_udf_module()
        self.field = udf_module.load_distance_field(_resolve(str(sdf_npz)))

        wp_cfg = cfg.get("workpiece", {})
        self._wp_pos = np.asarray(wp_cfg.get("position", [0.0, 0.0, 0.0]), dtype=float)
        wp_deg = np.asarray(wp_cfg.get("orientation_deg", [0.0, 0.0, 0.0]), dtype=float)
        self._r_inv = Rotation.from_euler("xyz", wp_deg, degrees=True).as_matrix().T

        origin_pb = np.zeros((1, 3), dtype=float)
        basis_pb = np.eye(3, dtype=float)
        mapped_origin = self.pb2sdf(origin_pb)[0]
        mapped_basis = self.pb2sdf(basis_pb)
        self._pb_to_sdf_linear = (mapped_basis - mapped_origin.reshape(1, 3)).T

    @staticmethod
    def _load_udf_module():
        path = Path(__file__).resolve().parent / "4_1_udf.py"
        spec = importlib.util.spec_from_file_location("geometry_udf_runtime", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def _resolve_link_indices(self) -> list[int]:
        raw = self.geometry_cfg.get("cbf_links", "auto")
        if raw == "auto":
            return list(getattr(self.robot, "cbf_link_indices", self.robot.active_joints))
        if isinstance(raw, list):
            return [int(x) for x in raw]
        return list(getattr(self.robot, "cbf_link_indices", self.robot.active_joints))

    def pb2sdf(self, pts_pb: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_pb, dtype=float).reshape(-1, 3)
        return (pts - self._wp_pos.reshape(1, 3)) @ self._r_inv.T

    def sdf2pb(self, pts_sdf: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_sdf, dtype=float).reshape(-1, 3)
        r_fwd = self._r_inv
        return pts @ r_fwd.T + self._wp_pos.reshape(1, 3)

    def query_sdf(self, pts_pb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_pb = np.asarray(pts_pb, dtype=float).reshape(-1, 3)
        points_sdf = np.asarray(self.pb2sdf(points_pb), dtype=np.float32)
        values, grads = self.field.query_with_gradient(points_sdf, kind=self.kind)
        distances = np.asarray(values, dtype=float).reshape(-1)
        grad_sdf = np.asarray(grads, dtype=float).reshape(-1, 3)
        grad_pb = grad_sdf @ self._pb_to_sdf_linear
        normals_pb = _normalize_rows(grad_pb)
        return distances, normals_pb

    @staticmethod
    def _import_trimesh():
        import trimesh  # type: ignore[reportMissingImports]

        return trimesh

    @staticmethod
    def _visual_shapes_for_link(body_id: int, link_index: int):
        return [shape for shape in p.getVisualShapeData(int(body_id)) if int(shape[1]) == int(link_index)]

    @staticmethod
    def _resolve_mesh_path(mesh_file) -> str | None:
        if isinstance(mesh_file, (bytes, bytearray)):
            mesh_file = mesh_file.decode("utf-8", errors="replace")
        mesh_path = str(mesh_file).strip()
        if not mesh_path:
            return None
        candidates = [Path(mesh_path)]
        if not os.path.isabs(mesh_path):
            candidates.append(REPO_ROOT / mesh_path)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return None

    def _build_shape_local_mesh(self, shape_data, *, source_kind: str):
        trimesh = self._import_trimesh()
        geom_type = int(shape_data[2])
        dims = tuple(float(v) for v in np.asarray(shape_data[3], dtype=float).reshape(-1))
        mesh_file = shape_data[4]
        local_pos = np.asarray(shape_data[5], dtype=float).reshape(3)
        local_quat = np.asarray(shape_data[6], dtype=float).reshape(4)
        mesh = None
        if geom_type == p.GEOM_MESH:
            resolved_mesh = self._resolve_mesh_path(mesh_file)
            if resolved_mesh is None:
                return None
            mesh = trimesh.load(resolved_mesh, force="mesh")
            mesh = mesh.copy()
            if dims:
                scale = np.asarray(dims[:3] if len(dims) >= 3 else [dims[0]] * 3, dtype=float).reshape(3)
                mesh.apply_scale(scale)
        elif geom_type == p.GEOM_BOX:
            mesh = trimesh.creation.box(extents=np.asarray(dims[:3], dtype=float))
        elif geom_type == p.GEOM_SPHERE:
            mesh = trimesh.creation.icosphere(radius=float(dims[0]), subdivisions=2)
        elif geom_type == p.GEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=float(dims[1]), height=float(dims[0]), sections=24)
        elif geom_type == p.GEOM_CAPSULE:
            mesh = trimesh.creation.capsule(radius=float(dims[1]), height=float(dims[0]), count=[8, 16])
        if mesh is None:
            return None
        rot = np.asarray(p.getMatrixFromQuaternion(local_quat.tolist()), dtype=float).reshape(3, 3)
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rot
        transform[:3, 3] = local_pos
        mesh.apply_transform(transform)
        return mesh

    def _collect_shape_meshes(self, body_id: int, link_index: int, *, source_kind: str) -> list[object]:
        if source_kind == "collision":
            shape_datas = p.getCollisionShapeData(int(body_id), int(link_index))
        else:
            shape_datas = self._visual_shapes_for_link(int(body_id), int(link_index))
        meshes = []
        for shape_data in shape_datas or []:
            try:
                mesh = self._build_shape_local_mesh(shape_data, source_kind=source_kind)
            except Exception:
                mesh = None
            if mesh is not None and len(getattr(mesh, "vertices", [])) > 0:
                meshes.append(mesh)
        return meshes

    def _resolve_surface_sources(self) -> list[str]:
        source = str(getattr(self, "surface_source", "collision_preferred")).lower()
        fallback = bool(getattr(self, "surface_fallback_to_visual", True))
        if source == "visual_preferred":
            return ["visual", "collision"] if fallback else ["visual"]
        if source == "collision_preferred":
            return ["collision", "visual"] if fallback else ["collision"]
        if source == "visual_only":
            return ["visual"]
        return ["collision"]

    def _load_link_surface_mesh(self, body_id: int, link_index: int) -> tuple[object | None, str | None]:
        trimesh = self._import_trimesh()
        for source_kind in self._resolve_surface_sources():
            meshes = self._collect_shape_meshes(body_id, link_index, source_kind=source_kind)
            if not meshes:
                continue
            if len(meshes) == 1:
                return meshes[0], source_kind
            try:
                return trimesh.util.concatenate(meshes), source_kind
            except Exception:
                return meshes[0], source_kind
        return None, None

    def _build_external_cbf_entries(self, robot: Robot, q: np.ndarray) -> list[dict]:
        by_link = self.sample_link_surfaces(robot, q)
        if not by_link:
            return []
        points = []
        link_indices = []
        for link_index, pts in by_link.items():
            pts = np.asarray(pts, dtype=float).reshape(-1, 3)
            points.append(pts)
            link_indices.extend([int(link_index)] * pts.shape[0])
        stacked = np.vstack(points) if points else np.zeros((0, 3), dtype=float)
        if stacked.shape[0] == 0:
            return []
        distances, normals = self.query_sdf(stacked)
        if getattr(self, "top_k_per_link", 0) > 0 and len(link_indices) > 0:
            selected_idx = []
            link_indices_arr = np.asarray(link_indices, dtype=int)
            for link_index in np.unique(link_indices_arr):
                mask = np.where(link_indices_arr == link_index)[0]
                order = mask[np.argsort(distances[mask])]
                selected_idx.extend(order[: self.top_k_per_link].tolist())
            selected_idx = np.asarray(sorted(selected_idx), dtype=int)
            stacked = stacked[selected_idx]
            distances = distances[selected_idx]
            normals = normals[selected_idx]
            link_indices = link_indices_arr[selected_idx].tolist()
        entries = []
        for idx, link_index in enumerate(link_indices):
            entries.append({
                "kind": "environment",
                "link_index": int(link_index),
                "distance": float(distances[idx]),
                "normal": np.asarray(normals[idx], dtype=float).reshape(3),
                "point_on_link": np.asarray(stacked[idx], dtype=float).reshape(3),
            })
        return entries

    def _resolve_self_collision_pairs(self, robot: Robot) -> list[tuple[int, int]]:
        if not bool(getattr(self, "self_collision_enabled", False)):
            return []
        active_links = [int(link) for link in getattr(robot, "active_joints", [])]
        min_gap = max(int(getattr(self, "self_collision_min_index_gap", 2)), 1)
        pairs: list[tuple[int, int]] = []
        for i, link_a in enumerate(active_links):
            for j in range(i + 1, len(active_links)):
                if (j - i) < min_gap:
                    continue
                pairs.append((link_a, active_links[j]))
        return pairs

    def _build_self_collision_entries(self, robot: Robot) -> list[dict]:
        entries: list[dict] = []
        query_dist = float(getattr(self, "self_collision_query_distance", 0.12))
        if query_dist <= 0.0:
            return entries
        for link_a, link_b in self._resolve_self_collision_pairs(robot):
            contacts = p.getClosestPoints(
                int(robot.body_id),
                int(robot.body_id),
                query_dist,
                linkIndexA=int(link_a),
                linkIndexB=int(link_b),
            )
            if not contacts:
                continue
            best = min(contacts, key=lambda contact: float(contact[8]))
            point_a = np.asarray(best[5], dtype=float).reshape(3)
            point_b = np.asarray(best[6], dtype=float).reshape(3)
            normal_on_b = np.asarray(best[7], dtype=float).reshape(3)
            normal = _normalize_rows(normal_on_b.reshape(1, 3))[0]
            entries.append({
                "kind": "self_collision",
                "link_index": int(link_a),
                "other_link_index": int(link_b),
                "distance": float(best[8]),
                "normal": normal,
                "point_on_link": point_a,
                "point_on_other_link": point_b,
            })
        return entries

    @staticmethod
    def _surface_grid_from_aabb(aabb_min: np.ndarray, aabb_max: np.ndarray, n_samples: int) -> np.ndarray:
        aabb_min = np.asarray(aabb_min, dtype=float).reshape(3)
        aabb_max = np.asarray(aabb_max, dtype=float).reshape(3)
        center = 0.5 * (aabb_min + aabb_max)
        ext = np.maximum(aabb_max - aabb_min, 1e-4)
        side = max(2, int(np.ceil(np.sqrt(max(n_samples // 6, 1)))))
        xs = np.linspace(aabb_min[0], aabb_max[0], side)
        ys = np.linspace(aabb_min[1], aabb_max[1], side)
        zs = np.linspace(aabb_min[2], aabb_max[2], side)

        points = []
        for x in xs:
            for y in ys:
                points.append([x, y, aabb_min[2]])
                points.append([x, y, aabb_max[2]])
        for x in xs:
            for z in zs:
                points.append([x, aabb_min[1], z])
                points.append([x, aabb_max[1], z])
        for y in ys:
            for z in zs:
                points.append([aabb_min[0], y, z])
                points.append([aabb_max[0], y, z])
        points.append(center.tolist())
        unique = np.unique(np.asarray(points, dtype=float), axis=0)
        if unique.shape[0] <= n_samples:
            return unique
        idx = np.linspace(0, unique.shape[0] - 1, n_samples, dtype=int)
        return unique[idx]

    @staticmethod
    def _world_to_link_local(points_world: np.ndarray, link_pos, link_quat) -> np.ndarray:
        pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
        origin = np.asarray(link_pos, dtype=float).reshape(1, 3)
        rot = np.asarray(p.getMatrixFromQuaternion(link_quat), dtype=float).reshape(3, 3)
        return (pts - origin) @ rot

    @staticmethod
    def _link_local_to_world(points_local: np.ndarray, link_pos, link_quat) -> np.ndarray:
        pts = np.asarray(points_local, dtype=float).reshape(-1, 3)
        origin = np.asarray(link_pos, dtype=float).reshape(1, 3)
        rot = np.asarray(p.getMatrixFromQuaternion(link_quat), dtype=float).reshape(3, 3)
        return pts @ rot.T + origin

    def _init_local_surface_samples(self, robot: Robot, q: np.ndarray):
        robot.set_joint_state(q)
        if not hasattr(self, "_local_surface_samples"):
            self._local_surface_samples = {}
        if not hasattr(self, "_local_surface_meshes"):
            self._local_surface_meshes = {}
        if not hasattr(self, "_surface_sample_sources"):
            self._surface_sample_sources = {}
        for link_index in self.link_indices:
            mesh, source_kind = self._load_link_surface_mesh(robot.body_id, int(link_index))
            if mesh is None:
                continue
            area = float(max(getattr(mesh, "area", 0.0), 1e-6))
            n_samples = int(np.clip(round(area * self.sample_density), self.min_samples, self.max_samples))
            try:
                samples_local, _face_idx = self._import_trimesh().sample.sample_surface(mesh, n_samples)
            except Exception:
                continue
            samples_local = np.asarray(samples_local, dtype=float).reshape(-1, 3)
            if samples_local.shape[0] == 0:
                continue
            self._local_surface_samples[int(link_index)] = samples_local
            self._local_surface_meshes[int(link_index)] = mesh
            self._surface_sample_sources[int(link_index)] = str(source_kind)

    def sample_link_surfaces(self, robot: Robot, q: np.ndarray) -> dict[int, np.ndarray]:
        if not self._local_surface_samples:
            self._init_local_surface_samples(robot, q)
        robot.set_joint_state(q)
        samples = {}
        for link_index, local_samples in self._local_surface_samples.items():
            link_state = p.getLinkState(robot.body_id, int(link_index), computeForwardKinematics=True)
            link_pos, link_quat = link_state[4], link_state[5]
            samples[int(link_index)] = self._link_local_to_world(local_samples, link_pos, link_quat)
        return samples

    def get_cbf_distances(self, robot: Robot, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
        if not getattr(self, "enabled", True):
            self.last_query_points = np.zeros((0, 3), dtype=float)
            self.last_query_meta = []
            return np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float), []
        entries = self._build_external_cbf_entries(robot, q)
        entries.extend(self._build_self_collision_entries(robot))
        if not entries:
            self.last_query_points = np.zeros((0, 3), dtype=float)
            self.last_query_meta = []
            return np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float), []
        self.last_query_meta = [dict(entry) for entry in entries]
        self.last_query_points = np.asarray([entry["point_on_link"] for entry in entries], dtype=float).reshape(-1, 3)
        distances = np.asarray([entry["distance"] for entry in entries], dtype=float).reshape(-1)
        normals = np.asarray([entry["normal"] for entry in entries], dtype=float).reshape(-1, 3)
        link_indices = [int(entry["link_index"]) for entry in entries]
        return distances, normals, link_indices


def load_geometry_engine(cfg_path: str | Path | None = None, robot: Robot | None = None) -> GeometryEngine:
    cfg = load_config(cfg_path)
    if robot is None:
        robot = Robot(cfg)
    return GeometryEngine(cfg, robot)

