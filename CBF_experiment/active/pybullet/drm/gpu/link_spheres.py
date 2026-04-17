"""mesh → 球链拟合（cuRobo 风格的 VOXEL_VOLUME 算法）。

核心思路（与 NVIDIA cuRobo `SphereFitType.VOXEL_VOLUME_INSIDE` 一致）：
  1. 对 mesh 体素化（pitch = 球直径），取所有内部 voxel 中心作为球心。
  2. 半径 = pitch * sqrt(3) / 2（让单球覆盖整个 voxel 含对角线 → 球链是 mesh 的超集）。
  3. 太薄/壳 mesh 退化时，fallback 用表面采样 + 贪心覆盖。

得到的球链满足：球链不碰 ⇒ 真实 mesh 不碰（保守安全），适合做 GPU 粗筛。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as p
import trimesh


@dataclass
class LinkSpheres:
    """单个 link 的球链（局部坐标）。

    centers : (N, 3) float32, link 局部系下的球心
    radii   : (N,)   float32, 每个球半径
    """
    link_index: int
    link_name: str
    centers: np.ndarray
    radii: np.ndarray

    @property
    def n(self) -> int:
        return int(self.centers.shape[0])

    def to_dict(self) -> dict:
        return {
            "link_index": int(self.link_index),
            "link_name": self.link_name,
            "centers": self.centers.tolist(),
            "radii": self.radii.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LinkSpheres":
        return cls(
            link_index=int(d["link_index"]),
            link_name=str(d["link_name"]),
            centers=np.asarray(d["centers"], dtype=np.float32),
            radii=np.asarray(d["radii"], dtype=np.float32),
        )


# ── 1. 单 mesh 拟合 ──────────────────────────────────


def fit_spheres_voxel_volume(
    vertices: np.ndarray,
    faces: np.ndarray | None,
    pitch: float,
    max_spheres: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """mesh → 保守球链。

    Parameters
    ----------
    vertices : (V, 3) 顶点（任意坐标系，输出球心在同一系）
    faces    : (F, 3) 面索引（None 时按点云处理）
    pitch    : 体素边长，决定球链密度（典型 0.02 = 2cm）
    max_spheres : 球数上限（超出则按距离贪心下采样）

    Returns
    -------
    centers (N, 3), radii (N,)
    """
    verts = np.asarray(vertices, dtype=np.float32)
    if verts.shape[0] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)

    radius = float(pitch) * math.sqrt(3.0) / 2.0  # 保证球覆盖整个 voxel 对角线
    centers: np.ndarray | None = None

    if faces is not None and len(faces) > 0:
        try:
            mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64),
                                   process=False)
            vox = mesh.voxelized(pitch=float(pitch))
            try:
                vox = vox.fill()
            except Exception:
                pass
            pts = np.asarray(vox.points, dtype=np.float32)
            if pts.shape[0] > 0:
                centers = pts
        except Exception:
            centers = None

    if centers is None or centers.shape[0] == 0:
        # fallback：表面顶点 + 贪心去重
        centers = _greedy_downsample(verts, min_dist=pitch)

    if centers.shape[0] > max_spheres:
        centers, radii = _greedy_cover_downsample(centers, base_radius=radius,
                                                  target=int(max_spheres))
    else:
        radii = np.full((centers.shape[0],), radius, dtype=np.float32)
    return centers, radii


def _greedy_downsample(pts: np.ndarray, *, min_dist: float, target: int | None = None) -> np.ndarray:
    """贪心保证球心间距 ≥ min_dist；可选目标球数。仅用于 fallback 表面去重。"""
    if pts.shape[0] == 0:
        return pts
    pts = np.asarray(pts, dtype=np.float32)
    keep = [pts[0]]
    md2 = float(min_dist) ** 2
    for q in pts[1:]:
        kept = np.stack(keep)
        if np.min(np.sum((kept - q) ** 2, axis=1)) >= md2:
            keep.append(q)
        if target is not None and len(keep) >= target:
            break
    return np.stack(keep).astype(np.float32)


def _greedy_cover_downsample(
    pts: np.ndarray, *, base_radius: float, target: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Farthest-point sampling 选 target 个球心，半径扩张以覆盖所有原始球。

    保证 ``union(原球链) ⊆ union(下采样球链)`` —— 即下采样仍是 mesh 的超集。

    Returns
    -------
    centers : (target, 3)
    radii   : (target,)  ≥ base_radius
    """
    pts = np.asarray(pts, dtype=np.float32)
    N = pts.shape[0]
    if N <= target:
        return pts, np.full(N, base_radius, dtype=np.float32)

    # 1) farthest-point sampling
    keep_idx = [0]
    d2 = np.sum((pts - pts[0]) ** 2, axis=1)
    for _ in range(target - 1):
        nxt = int(np.argmax(d2))
        keep_idx.append(nxt)
        d2 = np.minimum(d2, np.sum((pts - pts[nxt]) ** 2, axis=1))
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    centers = pts[keep_idx]

    # 2) 每个原球 → 最近保留球；保留球半径覆盖所有被代表球
    # diffs (N, K, 3) → (N, K)
    dists = np.sqrt(np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=-1))
    nearest = np.argmin(dists, axis=1)
    max_d = np.zeros(target, dtype=np.float32)
    for i in range(N):
        k = int(nearest[i])
        if dists[i, k] > max_d[k]:
            max_d[k] = dists[i, k]
    radii = (max_d + base_radius).astype(np.float32)
    return centers, radii


# ── 2. 从 PyBullet collision 几何提取 ─────────────────


def _shape_to_local_mesh(shape) -> tuple[np.ndarray, np.ndarray | None] | None:
    """把单个 PyBullet collision shape 转换为局部坐标的 (vertices, faces)。"""
    geom_type = int(shape[2])
    dims = shape[3]
    local_pos = np.array(shape[5], dtype=float)
    local_quat = np.array(shape[6], dtype=float)
    R = np.array(p.getMatrixFromQuaternion(local_quat), dtype=float).reshape(3, 3)

    mesh_file = shape[4]
    if isinstance(mesh_file, bytes):
        mesh_file = mesh_file.decode("utf-8", errors="replace").strip()

    verts: np.ndarray | None = None
    faces: np.ndarray | None = None

    if geom_type == p.GEOM_MESH and mesh_file:
        mp = Path(mesh_file)
        if mp.exists():
            try:
                m = trimesh.load(str(mp), force="mesh")
                scale = np.array(dims[:3], dtype=float) if len(dims) >= 3 else np.ones(3)
                verts = np.asarray(m.vertices, dtype=float) * scale
                faces = np.asarray(m.faces, dtype=np.int64)
            except Exception:
                pass
    elif geom_type == p.GEOM_BOX:
        hx, hy, hz = float(dims[0]) / 2, float(dims[1]) / 2, float(dims[2]) / 2
        m = trimesh.creation.box(extents=(hx * 2, hy * 2, hz * 2))
        verts, faces = np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int64)
    elif geom_type == p.GEOM_SPHERE:
        m = trimesh.creation.icosphere(radius=float(dims[0]), subdivisions=2)
        verts, faces = np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int64)
    elif geom_type == p.GEOM_CYLINDER:
        # PyBullet cylinder dims: (length, radius, radius)
        m = trimesh.creation.cylinder(radius=float(dims[1]), height=float(dims[0]), sections=24)
        verts, faces = np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int64)

    if verts is None or len(verts) == 0:
        return None
    verts = (R @ verts.T).T + local_pos
    return verts.astype(np.float32), (faces.astype(np.int64) if faces is not None else None)


def fit_link_spheres(
    body_id: int,
    link_index: int,
    link_name: str,
    pitch: float,
    max_spheres_per_link: int = 80,
) -> LinkSpheres:
    """从已加载的 PyBullet body 提取一个 link 的 collision mesh，并拟合球链。"""
    shapes = p.getCollisionShapeData(body_id, link_index)
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    v_offset = 0
    for shape in shapes:
        result = _shape_to_local_mesh(shape)
        if result is None:
            continue
        v, f = result
        all_verts.append(v)
        if f is not None:
            all_faces.append(f + v_offset)
        v_offset += v.shape[0]

    if not all_verts:
        return LinkSpheres(link_index, link_name, np.zeros((0, 3), np.float32),
                           np.zeros((0,), np.float32))

    V = np.vstack(all_verts).astype(np.float32)
    F = np.vstack(all_faces).astype(np.int64) if all_faces else None
    centers, radii = fit_spheres_voxel_volume(V, F, pitch=pitch,
                                              max_spheres=max_spheres_per_link)
    return LinkSpheres(link_index, link_name, centers, radii)


# ── 3. 全机器人球链拟合 + JSON 缓存 ──────────────────


def fit_robot_spheres(
    body_id: int,
    monitored_links: list[int],
    pitch: float = 0.025,
    max_spheres_per_link: int = 80,
) -> dict[int, LinkSpheres]:
    """对所有受监控的 link 拟合球链。"""
    out: dict[int, LinkSpheres] = {}
    for li in monitored_links:
        info = p.getJointInfo(body_id, li) if li >= 0 else None
        link_name = info[12].decode() if info is not None else f"link_{li}"
        out[int(li)] = fit_link_spheres(body_id, int(li), link_name,
                                        pitch=pitch,
                                        max_spheres_per_link=max_spheres_per_link)
    return out


def save_link_spheres(spheres: dict[int, LinkSpheres], path: str | Path):
    p_ = Path(path)
    p_.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): v.to_dict() for k, v in spheres.items()}
    with open(p_, "w") as f:
        json.dump(payload, f, indent=2)


def load_link_spheres(path: str | Path) -> dict[int, LinkSpheres]:
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): LinkSpheres.from_dict(v) for k, v in raw.items()}


def summarize(spheres: dict[int, LinkSpheres]) -> str:
    lines = ["LinkSpheres summary:"]
    total = 0
    for li, s in spheres.items():
        total += s.n
        lines.append(f"  link {li:>3} ({s.link_name:<30}) → {s.n:>3} spheres, "
                     f"r={s.radii.mean() if s.n else 0:.4f}m")
    lines.append(f"  total: {total} spheres, {len(spheres)} links")
    return "\n".join(lines)
