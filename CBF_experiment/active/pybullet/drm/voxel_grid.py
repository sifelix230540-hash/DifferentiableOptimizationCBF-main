"""体素网格：AABB 空间离散化、坐标转换、碰撞 mesh 顶点加载。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p


class VoxelGrid:
    """以 robobase 为中心的 axis-aligned 体素网格。"""

    def __init__(self, origin: np.ndarray, half_extents: np.ndarray, resolution: float):
        self.origin = np.asarray(origin, dtype=float)
        self.half = np.asarray(half_extents, dtype=float)
        self.res = float(resolution)
        self.lo = self.origin - self.half
        self.hi = self.origin + self.half
        self.shape = np.ceil((self.hi - self.lo) / self.res).astype(int)
        self.nx, self.ny, self.nz = int(self.shape[0]), int(self.shape[1]), int(self.shape[2])

    def world_to_ijk(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) world → (N,3) int voxel indices, 不做越界裁剪。"""
        return np.floor((pts - self.lo) / self.res).astype(np.int32)

    def ijk_to_key(self, ijk: np.ndarray) -> np.ndarray:
        """(N,3) ijk → (N,) int64 key。越界 key 为 -1。"""
        valid = np.all((ijk >= 0) & (ijk < self.shape), axis=1)
        keys = ijk[:, 0].astype(np.int64) * self.ny * self.nz + ijk[:, 1] * self.nz + ijk[:, 2]
        keys[~valid] = -1
        return keys

    def world_to_keys(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) world → (N,) voxel key，一步到位。"""
        return self.ijk_to_key(self.world_to_ijk(pts))

    def key_to_center(self, key: int) -> np.ndarray:
        iz = key % self.nz
        iy = (key // self.nz) % self.ny
        ix = key // (self.ny * self.nz)
        return self.lo + (np.array([ix, iy, iz]) + 0.5) * self.res

    def total_voxels(self) -> int:
        return self.nx * self.ny * self.nz

    def __repr__(self):
        return (f"VoxelGrid(shape={self.shape.tolist()}, res={self.res:.3f}m, "
                f"total={self.total_voxels()}, lo={self.lo.tolist()}, hi={self.hi.tolist()})")


# ── Mesh 顶点加载 ─────────────────────────────────────


def load_link_collision_vertices(body_id: int, link_indices: list[int]) -> dict[int, np.ndarray]:
    """从 PyBullet 的碰撞几何中提取每个连杆的局部坐标顶点。

    返回 dict[link_index → vertices_local (M, 3)]。
    对 GEOM_MESH 类型用 trimesh 加载 .stl；对简单几何体则生成采样点。
    """
    result: dict[int, np.ndarray] = {}
    for li in link_indices:
        shapes = p.getCollisionShapeData(body_id, li)
        all_verts: list[np.ndarray] = []
        for shape in shapes:
            geom_type = int(shape[2])
            dims = shape[3]
            local_pos = np.array(shape[5], dtype=float)
            local_quat = np.array(shape[6], dtype=float)
            R = np.array(p.getMatrixFromQuaternion(local_quat), dtype=float).reshape(3, 3)
            mesh_file = shape[4]
            if isinstance(mesh_file, bytes):
                mesh_file = mesh_file.decode("utf-8", errors="replace").strip()

            verts_local = None
            if geom_type == p.GEOM_MESH and mesh_file:
                mesh_path = Path(mesh_file)
                if mesh_path.exists():
                    try:
                        import trimesh
                        mesh = trimesh.load(str(mesh_path), force="mesh")
                        scale = np.array(dims[:3], dtype=float) if len(dims) >= 3 else np.ones(3)
                        verts_local = np.asarray(mesh.vertices, dtype=float) * scale
                    except Exception:
                        pass
            elif geom_type == p.GEOM_BOX:
                hx, hy, hz = float(dims[0]) / 2, float(dims[1]) / 2, float(dims[2]) / 2
                corners = np.array([[s * hx, t * hy, u * hz]
                                    for s in (-1, 1) for t in (-1, 1) for u in (-1, 1)])
                verts_local = corners
            elif geom_type == p.GEOM_SPHERE:
                r = float(dims[0])
                n = 20
                phi = np.linspace(0, np.pi, n)
                theta = np.linspace(0, 2 * np.pi, n * 2)
                pp, tt = np.meshgrid(phi, theta)
                verts_local = np.column_stack([
                    r * np.sin(pp.ravel()) * np.cos(tt.ravel()),
                    r * np.sin(pp.ravel()) * np.sin(tt.ravel()),
                    r * np.cos(pp.ravel()),
                ])
            elif geom_type == p.GEOM_CYLINDER:
                r, length = float(dims[1]), float(dims[0])
                angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
                ring = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
                z_vals = np.linspace(-length / 2, length / 2, 8)
                pts = []
                for z in z_vals:
                    pts.append(np.column_stack([ring, np.full(len(angles), z)]))
                verts_local = np.vstack(pts)

            if verts_local is not None and len(verts_local) > 0:
                transformed = (R @ verts_local.T).T + local_pos
                all_verts.append(transformed)

        if all_verts:
            result[li] = np.vstack(all_verts).astype(float)
    return result


def compute_node_occupied_voxels(
    robot,
    q_full: np.ndarray,
    link_meshes: dict[int, np.ndarray],
    voxel_grid: VoxelGrid,
) -> set[int]:
    """给定一个完整关节配置，计算该配置下机器人占据的体素 key 集合。"""
    robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
    occupied: set[int] = set()
    for li, verts_local in link_meshes.items():
        state = p.getLinkState(robot.body_id, li, computeForwardKinematics=True)
        pos = np.array(state[4], dtype=float)
        R = np.array(p.getMatrixFromQuaternion(state[5]), dtype=float).reshape(3, 3)
        verts_world = (R @ verts_local.T).T + pos
        keys = voxel_grid.world_to_keys(verts_world)
        occupied.update(k for k in keys.tolist() if k >= 0)
    return occupied
