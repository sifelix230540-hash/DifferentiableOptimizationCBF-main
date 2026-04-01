"""Global UDF/SDF bake helpers: voxel layout, distance field I/O, trilinear query, URDF assembly."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_MAX_MEMORY_MB = 1024.0
DEFAULT_MAX_POINTS_PER_BATCH = 4096
DEFAULT_SCENE_URDF_PATH = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "cad_exports"
    / "model_CAD"
    / "scene"
    / "urdf"
    / "中组立0725(1).stp.SLDASM.urdf"
)
DEFAULT_URDF_ARG = (
    str(DEFAULT_SCENE_URDF_PATH) if DEFAULT_SCENE_URDF_PATH.is_file() else None
)
DEFAULT_OUTPUT_NPZ = str(
    DEFAULT_SCENE_URDF_PATH.parent / (DEFAULT_SCENE_URDF_PATH.stem + "_udf.npz")
)
DEFAULT_ARTIFACT_DIR = str(
    DEFAULT_SCENE_URDF_PATH.parent / (DEFAULT_SCENE_URDF_PATH.stem + "_udf_plots")
)

# =============================================================================
#  ★  USER CONFIG — 修改这里即可；CLI 参数仍可覆盖任意设置  ★
# =============================================================================

# ── 输入 / 输出 ─────────────────────────────────────────────────────────────
# 已烘焙好的 .npz 路径。设为此值可直接跳过重新烘焙，只做渲染。
# 若要从 URDF 重新烘焙，请将此项改为 None。
LOAD_NPZ: str | None = DEFAULT_OUTPUT_NPZ       # ← 直接加载上次结果
#LOAD_NPZ: str | None = None

# 当 LOAD_NPZ = None 时从此 URDF 烘焙（有默认场景时自动填入）。
URDF_PATH: str | None = DEFAULT_URDF_ARG

# 烘焙结果的保存路径。设为 None 则不保存。
OUTPUT_NPZ: str | None = DEFAULT_OUTPUT_NPZ

# 所有渲染产物（PNG / HTML）的输出目录。
ARTIFACT_DIR: str | None = DEFAULT_ARTIFACT_DIR

# ── 2-D 渲染（切片热图、散点对比、符号诊断）──────────────────────────────
RENDER_2D: bool = True          # True = 同时生成 2-D PNG 套件

# ── 3-D 渲染（等值面 PNG + 交互式 HTML）─────────────────────────────────────
RENDER_3D: bool = True           # ← 默认开启，直接从 LOAD_NPZ 渲染

# 等值面提取的 UDF 阈值（单位：米）。
# 越小越贴近网格面；越大等于给模型加一层"膨胀壳"。
RENDER_3D_THRESHOLD: float = 0.03

# 运行 marching cubes 前将 UDF 网格最长轴降采样到该值以内。
# 越大细节越好，但渲染速度与 HTML 文件体积随之增加。
RENDER_3D_MAX_SIDE: int = 128

# ── 3-D 体积热图（整个空间颜色 = UDF 距离）──────────────────────────────────
RENDER_3D_HEATMAP: bool = True       # 生成体积热图 HTML（需要 plotly）
HEATMAP_MAX_SIDE: int = 64           # 降采样上限（建议 48~80）
HEATMAP_DIST_MIN: float = 0.0        # 显示距离下限（米）
HEATMAP_DIST_MAX: float = 1.3        # 显示距离上限（米，超出部分透明）
HEATMAP_SURFACE_COUNT: int = 20      # 渲染等值面层数
HEATMAP_OPACITY: float = 0.15        # 体积不透明度（越小越透明）
HEATMAP_OVERLAY_ISOSURFACE: bool = True  # 叠加等值面轮廓（需要 scikit-image）

# ── 烘焙参数（仅 LOAD_NPZ = None 时生效）────────────────────────────────────
BAKE_SPACING: float = 0.02               # 体素边长（米）
BAKE_MARGIN: float = 0.0                 # bbox 外扩边距（米）
BAKE_UDF_ONLY: bool = False              # True = 跳过 libigl / Open3D SDF
BAKE_MAX_MEMORY_MB: float = DEFAULT_MAX_MEMORY_MB
BAKE_MAX_POINTS_PER_BATCH: int = DEFAULT_MAX_POINTS_PER_BATCH

# =============================================================================


class DistanceFieldQueryOutOfBoundsError(ValueError):
    """Raised when ``clip=False`` and the query point lies outside the valid domain."""


class GridMemoryBudgetError(MemoryError):
    """Raised when estimated voxel grid storage exceeds ``max_memory_bytes``."""


def compute_voxel_centers(
    origin: np.ndarray, spacing: float, shape: tuple[int, int, int]
) -> np.ndarray:
    """Voxel center positions: origin + (i + 0.5) * spacing per axis.

    Returns array of shape (nx, ny, nz, 3) with last axis (x, y, z).
    """
    nx, ny, nz = shape
    grid = np.indices((nx, ny, nz), dtype=np.float32)
    centers = origin.reshape(3, 1, 1, 1) + (grid + 0.5) * float(spacing)
    return np.moveaxis(centers, 0, -1)


def _grid_for_kind(df: "DistanceField", kind: str) -> np.ndarray:
    if kind == "udf":
        return df.udf_grid
    if kind == "igl_sdf":
        return df.igl_sdf_grid
    if kind == "o3d_sdf":
        return df.o3d_sdf_grid
    raise ValueError(
        f"unsupported distance field kind {kind!r}; expected "
        "'udf', 'igl_sdf', or 'o3d_sdf'"
    )


def _axis_i0_i1_t(
    coord_axis: float, n: int, *, clip_indices: bool
) -> tuple[int, int, np.float32]:
    """Integer voxel indices and local parameter for one trilinear axis."""
    if n < 1:
        raise ValueError("distance field grid axis has length < 1")
    if n == 1:
        return 0, 0, np.float32(0.0)

    i0 = int(np.floor(float(coord_axis)))
    if clip_indices:
        i0 = int(np.clip(i0, 0, n - 2))
    elif i0 < 0 or i0 > n - 2:
        raise DistanceFieldQueryOutOfBoundsError(
            "query maps outside grid index range along an axis (clip=False); "
            f"axis_len={n}, grid_coord={coord_axis!r}"
        )
    i1 = i0 + 1
    t = np.float32(coord_axis - i0)
    return i0, i1, t


def _trilinear_sample(
    grid: np.ndarray,
    pos: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    *,
    clip: bool,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.float32:
    p = np.asarray(pos, dtype=np.float32).reshape(3).copy()
    bmin = np.asarray(bbox_min, dtype=np.float32).reshape(3)
    bmax = np.asarray(bbox_max, dtype=np.float32).reshape(3)

    if not clip:
        if np.any(p < bmin) or np.any(p > bmax):
            raise DistanceFieldQueryOutOfBoundsError(
                "query point outside distance field bbox (clip=False); "
                f"bbox_min={bmin.tolist()} bbox_max={bmax.tolist()} point={p.tolist()}"
            )
    else:
        p = np.clip(p, bmin, bmax)

    org = np.asarray(origin, dtype=np.float32).reshape(3)
    s = np.float32(spacing)
    coord = (p - org) / s - np.float32(0.5)

    nx, ny, nz = grid.shape
    clip_idx = clip
    ix0, ix1, tx = _axis_i0_i1_t(float(coord[0]), nx, clip_indices=clip_idx)
    iy0, iy1, ty = _axis_i0_i1_t(float(coord[1]), ny, clip_indices=clip_idx)
    iz0, iz1, tz = _axis_i0_i1_t(float(coord[2]), nz, clip_indices=clip_idx)

    c000 = grid[ix0, iy0, iz0]
    c001 = grid[ix0, iy0, iz1]
    c010 = grid[ix0, iy1, iz0]
    c011 = grid[ix0, iy1, iz1]
    c100 = grid[ix1, iy0, iz0]
    c101 = grid[ix1, iy0, iz1]
    c110 = grid[ix1, iy1, iz0]
    c111 = grid[ix1, iy1, iz1]

    corners = (c000, c001, c010, c011, c100, c101, c110, c111)
    if any(np.isnan(np.float32(c)) for c in corners):
        return np.float32(np.nan)

    c00 = c000 * (1.0 - tz) + c001 * tz
    c01 = c010 * (1.0 - tz) + c011 * tz
    c10 = c100 * (1.0 - tz) + c101 * tz
    c11 = c110 * (1.0 - tz) + c111 * tz
    c0 = c00 * (1.0 - ty) + c01 * ty
    c1 = c10 * (1.0 - ty) + c11 * ty
    return np.float32(c0 * (1.0 - tx) + c1 * tx)


def _trilinear_sample_with_gradient(
    grid: np.ndarray,
    pos: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    *,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> tuple[np.float32, np.ndarray]:
    """Trilinear interpolation with analytical spatial gradient.

    For **in-bounds** points the function returns the exact trilinear value and
    its closed-form gradient  ``∂d/∂(x, y, z)``.

    For **out-of-bounds** points the nearest surface point is estimated from
    the boundary value and gradient, then the distance and direction from
    that estimate to the query point are returned:

        s ≈ p_c − d(p_c) · ∇d / |∇d|²     (nearest surface estimate)
        d(p) = |p − s|
        ∇d(p) = (p − s) / |p − s|

    When the boundary gradient is degenerate (|∇d| ≈ 0, e.g. the mesh
    surface coincides with the bbox edge), the extrapolated distance falls
    back to ``|p − p_c|`` with gradient pointing outward from the bbox.

    Returns
    -------
    value : np.float32
        Scalar distance (or its extrapolation).
    gradient : np.ndarray, shape (3,), dtype float32
        Spatial gradient  ∂d/∂x, ∂d/∂y, ∂d/∂z  in world coordinates.
    """
    p = np.asarray(pos, dtype=np.float32).reshape(3).copy()
    bmin = np.asarray(bbox_min, dtype=np.float32).reshape(3)
    bmax = np.asarray(bbox_max, dtype=np.float32).reshape(3)
    org = np.asarray(origin, dtype=np.float32).reshape(3)
    s = float(spacing)

    p_clamped = np.clip(p, bmin, bmax)
    d_out_vec = p - p_clamped
    d_out = float(np.linalg.norm(d_out_vec))
    outside = d_out > 0.0

    coord = (p_clamped - org) / s - np.float32(0.5)

    nx, ny, nz = grid.shape
    ix0, ix1, tx = _axis_i0_i1_t(float(coord[0]), nx, clip_indices=True)
    iy0, iy1, ty = _axis_i0_i1_t(float(coord[1]), ny, clip_indices=True)
    iz0, iz1, tz = _axis_i0_i1_t(float(coord[2]), nz, clip_indices=True)

    c000 = float(grid[ix0, iy0, iz0])
    c001 = float(grid[ix0, iy0, iz1])
    c010 = float(grid[ix0, iy1, iz0])
    c011 = float(grid[ix0, iy1, iz1])
    c100 = float(grid[ix1, iy0, iz0])
    c101 = float(grid[ix1, iy0, iz1])
    c110 = float(grid[ix1, iy1, iz0])
    c111 = float(grid[ix1, iy1, iz1])

    if any(math.isnan(c) for c in (c000, c001, c010, c011, c100, c101, c110, c111)):
        return np.float32(np.nan), np.full(3, np.nan, dtype=np.float32)

    # --- trilinear value ---
    c00 = c000 * (1.0 - tz) + c001 * tz
    c01 = c010 * (1.0 - tz) + c011 * tz
    c10 = c100 * (1.0 - tz) + c101 * tz
    c11 = c110 * (1.0 - tz) + c111 * tz
    c0 = c00 * (1.0 - ty) + c01 * ty
    c1 = c10 * (1.0 - ty) + c11 * ty
    value = float(c0 * (1.0 - tx) + c1 * tx)

    # --- analytical gradient in grid-index coords ---
    dfdtx = (
        (c100 - c000) * (1.0 - ty) * (1.0 - tz)
        + (c101 - c001) * (1.0 - ty) * tz
        + (c110 - c010) * ty * (1.0 - tz)
        + (c111 - c011) * ty * tz
    )
    dfdty = (
        (c010 - c000) * (1.0 - tx) * (1.0 - tz)
        + (c011 - c001) * (1.0 - tx) * tz
        + (c110 - c100) * tx * (1.0 - tz)
        + (c111 - c101) * tx * tz
    )
    dfdtz = (
        (c001 - c000) * (1.0 - tx) * (1.0 - ty)
        + (c011 - c010) * (1.0 - tx) * ty
        + (c101 - c100) * tx * (1.0 - ty)
        + (c111 - c110) * tx * ty
    )

    inv_s = 1.0 / s
    gradient = np.array(
        [dfdtx * inv_s, dfdty * inv_s, dfdtz * inv_s], dtype=np.float32
    )

    if outside:
        grad_norm_sq = float(np.dot(gradient, gradient))
        if grad_norm_sq > 0.25:
            surf_est = p_clamped - np.float32(value / grad_norm_sq) * gradient
            diff = p - surf_est
            new_dist = float(np.linalg.norm(diff))
            if new_dist > 1e-12:
                value = new_dist
                gradient = (diff / new_dist).astype(np.float32)
            else:
                value = 0.0
        else:
            value = d_out
            if d_out > 1e-12:
                gradient = (d_out_vec / np.float32(d_out)).astype(np.float32)
            else:
                gradient = np.zeros(3, dtype=np.float32)

    return np.float32(value), gradient


@dataclass
class DistanceField:
    origin: np.ndarray
    spacing: np.float32
    udf_grid: np.ndarray
    igl_sdf_grid: np.ndarray
    o3d_sdf_grid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    status_flags: dict | None = None
    failure_reasons: list[str] | None = field(default_factory=list)
    build_config: dict | None = None

    def query_single(
        self,
        position: np.ndarray,
        kind: str = "udf",
        clip: bool = False,
    ) -> np.float32:
        """Trilinear sample at ``position``; see :meth:`query`."""
        grid = _grid_for_kind(self, kind)
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        return _trilinear_sample(
            grid,
            pos,
            self.origin,
            float(self.spacing),
            clip=clip,
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
        )

    def query(
        self,
        points: np.ndarray,
        kind: str = "udf",
        clip: bool = False,
    ) -> np.floating | np.ndarray:
        """Query distance at ``points`` (shape ``(3,)`` or ``(N, 3)``).

        Returns a 0-d float32 array for a single point and shape ``(N,)`` float32
        for batches. Unsupported ``kind`` raises ``ValueError``.
        """
        pts = np.asarray(points, dtype=np.float32)
        if pts.shape == (3,):
            return self.query_single(pts, kind=kind, clip=clip)
        if pts.ndim == 2 and pts.shape[1] == 3:
            out = np.empty((pts.shape[0],), dtype=np.float32)
            for i in range(pts.shape[0]):
                out[i] = self.query_single(pts[i], kind=kind, clip=clip)
            return out
        raise ValueError(
            f"points must have shape (3,) or (N, 3); got shape {pts.shape}"
        )

    # ----- distance + gradient (for CBF-QP) -----

    def query_single_with_gradient(
        self,
        position: np.ndarray,
        kind: str = "udf",
    ) -> tuple[np.float32, np.ndarray]:
        """Distance and spatial gradient at one point.

        Always succeeds: out-of-bounds points are handled by linear
        extrapolation from the nearest grid boundary (no exception, no need
        to enlarge the bounding box).

        Returns
        -------
        distance : np.float32
            Scalar distance (extrapolated if outside domain).
        gradient : np.ndarray, shape (3,), dtype float32
            ∂d/∂x, ∂d/∂y, ∂d/∂z.  ``-gradient / |gradient|`` points
            toward the nearest obstacle surface (the repulsion direction
            for CBF-QP).
        """
        grid = _grid_for_kind(self, kind)
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        return _trilinear_sample_with_gradient(
            grid,
            pos,
            self.origin,
            float(self.spacing),
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
        )

    def query_with_gradient(
        self,
        points: np.ndarray,
        kind: str = "udf",
    ) -> tuple[np.floating | np.ndarray, np.ndarray]:
        """Distance and gradient for one or many points.

        Parameters
        ----------
        points : (3,) or (N, 3) array
        kind : ``'udf'``, ``'igl_sdf'``, or ``'o3d_sdf'``

        Returns
        -------
        distances : float32 scalar or (N,) float32
        gradients : (3,) or (N, 3) float32
            Spatial gradient.  For CBF-QP the repulsion direction is
            ``-gradients / |gradients|`` and the nearest obstacle point
            estimate is ``points - distances * gradients / |gradients|``.
        """
        pts = np.asarray(points, dtype=np.float32)
        if pts.shape == (3,):
            return self.query_single_with_gradient(pts, kind=kind)
        if pts.ndim == 2 and pts.shape[1] == 3:
            n = pts.shape[0]
            dists = np.empty(n, dtype=np.float32)
            grads = np.empty((n, 3), dtype=np.float32)
            for i in range(n):
                dists[i], grads[i] = self.query_single_with_gradient(
                    pts[i], kind=kind
                )
            return dists, grads
        raise ValueError(
            f"points must have shape (3,) or (N, 3); got shape {pts.shape}"
        )


def _validate_grid_shapes(
    udf_grid: np.ndarray,
    igl_sdf_grid: np.ndarray,
    o3d_sdf_grid: np.ndarray,
    expected_shape: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    udf_shape = tuple(int(x) for x in np.asarray(udf_grid).shape)
    igl_shape = tuple(int(x) for x in np.asarray(igl_sdf_grid).shape)
    o3d_shape = tuple(int(x) for x in np.asarray(o3d_sdf_grid).shape)

    if udf_shape != igl_shape or udf_shape != o3d_shape:
        raise ValueError(
            "distance field grids must share the same shape; "
            f"udf_grid={udf_shape}, igl_sdf_grid={igl_shape}, o3d_sdf_grid={o3d_shape}"
        )
    if expected_shape is not None and udf_shape != expected_shape:
        raise ValueError(
            f"npz 'shape' {expected_shape} does not match grid shape {udf_shape}"
        )
    return udf_shape


def _decode_json_field(data: np.lib.npyio.NpzFile, key: str) -> object | None:
    try:
        raw = str(np.asarray(data[key]).item())
        if raw == "":
            return None
        return json.loads(raw)
    except KeyError as exc:
        raise ValueError(f"missing required npz field {key!r}") from exc
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON payload for field {key!r}") from exc


def _validate_json_container(
    value: object | None,
    key: str,
    expected_type: type[dict] | type[list],
) -> dict | list | None:
    if value is None:
        return None
    if isinstance(value, expected_type):
        return value
    raise ValueError(
        f"field {key!r} must decode to {expected_type.__name__} or None, "
        f"got {type(value).__name__}"
    )


def _validate_string_list(value: object | None, key: str) -> list[str] | None:
    items = _validate_json_container(value, key, list)
    if items is None:
        return None
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise ValueError(
                f"field {key!r} must contain only str items; "
                f"index {i} has {type(item).__name__}"
            )
    return items


def _get_required_npz_field(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    try:
        return data[key]
    except KeyError as exc:
        raise ValueError(f"missing required npz field {key!r}") from exc


def save_distance_field(path: str | Path, field: DistanceField) -> None:
    """Serialize ``field`` to a compressed ``.npz`` (required bake metadata + grids)."""
    path = Path(path)
    grid_shape = _validate_grid_shapes(
        field.udf_grid,
        field.igl_sdf_grid,
        field.o3d_sdf_grid,
    )
    failure_reasons = _validate_string_list(field.failure_reasons, "failure_reasons")
    shape_arr = np.asarray(grid_shape, dtype=np.int64)
    np.savez_compressed(
        path,
        origin=np.asarray(field.origin, dtype=np.float32),
        spacing=np.float32(field.spacing),
        shape=shape_arr,
        udf_grid=np.asarray(field.udf_grid, dtype=np.float32),
        igl_sdf_grid=np.asarray(field.igl_sdf_grid, dtype=np.float32),
        o3d_sdf_grid=np.asarray(field.o3d_sdf_grid, dtype=np.float32),
        bbox_min=np.asarray(field.bbox_min, dtype=np.float32),
        bbox_max=np.asarray(field.bbox_max, dtype=np.float32),
        status_flags=np.array(
            ""
            if field.status_flags is None
            else json.dumps(field.status_flags, ensure_ascii=False),
            dtype=np.str_,
        ),
        failure_reasons=np.array(
            ""
            if failure_reasons is None
            else json.dumps(failure_reasons, ensure_ascii=False),
            dtype=np.str_,
        ),
        build_config=np.array(
            ""
            if field.build_config is None
            else json.dumps(field.build_config, ensure_ascii=False),
            dtype=np.str_,
        ),
    )


def load_distance_field(path: str | Path) -> DistanceField:
    """Load :class:`DistanceField` written by :func:`save_distance_field`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        shape_tuple = tuple(
            int(x) for x in np.asarray(_get_required_npz_field(data, "shape")).reshape(-1)
        )
        udf_grid = np.asarray(_get_required_npz_field(data, "udf_grid"), dtype=np.float32)
        igl_sdf_grid = np.asarray(
            _get_required_npz_field(data, "igl_sdf_grid"), dtype=np.float32
        )
        o3d_sdf_grid = np.asarray(
            _get_required_npz_field(data, "o3d_sdf_grid"), dtype=np.float32
        )
        _validate_grid_shapes(
            udf_grid,
            igl_sdf_grid,
            o3d_sdf_grid,
            expected_shape=shape_tuple,
        )

        status_flags = _validate_json_container(
            _decode_json_field(data, "status_flags"),
            "status_flags",
            dict,
        )
        failure_reasons = _validate_string_list(
            _decode_json_field(data, "failure_reasons"),
            "failure_reasons",
        )
        build_config = _validate_json_container(
            _decode_json_field(data, "build_config"),
            "build_config",
            dict,
        )

        return DistanceField(
            origin=np.asarray(_get_required_npz_field(data, "origin"), dtype=np.float32),
            spacing=np.float32(
                np.asarray(_get_required_npz_field(data, "spacing")).reshape(())
            ),
            udf_grid=udf_grid,
            igl_sdf_grid=igl_sdf_grid,
            o3d_sdf_grid=o3d_sdf_grid,
            bbox_min=np.asarray(_get_required_npz_field(data, "bbox_min"), dtype=np.float32),
            bbox_max=np.asarray(_get_required_npz_field(data, "bbox_max"), dtype=np.float32),
            status_flags=status_flags,
            failure_reasons=failure_reasons,
            build_config=build_config,
        )


# ---------------------------------------------------------------------------
# Global UDF grid bake (unsigned point–triangle distance, no PyBullet)
# ---------------------------------------------------------------------------


def estimate_grid_memory(
    shape: tuple[int, int, int],
    num_enabled_grids: int,
    dtype: type[np.floating] | np.dtype = np.float32,
) -> int:
    """Return bytes needed for ``num_enabled_grids`` full 3D arrays of ``shape``."""
    if num_enabled_grids < 0:
        raise ValueError("num_enabled_grids must be non-negative")
    n = int(np.prod(shape, dtype=np.int64))
    return n * int(num_enabled_grids) * int(np.dtype(dtype).itemsize)


def compute_grid_domain(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    spacing: float,
    margin: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Expanded AABB grid: ``origin`` is ``bbox_min - margin``; size from ``bbox_max + margin``.

    ``bbox_min`` / ``bbox_max`` remain the *original* assembly AABB (no margin). Voxel centers follow
    ``origin + (i + 0.5) * spacing`` per :func:`compute_voxel_centers`.
    """
    bmin = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bmax = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    s = float(spacing)
    m = float(margin)
    if s <= 0:
        raise ValueError("spacing must be positive")
    if m < 0:
        raise ValueError("margin must be non-negative")
    exp_min = bmin - m
    exp_max = bmax + m
    extent = exp_max - exp_min
    shape = tuple(max(1, int(math.ceil(float(extent[i]) / s))) for i in range(3))
    origin = exp_min.astype(np.float32)
    return origin, shape


def _dist_sq_point_segment_batch(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Squared distance from points ``p`` (B,3) to segment ``ab``."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-30:
        return np.sum((p - a) ** 2, axis=1)
    ap = p - a
    t = np.einsum("ij,j->i", ap, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + np.outer(t, ab)
    return np.sum((p - proj) ** 2, axis=1)


def _point_triangle_dist_sq(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Squared unsigned distance from points ``p`` (B,3) to triangle ``tri`` (3,3)."""
    a = np.asarray(tri[0], dtype=np.float64)
    b = np.asarray(tri[1], dtype=np.float64)
    c = np.asarray(tri[2], dtype=np.float64)
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    nn = float(np.dot(normal, normal))
    if nn < 1e-30:
        d0 = _dist_sq_point_segment_batch(p, a, b)
        d1 = _dist_sq_point_segment_batch(p, b, c)
        d2 = _dist_sq_point_segment_batch(p, c, a)
        return np.minimum(np.minimum(d0, d1), d2)

    ap = p - a
    t = np.einsum("ij,j->i", ap, normal) / nn
    proj = p - np.outer(t, normal)

    v0 = ac
    v1 = ab
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot11 = float(np.dot(v1, v1))
    v2 = proj - a
    dot02 = np.einsum("bi,i->b", v2, v0)
    dot12 = np.einsum("bi,i->b", v2, v1)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-30:
        d0 = _dist_sq_point_segment_batch(p, a, b)
        d1 = _dist_sq_point_segment_batch(p, b, c)
        d2 = _dist_sq_point_segment_batch(p, c, a)
        return np.minimum(np.minimum(d0, d1), d2)

    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    eps = 1e-9
    inside = (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps)
    dist_face = (t * t) * nn
    d_ab = _dist_sq_point_segment_batch(p, a, b)
    d_bc = _dist_sq_point_segment_batch(p, b, c)
    d_ca = _dist_sq_point_segment_batch(p, c, a)
    d_edge = np.minimum(np.minimum(d_ab, d_bc), d_ca)
    return np.where(inside, dist_face, d_edge)


def _voxel_centers_from_flat_indices(
    origin: np.ndarray,
    spacing: float,
    shape: tuple[int, int, int],
    flat_idx: np.ndarray,
) -> np.ndarray:
    """World positions of voxel centers for C-order flat indices (shape ``(B, 3)``)."""
    _nx, ny, nz = shape
    idx = np.asarray(flat_idx, dtype=np.int64)
    iz = idx % nz
    rem = idx // nz
    iy = rem % ny
    ix = rem // ny
    org = np.asarray(origin, dtype=np.float64).reshape(1, 3)
    sp = float(spacing)
    pts = np.empty((idx.shape[0], 3), dtype=np.float64)
    pts[:, 0] = org[0, 0] + (ix + 0.5) * sp
    pts[:, 1] = org[0, 1] + (iy + 0.5) * sp
    pts[:, 2] = org[0, 2] + (iz + 0.5) * sp
    return pts


def build_nan_grid(shape: tuple[int, int, int], dtype=np.float32) -> np.ndarray:
    """Placeholder grid filled with ``NaN`` (libigl / Open3D SDF not computed yet)."""
    return np.full(shape, np.nan, dtype=dtype)


def _import_igl_module():
    """Import libigl Python bindings (``import igl``; PyPI package name is commonly ``libigl``)."""
    try:
        import igl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "libigl Python bindings are not installed (pip package often named 'libigl'; import as 'igl')"
        ) from exc
    return igl


def _import_open3d_module():
    try:
        import open3d as o3d  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("open3d is not installed") from exc
    return o3d


def _backend_unavailable_reason(name: str, exc: Exception) -> str:
    return f"{name}: backend unavailable: {type(exc).__name__}: {exc}"


def _igl_signed_distance_pseudonormal(
    igl, p: np.ndarray, v: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """Signed distance with pseudonormal signing when the installed binding exposes it."""
    p64 = np.asarray(p, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    fi = np.asarray(f, dtype=np.int32)
    pseudo = getattr(igl, "SIGNED_DISTANCE_TYPE_PSEUDONORMAL", None)
    if pseudo is not None:
        try:
            s, *_ = igl.signed_distance(
                p64, v64, fi, sign_type=pseudo, return_normals=False
            )
            return np.asarray(np.ravel(s), dtype=np.float64)
        except TypeError:
            try:
                s, *_ = igl.signed_distance(p64, v64, fi, pseudo)
                return np.asarray(np.ravel(s), dtype=np.float64)
            except TypeError:
                pass
    s, *_ = igl.signed_distance(p64, v64, fi, return_normals=False)
    return np.asarray(np.ravel(s), dtype=np.float64)


def bake_igl_sdf_grid(
    triangles: np.ndarray,
    origin: np.ndarray,
    shape: tuple[int, int, int],
    spacing: float,
    *,
    max_memory_bytes: int | None = None,
    point_batch_size: int = DEFAULT_MAX_POINTS_PER_BATCH,
) -> tuple[np.ndarray, str | None]:
    """Voxel SDF via libigl :func:`igl.signed_distance` (pseudonormal when available).

    Returns ``(grid, failure_reason)``. On success ``failure_reason`` is ``None``.
    Missing ``igl`` or runtime failures yield an all-``NaN`` grid and a non-empty reason string
    without raising.
    """
    if point_batch_size < 1:
        raise ValueError("point_batch_size must be >= 1")
    tris = np.asarray(triangles, dtype=np.float64)
    if tris.ndim != 3 or tris.shape[-2:] != (3, 3):
        raise ValueError(f"triangles must have shape (N, 3, 3); got {tris.shape}")
    if tris.shape[0] == 0:
        raise ValueError("triangles must be non-empty")

    need = estimate_grid_memory(shape, 1, np.float32)
    if max_memory_bytes is not None and need > max_memory_bytes:
        return build_nan_grid(shape), (
            f"igl_sdf: estimated output grid memory {need} bytes exceeds budget "
            f"{max_memory_bytes}"
        )

    try:
        igl = _import_igl_module()
    except Exception as exc:
        return build_nan_grid(shape), _backend_unavailable_reason("igl_sdf", exc)

    n = int(tris.shape[0])
    v = tris.reshape(-1, 3)
    f_idx = np.arange(3 * n, dtype=np.int32).reshape(n, 3)

    out = np.empty(shape, dtype=np.float32)
    out_flat = out.reshape(-1)
    nx, ny, nz = shape
    flat_n = int(nx * ny * nz)
    bs = int(point_batch_size)

    for start in range(0, flat_n, bs):
        end = min(start + bs, flat_n)
        idx = np.arange(start, end, dtype=np.int64)
        pts = _voxel_centers_from_flat_indices(origin, spacing, shape, idx)
        try:
            s = _igl_signed_distance_pseudonormal(igl, pts, v, f_idx)
        except Exception as exc:
            return build_nan_grid(shape), f"igl_sdf: signed_distance failed: {exc}"
        if int(s.shape[0]) != end - start:
            return build_nan_grid(shape), "igl_sdf: unexpected distance vector length"
        out_flat[start:end] = s.astype(np.float32, copy=False)

    return out, None


def bake_open3d_sdf_grid(
    triangles: np.ndarray,
    origin: np.ndarray,
    shape: tuple[int, int, int],
    spacing: float,
    *,
    max_memory_bytes: int | None = None,
    point_batch_size: int = DEFAULT_MAX_POINTS_PER_BATCH,
) -> tuple[np.ndarray, str | None]:
    """Voxel SDF via Open3D :meth:`open3d.t.geometry.RaycastingScene.compute_signed_distance`.

    Returns ``(grid, failure_reason)``. On success ``failure_reason`` is ``None``.
    Missing ``open3d`` or backend failures yield an all-``NaN`` grid and a reason string
    without raising.
    """
    if point_batch_size < 1:
        raise ValueError("point_batch_size must be >= 1")
    tris = np.asarray(triangles, dtype=np.float64)
    if tris.ndim != 3 or tris.shape[-2:] != (3, 3):
        raise ValueError(f"triangles must have shape (N, 3, 3); got {tris.shape}")
    if tris.shape[0] == 0:
        raise ValueError("triangles must be non-empty")

    need = estimate_grid_memory(shape, 1, np.float32)
    if max_memory_bytes is not None and need > max_memory_bytes:
        return build_nan_grid(shape), (
            f"open3d_sdf: estimated output grid memory {need} bytes exceeds budget "
            f"{max_memory_bytes}"
        )

    try:
        o3d = _import_open3d_module()
    except Exception as exc:
        return build_nan_grid(shape), _backend_unavailable_reason("open3d_sdf", exc)

    n = int(tris.shape[0])
    v_np = tris.reshape(-1, 3).astype(np.float32, copy=False)
    f_np = np.arange(3 * n, dtype=np.uint32).reshape(n, 3)

    try:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(v_np), o3d.core.Tensor(f_np))
    except Exception as exc:
        return build_nan_grid(shape), f"open3d_sdf: RaycastingScene setup failed: {exc}"

    out = np.empty(shape, dtype=np.float32)
    out_flat = out.reshape(-1)
    nx, ny, nz = shape
    flat_n = int(nx * ny * nz)
    bs = int(point_batch_size)

    for start in range(0, flat_n, bs):
        end = min(start + bs, flat_n)
        idx = np.arange(start, end, dtype=np.int64)
        pts = _voxel_centers_from_flat_indices(origin, spacing, shape, idx).astype(
            np.float32, copy=False
        )
        try:
            q = o3d.core.Tensor(pts)
            d = scene.compute_signed_distance(q)
            d_np = np.asarray(d.numpy(), dtype=np.float32).reshape(-1)
        except Exception as exc:
            return build_nan_grid(shape), f"open3d_sdf: compute_signed_distance failed: {exc}"
        if int(d_np.shape[0]) != end - start:
            return build_nan_grid(shape), "open3d_sdf: unexpected distance tensor shape"
        out_flat[start:end] = d_np

    return out, None


# ---------------------------------------------------------------------------
# GPU-accelerated UDF helpers (PyTorch CUDA)
# ---------------------------------------------------------------------------

def _seg_dist_sq_torch(
    p: "torch.Tensor",  # (B, T, 3)
    a: "torch.Tensor",  # (1, T, 3)
    b: "torch.Tensor",  # (1, T, 3)
) -> "torch.Tensor":  # (B, T)
    """Squared distance from B points to T segments, fully vectorized."""
    import torch  # noqa: PLC0415
    ab = b - a                                  # (1, T, 3)
    denom = (ab * ab).sum(-1).clamp(min=1e-30)  # (1, T)
    ap = p - a                                  # (B, T, 3)
    t = (ap * ab).sum(-1) / denom              # (B, T)
    t = t.clamp(0.0, 1.0)
    proj = a + t.unsqueeze(-1) * ab            # (B, T, 3)
    return ((p - proj) ** 2).sum(-1)           # (B, T)


def _pt_tri_dist_sq_torch(
    p: "torch.Tensor",     # (B, 3)
    tris: "torch.Tensor",  # (T, 3, 3)
) -> "torch.Tensor":  # (B, T)
    """Unsigned squared distance from B query points to T triangles, no Python loops."""
    import torch  # noqa: PLC0415
    a = tris[:, 0].unsqueeze(0)   # (1, T, 3)
    b = tris[:, 1].unsqueeze(0)   # (1, T, 3)
    c = tris[:, 2].unsqueeze(0)   # (1, T, 3)
    p_ = p.unsqueeze(1)            # (B, 1, 3)

    ab = b - a                     # (1, T, 3)
    ac = c - a                     # (1, T, 3)
    ap = p_ - a                    # (B, T, 3)

    normal = torch.linalg.cross(
        ab.expand(p_.shape[0], -1, -1),
        ac.expand(p_.shape[0], -1, -1),
        dim=-1,
    )                              # (B, T, 3)
    nn = (normal * normal).sum(-1)  # (B, T)
    degenerate = nn < 1e-30

    t_face = (ap * normal).sum(-1) / nn.clamp(min=1e-30)  # (B, T)
    proj = p_ - t_face.unsqueeze(-1) * normal              # (B, T, 3)
    del normal

    v2 = proj - a                    # (B, T, 3)
    dot00 = (ac * ac).sum(-1)        # (1, T)
    dot01 = (ac * ab).sum(-1)        # (1, T)
    dot11 = (ab * ab).sum(-1)        # (1, T)
    dot02 = (v2 * ac.expand_as(v2)).sum(-1)  # (B, T)
    dot12 = (v2 * ab.expand_as(v2)).sum(-1)  # (B, T)
    del v2, proj

    denom = (dot00 * dot11 - dot01 * dot01).clamp(min=1e-30)  # (1, T)
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv   # (B, T)
    v = (dot00 * dot12 - dot01 * dot02) * inv   # (B, T)

    eps = 1e-9
    inside = (u >= -eps) & (v >= -eps) & ((u + v) <= 1.0 + eps) & ~degenerate

    dist_face = (t_face * t_face) * nn   # (B, T)
    del t_face, nn, u, v, degenerate

    d_ab = _seg_dist_sq_torch(p_ - 0, a, b)
    d_bc = _seg_dist_sq_torch(p_ - 0, b, c)
    d_ca = _seg_dist_sq_torch(p_ - 0, c, a)
    d_edge = torch.minimum(torch.minimum(d_ab, d_bc), d_ca)

    return torch.where(inside, dist_face, d_edge)


def _bake_udf_torch(
    tris: np.ndarray,       # (N, 3, 3) float64
    origin: np.ndarray,
    shape: tuple[int, int, int],
    spacing: float,
    *,
    point_batch_size: int = 50_000,
    tri_chunk_size: int = 2_000,
) -> np.ndarray:
    """GPU UDF bake via PyTorch CUDA.  Returns (nx, ny, nz) float32 array."""
    import torch

    device = torch.device("cuda")
    tris_gpu = torch.from_numpy(tris.astype(np.float32)).to(device)   # (N, 3, 3)
    N_tri = tris_gpu.shape[0]

    nx, ny, nz = shape
    flat_n = int(nx * ny * nz)
    udf_flat = np.empty(flat_n, dtype=np.float32)

    n_pt_batches = math.ceil(flat_n / point_batch_size)
    n_tri_chunks = math.ceil(N_tri / tri_chunk_size)
    print(
        f"[bake_udf_grid] torch CUDA  device={torch.cuda.get_device_name(0)}  "
        f"tris={N_tri}  voxels={flat_n}  "
        f"pt_batches={n_pt_batches}  tri_chunks={n_tri_chunks}  "
        f"pt_bs={point_batch_size}  tri_chunk={tri_chunk_size}",
        flush=True,
    )

    with torch.no_grad():
        for batch_i, start in enumerate(range(0, flat_n, point_batch_size)):
            end = min(start + point_batch_size, flat_n)
            idx = np.arange(start, end, dtype=np.int64)
            pts_np = _voxel_centers_from_flat_indices(origin, spacing, shape, idx).astype(
                np.float32
            )
            pts = torch.from_numpy(pts_np).to(device)  # (B, 3)
            d_sq_min = torch.full((end - start,), float("inf"), device=device)

            for tri_start in range(0, N_tri, tri_chunk_size):
                tri_end = min(tri_start + tri_chunk_size, N_tri)
                chunk = tris_gpu[tri_start:tri_end]           # (T, 3, 3)
                d_sq_chunk = _pt_tri_dist_sq_torch(pts, chunk)  # (B, T)
                d_sq_min = torch.minimum(d_sq_min, d_sq_chunk.min(dim=1).values)
                del d_sq_chunk

            udf_flat[start:end] = d_sq_min.sqrt().cpu().numpy()
            del pts, d_sq_min

            if batch_i % max(1, n_pt_batches // 20) == 0:
                print(
                    f"  {100 * end / flat_n:5.1f}%  pt_batch {batch_i + 1}/{n_pt_batches}",
                    flush=True,
                )

    return udf_flat.reshape(shape)

def bake_udf_grid(
    triangles: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    spacing: float,
    margin: float,
    *,
    max_memory_bytes: int | None = None,
    point_batch_size: int = DEFAULT_MAX_POINTS_PER_BATCH,
) -> np.ndarray:
    """Voxel UDF = min unsigned distance from each voxel center to all triangles.

    Priority: (1) PyTorch CUDA GPU, (2) trimesh BVH (CPU), (3) brute-force NumPy.

    Returns ``udf_grid`` only. Callers that also need ``origin`` / ``shape`` should
    derive them separately via :func:`compute_grid_domain`.
    """
    if point_batch_size < 1:
        raise ValueError("point_batch_size must be >= 1")
    origin, shape = compute_grid_domain(bbox_min, bbox_max, spacing, margin)
    need_udf = estimate_grid_memory(shape, 1, np.float32)
    if max_memory_bytes is not None and need_udf > max_memory_bytes:
        raise GridMemoryBudgetError(
            f"UDF grid would need ~{need_udf} bytes (limit {max_memory_bytes}); "
            f"shape={shape}, spacing={spacing}, margin={margin}"
        )

    tris = np.asarray(triangles, dtype=np.float64)
    if tris.ndim != 3 or tris.shape[-2:] != (3, 3):
        raise ValueError(f"triangles must have shape (N, 3, 3); got {tris.shape}")
    if tris.shape[0] == 0:
        raise ValueError("triangles must be non-empty")

    nx, ny, nz = shape
    flat_n = int(nx * ny * nz)
    udf_flat = np.empty(flat_n, dtype=np.float32)

    # ------------------------------------------------------------------
    # Fast path 1: PyTorch CUDA GPU
    # ------------------------------------------------------------------
    try:
        import torch
        if torch.cuda.is_available():
            return _bake_udf_torch(
                tris,
                origin,
                shape,
                spacing,
                point_batch_size=max(int(point_batch_size), 50_000),
                tri_chunk_size=2_000,
            )
        print("[bake_udf_grid] torch available but CUDA not found — trying trimesh BVH", flush=True)
    except ImportError:
        print("[bake_udf_grid] torch not available — trying trimesh BVH", flush=True)
    except Exception as exc:
        print(
            f"[bake_udf_grid] torch GPU path failed ({exc!r}) — trying trimesh BVH",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Fast path 2: trimesh BVH (CPU, O(N log N))
    # ------------------------------------------------------------------
    _trimesh_ok = False
    try:
        import trimesh
        import trimesh.proximity as _trx_prox

        verts = tris.reshape(-1, 3)
        faces = np.arange(len(verts), dtype=np.int32).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        bs = max(int(point_batch_size), 50_000)
        n_batches = math.ceil(flat_n / bs)
        print(
            f"[bake_udf_grid] trimesh BVH  tris={tris.shape[0]}  "
            f"voxels={flat_n}  batches={n_batches}  batch_size={bs}",
            flush=True,
        )
        for batch_i, start in enumerate(range(0, flat_n, bs)):
            end = min(start + bs, flat_n)
            idx = np.arange(start, end, dtype=np.int64)
            pts = _voxel_centers_from_flat_indices(origin, spacing, shape, idx).astype(
                np.float64
            )
            _, dists, _ = _trx_prox.closest_point(mesh, pts)
            udf_flat[start:end] = dists.astype(np.float32)
            if batch_i % max(1, n_batches // 20) == 0:
                print(
                    f"  {100 * end / flat_n:5.1f}%  batch {batch_i + 1}/{n_batches}",
                    flush=True,
                )
        _trimesh_ok = True
    except ImportError:
        print(
            "[bake_udf_grid] trimesh not available — falling back to brute-force NumPy "
            "(install trimesh for 100× speedup)",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[bake_udf_grid] trimesh fast path failed ({exc!r}) "
            "— falling back to brute-force NumPy",
            flush=True,
        )

    if _trimesh_ok:
        return udf_flat.reshape(shape)

    # ------------------------------------------------------------------
    # Fallback: brute-force O(N_tri × N_pts)  — slow for large scenes
    # ------------------------------------------------------------------
    bs = int(point_batch_size)
    n_batches = math.ceil(flat_n / bs)
    print(
        f"[bake_udf_grid] brute-force  tris={tris.shape[0]}  "
        f"voxels={flat_n}  batches={n_batches}  batch_size={bs}",
        flush=True,
    )
    for batch_i, start in enumerate(range(0, flat_n, bs)):
        end = min(start + bs, flat_n)
        idx = np.arange(start, end, dtype=np.int64)
        pts = _voxel_centers_from_flat_indices(origin, spacing, shape, idx)
        d_sq = np.full(end - start, np.inf, dtype=np.float64)
        for ti in range(tris.shape[0]):
            d_sq = np.minimum(d_sq, _point_triangle_dist_sq(pts, tris[ti]))
        udf_flat[start:end] = np.sqrt(d_sq).astype(np.float32)
        if batch_i % max(1, n_batches // 20) == 0:
            print(
                f"  {100 * end / flat_n:5.1f}%  batch {batch_i + 1}/{n_batches}",
                flush=True,
            )

    return udf_flat.reshape(shape)


# ---------------------------------------------------------------------------
# URDF assembly (fixed joints at zero pose; mesh geometry → world triangles)
# ---------------------------------------------------------------------------


@dataclass
class UrdfAssembly:
    """Triangle soup for all meshed links in one world frame, plus metadata."""

    triangles: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    link_names: list[str]
    mesh_paths: list[str]


def _urdf_local_tag(elem: ET.Element) -> str:
    if "}" in elem.tag:
        return elem.tag.split("}", 1)[1]
    return elem.tag


def _parse_float_vec(attr: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not attr:
        return np.asarray(default, dtype=np.float64)
    parts = attr.split()
    if len(parts) != 3:
        raise ValueError(f"expected 3 floats, got {attr!r}")
    return np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Fixed-axis roll–pitch–yaw (URDF): roll about X, then pitch about Y, then yaw about Z."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def origin_to_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """4×4: maps child column vector to parent frame (URDF joint/visual origin)."""
    r = rpy_to_rotation_matrix(rpy)
    t = np.asarray(xyz, dtype=np.float64).reshape(3)
    t4 = np.eye(4, dtype=np.float64)
    t4[:3, :3] = r
    t4[:3, 3] = t
    return t4


def _transform_points(t4: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    r = t4[:3, :3]
    t = t4[:3, 3]
    return (r @ pts.T).T + t


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def resolve_urdf_mesh_uri(urdf_path: str | Path, uri: str) -> Path:
    """Resolve a ``mesh`` ``filename`` to an on-disk path.

    - ``package://<robot_pkg>/meshes/...`` → ``<urdf_dir.parent>/<rest after pkg/>``
      (standard layout: ``.../package/urdf/file.urdf`` and ``.../package/meshes/...``).
    - Otherwise relative to the URDF file directory.
    """
    urdf_path = Path(urdf_path).resolve()
    raw = uri.strip()
    if raw.startswith("package://"):
        rest = raw[len("package://") :]
        slash = rest.find("/")
        if slash < 0 or slash == len(rest) - 1:
            raise ValueError(f"invalid package mesh URI: {uri!r}")
        subpath = rest[slash + 1 :]
        candidate = (urdf_path.parent.parent / subpath).resolve()
    else:
        candidate = (urdf_path.parent / raw).resolve()
    return _resolve_mesh_path_with_fallbacks(candidate)


def _resolve_mesh_path_with_fallbacks(path: Path) -> Path:
    if path.is_file():
        return path
    lower = path.with_name(path.name.lower())
    if lower.is_file():
        return lower
    if path.suffix.lower() in (".stl", ".STL"):
        alt = path.with_suffix(".obj")
        if alt.is_file():
            return alt
    if path.suffix.lower() == ".obj":
        alt = path.with_suffix(".STL")
        if alt.is_file():
            return alt
        alt2 = path.with_suffix(".stl")
        if alt2.is_file():
            return alt2
    raise FileNotFoundError(f"mesh file not found (tried variants): {path}")


def _import_trimesh():
    try:
        import trimesh  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("URDF mesh loading requires the trimesh package") from exc
    return trimesh


def _load_mesh_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    trimesh = _import_trimesh()
    loaded = trimesh.load(str(path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        parts = [
            g
            for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        if not parts:
            raise ValueError(f"no Trimesh geometry in scene: {path}")
        mesh = trimesh.util.concatenate(parts)
    else:
        mesh = loaded
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return verts, faces


def _mesh_specs_from_group(
    link_el: ET.Element,
    *,
    group_tag: str,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """List of (uri, scale_xyz, local_xyz, local_rpy) from one URDF group tag.

    The XML child order inside ``<visual>`` / ``<collision>`` must not matter:
    ``<origin>`` may appear before or after ``<geometry>``.
    """
    out: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for child in link_el:
        if _urdf_local_tag(child) != group_tag:
            continue
        local_xyz = np.zeros(3, dtype=np.float64)
        local_rpy = np.zeros(3, dtype=np.float64)
        geometry_nodes: list[ET.Element] = []
        for sub in child:
            tag = _urdf_local_tag(sub)
            if tag == "origin":
                local_xyz = _parse_float_vec(sub.get("xyz"), (0.0, 0.0, 0.0))
                local_rpy = _parse_float_vec(sub.get("rpy"), (0.0, 0.0, 0.0))
            elif tag == "geometry":
                geometry_nodes.append(sub)
        for geometry in geometry_nodes:
            for g in geometry.iter():
                if _urdf_local_tag(g) != "mesh":
                    continue
                fn = g.get("filename")
                if not fn:
                    continue
                scale = _parse_float_vec(g.get("scale"), (1.0, 1.0, 1.0))
                out.append((fn, scale, local_xyz.copy(), local_rpy.copy()))
    return out


def _visual_mesh_specs(link_el: ET.Element) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """List of (uri, scale_xyz, visual_xyz, visual_rpy) from ``<visual>`` only."""
    return _mesh_specs_from_group(link_el, group_tag="visual")


def _collision_mesh_specs(link_el: ET.Element) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    return _mesh_specs_from_group(link_el, group_tag="collision")


def _link_mesh_specs(link_el: ET.Element) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    specs = _visual_mesh_specs(link_el)
    if specs:
        return specs
    return _collision_mesh_specs(link_el)


def _parse_urdf_links_and_joints(
    root: ET.Element,
) -> tuple[dict[str, ET.Element], list[tuple[str, str, str, np.ndarray, np.ndarray]]]:
    links: dict[str, ET.Element] = {}
    joints: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []
    for elem in root:
        tag = _urdf_local_tag(elem)
        if tag == "link":
            name = elem.get("name")
            if not name:
                raise ValueError("link without name")
            links[name] = elem
        elif tag == "joint":
            jtype = elem.get("type") or "fixed"
            parent_el = child_el = None
            j_xyz = np.zeros(3, dtype=np.float64)
            j_rpy = np.zeros(3, dtype=np.float64)
            for ch in elem:
                ct = _urdf_local_tag(ch)
                if ct == "parent":
                    parent_el = ch
                elif ct == "child":
                    child_el = ch
                elif ct == "origin":
                    j_xyz = _parse_float_vec(ch.get("xyz"), (0.0, 0.0, 0.0))
                    j_rpy = _parse_float_vec(ch.get("rpy"), (0.0, 0.0, 0.0))
            if parent_el is None or child_el is None:
                raise ValueError("joint must contain exactly one parent and one child link reference")
            pn = parent_el.get("link")
            cn = child_el.get("link")
            if not pn or not cn:
                raise ValueError("joint parent/child link attribute must be non-empty")
            joints.append((pn, cn, jtype, j_xyz, j_rpy))
    return links, joints


def _find_root_link(links: dict[str, ET.Element], joints: list[tuple[str, str, str, np.ndarray, np.ndarray]]) -> str:
    children = {c for _, c, _, _, _ in joints}
    roots = [n for n in links if n not in children]
    if not roots:
        raise ValueError("could not find URDF root link")
    if len(roots) > 1:
        raise ValueError(f"multiple root links unsupported: {roots}")
    return roots[0]


def _world_link_transforms(
    root: str,
    joints: list[tuple[str, str, str, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Static assembly: all joint types use the declared ``origin`` (zero articulation)."""
    children_map: dict[str, list[tuple[str, np.ndarray]]] = {}
    for pn, cn, _jt, j_xyz, j_rpy in joints:
        t_joint = origin_to_transform(j_xyz, j_rpy)
        children_map.setdefault(pn, []).append((cn, t_joint))

    world_t: dict[str, np.ndarray] = {}
    identity = np.eye(4, dtype=np.float64)
    world_t[root] = identity
    stack = [root]
    seen = {root}
    while stack:
        parent = stack.pop()
        t_parent_world = world_t[parent]
        for child, t_parent_child in children_map.get(parent, []):
            t_child_world = _compose(t_parent_world, t_parent_child)
            if child in seen and not np.allclose(t_child_world, world_t[child], atol=1e-9):
                raise ValueError(f"conflicting transforms for link {child!r}")
            if child not in seen:
                world_t[child] = t_child_world
                seen.add(child)
                stack.append(child)
    return world_t


def load_assembly_from_urdf(urdf_path: str | Path) -> UrdfAssembly:
    """Load meshes for each link, accumulate fixed (and static) joint origins, return world triangles."""
    urdf_path = Path(urdf_path).resolve()
    tree = ET.parse(urdf_path)
    el_root = tree.getroot()
    if _urdf_local_tag(el_root) != "robot":
        raise ValueError("URDF root must be <robot>")

    links, joints = _parse_urdf_links_and_joints(el_root)
    root_name = _find_root_link(links, joints)
    world_t_link = _world_link_transforms(root_name, joints)

    all_tris: list[np.ndarray] = []
    link_names: list[str] = []
    mesh_paths: list[str] = []

    for link_name, t_w_l in sorted(world_t_link.items(), key=lambda x: x[0]):
        link_el = links.get(link_name)
        if link_el is None:
            continue
        specs = _link_mesh_specs(link_el)
        if not specs:
            continue
        for uri, scale, v_xyz, v_rpy in specs:
            resolved = resolve_urdf_mesh_uri(urdf_path, uri)
            verts, faces = _load_mesh_vertices_faces(resolved)
            verts = verts * scale.reshape(1, 3)
            t_link_visual = origin_to_transform(v_xyz, v_rpy)
            t_w_visual = _compose(t_w_l, t_link_visual)
            vw = _transform_points(t_w_visual, verts)
            tri = vw[faces].reshape(-1, 3, 3)
            all_tris.append(tri)
            link_names.append(link_name)
            mesh_paths.append(str(resolved))

    if not all_tris:
        raise ValueError(f"no mesh geometry found in URDF: {urdf_path}")

    triangles = np.concatenate(all_tris, axis=0).astype(np.float32, copy=False)
    flat = triangles.reshape(-1, 3)
    bbox_min = np.asarray(flat.min(axis=0), dtype=np.float32)
    bbox_max = np.asarray(flat.max(axis=0), dtype=np.float32)
    return UrdfAssembly(
        triangles=triangles,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        link_names=link_names,
        mesh_paths=mesh_paths,
    )


def bake_distance_field(assembly: UrdfAssembly, args: argparse.Namespace) -> DistanceField:
    """Bake global UDF plus optional libigl / Open3D SDF grids on the same voxel layout.

    The UDF bake is always attempted first. libigl and Open3D backends are best-effort:
    failures degrade to all-``NaN`` SDF grids with entries in ``failure_reasons`` without
    affecting ``udf_grid``.
    """
    spacing = float(args.spacing)
    margin = float(args.margin)
    max_mem = getattr(args, "max_memory_bytes", None)
    if max_mem is None:
        mb = getattr(args, "max_memory_mb", DEFAULT_MAX_MEMORY_MB)
        max_mem = int(float(mb) * 1024**2)
    udf_only = bool(getattr(args, "udf_only", False))

    origin, shape = compute_grid_domain(assembly.bbox_min, assembly.bbox_max, spacing, margin)
    enabled_grids = 1 if udf_only else 3
    need_bytes = estimate_grid_memory(shape, enabled_grids, np.float32)
    if max_mem is not None and need_bytes > max_mem:
        label = "UDF-only distance field" if udf_only else "Distance field (3 grids)"
        raise GridMemoryBudgetError(
            f"{label} would need ~{need_bytes} bytes (limit {max_mem}); "
            f"shape={shape}, spacing={spacing}, margin={margin}"
        )

    pb = int(getattr(args, "max_points_per_batch", DEFAULT_MAX_POINTS_PER_BATCH))
    udf_grid = bake_udf_grid(
        assembly.triangles,
        assembly.bbox_min,
        assembly.bbox_max,
        spacing,
        margin,
        max_memory_bytes=None,
        point_batch_size=pb,
    )
    if tuple(int(x) for x in udf_grid.shape) != shape:
        raise RuntimeError("internal grid layout mismatch after bake_udf_grid")

    if udf_only:
        igl_grid = build_nan_grid(shape)
        o3d_grid = build_nan_grid(shape)
        igl_reason = "igl_sdf: skipped (--udf-only)"
        o3d_reason = "open3d_sdf: skipped (--udf-only)"
    else:
        igl_grid, igl_reason = bake_igl_sdf_grid(
            assembly.triangles,
            origin,
            shape,
            spacing,
            max_memory_bytes=None,
            point_batch_size=pb,
        )
        o3d_grid, o3d_reason = bake_open3d_sdf_grid(
            assembly.triangles,
            origin,
            shape,
            spacing,
            max_memory_bytes=None,
            point_batch_size=pb,
        )

    status_flags = {
        "udf_ok": True,
        "igl_ok": igl_reason is None,
        "o3d_ok": o3d_reason is None,
    }
    failure_reasons: list[str] = []
    if igl_reason:
        failure_reasons.append(igl_reason)
    if o3d_reason:
        failure_reasons.append(o3d_reason)
    build_config = {
        "spacing": spacing,
        "margin": margin,
        "shape": list(shape),
        "udf_only": udf_only,
        "max_points_per_batch": pb,
        "version": 1,
    }

    return DistanceField(
        origin=origin,
        spacing=np.float32(spacing),
        udf_grid=udf_grid,
        igl_sdf_grid=igl_grid,
        o3d_sdf_grid=o3d_grid,
        bbox_min=np.asarray(assembly.bbox_min, dtype=np.float32),
        bbox_max=np.asarray(assembly.bbox_max, dtype=np.float32),
        status_flags=status_flags,
        failure_reasons=failure_reasons,
        build_config=build_config,
    )


# ---------------------------------------------------------------------------
# Field visualization (static PNG via matplotlib)
# ---------------------------------------------------------------------------


def prepare_slice_for_plot(
    grid_3d: np.ndarray,
    axis: int,
    index: int,
) -> np.ma.MaskedArray:
    """Take a 2D slice from a 3D scalar grid; mask invalid values (NaN/inf), never as zero."""
    g = np.asarray(grid_3d)
    if g.ndim != 3:
        raise ValueError(f"expected 3D grid, got shape {g.shape}")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    n = int(g.shape[axis])
    if index < 0 or index >= n:
        raise ValueError(f"index {index} out of range for axis {axis} (len={n})")
    if axis == 0:
        sl = g[index, :, :]
    elif axis == 1:
        sl = g[:, index, :]
    else:
        sl = g[:, :, index]
    return np.ma.masked_invalid(np.asarray(sl, dtype=np.float64))


def _slice_2d_array(arr: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return arr[index, :, :]
    if axis == 1:
        return arr[:, index, :]
    return arr[:, :, index]


def _matplotlib_pyplot_agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_masked_heatmap(
    data_ma: np.ma.MaskedArray,
    title: str,
    path: Path,
    *,
    cmap: str = "viridis",
    symmetric: bool = False,
) -> Path:
    plt = _matplotlib_pyplot_agg()
    fig, ax = plt.subplots(figsize=(6, 5))
    if symmetric:
        filled = np.ma.filled(data_ma, np.nan)
        if np.any(np.isfinite(filled)):
            m = float(np.nanmax(np.abs(filled)))
            if m > 0:
                im = ax.imshow(
                    data_ma, origin="lower", cmap=cmap, vmin=-m, vmax=m, aspect="auto"
                )
            else:
                im = ax.imshow(
                    data_ma, origin="lower", cmap=cmap, vmin=-1e-9, vmax=1e-9, aspect="auto"
                )
        else:
            im = ax.imshow(data_ma, origin="lower", cmap=cmap, aspect="auto")
    else:
        im = ax.imshow(data_ma, origin="lower", cmap=cmap, aspect="auto")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def render_slice_comparison(
    field: DistanceField,
    output_dir: str | Path,
    *,
    axis: int = 2,
    index: int | None = None,
    file_prefix: str = "slice",
) -> list[Path]:
    """Write PNGs for UDF, igl SDF, o3d SDF, and absolute differences vs UDF on one grid slice."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(x) for x in field.udf_grid.shape)
    if index is None:
        index = int(shape[axis] // 2)
    idx = int(index)
    axis_name = ("x", "y", "z")[axis]
    tag = f"{file_prefix}_axis{axis}_{axis_name}_idx{idx}"

    paths: list[Path] = []
    udf_ma = prepare_slice_for_plot(field.udf_grid, axis, idx)
    paths.append(
        _save_masked_heatmap(udf_ma, f"UDF ({axis_name}={idx})", out / f"{tag}_udf.png")
    )

    igl_ma = prepare_slice_for_plot(field.igl_sdf_grid, axis, idx)
    paths.append(
        _save_masked_heatmap(
            igl_ma,
            f"libigl SDF ({axis_name}={idx})",
            out / f"{tag}_igl_sdf.png",
            cmap="coolwarm",
            symmetric=True,
        )
    )

    o3d_ma = prepare_slice_for_plot(field.o3d_sdf_grid, axis, idx)
    paths.append(
        _save_masked_heatmap(
            o3d_ma,
            f"Open3D SDF ({axis_name}={idx})",
            out / f"{tag}_o3d_sdf.png",
            cmap="coolwarm",
            symmetric=True,
        )
    )

    u = np.asarray(_slice_2d_array(field.udf_grid, axis, idx), dtype=np.float64)
    ig = np.asarray(_slice_2d_array(field.igl_sdf_grid, axis, idx), dtype=np.float64)
    od = np.asarray(_slice_2d_array(field.o3d_sdf_grid, axis, idx), dtype=np.float64)
    mask_igl = ~np.isfinite(u) | ~np.isfinite(ig)
    diff_igl = np.ma.masked_where(mask_igl, np.abs(ig - u))
    paths.append(
        _save_masked_heatmap(
            diff_igl,
            f"abs(igl_sdf - UDF) ({axis_name}={idx})",
            out / f"{tag}_abs_diff_igl_udf.png",
            cmap="magma",
        )
    )
    mask_o3d = ~np.isfinite(u) | ~np.isfinite(od)
    diff_o3d = np.ma.masked_where(mask_o3d, np.abs(od - u))
    paths.append(
        _save_masked_heatmap(
            diff_o3d,
            f"abs(o3d_sdf - UDF) ({axis_name}={idx})",
            out / f"{tag}_abs_diff_o3d_udf.png",
            cmap="magma",
        )
    )
    return paths


def _sign_mismatch_mask(igl: np.ndarray, o3d: np.ndarray) -> np.ndarray:
    """Voxels where finite igl and o3d have strictly opposite signs.

    Zeros are treated as ambiguous (surface / unsigned): only finite, strictly
    opposite signs with both values nonzero count as disagreement.
    """
    ig = np.asarray(igl, dtype=np.float64)
    od = np.asarray(o3d, dtype=np.float64)
    both_finite = np.isfinite(ig) & np.isfinite(od)
    both_nonzero = (ig != 0.0) & (od != 0.0)
    opposite = ((ig > 0.0) & (od < 0.0)) | ((ig < 0.0) & (od > 0.0))
    return both_finite & both_nonzero & opposite


def render_sample_point_comparison(
    field: DistanceField,
    output_dir: str | Path,
    *,
    n_samples: int = 2048,
    seed: int = 0,
    file_name: str = "sample_points_xy_comparison.png",
) -> list[Path]:
    """Uniform random samples inside ``bbox``; scatter UDF / SDF / deltas in the XY plane."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = int(max(1, n_samples))
    rng = np.random.default_rng(seed)
    lo = np.asarray(field.bbox_min, dtype=np.float64).reshape(1, 3)
    hi = np.asarray(field.bbox_max, dtype=np.float64).reshape(1, 3)
    u01 = rng.random((n, 3))
    pts = (lo + u01 * (hi - lo)).astype(np.float32, copy=False)

    udf_v = field.query(pts, kind="udf", clip=True)
    igl_v = field.query(pts, kind="igl_sdf", clip=True)
    o3d_v = field.query(pts, kind="o3d_sdf", clip=True)
    if udf_v.ndim == 0:
        udf_v = np.array([float(udf_v)], dtype=np.float32)
        igl_v = np.array([float(igl_v)], dtype=np.float32)
        o3d_v = np.array([float(o3d_v)], dtype=np.float32)

    plt = _matplotlib_pyplot_agg()
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    x = pts[:, 0]
    y = pts[:, 1]

    udf_mask = np.isfinite(udf_v)
    sc0 = axes[0, 0].scatter(
        x[udf_mask],
        y[udf_mask],
        c=udf_v[udf_mask],
        s=8,
        cmap="viridis",
        alpha=0.85,
    )
    axes[0, 0].set_title("UDF (XY, bbox samples)")
    fig.colorbar(sc0, ax=axes[0, 0], shrink=0.8)

    igl_mask = np.isfinite(igl_v)
    sc1 = axes[0, 1].scatter(
        x[igl_mask], y[igl_mask], c=igl_v[igl_mask], s=8, cmap="coolwarm", alpha=0.85
    )
    axes[0, 1].set_title("libigl SDF (finite only)")
    fig.colorbar(sc1, ax=axes[0, 1], shrink=0.8)

    o3d_mask = np.isfinite(o3d_v)
    sc2 = axes[1, 0].scatter(
        x[o3d_mask], y[o3d_mask], c=o3d_v[o3d_mask], s=8, cmap="coolwarm", alpha=0.85
    )
    axes[1, 0].set_title("Open3D SDF (finite only)")
    fig.colorbar(sc2, ax=axes[1, 0], shrink=0.8)

    d_igl = igl_v.astype(np.float64) - udf_v.astype(np.float64)
    d_o3d = o3d_v.astype(np.float64) - udf_v.astype(np.float64)
    dm = udf_mask & np.isfinite(d_igl) & np.isfinite(d_o3d)
    sc3 = axes[1, 1].scatter(
        x[dm],
        y[dm],
        c=np.maximum(np.abs(d_igl[dm]), np.abs(d_o3d[dm])),
        s=8,
        cmap="magma",
        alpha=0.85,
    )
    axes[1, 1].set_title("max(|igl-udf|, |o3d-udf|) (both finite)")
    fig.colorbar(sc3, ax=axes[1, 1], shrink=0.8)

    for ax in axes.ravel():
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    path = out / file_name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [path]


def render_sign_diagnostics(
    field: DistanceField,
    output_dir: str | Path,
    *,
    axis: int = 2,
    index: int | None = None,
    udf_far_threshold: float | None = None,
) -> list[Path]:
    """PNG diagnostics: igl vs o3d sign disagreement, and SDF<0 with UDF above threshold."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    spacing = float(field.spacing)
    thresh = float(udf_far_threshold) if udf_far_threshold is not None else 2.0 * spacing

    udf = np.asarray(field.udf_grid, dtype=np.float64)
    igl = np.asarray(field.igl_sdf_grid, dtype=np.float64)
    o3d = np.asarray(field.o3d_sdf_grid, dtype=np.float64)

    sign_mismatch = _sign_mismatch_mask(igl, o3d)

    neg_igl = np.isfinite(igl) & (igl < 0.0) & np.isfinite(udf) & (udf > thresh)
    neg_o3d = np.isfinite(o3d) & (o3d < 0.0) & np.isfinite(udf) & (udf > thresh)
    anomaly = neg_igl | neg_o3d

    shape = udf.shape
    if index is None:
        index = int(shape[axis] // 2)
    idx = int(index)
    axis_name = ("x", "y", "z")[axis]

    sm_slice = _slice_2d_array(sign_mismatch, axis, idx)
    sm_sl = sm_slice.astype(np.float32)
    an_slice = _slice_2d_array(anomaly, axis, idx)
    an_sl = an_slice.astype(np.float32)

    plt = _matplotlib_pyplot_agg()
    paths: list[Path] = []

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im1 = ax1.imshow(sm_sl, origin="lower", cmap="hot", vmin=0.0, vmax=1.0, aspect="auto")
    ax1.set_title(
        f"igl vs o3d sign mismatch ({axis_name}={idx}); "
        f"voxels={int(np.sum(sm_slice))}"
    )
    fig1.colorbar(im1, ax=ax1, shrink=0.8)
    fig1.tight_layout()
    p1 = out / f"sign_diag_igl_o3d_mismatch_{axis_name}{idx}.png"
    fig1.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig1)
    paths.append(p1)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = ax2.imshow(an_sl, origin="lower", cmap="Oranges", vmin=0.0, vmax=1.0, aspect="auto")
    ax2.set_title(
        f"SDF<0 & UDF>{thresh:g} ({axis_name}={idx}); voxels={int(np.sum(an_slice))}"
    )
    fig2.colorbar(im2, ax=ax2, shrink=0.8)
    fig2.tight_layout()
    p2 = out / f"sign_diag_neg_sdf_large_udf_{axis_name}{idx}.png"
    fig2.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    paths.append(p2)

    return paths


# ---------------------------------------------------------------------------
# 3-D visualization helpers
# ---------------------------------------------------------------------------


def _import_skimage_measure():
    """Lazy import of skimage.measure (marching cubes).  pip install scikit-image."""
    try:
        from skimage import measure  # type: ignore[import-not-found]

        return measure
    except ImportError as exc:
        raise ImportError(
            "3-D visualization requires scikit-image.  "
            "Install with: pip install scikit-image"
        ) from exc


def _downsample_grid(grid: np.ndarray, max_side: int = 128) -> tuple[np.ndarray, int]:
    """Uniformly subsample a 3-D grid so its longest axis <= max_side.

    Returns (downsampled_grid, stride).
    """
    stride = max(1, int(math.ceil(max(grid.shape) / max_side)))
    if stride == 1:
        return grid, 1
    return grid[::stride, ::stride, ::stride], stride


def _udf_to_solid(udf_raw: np.ndarray) -> np.ndarray:
    """Fill NaN/inf cells with the finite maximum so marching cubes stays clean."""
    finite = np.isfinite(udf_raw)
    fill = float(np.nanmax(udf_raw[finite])) if finite.any() else 1.0
    return np.where(finite, udf_raw, fill)


def _sdf_to_solid(sdf_raw: np.ndarray) -> np.ndarray:
    """Fill NaN/inf cells with a large positive value for marching cubes at level=0."""
    finite = np.isfinite(sdf_raw)
    fill = float(np.nanmax(np.abs(sdf_raw[finite]))) + 1.0 if finite.any() else 1.0
    return np.where(finite, sdf_raw, fill)


def _grid_has_data(grid: np.ndarray) -> bool:
    """True if the grid contains at least one finite value."""
    return bool(np.any(np.isfinite(grid)))


def _render_isosurface_png(
    verts: np.ndarray,
    faces: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    save_path: Path,
    suptitle: str,
    facecolor: str = "steelblue",
) -> Path:
    """Render 3-panel (perspective / top / side) isosurface PNG via matplotlib."""
    plt = _matplotlib_pyplot_agg()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-not-found]

    fig = plt.figure(figsize=(15, 5))
    views: list[tuple[int, int]] = [(25, -55), (90, -90), (0, -90)]
    titles = ["Perspective", "Top (XY)", "Side (XZ)"]
    mesh_polys = verts[faces]
    for col, ((elev, azim), title) in enumerate(zip(views, titles)):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        poly = Poly3DCollection(
            mesh_polys, alpha=0.40, facecolor=facecolor, edgecolor="none", linewidth=0,
        )
        ax.add_collection3d(poly)
        ax.set_xlim(float(bbox_min[0]), float(bbox_max[0]))
        ax.set_ylim(float(bbox_min[1]), float(bbox_max[1]))
        ax.set_zlim(float(bbox_min[2]), float(bbox_max[2]))
        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Y (m)", fontsize=7)
        ax.set_zlabel("Z (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=9)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def render_3d_isosurface(
    field: DistanceField,
    output_dir: str | Path,
    *,
    threshold: float = 0.01,
    max_side: int = 128,
    file_name: str = "udf_3d_isosurface.png",
) -> list[Path]:
    """Extract isosurfaces for UDF, igl SDF, and Open3D SDF via marching cubes.

    UDF uses *threshold* as the level; SDF grids use level=0 (zero-crossing).
    Each grid that contains valid data produces a separate 3-panel PNG.
    Requires scikit-image (pip install scikit-image).
    """
    try:
        measure = _import_skimage_measure()
    except ImportError as exc:
        print(f"[render_3d_isosurface] skipped: {exc}", file=sys.stderr)
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    origin = np.asarray(field.origin, dtype=np.float64)
    bbox_min = np.asarray(field.bbox_min, dtype=np.float64)
    bbox_max = np.asarray(field.bbox_max, dtype=np.float64)
    paths: list[Path] = []

    grids_to_render: list[tuple[str, np.ndarray, float, str, str]] = [
        ("UDF", field.udf_grid, threshold, file_name, "steelblue"),
    ]
    if _grid_has_data(field.igl_sdf_grid):
        stem = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
        grids_to_render.append(
            ("libigl SDF", field.igl_sdf_grid, 0.0, f"{stem}_igl.png", "coral")
        )
    if _grid_has_data(field.o3d_sdf_grid):
        stem = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
        grids_to_render.append(
            ("Open3D SDF", field.o3d_sdf_grid, 0.0, f"{stem}_o3d.png", "seagreen")
        )

    for label, raw_grid, level, fname, color in grids_to_render:
        raw = np.asarray(raw_grid, dtype=np.float64)
        is_sdf = level == 0.0 and label != "UDF"
        solid = _sdf_to_solid(raw) if is_sdf else _udf_to_solid(raw)
        grid_ds, stride = _downsample_grid(solid, max_side)
        eff_spacing = float(field.spacing) * stride

        if is_sdf:
            has_crossing = float(grid_ds.min()) <= level <= float(grid_ds.max())
        else:
            has_crossing = float(grid_ds.min()) < level
        if not has_crossing:
            print(
                f"[render_3d_isosurface] {label}: no isosurface at level={level:.4f} — skipping",
                file=sys.stderr,
            )
            continue

        try:
            verts, faces, _, _ = measure.marching_cubes(
                grid_ds, level=level, spacing=(eff_spacing,) * 3
            )
        except Exception as exc:
            print(f"[render_3d_isosurface] {label}: marching_cubes failed: {exc}", file=sys.stderr)
            continue

        verts = verts + origin
        n_v, n_f = len(verts), len(faces)
        level_desc = f"level=0 (surface)" if is_sdf else f"threshold={level:.4f} m"
        suptitle = (
            f"{label} Isosurface  |  {level_desc}  |  "
            f"{n_v:,} verts  {n_f:,} faces  (grid stride={stride})"
        )
        p = _render_isosurface_png(verts, faces, bbox_min, bbox_max, out / fname, suptitle, color)
        print(f"[render_3d_isosurface] saved {p}  ({n_v:,} verts, {n_f:,} faces)")
        paths.append(p)

    return paths


def render_3d_plotly(
    field: DistanceField,
    output_dir: str | Path,
    *,
    threshold: float = 0.01,
    max_side: int = 96,
    file_name: str = "udf_3d_interactive.html",
) -> list[Path]:
    """Interactive 3-D isosurface rendered as a self-contained HTML file via plotly.

    UDF, libigl SDF, and Open3D SDF isosurfaces are overlaid as separate toggleable
    traces when the corresponding grids contain valid data.
    Requires both plotly and scikit-image (pip install plotly scikit-image).
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import-not-found]
        import plotly.io as pio  # type: ignore[import-not-found]
    except ImportError:
        print(
            "[render_3d_plotly] skipped: plotly not installed.  "
            "Install with: pip install plotly",
            file=sys.stderr,
        )
        return []

    try:
        measure = _import_skimage_measure()
    except ImportError as exc:
        print(f"[render_3d_plotly] skipped: {exc}", file=sys.stderr)
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    origin = np.asarray(field.origin, dtype=np.float64)
    bbox_min = np.asarray(field.bbox_min, dtype=np.float64)
    bbox_max = np.asarray(field.bbox_max, dtype=np.float64)

    traces: list = []
    title_parts: list[str] = []

    layer_defs: list[tuple[str, np.ndarray, float, bool, str, str, float]] = [
        ("UDF", field.udf_grid, threshold, False, "Blues", "steelblue", 0.70),
    ]
    if _grid_has_data(field.igl_sdf_grid):
        layer_defs.append(
            ("libigl SDF", field.igl_sdf_grid, 0.0, True, "Reds", "coral", 0.50)
        )
    if _grid_has_data(field.o3d_sdf_grid):
        layer_defs.append(
            ("Open3D SDF", field.o3d_sdf_grid, 0.0, True, "Greens", "seagreen", 0.50)
        )

    for label, raw_grid, level, is_sdf, cscale, color, opa in layer_defs:
        raw = np.asarray(raw_grid, dtype=np.float64)
        solid = _sdf_to_solid(raw) if is_sdf else _udf_to_solid(raw)
        grid_ds, stride = _downsample_grid(solid, max_side)
        eff_spacing = float(field.spacing) * stride

        if is_sdf:
            has_crossing = float(grid_ds.min()) <= level <= float(grid_ds.max())
        else:
            has_crossing = float(grid_ds.min()) < level
        if not has_crossing:
            print(
                f"[render_3d_plotly] {label}: no isosurface at level={level:.4f} — skipping",
                file=sys.stderr,
            )
            continue

        try:
            verts, faces, _, _ = measure.marching_cubes(
                grid_ds, level=level, spacing=(eff_spacing,) * 3
            )
        except Exception as exc:
            print(f"[render_3d_plotly] {label}: marching_cubes failed: {exc}", file=sys.stderr)
            continue

        verts = verts + origin
        n_v, n_f = len(verts), len(faces)
        level_str = "level=0" if is_sdf else f"thr={level:.4f}"
        title_parts.append(f"{label} {n_v:,}v/{n_f:,}f ({level_str}, s{stride})")

        traces.append(go.Mesh3d(
            x=verts[:, 0].tolist(),
            y=verts[:, 1].tolist(),
            z=verts[:, 2].tolist(),
            i=faces[:, 0].tolist(),
            j=faces[:, 1].tolist(),
            k=faces[:, 2].tolist(),
            intensity=verts[:, 2].tolist(),
            colorscale=cscale,
            showscale=(label == "UDF"),
            colorbar=dict(title="Z (m)", thickness=14) if label == "UDF" else None,
            opacity=opa,
            flatshading=False,
            lighting=dict(ambient=0.35, diffuse=0.85, specular=0.25, roughness=0.45),
            lightposition=dict(x=100, y=200, z=300),
            name=f"{label} isosurface",
            showlegend=True,
            visible=True,
        ))

    if not traces:
        print("[render_3d_plotly] no valid isosurface produced — skipping", file=sys.stderr)
        return []

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=" &nbsp;|&nbsp; ".join(title_parts) if title_parts else "3-D Isosurface",
            x=0.5,
            font=dict(size=12),
        ),
        scene=dict(
            xaxis=dict(title="X (m)", range=[float(bbox_min[0]), float(bbox_max[0])]),
            yaxis=dict(title="Y (m)", range=[float(bbox_min[1]), float(bbox_max[1])]),
            zaxis=dict(title="Z (m)", range=[float(bbox_min[2]), float(bbox_max[2])]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, t=55, b=0),
        height=720,
    )

    p = out / file_name
    pio.write_html(fig, file=str(p), include_plotlyjs="cdn", full_html=True)
    print(f"[render_3d_plotly] saved {p}  ({len(traces)} layer(s))")
    return [p]


def render_3d_heatmap_plotly(
    field,
    output_dir,
    *,
    max_side: int = 64,
    dist_min: float = 0.0,
    dist_max: float = 0.5,
    surface_count: int = 20,
    opacity: float = 0.15,
    overlay_isosurface: bool = True,
    iso_threshold: float = 0.01,
    file_name: str = "udf_3d_heatmap.html",
):
    """Interactive 3-D volume heatmap for UDF, libigl SDF, and Open3D SDF via plotly.

    Each grid with valid data is rendered as a semi-transparent Volume trace
    (toggleable via legend).  An isosurface shell overlay is optionally added
    for each grid.  Requires: pip install plotly scikit-image.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print(
            "[render_3d_heatmap_plotly] skipped: plotly not installed. "
            "Install with: pip install plotly",
            file=sys.stderr,
        )
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    origin   = np.asarray(field.origin,   dtype=np.float64)
    bbox_min = np.asarray(field.bbox_min, dtype=np.float64)
    bbox_max = np.asarray(field.bbox_max, dtype=np.float64)

    traces: list = []
    title_parts: list[str] = []

    volume_defs: list[tuple[str, np.ndarray, bool, str, str]] = [
        ("UDF", field.udf_grid, False, "Plasma", "UDF (m)"),
    ]
    if _grid_has_data(field.igl_sdf_grid):
        volume_defs.append(
            ("libigl SDF", field.igl_sdf_grid, True, "RdBu_r", "igl SDF (m)")
        )
    if _grid_has_data(field.o3d_sdf_grid):
        volume_defs.append(
            ("Open3D SDF", field.o3d_sdf_grid, True, "PRGn_r", "o3d SDF (m)")
        )

    for vi, (label, raw_grid, is_sdf, cscale, cbar_title) in enumerate(volume_defs):
        raw = np.asarray(raw_grid, dtype=np.float64)
        grid_ds, stride = _downsample_grid(raw, max_side)
        eff_sp = float(field.spacing) * stride
        nx, ny, nz = grid_ds.shape

        cx = origin[0] + (np.arange(nx) + 0.5) * eff_sp
        cy = origin[1] + (np.arange(ny) + 0.5) * eff_sp
        cz = origin[2] + (np.arange(nz) + 0.5) * eff_sp
        XX, YY, ZZ = np.meshgrid(cx, cy, cz, indexing="ij")

        if is_sdf:
            finite_vals = raw[np.isfinite(raw)]
            if finite_vals.size == 0:
                continue
            sdf_abs_max = float(np.max(np.abs(finite_vals)))
            v_min, v_max = -sdf_abs_max, sdf_abs_max
            fill_val = sdf_abs_max + 1.0
        else:
            v_min, v_max = float(dist_min), float(dist_max)
            fill_val = float(dist_max) + 1.0

        val = np.where(np.isfinite(grid_ds), grid_ds, fill_val)
        x_flat = XX.ravel().astype(np.float32)
        y_flat = YY.ravel().astype(np.float32)
        z_flat = ZZ.ravel().astype(np.float32)
        v_flat = val.ravel().astype(np.float32)

        traces.append(go.Volume(
            x=x_flat.tolist(),
            y=y_flat.tolist(),
            z=z_flat.tolist(),
            value=v_flat.tolist(),
            isomin=v_min,
            isomax=v_max,
            opacity=float(opacity),
            surface_count=int(surface_count),
            colorscale=cscale,
            colorbar=dict(
                title=dict(text=cbar_title, side="right"),
                thickness=14,
                len=0.6,
                x=1.0 + vi * 0.06,
            ),
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=f"{label} volume",
            showlegend=True,
            visible=True if vi == 0 else "legendonly",
        ))
        title_parts.append(label)

    # ── 可选：叠加等值面轮廓 ──────────────────────────────────────────────────
    if overlay_isosurface:
        try:
            measure = _import_skimage_measure()
        except ImportError:
            measure = None

        if measure is not None:
            iso_defs: list[tuple[str, np.ndarray, float, bool, str]] = [
                ("UDF", field.udf_grid, iso_threshold, False, "rgba(200,230,255,0.5)"),
            ]
            if _grid_has_data(field.igl_sdf_grid):
                iso_defs.append(
                    ("libigl SDF", field.igl_sdf_grid, 0.0, True, "rgba(255,180,180,0.5)")
                )
            if _grid_has_data(field.o3d_sdf_grid):
                iso_defs.append(
                    ("Open3D SDF", field.o3d_sdf_grid, 0.0, True, "rgba(180,230,180,0.5)")
                )

            for label, raw_grid, level, is_sdf, color in iso_defs:
                raw = np.asarray(raw_grid, dtype=np.float64)
                solid = _sdf_to_solid(raw) if is_sdf else _udf_to_solid(raw)
                iso_ds, iso_stride = _downsample_grid(solid, max_side)
                iso_sp = float(field.spacing) * iso_stride
                if is_sdf:
                    has_crossing = float(iso_ds.min()) <= level <= float(iso_ds.max())
                else:
                    has_crossing = float(iso_ds.min()) < level
                if not has_crossing:
                    continue
                try:
                    verts_iso, faces_iso, _, _ = measure.marching_cubes(
                        iso_ds, level=level, spacing=(iso_sp,) * 3
                    )
                except Exception:
                    continue
                verts_iso = verts_iso + origin
                level_str = "level=0" if is_sdf else f"d={level:.3f} m"
                traces.append(go.Mesh3d(
                    x=verts_iso[:, 0].tolist(),
                    y=verts_iso[:, 1].tolist(),
                    z=verts_iso[:, 2].tolist(),
                    i=faces_iso[:, 0].tolist(),
                    j=faces_iso[:, 1].tolist(),
                    k=faces_iso[:, 2].tolist(),
                    color=color,
                    opacity=0.45,
                    flatshading=False,
                    lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1),
                    name=f"{label} iso ({level_str})",
                    showlegend=True,
                ))

    fig = go.Figure(data=traces)
    n_pts = int(np.prod(np.asarray(field.udf_grid).shape))
    fig.update_layout(
        title=dict(
            text=(
                f"3-D Volume Heatmap ({', '.join(title_parts)}) \u00a0|\u00a0 "
                f"dist [{dist_min:.3f}, {dist_max:.3f}] m"
            ),
            x=0.5, font=dict(size=13),
        ),
        scene=dict(
            xaxis=dict(title="X (m)", range=[float(bbox_min[0]), float(bbox_max[0])]),
            yaxis=dict(title="Y (m)", range=[float(bbox_min[1]), float(bbox_max[1])]),
            zaxis=dict(title="Z (m)", range=[float(bbox_min[2]), float(bbox_max[2])]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.2, z=0.8)),
            bgcolor="rgb(10,10,20)",
        ),
        paper_bgcolor="rgb(20,20,30)",
        font=dict(color="white"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)", font=dict(color="white")),
        margin=dict(l=0, r=0, t=55, b=0),
        height=760,
    )

    p = out / file_name
    pio.write_html(fig, file=str(p), include_plotlyjs="cdn", full_html=True)
    print(f"[render_3d_heatmap_plotly] saved {p}  ({len(traces)} trace(s))")
    return [p]


def _default_artifact_dir(
    args: argparse.Namespace, output_path: Path | None
) -> Path:
    """Default PNG output directory when ``--render`` is set but ``--artifact-dir`` is omitted."""
    if output_path is not None:
        return output_path.parent / f"{output_path.stem}_plots"
    if getattr(args, "load_path", None):
        p = Path(args.load_path)
        return p.parent / f"{p.stem}_plots"
    if args.urdf:
        return Path.cwd() / f"{Path(args.urdf).stem}_plots"
    return Path.cwd() / "global_udf_plots"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bake, load, inspect, and render global distance fields for URDF assemblies."
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=URDF_PATH,
        help="Path to URDF file (default: %(default)s)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Parse URDF, load meshes, print links / paths / bbox only",
    )
    parser.add_argument(
        "--udf-only",
        action="store_true",
        default=BAKE_UDF_ONLY,
        help="Bake unsigned distance field (UDF) only and write compressed .npz",
    )
    parser.add_argument(
        "--no-udf-only",
        action="store_false",
        dest="udf_only",
        help="Override BAKE_UDF_ONLY=True from the config block",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=BAKE_SPACING,
        help="Voxel spacing for UDF bake (default: %(default)s m)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=BAKE_MARGIN,
        help="Extra padding around assembly bbox for the grid domain (default: %(default)s m)",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=float,
        default=BAKE_MAX_MEMORY_MB,
        help="Abort if estimated float32 grid storage (3 grids) exceeds this budget",
    )
    parser.add_argument(
        "--max-points-per-batch",
        type=int,
        default=BAKE_MAX_POINTS_PER_BATCH,
        help="Maximum voxel-center queries processed per UDF batch",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_NPZ,
        help="Output .npz path when baking from URDF (default: %(default)s)",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=LOAD_NPZ,
        dest="load_path",
        metavar="NPZ",
        help="Load a baked distance field .npz instead of baking from --urdf (default: %(default)s)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=RENDER_2D,
        help="Write slice / sample / sign diagnostic PNGs under --artifact-dir",
    )
    parser.add_argument(
        "--no-render",
        action="store_false",
        dest="render",
        help="Disable 2-D PNG rendering (overrides RENDER_2D=True in config block)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=ARTIFACT_DIR,
        dest="artifact_dir",
        metavar="DIR",
        help="Output directory for rendered artefacts (default: %(default)s)",
    )
    parser.add_argument(
        "--render-plots-npz",
        type=str,
        default=None,
        metavar="NPZ",
        help="Legacy alias: same as --load NPZ with rendering enabled (use --render-plots-out or --artifact-dir for output dir)",
    )
    parser.add_argument(
        "--render-plots-out",
        type=str,
        default=None,
        metavar="DIR",
        help="Legacy alias for --artifact-dir when using --render-plots-npz",
    )
    parser.add_argument(
        "--render-3d",
        action="store_true",
        default=RENDER_3D,
        dest="render_3d",
        help=(
            "Write a 3-view isosurface PNG and (if plotly is installed) an interactive "
            "HTML file.  Requires scikit-image (pip install scikit-image)."
        ),
    )
    parser.add_argument(
        "--no-render-3d",
        action="store_false",
        dest="render_3d",
        help="Disable 3-D rendering (overrides RENDER_3D=True in config block)",
    )
    parser.add_argument(
        "--render-3d-threshold",
        type=float,
        default=RENDER_3D_THRESHOLD,
        dest="render_3d_threshold",
        metavar="DIST",
        help="UDF distance threshold for isosurface extraction in metres (default: %(default)s)",
    )
    parser.add_argument(
        "--render-3d-max-side",
        type=int,
        default=RENDER_3D_MAX_SIDE,
        dest="render_3d_max_side",
        metavar="N",
        help=(
            "Downsample UDF grid so longest axis <= N voxels before marching cubes "
            "(default: %(default)s).  Increase for higher fidelity at the cost of speed."
        ),
    )
    parser.add_argument(
        "--render-3d-heatmap",
        action="store_true",
        default=RENDER_3D_HEATMAP,
        dest="render_3d_heatmap",
        help="Write interactive 3-D volume heatmap HTML (requires plotly)",
    )
    parser.add_argument(
        "--no-render-3d-heatmap",
        action="store_false",
        dest="render_3d_heatmap",
        help="Disable 3-D heatmap (overrides RENDER_3D_HEATMAP=True in config block)",
    )
    parser.add_argument(
        "--heatmap-max-side",
        type=int,
        default=HEATMAP_MAX_SIDE,
        dest="heatmap_max_side",
        metavar="N",
        help="Heatmap grid downsample limit per axis (default: %(default)s)",
    )
    parser.add_argument(
        "--heatmap-dist-max",
        type=float,
        default=HEATMAP_DIST_MAX,
        dest="heatmap_dist_max",
        metavar="D",
        help="Max UDF distance shown in heatmap in metres (default: %(default)s)",
    )
    parser.add_argument(
        "--heatmap-opacity",
        type=float,
        default=HEATMAP_OPACITY,
        dest="heatmap_opacity",
        metavar="A",
        help="Volume opacity 0-1 (default: %(default)s)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    legacy_render_plots = bool(args.render_plots_npz)
    if args.render_plots_npz and not args.load_path:
        args.load_path = args.render_plots_npz
    if args.render_plots_out and args.artifact_dir is None:
        args.artifact_dir = args.render_plots_out

    do_render = bool(args.render or legacy_render_plots)
    output_path = Path(args.output) if args.output else None

    if args.inspect_only:
        if not args.urdf:
            print("error: --urdf is required for --inspect-only", file=sys.stderr)
            return 2
        assy = load_assembly_from_urdf(args.urdf)
        print("link_names:")
        for name in assy.link_names:
            print(f"  {name}")
        print("mesh_paths:")
        for mp in assy.mesh_paths:
            print(f"  {mp}")
        print("bbox_min:", np.asarray(assy.bbox_min).tolist())
        print("bbox_max:", np.asarray(assy.bbox_max).tolist())
        return 0

    field: DistanceField | None = None

    if args.load_path:
        field = load_distance_field(Path(args.load_path))
    elif args.urdf:
        if args.udf_only and not args.output and not do_render:
            print(
                "error: --output and/or --render is required with --udf-only",
                file=sys.stderr,
            )
            return 2
        if not args.udf_only and not args.output and not do_render:
            print(
                "error: no action requested for --urdf input; use --inspect-only, "
                "--output, and/or --render",
                file=sys.stderr,
            )
            return 2
        assy = load_assembly_from_urdf(args.urdf)
        field = bake_distance_field(assy, args)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_distance_field(output_path, field)
    else:
        print(
            "error: --urdf is required unless a default scene URDF is available, "
            "or unless --load / --render-plots-npz is used",
            file=sys.stderr,
        )
        return 2

    assert field is not None

    do_render_3d     = bool(getattr(args, "render_3d", False))
    do_render_heatmap = bool(getattr(args, "render_3d_heatmap", False))

    if do_render or do_render_3d or do_render_heatmap:
        plot_dir = (
            Path(args.artifact_dir)
            if args.artifact_dir
            else _default_artifact_dir(args, output_path)
        )
        plot_dir.mkdir(parents=True, exist_ok=True)

        if do_render:
            render_slice_comparison(field, plot_dir)
            render_sample_point_comparison(field, plot_dir)
            render_sign_diagnostics(field, plot_dir)

        if do_render_3d:
            thr = float(getattr(args, "render_3d_threshold", 0.01))
            ms = int(getattr(args, "render_3d_max_side", 128))
            render_3d_isosurface(field, plot_dir, threshold=thr, max_side=ms)
            render_3d_plotly(field, plot_dir, threshold=thr, max_side=ms)

        if getattr(args, "render_3d_heatmap", False):
            render_3d_heatmap_plotly(
                field,
                plot_dir,
                max_side=int(getattr(args, "heatmap_max_side", HEATMAP_MAX_SIDE)),
                dist_min=float(HEATMAP_DIST_MIN),
                dist_max=float(getattr(args, "heatmap_dist_max", HEATMAP_DIST_MAX)),
                surface_count=int(HEATMAP_SURFACE_COUNT),
                opacity=float(getattr(args, "heatmap_opacity", HEATMAP_OPACITY)),
                overlay_isosurface=bool(HEATMAP_OVERLAY_ISOSURFACE),
                iso_threshold=float(RENDER_3D_THRESHOLD),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
