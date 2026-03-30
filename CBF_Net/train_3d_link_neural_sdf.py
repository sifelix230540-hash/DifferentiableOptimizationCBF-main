"""逐连杆 3D Neural SDF 训练管线。

从 URDF 解析连杆 collision mesh，生成 SDF 训练数据，
为每个连杆训练独立的 NeuralSDF3D 网络，并输出可视化结果。

用法:
    python CBF_Net/train_3d_link_neural_sdf.py \
        --output-dir artifacts/neural_sdf/9_axis_links

默认 --urdf 指向仓库根目录下 assets/robots/9_axis/urdf/9_axis.urdf（与 CWD 无关）。

进度条依赖 tqdm（``pip install tqdm``）；未安装或 ``--no-progress`` 时退化为普通输出。

默认启用 TF32。AMP 需显式 ``--amp`` 开启（高阶导数训练可能不稳定）。
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm import tqdm as _tqdm_bar
except ImportError:
    def _tqdm_bar(iterable, **_kwargs):  # type: ignore[misc]
        return iterable

# 仓库根目录（本文件位于 CBF_Net/ 下）
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_URDF = _REPO_ROOT / "assets/robots/9_axis/urdf/9_axis.urdf"

# ---------------------------------------------------------------------------
#  默认连杆白名单（后六轴 + 焊枪 + 焊点）
# ---------------------------------------------------------------------------

DEFAULT_LINK_NAMES: list[str] = [
    "robobase",
    "link04", "link05", "link06",
    "link07", "link08", "link09",
    "welding_gun_base", "weld_point",
]


# ========================================================================== #
#  1. URDF 解析                                                              #
# ========================================================================== #

@dataclass
class LinkMeshInfo:
    link_name: str
    mesh_path: Path
    collision_origin_xyz: tuple[float, ...] = (0.0, 0.0, 0.0)
    collision_origin_rpy: tuple[float, ...] = (0.0, 0.0, 0.0)


def parse_urdf_links(
    urdf_path: str | Path,
    link_names: Sequence[str] | None = None,
) -> list[LinkMeshInfo]:
    """解析 URDF 中每个 link 的 collision mesh 路径。

    ``package://9_axis/meshes/xxx.STL`` 会被解析为相对于 URDF 所在目录
    的上一级 + ``meshes/xxx.STL``（即 ``<urdf_dir>/../meshes/xxx.STL``）。
    同时自动检测同名 ``.obj`` 文件作为后备。
    """
    urdf_path = Path(urdf_path).resolve()
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    results: list[LinkMeshInfo] = []
    for link_elem in root.findall("link"):
        name = link_elem.get("name", "")
        if link_names is not None and name not in link_names:
            continue

        collision = link_elem.find("collision")
        if collision is None:
            continue
        geom = collision.find("geometry")
        if geom is None:
            continue
        mesh_elem = geom.find("mesh")
        if mesh_elem is None:
            continue

        filename = mesh_elem.get("filename", "")

        if filename.startswith("package://"):
            parts = filename.replace("package://", "").split("/", 1)
            rel = parts[1] if len(parts) == 2 else parts[0]
            mesh_path = (urdf_path.parent.parent / rel).resolve()
        else:
            mesh_path = (urdf_path.parent / filename).resolve()

        # 如果 STL 不存在，尝试 .obj 后备
        if not mesh_path.exists():
            alt = mesh_path.with_suffix(".obj")
            if alt.exists():
                mesh_path = alt

        origin = collision.find("origin")
        xyz = (0.0, 0.0, 0.0)
        rpy = (0.0, 0.0, 0.0)
        if origin is not None:
            xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
            rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split())

        results.append(LinkMeshInfo(
            link_name=name,
            mesh_path=mesh_path,
            collision_origin_xyz=xyz,
            collision_origin_rpy=rpy,
        ))
    return results


# ========================================================================== #
#  2. SDF 训练数据生成                                                        #
# ========================================================================== #

@dataclass
class SDFDataset:
    surface_points: np.ndarray  # (M, 3) float32
    surface_normals: np.ndarray # (M, 3) float32
    aabb_min: np.ndarray        # (3,) 归一化空间的采样下界
    aabb_max: np.ndarray        # (3,) 归一化空间的采样上界
    center: np.ndarray          # (3,) 原始 mesh 质心
    scale: float                # 归一化缩放因子


def _try_repair_mesh(mesh):
    """尝试修补非封闭 mesh 以改善 signed-distance 计算。"""
    import trimesh
    if not mesh.is_watertight:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
    return mesh


def generate_pointcloud_data(
    mesh_path: Path,
    n_surface: int = 30_000,
    grid_range: float = 1.1,
) -> tuple[SDFDataset, dict]:
    """为无监督训练生成点云数据 (只采样表面点+法向，不计算 SDF 标签)。

    对齐 StEik 源码: 归一化到 [-1,1] 后只保留 surface points + normals。
    """
    import trimesh

    mesh: trimesh.Trimesh = trimesh.load(str(mesh_path), force="mesh")
    if hasattr(mesh, "fix_normals"):
        mesh.fix_normals()
    mesh = _try_repair_mesh(mesh)

    n_verts = int(mesh.vertices.shape[0])
    n_faces = int(mesh.faces.shape[0])

    center = mesh.bounding_box.centroid.copy()
    extent = np.max(mesh.bounding_box.extents) / 2.0
    scale = float(extent) if extent > 1e-8 else 1.0

    mesh_n = mesh.copy()
    mesh_n.apply_translation(-center)
    mesh_n.apply_scale(1.0 / scale)

    surface_pts, face_ids = trimesh.sample.sample_surface(mesh_n, n_surface)
    surface_pts = surface_pts.astype(np.float32)
    surface_normals = mesh_n.face_normals[face_ids].astype(np.float32)

    # 归一化法向
    norms = np.linalg.norm(surface_normals, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    surface_normals = surface_normals / norms

    aabb_lo = np.full(3, -grid_range, dtype=np.float32)
    aabb_hi = np.full(3, grid_range, dtype=np.float32)

    dataset = SDFDataset(
        surface_points=surface_pts,
        surface_normals=surface_normals,
        aabb_min=aabb_lo,
        aabb_max=aabb_hi,
        center=center.astype(np.float32),
        scale=scale,
    )
    meta = dict(
        n_vertices=n_verts, n_faces=n_faces,
        n_train_points=n_surface,
        is_watertight=bool(mesh_n.is_watertight),
    )
    return dataset, meta


# ========================================================================== #
#  3. 网络结构                                                                #
# ========================================================================== #

class FourierFeatures(nn.Module):
    """Positional encoding: x -> [x, sin(2^k pi x), cos(2^k pi x)]."""
    def __init__(self, input_dim: int = 3, n_frequencies: int = 6):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.output_dim = input_dim * (1 + 2 * n_frequencies)
        freqs = 2.0 ** torch.arange(n_frequencies, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x]
        for f in self.freqs:
            parts.append(torch.sin(f * torch.pi * x))
            parts.append(torch.cos(f * torch.pi * x))
        return torch.cat(parts, dim=-1)


class SinActivation(nn.Module):
    """正弦激活函数 (SIREN 风格), omega_0 控制初始频率。"""
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class QuadraticLayer(nn.Module):
    """二次神经元层 (StEik, NeurIPS 2023).

    a(x) = (W1x + b1) ◦ (W2x + b2) + W3x² + b3

    关键初始化 (来自 StEik 源码 init_lin2_lin3):
      lin2.weight ≈ 0 (std=1e-5), lin2.bias = 1
      lin3.weight ≈ 0 (std=1e-5), lin3.bias = 0
    这使得层在训练初期近似于 lin1(x)·1 + 0 = lin1(x)，保证训练稳定性。
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin1 = nn.Linear(in_features, out_features)
        self.lin2 = nn.Linear(in_features, out_features)
        self.lin3 = nn.Linear(in_features, out_features)
        self._init_lin2_lin3()

    @torch.no_grad()
    def _init_lin2_lin3(self):
        """StEik 源码中的关键初始化: 让二次项和乘法项在初始时近似为 identity。"""
        nn.init.normal_(self.lin2.weight, mean=0.0, std=1e-5)
        nn.init.ones_(self.lin2.bias)
        nn.init.normal_(self.lin3.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.lin3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin1(x) * self.lin2(x) + self.lin3(torch.square(x))


class NeuralSDF3D(nn.Module):
    """R^3 -> R 的 Neural SDF 网络。

    Args:
        hidden: 隐藏层宽度
        n_layers: 隐藏层数
        n_frequencies: Fourier 编码频率数
        use_fourier: 是否启用 Fourier 位置编码
        quadratic: 使用 QuadraticLayer 替代 nn.Linear (StEik)
        activation: "softplus" 或 "sin" (SIREN)
        omega_0: SinActivation 的频率参数
        init_type: 初始化方式 "siren" | "mfgi" | "xavier"
    """
    def __init__(
        self,
        hidden: int = 256,
        n_layers: int = 8,
        n_frequencies: int = 6,
        use_fourier: bool = False,
        quadratic: bool = True,
        activation: str = "softplus",
        omega_0: float = 30.0,
        init_type: str = "mfgi",
        sphere_init_params: tuple[float, float] = (1.6, 1.0),
    ):
        super().__init__()
        self._use_sphere_transform = init_type in ("mfgi", "geometric_sine")
        self._sphere_radius = sphere_init_params[0]
        self._sphere_scaling = sphere_init_params[1]
        self.config = dict(
            hidden=hidden, n_layers=n_layers,
            n_frequencies=n_frequencies, use_fourier=use_fourier,
            quadratic=quadratic, activation=activation, omega_0=omega_0,
            init_type=init_type,
            sphere_init_params=list(sphere_init_params),
        )
        if use_fourier:
            self.encoding = FourierFeatures(3, n_frequencies)
            in_dim = self.encoding.output_dim
        else:
            self.encoding = None
            in_dim = 3

        dims = [in_dim] + [hidden for _ in range(n_layers)] + [1]
        layers: list[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            if quadratic:
                layers.append(QuadraticLayer(d_in, d_out))
            else:
                layers.append(nn.Linear(d_in, d_out))
            if i != len(dims) - 2:
                if activation == "sin":
                    layers.append(SinActivation(omega_0))
                else:
                    layers.append(nn.Softplus(beta=100))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  MFGI 多频率几何初始化 (来自 StEik/DiGS 源码)
    # ------------------------------------------------------------------ #
    _mfgi_periods = [1, 30]
    _mfgi_portion = np.array([0.25, 0.75])

    def _init_weights(self):
        init_type = self.config.get("init_type", "xavier")
        act = self.config.get("activation", "softplus")
        omega_0 = self.config.get("omega_0", 30.0)
        quadratic = self.config.get("quadratic", False)

        if act == "sin":
            if init_type == "mfgi" and quadratic:
                self._init_mfgi_quadratic()
            elif init_type == "mfgi" and not quadratic:
                self._init_mfgi_linear()
            else:
                self._init_siren(omega_0, quadratic)
        else:
            if init_type == "mfgi" and quadratic:
                self._init_mfgi_quadratic()
            elif init_type == "mfgi" and not quadratic:
                self._init_mfgi_linear()
            else:
                self._init_xavier(quadratic)

    def _get_layer_modules(self):
        """提取网络中的线性/二次层列表 (不含激活函数)。"""
        return [m for m in self.net if isinstance(m, (nn.Linear, QuadraticLayer))]

    @torch.no_grad()
    def _init_xavier(self, quadratic: bool):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, QuadraticLayer):
                nn.init.xavier_uniform_(m.lin1.weight)
                nn.init.zeros_(m.lin1.bias)
                m._init_lin2_lin3()

    @torch.no_grad()
    def _init_siren(self, omega_0: float, quadratic: bool):
        first_layer = True
        for m in self.net:
            if isinstance(m, QuadraticLayer):
                fan_in = m.in_features
                if first_layer:
                    nn.init.uniform_(m.lin1.weight, -1.0 / fan_in, 1.0 / fan_in)
                    first_layer = False
                else:
                    bound = np.sqrt(6.0 / fan_in) / omega_0
                    nn.init.uniform_(m.lin1.weight, -bound, bound)
                nn.init.zeros_(m.lin1.bias)
                m._init_lin2_lin3()
            elif isinstance(m, nn.Linear):
                fan_in = m.in_features
                if first_layer:
                    nn.init.uniform_(m.weight, -1.0 / fan_in, 1.0 / fan_in)
                    first_layer = False
                else:
                    bound = np.sqrt(6.0 / fan_in) / omega_0
                    nn.init.uniform_(m.weight, -bound, bound)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _init_mfgi_quadratic(self):
        """StEik 源码中的 MFGI 初始化 (用于 QuadraticLayer)。"""
        layer_modules = self._get_layer_modules()
        n = len(layer_modules)
        for i, m in enumerate(layer_modules):
            if isinstance(m, QuadraticLayer):
                num_out = m.lin1.weight.size(0)
                num_in = m.lin1.weight.size(-1)
                if i == 0:
                    num_per_period = (self._mfgi_portion * num_out).astype(int)
                    weights = []
                    for j, period in enumerate(self._mfgi_periods):
                        scale = 30.0 / period
                        w = torch.zeros(num_per_period[j], num_in).uniform_(
                            -np.sqrt(3.0 / num_in) / scale,
                            np.sqrt(3.0 / num_in) / scale,
                        )
                        weights.append(w)
                    m.lin1.weight.data = torch.cat(weights, dim=0)
                    m._init_lin2_lin3()
                elif i == 1:
                    num_per_period = (self._mfgi_portion * num_in).astype(int)
                    k = num_per_period[0]
                    W1_new = torch.zeros(num_out, num_in).uniform_(
                        -np.sqrt(3.0 / num_in),
                        np.sqrt(3.0 / num_in) / 30,
                    ) * 0.0005
                    W1_new_1 = torch.zeros(k, k).uniform_(
                        -np.sqrt(3.0 / num_in) / 30,
                        np.sqrt(3.0 / num_in) / 30,
                    )
                    W1_new[:k, :k] = W1_new_1
                    m.lin1.weight.data = W1_new
                    m._init_lin2_lin3()
                elif i == n - 2:
                    m.lin1.weight.data = (
                        0.5 * np.pi * torch.eye(num_out)
                        + 0.001 * torch.randn(num_out, num_out)
                    )
                    m.lin1.bias.data = (
                        0.5 * np.pi * torch.ones(num_out)
                        + 0.001 * torch.randn(num_out)
                    )
                    m.lin1.weight.data /= 30
                    m.lin1.bias.data /= 30
                    m._init_lin2_lin3()
                elif i == n - 1 and isinstance(m, QuadraticLayer):
                    m.lin1.weight.data = (
                        -1 * torch.ones(1, num_in) + 1e-5 * torch.randn(num_in)
                    )
                    m.lin1.bias.data = torch.zeros(1) + num_in
                    m._init_lin2_lin3()
                else:
                    m.lin1.weight.uniform_(
                        -np.sqrt(3.0 / num_out), np.sqrt(3.0 / num_out)
                    )
                    m.lin1.bias.uniform_(
                        -1.0 / (num_out * 1000), 1.0 / (num_out * 1000)
                    )
                    m.lin1.weight.data /= 30
                    m.lin1.bias.data /= 30
                    m._init_lin2_lin3()
            elif isinstance(m, nn.Linear):
                if i == n - 1:
                    num_in = m.weight.size(-1)
                    m.weight.data = (
                        -1 * torch.ones(1, num_in) + 1e-5 * torch.randn(num_in)
                    )
                    m.bias.data = torch.zeros(1) + num_in
                else:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _init_mfgi_linear(self):
        """MFGI 初始化 (用于普通 nn.Linear 层)。"""
        layer_modules = self._get_layer_modules()
        n = len(layer_modules)
        for i, m in enumerate(layer_modules):
            if not isinstance(m, nn.Linear):
                continue
            num_out = m.weight.size(0)
            num_in = m.weight.size(-1)
            if i == 0:
                num_per_period = (self._mfgi_portion * num_out).astype(int)
                weights = []
                for j, period in enumerate(self._mfgi_periods):
                    scale = 30.0 / period
                    w = torch.zeros(num_per_period[j], num_in).uniform_(
                        -np.sqrt(3.0 / num_in) / scale,
                        np.sqrt(3.0 / num_in) / scale,
                    )
                    weights.append(w)
                m.weight.data = torch.cat(weights, dim=0)
            elif i == 1:
                num_per_period = (self._mfgi_portion * num_in).astype(int)
                k = num_per_period[0]
                W1_new = torch.zeros(num_out, num_in).uniform_(
                    -np.sqrt(3.0 / num_in),
                    np.sqrt(3.0 / num_in) / 30,
                ) * 0.0005
                W1_new_1 = torch.zeros(k, k).uniform_(
                    -np.sqrt(3.0 / num_in) / 30,
                    np.sqrt(3.0 / num_in) / 30,
                )
                W1_new[:k, :k] = W1_new_1
                m.weight.data = W1_new
            elif i == n - 2:
                m.weight.data = (
                    0.5 * np.pi * torch.eye(num_out)
                    + 0.001 * torch.randn(num_out, num_out)
                )
                m.bias.data = (
                    0.5 * np.pi * torch.ones(num_out)
                    + 0.001 * torch.randn(num_out)
                )
                m.weight.data /= 30
                m.bias.data /= 30
            elif i == n - 1:
                m.weight.data = (
                    -1 * torch.ones(1, num_in) + 1e-5 * torch.randn(num_in)
                )
                m.bias.data = torch.zeros(1) + num_in
            else:
                m.weight.uniform_(
                    -np.sqrt(3.0 / num_out), np.sqrt(3.0 / num_out)
                )
                m.bias.uniform_(
                    -1.0 / (num_out * 1000), 1.0 / (num_out * 1000)
                )
                m.weight.data /= 30
                m.bias.data /= 30

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding is not None:
            x = self.encoding(x)
        out = self.net(x).squeeze(-1)
        if self._use_sphere_transform:
            out = torch.sign(out) * torch.sqrt(out.abs() + 1e-8)
            out = (out - self._sphere_radius) * self._sphere_scaling
        return out

    @staticmethod
    def from_checkpoint(ckpt_path: str | Path, device: str = "cpu") -> "NeuralSDF3D":
        """从 checkpoint 文件恢复模型。"""
        data = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = data["model_config"]
        model = NeuralSDF3D(**cfg)
        model.load_state_dict(data["model_state_dict"])
        model.to(device)
        model.eval()
        return model


# ========================================================================== #
#  4. 训练                                                                    #
# ========================================================================== #


def _build_div_decay_knots(params: Sequence[float]) -> list[tuple[float, float]]:
    """StEik losses.py 结点格式: ``w0, t1, w1, t2, w2, ..., w_end``。

    ``t`` 为训练进度比例，首末时刻固定为 0 与 1。中间须为成对的 (时刻, 权重)。
    """
    p = tuple(float(x) for x in params)
    if len(p) < 2:
        raise ValueError(
            "div_decay_params 至少需要 2 个数: w_start, w_end。"
        )
    mid = p[1:-1]
    if len(mid) % 2 != 0:
        raise ValueError(
            "div_decay_params 须为 StEik 格式: w0 t1 w1 t2 w2 ... w_end "
            "(中间为成对的归一化时刻、散度项权重)。"
            " 示例: --div-decay-params 1e1 0.5 1e1 0.75 0 0"
        )
    weights = [p[0], *mid[1::2], p[-1]]
    times = [0.0, *mid[::2], 1.0]
    if len(weights) != len(times):
        raise RuntimeError("div_decay_params 内部解析失败: 时刻与权重数量不一致。")
    for i in range(1, len(times)):
        if times[i] + 1e-15 < times[i - 1]:
            raise ValueError("div_decay_params: 时刻须非递减。")
    return list(zip(weights, times))


def _div_weight_at_iteration(
    iteration: int,
    n_iterations: int,
    div_decay: str,
    knots: list[tuple[float, float]],
) -> float:
    """按 StEik ``update_div_weight`` 计算当前迭代的散度项标量权重。"""
    n_it = max(n_iterations, 1)
    curr = iteration / n_it
    upward = [tup for tup in knots if tup[1] >= curr - 1e-15]
    downward = [tup for tup in knots if tup[1] <= curr + 1e-15]
    if not upward:
        upward = [knots[-1]]
    if not downward:
        downward = [knots[0]]
    we, e = min(upward, key=lambda t: t[1])
    w0, s = max(downward, key=lambda t: t[1])
    ci = iteration

    if div_decay == "linear":
        if ci < s * n_it:
            return w0
        if ci < e * n_it and abs(e - s) > 1e-18:
            return w0 + (we - w0) * (ci / n_it - s) / (e - s)
        return we
    if div_decay == "quintic":
        if ci < s * n_it:
            return w0
        if ci < e * n_it and abs(e - s) > 1e-18:
            t = (ci / n_it - s) / (e - s)
            return w0 + (we - w0) * (1.0 - (1.0 - t) ** 5)
        return we
    if div_decay == "step":
        if ci < s * n_it:
            return w0
        return we
    raise ValueError(f"不支持的 div_decay: {div_decay}")


def _directional_divergence(
    points: torch.Tensor,
    grads: torch.Tensor,
) -> torch.Tensor:
    """Directional divergence, 对齐 StEik losses.directional_div。"""
    dot_grad = torch.sum(grads * grads, dim=-1, keepdim=True)
    hvp = 0.5 * torch.autograd.grad(
        dot_grad,
        points,
        grad_outputs=torch.ones_like(dot_grad),
        retain_graph=True,
        create_graph=True,
    )[0]
    return torch.sum(grads * hvp, dim=-1) / (
        torch.sum(grads * grads, dim=-1) + 1e-5
    )


@dataclass
class TrainResult:
    model: NeuralSDF3D
    loss_history: list[dict] = field(default_factory=list)
    final_loss: float = float("inf")
    best_loss: float = float("inf")
    elapsed_seconds: float = 0.0
    iterations_run: int = 0
    stop_reason: str = "max_iterations"


def train_link_sdf_unsupervised(
    dataset: SDFDataset,
    *,
    n_iterations: int = 5000,
    n_points: int = 30000,
    lr: float = 1e-4,
    device: str = "cpu",
    loss_weights: tuple[float, ...] = (3e3, 1e2, 1e2, 5e1, 1e1),
    loss_type: str = "siren_wo_n_w_div",
    div_decay: str = "linear",
    div_decay_params: tuple[float, ...] = (1e1, 0.5, 1e1, 0.75, 0.0, 0.0),
    div_type: str = "dir_l1",
    eikonal_type: str = "abs",
    hidden: int = 256,
    n_layers: int = 8,
    n_frequencies: int = 6,
    use_fourier: bool = False,
    quadratic: bool = True,
    activation: str = "softplus",
    omega_0: float = 30.0,
    init_type: str = "mfgi",
    sphere_init_params: tuple[float, float] = (1.6, 0.1),
    grad_clip: float = 10.0,
    grid_range: float = 1.1,
    show_progress: bool = True,
    progress_desc: str | None = None,
    use_amp: bool = False,
    amp_dtype: str = "float16",
    use_tf32: bool = True,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    lr_schedule: str = "none",
    lr_min_ratio: float = 0.3,
    plateau_factor: float = 0.5,
    plateau_patience: int = 300,
    plateau_threshold: float = 1e-4,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    early_stop_warmup: int = 0,
    restore_best: bool = True,
) -> TrainResult:
    """无监督训练 (完全对齐 StEik 源码).

    不需要 SDF 标签，只用表面点+法向 + 空间均匀采样点。
    损失结构:
      L = w_sdf * |f(x_surf)| + w_inter * exp(-100|f(x_off)|)
        + w_normal * (1 - |cos(∇f, n)|) + w_eik * |‖∇f‖ - 1|
        + w_div * directional_div
    """
    w_sdf, w_inter, w_normal, w_eikonal, w_div = loss_weights
    use_div_loss = ("div" in loss_type) and (w_div > 0)
    use_normal_loss = ("wo_n" not in loss_type)
    use_inter_loss = ("igr" not in loss_type)

    is_cuda = str(device).startswith("cuda")
    requested_amp = bool(use_amp and is_cuda)
    # 当前目标包含 create_graph=True 的高阶导数与 HVP，半精度极易数值不稳定。
    # 为避免训练中出现 NaN/Inf，这里默认强制关闭 AMP。
    use_amp = False
    if requested_amp:
        print("  [警告] 检测到高阶导数训练，已自动关闭 AMP 以避免 NaN。")
    adtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cudnn.benchmark = True

    model = NeuralSDF3D(
        hidden, n_layers, n_frequencies, use_fourier,
        quadratic=quadratic, activation=activation, omega_0=omega_0,
        init_type=init_type,
        sphere_init_params=sphere_init_params,
    ).to(device)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    scheduler = None
    scheduler_on_metric = False
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, n_iterations),
            eta_min=lr * max(0.0, float(lr_min_ratio)),
        )
    elif lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=max(1, plateau_patience),
            threshold=plateau_threshold,
            min_lr=lr * max(0.0, float(lr_min_ratio)),
        )
        scheduler_on_metric = True
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    surf_pts = torch.as_tensor(
        dataset.surface_points, dtype=torch.float32, device=device
    )
    surf_normals = torch.as_tensor(
        dataset.surface_normals, dtype=torch.float32, device=device
    )
    n_surf = surf_pts.shape[0]

    div_knots: list[tuple[float, float]] | None = None
    if div_decay != "none":
        div_knots = _build_div_decay_knots(div_decay_params)
        p0 = float(div_decay_params[0])
        if abs(p0) > 1e-30:
            scale = w_div / p0
            div_knots = [(w * scale, t) for w, t in div_knots]

    log_interval = max(1, n_iterations // 20)
    loss_history: list[dict] = []
    t0 = time.time()

    it_range = range(n_iterations)
    if show_progress:
        it_range = _tqdm_bar(
            it_range,
            desc=progress_desc or "SDF",
            total=n_iterations,
            unit="it",
            dynamic_ncols=True,
            mininterval=0.2,
            leave=False,
        )

    amp_ctx = contextlib.nullcontext()
    if use_amp:
        if hasattr(torch, "autocast"):
            amp_ctx = torch.autocast(device_type="cuda", dtype=adtype)
        else:
            amp_ctx = torch.cuda.amp.autocast(dtype=adtype)

    best_loss = float("inf")
    no_improve_steps = 0
    iterations_run = 0
    stop_reason = "max_iterations"
    best_state_dict: dict | None = None

    for iteration in it_range:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # 随机采样 manifold (表面) 点
        mnfld_idx = torch.randint(0, n_surf, (n_points,), device=device)
        mnfld_pts = surf_pts[mnfld_idx].detach().clone()
        mnfld_pts.requires_grad_(True)
        mnfld_n_gt = surf_normals[mnfld_idx]

        # 均匀采样 non-manifold 点（直接在 device 上分配，避免 CPU->GPU 拷贝）
        nonmnfld_pts = torch.empty(
            (n_points, 3), device=device, dtype=torch.float32
        )
        nonmnfld_pts.uniform_(-grid_range, grid_range)
        nonmnfld_pts.requires_grad_(True)

        with amp_ctx:
            # forward
            mnfld_pred = model(mnfld_pts)
            nonmnfld_pred = model(nonmnfld_pts)

            # gradients
            mnfld_grad, nonmnfld_grad = torch.autograd.grad(
                outputs=[mnfld_pred.sum(), nonmnfld_pred.sum()],
                inputs=[mnfld_pts, nonmnfld_pts],
                create_graph=True, retain_graph=True,
            )

            # ----- 1. SDF term: 表面点预测值应为 0 -----
            sdf_term = torch.abs(mnfld_pred).mean()

            # ----- 2. Inter term: 防止空间点全部坍塌到 0 -----
            inter_term = torch.tensor(0.0, device=device)
            if use_inter_loss:
                inter_term = torch.exp(-1e2 * torch.abs(nonmnfld_pred)).mean()

            # ----- 3. Normal term: 表面梯度方向应与法向一致 -----
            normal_term = torch.tensor(0.0, device=device)
            if use_normal_loss:
                normal_term = (
                    1.0 - torch.abs(
                        torch.nn.functional.cosine_similarity(
                            mnfld_grad, mnfld_n_gt, dim=-1
                        )
                    )
                ).mean()

            # ----- 4. Eikonal term -----
            all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=0)
            if eikonal_type == "abs":
                eikonal_term = (all_grads.norm(2, dim=-1) - 1).abs().mean()
            else:
                eikonal_term = ((all_grads.norm(2, dim=-1) - 1) ** 2).mean()

            # ----- 5. Divergence term -----
            div_loss = torch.tensor(0.0, device=device)
            if use_div_loss:
                div_val = _directional_divergence(nonmnfld_pts, nonmnfld_grad)
                if div_type == "dir_l1":
                    div_loss = torch.abs(div_val).mean()
                elif div_type == "dir_l2":
                    div_loss = torch.square(div_val).mean()
            if loss_type == "siren":
                total = (w_sdf * sdf_term + w_inter * inter_term
                         + w_normal * normal_term + w_eikonal * eikonal_term)
            elif loss_type == "siren_wo_n":
                total = w_sdf * sdf_term + w_inter * inter_term + w_eikonal * eikonal_term
            elif loss_type == "igr":
                total = w_sdf * sdf_term + w_normal * normal_term + w_eikonal * eikonal_term
            elif loss_type == "igr_wo_n":
                total = w_sdf * sdf_term + w_eikonal * eikonal_term
            elif loss_type == "siren_w_div":
                total = (w_sdf * sdf_term + w_inter * inter_term
                         + w_normal * normal_term + w_eikonal * eikonal_term
                         + w_div * div_loss)
            elif loss_type == "siren_wo_n_w_div":
                total = (w_sdf * sdf_term + w_inter * inter_term
                         + w_eikonal * eikonal_term + w_div * div_loss)
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

        if not torch.isfinite(total):
            raise RuntimeError(
                "检测到非有限损失: "
                f"iter={iteration}, total={total.item()}, "
                f"sdf={sdf_term.item()}, inter={inter_term.item()}, "
                f"normal={normal_term.item()}, eik={eikonal_term.item()}, "
                f"div={div_loss.item()}"
            )

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_val = float(total.item())
        if scheduler is not None:
            if scheduler_on_metric:
                scheduler.step(total_val)
            else:
                scheduler.step()
        if total_val < best_loss - early_stop_min_delta:
            best_loss = total_val
            no_improve_steps = 0
            if restore_best:
                best_state_dict = copy.deepcopy(model.state_dict())
        elif iteration >= max(0, early_stop_warmup):
            no_improve_steps += 1

        if show_progress and hasattr(it_range, "set_postfix"):
            it_range.set_postfix(
                L=f"{total_val:.3e}",
                eik=f"{eikonal_term.item():.2e}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                refresh=False,
            )

        # ----- div weight decay（StEik losses.update_div_weight，每步末更新供下一轮使用）-----
        if div_knots is not None and len(div_knots) >= 2:
            w_div = _div_weight_at_iteration(
                iteration, n_iterations, div_decay, div_knots
            )

        if iteration % log_interval == 0 or iteration == n_iterations - 1:
            loss_history.append(dict(
                epoch=iteration,
                loss_total=total.item(),
                loss_sdf=sdf_term.item(),
                loss_eikonal=eikonal_term.item(),
                loss_dd=div_loss.item(),
                loss_normal=normal_term.item(),
                loss_inter=inter_term.item(),
            ))
        iterations_run = iteration + 1

        if early_stop_patience > 0 and no_improve_steps >= early_stop_patience:
            stop_reason = "early_stop"
            loss_history.append(dict(
                epoch=iteration,
                loss_total=total_val,
                loss_sdf=sdf_term.item(),
                loss_eikonal=eikonal_term.item(),
                loss_dd=div_loss.item(),
                loss_normal=normal_term.item(),
                loss_inter=inter_term.item(),
            ))
            break

    if restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    elapsed = time.time() - t0
    return TrainResult(
        model=model,
        loss_history=loss_history,
        final_loss=loss_history[-1]["loss_total"] if loss_history else float("inf"),
        best_loss=best_loss if best_loss < float("inf") else float("inf"),
        elapsed_seconds=elapsed,
        iterations_run=iterations_run,
        stop_reason=stop_reason,
    )


# ========================================================================== #
#  5. 可视化                                                                  #
# ========================================================================== #

def _save_loss_curve(history: list[dict], path: Path):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, [h["loss_total"]    for h in history], label="Total",    lw=2)
    ax.semilogy(epochs, [h["loss_sdf"]      for h in history], label="SDF",      lw=1.5, ls="--")
    ax.semilogy(epochs, [h["loss_eikonal"]  for h in history], label="Eikonal",  lw=1.5, ls=":")
    dd_vals = [h.get("loss_dd", 0.0) for h in history]
    if any(v > 0 for v in dd_vals):
        ax.semilogy(epochs, [max(v, 1e-12) for v in dd_vals],
                    label="Dir. Div. (LL.n.)", lw=1.5, ls="-.")
    normal_vals = [h.get("loss_normal", 0.0) for h in history]
    if any(v > 0 for v in normal_vals):
        ax.semilogy(epochs, [max(v, 1e-12) for v in normal_vals],
                    label="Normal", lw=1.5, ls=(0, (3, 1, 1, 1)))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_sdf_slices(
    model: NeuralSDF3D,
    dataset: SDFDataset,
    path: Path,
    resolution: int = 200,
    device: str = "cpu",
):
    """XY / XZ / YZ 三组正交切片的 SDF 热力图 + 零等值线。"""
    model.eval()
    lo, hi = dataset.aabb_min, dataset.aabb_max
    mid = (lo + hi) / 2.0

    slices = [
        ("XY  (z = mid)", 0, 1, 2),
        ("XZ  (y = mid)", 0, 2, 1),
        ("YZ  (x = mid)", 1, 2, 0),
    ]
    dim_label = "XYZ"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (title, da, db, df) in zip(axes, slices):
        a = np.linspace(lo[da], hi[da], resolution)
        b = np.linspace(lo[db], hi[db], resolution)
        aa, bb = np.meshgrid(a, b)

        grid = np.zeros((resolution * resolution, 3), dtype=np.float32)
        grid[:, da] = aa.ravel()
        grid[:, db] = bb.ravel()
        grid[:, df] = mid[df]

        with torch.no_grad():
            sdf = model(torch.tensor(grid, device=device)).cpu().numpy()
        sdf = sdf.reshape(resolution, resolution)

        vmax = float(np.percentile(np.abs(sdf), 95)) or 1.0
        c = ax.contourf(aa, bb, sdf, levels=50, cmap="RdBu", vmin=-vmax, vmax=vmax)
        fig.colorbar(c, ax=ax, shrink=0.8)
        ax.contour(aa, bb, sdf, levels=[0.0], colors="lime", linewidths=2)

        sp = dataset.surface_points
        tol = (hi[df] - lo[df]) * 0.05
        mask = np.abs(sp[:, df] - mid[df]) < tol
        if mask.sum() > 0:
            ax.scatter(sp[mask, da], sp[mask, db], s=0.5, c="black", alpha=0.3)

        ax.set_xlabel(dim_label[da])
        ax.set_ylabel(dim_label[db])
        ax.set_title(title)
        ax.set_aspect("equal")

    fig.suptitle("Learned SDF slices  (green = zero level set)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_surface_eval(
    model: NeuralSDF3D,
    dataset: SDFDataset,
    path: Path,
    device: str = "cpu",
):
    """表面点上的预测值分布 + 梯度模长分布。"""
    model.eval()
    sp = torch.tensor(dataset.surface_points, dtype=torch.float32, device=device)
    sp.requires_grad_(True)
    pred = model(sp)
    grad = torch.autograd.grad(
        pred, sp, grad_outputs=torch.ones_like(pred), create_graph=False,
    )[0]
    grad_norm = torch.linalg.norm(grad, dim=1).detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(pred_np, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", lw=2, ls="--")
    ax.set_xlabel("Predicted SDF at surface")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Surface SDF  (ideal = 0)\n"
        f"mean={pred_np.mean():.4f}  std={pred_np.std():.4f}"
    )

    ax = axes[1]
    ax.hist(grad_norm, bins=80, color="darkorange", edgecolor="white", alpha=0.8)
    ax.axvline(1, color="red", lw=2, ls="--")
    ax.set_xlabel("||grad f|| at surface")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Gradient norm  (ideal = 1)\n"
        f"mean={grad_norm.mean():.4f}  std={grad_norm.std():.4f}"
    )

    fig.suptitle("Surface Point Evaluation", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_isosurface(
    model: NeuralSDF3D,
    dataset: SDFDataset,
    png_path: Path,
    obj_path: Path | None = None,
    resolution: int = 80,
    device: str = "cpu",
):
    """用 marching cubes 提取零等值面，多视角渲染 + 可选导出 OBJ。"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("    [跳过 isosurface] 需要 scikit-image (pip install scikit-image)")
        return

    model.eval()
    lo, hi = dataset.aabb_min, dataset.aabb_max
    spacing = (hi - lo) / (resolution - 1)

    xs = np.linspace(lo[0], hi[0], resolution)
    ys = np.linspace(lo[1], hi[1], resolution)
    zs = np.linspace(lo[2], hi[2], resolution)

    # 分批评估避免显存不足
    volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    for i, x_val in enumerate(xs):
        plane = np.zeros((resolution * resolution, 3), dtype=np.float32)
        yy, zz = np.meshgrid(ys, zs, indexing="ij")
        plane[:, 0] = x_val
        plane[:, 1] = yy.ravel()
        plane[:, 2] = zz.ravel()
        with torch.no_grad():
            volume[i] = model(
                torch.tensor(plane, device=device)
            ).cpu().numpy().reshape(resolution, resolution)

    try:
        verts, faces, normals, _ = marching_cubes(
            volume, level=0.0, spacing=tuple(spacing.tolist()),
        )
    except (ValueError, RuntimeError):
        print("    [跳过 isosurface] 体积数据中未找到零交叉面")
        return

    verts += lo  # 平移到归一化坐标系

    # ---- 导出 OBJ (归一化坐标 + 原始坐标两份) ----
    if obj_path is not None:
        with open(obj_path, "w") as f:
            f.write(f"# Neural SDF isosurface (normalized coords)\n")
            f.write(f"# center = {dataset.center.tolist()}\n")
            f.write(f"# scale  = {dataset.scale}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for tri in faces:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    # ---- 多视角渲染 ----
    view_configs = [
        (25, -45, "Perspective"),
        (0,    0, "Front  (YZ)"),
        (90,   0, "Top    (XY)"),
        (0,   90, "Side   (XZ)"),
    ]
    fig = plt.figure(figsize=(22, 5))

    # 为全局坐标轴范围统一
    x_range = float(verts[:, 0].max() - verts[:, 0].min()) or 1.0
    y_range = float(verts[:, 1].max() - verts[:, 1].min()) or 1.0
    z_range = float(verts[:, 2].max() - verts[:, 2].min()) or 1.0
    max_range = max(x_range, y_range, z_range) / 2.0
    mid = verts.mean(axis=0)

    # 面数太多时子采样以加速渲染
    max_faces_render = 30_000
    if len(faces) > max_faces_render:
        idx = np.random.choice(len(faces), max_faces_render, replace=False)
        render_faces = faces[idx]
    else:
        render_faces = faces

    for i, (elev, azim, title) in enumerate(view_configs):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")

        # 渲染三角面
        tri_verts = verts[render_faces]
        poly = Poly3DCollection(
            tri_verts, alpha=0.6, edgecolor=(0.2, 0.2, 0.2, 0.05),
            facecolor=(0.35, 0.65, 0.90, 0.6), linewidth=0.1,
        )
        ax.add_collection3d(poly)

        # 叠加原始表面点
        sp = dataset.surface_points
        subsample = min(1500, len(sp))
        sp_idx = np.random.choice(len(sp), subsample, replace=False)
        ax.scatter(
            sp[sp_idx, 0], sp[sp_idx, 1], sp[sp_idx, 2],
            s=0.3, c="red", alpha=0.4, label="GT surface",
        )

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11)

    fig.suptitle(
        f"Zero Level Set (isosurface)   |   "
        f"{len(verts)} verts, {len(faces)} faces",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ========================================================================== #
#  6. 主流程                                                                  #
# ========================================================================== #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="逐连杆 3D Neural SDF 无监督训练 (对齐 StEik 源码)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--urdf", default=str(_DEFAULT_URDF),
                    help="URDF 文件路径（默认：仓库根下 9 轴模型）")
    p.add_argument("--link-names", nargs="+", default=None,
                    help="要训练的连杆名列表；不指定则使用默认后六轴集合")
    p.add_argument("--n-iterations", type=int,   default=10000,
                    help="无监督训练迭代数")
    p.add_argument("--n-points",     type=int,   default=15000,
                    help="每次迭代采样的 manifold / non-manifold 点数")
    p.add_argument("--n-surface",    type=int,   default=30000,
                    help="从 mesh 表面预采样的点云规模")
    p.add_argument("--grid-range",   type=float, default=1.1,
                    help="non-manifold 均匀采样范围 [-r, r]")
    p.add_argument("--device",       default="auto",
                    help="训练设备 (auto / cpu / cuda)")
    p.add_argument("--output-dir",   default="artifacts/neural_sdf/9_axis_links")
    p.add_argument("--lr",           type=float, default=1e-4,
                    help="学习率 (StEik 默认 1e-4)")
    p.add_argument("--optimizer",    default="adam",
                    choices=["adam", "adamw"],
                    help="优化器（torch.optim）")
    p.add_argument("--weight-decay", type=float, default=0.0,
                    help="优化器权重衰减")
    p.add_argument("--lr-schedule",  default="none",
                    choices=["none", "cosine", "plateau"],
                    help="学习率调度器")
    p.add_argument("--lr-min-ratio", type=float, default=0.3,
                    help="cosine 调度最小学习率比例 eta_min/lr")
    p.add_argument("--plateau-factor", type=float, default=0.5,
                    help="ReduceLROnPlateau 的衰减倍率")
    p.add_argument("--plateau-patience", type=int, default=300,
                    help="ReduceLROnPlateau 的 patience（步）")
    p.add_argument("--plateau-threshold", type=float, default=1e-4,
                    help="ReduceLROnPlateau 的阈值")
    p.add_argument("--hidden",       type=int,   default=128,
                    help="隐藏层宽度 (StEik 默认 256)")
    p.add_argument("--n-layers",     type=int,   default=5,
                    help="隐藏层数 (StEik 默认 8)")
    p.add_argument("--n-frequencies", type=int,  default=6)
    p.add_argument("--fourier",      action="store_true",
                    help="启用 Fourier 位置编码（默认关闭）")
    p.add_argument("--viz-resolution", type=int, default=200,
                    help="切片可视化网格分辨率")
    p.add_argument("--iso-resolution", type=int, default=128,
                    help="零等值面 marching-cubes 网格分辨率")
    # ---- 网络架构 (默认对齐 StEik 源码最优配置) ----
    p.add_argument("--no-quadratic", action="store_true",
                    help="禁用二次神经元层，退回线性层")
    p.add_argument("--activation", default="sin",
                    choices=["softplus", "sin"],
                    help="激活函数 (StEik 默认 softplus(beta=100))")
    p.add_argument("--sphere-init-params", nargs=2, type=float,
                    default=[1.6, 0.1], metavar=("RADIUS", "SCALING"),
                    help="球面初始化参数，和 StEik 配置一致")
    p.add_argument("--omega-0", type=float, default=30.0,
                    help="SIREN sin 激活的 omega_0 频率参数")
    p.add_argument("--init-type", default="mfgi",
                    choices=["mfgi", "siren", "xavier"],
                    help="权重初始化方式 (StEik 默认 mfgi)")
    # ---- 损失函数 ----
    p.add_argument("--loss-type", default="siren_wo_n_w_div",
                    choices=[
                        "siren", "siren_wo_n", "igr", "igr_wo_n",
                        "siren_w_div", "siren_wo_n_w_div",
                    ],
                    help="损失组合，和 StEik losses.py 分支一致")
    p.add_argument("--loss-weights", nargs=5, type=float,
                    default=[2e3, 1e2, 1e2, 5e1, 1e2],
                    metavar=("W_SDF", "W_INTER", "W_NORMAL", "W_EIK", "W_DIV"),
                    help="StEik 五项损失权重: sdf inter normal eikonal div")
    p.add_argument("--div-decay", default="linear",
                    choices=["none", "step", "linear", "quintic"],
                    help="方向散度权重退火策略")
    p.add_argument("--div-decay-params", nargs="+", type=float,
                    default=[1e2, 0.2, 1e2, 0.4, 0.0, 0.0],
                    help="StEik 格式 w0 t1 w1 ... w_end；默认与 loss 中 W_DIV=1e2 分段一致")
    p.add_argument("--div-type", default="dir_l1",
                    choices=["dir_l1", "dir_l2"],
                    help="方向散度项形式")
    p.add_argument("--eikonal-type", default="abs",
                    choices=["abs", "square"],
                    help="Eikonal 项形式")
    p.add_argument("--grad-clip", type=float, default=10.0,
                    help="梯度裁剪阈值 (StEik 默认 10.0)")
    p.add_argument("--early-stop-patience", type=int, default=0,
                    help="早停容忍步数；0 表示关闭")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0,
                    help="判定改进所需的最小下降量")
    p.add_argument("--early-stop-warmup", type=int, default=1000,
                    help="早停生效前的预热步数")
    p.add_argument("--no-restore-best", action="store_true",
                    help="关闭早停/训练结束后自动回滚到最佳权重")
    p.add_argument("--no-progress", action="store_true",
                    help="关闭 tqdm 进度条（日志重定向或无可 TTY 时可用）")
    p.add_argument("--amp", action="store_true",
                    help="启用 CUDA AMP（本任务含高阶导，可能不稳定，默认关闭）")
    p.add_argument("--bf16", action="store_true",
                    help="混合精度使用 bfloat16（推荐 Ampere sm_80+，数值通常比 float16 稳）")
    p.add_argument("--no-tf32", action="store_true",
                    help="禁用 CUDA TF32（默认开启，可显著加速矩阵乘）")
    return p


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device.startswith("cuda") and args.amp
    amp_dtype = "bfloat16" if (use_amp and args.bf16) else "float16"
    if use_amp and args.bf16:
        major, _minor = torch.cuda.get_device_capability()
        if major < 8:
            print("  [警告] BF16 推荐 sm_80+ (Ampere)，已回退 float16")
            amp_dtype = "float16"
    use_tf32 = device.startswith("cuda") and not args.no_tf32
    print(f"使用设备: {device}")
    if device.startswith("cuda"):
        print(
            f"  加速: AMP={'开' if use_amp else '关'} ({amp_dtype}), "
            f"TF32={'开' if use_tf32 else '关'}, cudnn.benchmark=开"
        )

    link_names = args.link_names or DEFAULT_LINK_NAMES
    print(f"目标连杆: {link_names}")

    # ---- 解析 URDF ----
    links = parse_urdf_links(args.urdf, link_names)
    if not links:
        print("错误: 未找到匹配的连杆。请检查 --urdf 和 --link-names 参数。")
        return

    print(f"已解析 {len(links)} 个连杆:")
    for li in links:
        tag = "OK" if li.mesh_path.exists() else "MISSING"
        print(f"  {li.link_name:<22s} {li.mesh_path.name:<28s} [{tag}]")

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    use_fourier = args.fourier
    quadratic = not args.no_quadratic
    activation = args.activation
    omega_0 = args.omega_0
    init_type = args.init_type
    sphere_init_params = tuple(args.sphere_init_params)
    grad_clip = args.grad_clip
    w_sdf, w_inter, w_normal, w_eikonal, w_div = args.loss_weights

    if activation == "sin" and use_fourier:
        print("  [警告] sin 激活 + Fourier 编码可能导致频率过高，建议去掉 --fourier")

    flags = []
    if quadratic:
        flags.append("quadratic")
    else:
        flags.append("linear")
    flags.append(f"init={init_type}")
    if use_fourier:
        flags.append("fourier")
    flags.append(f"eik={args.eikonal_type}")
    flags.append(f"loss={args.loss_type}")
    if activation == "sin":
        flags.append(f"sin(ω₀={omega_0})")
    else:
        flags.append("softplus(β=100)")
    if w_div > 0:
        flags.append(f"LL.n.={w_div}")
    if w_normal > 0:
        flags.append(f"normal={w_normal}")
    flags.append(f"clip={grad_clip}")
    flags.append(f"opt={args.optimizer}(wd={args.weight_decay:g})")
    flags.append(f"sphere={sphere_init_params}")
    if args.lr_schedule != "none":
        flags.append(f"lr_sched={args.lr_schedule}(min={args.lr_min_ratio:g})")
        if args.lr_schedule == "plateau":
            flags.append(
                f"plateau(f={args.plateau_factor:g},p={args.plateau_patience},th={args.plateau_threshold:g})"
            )
    if args.early_stop_patience > 0:
        flags.append(
            f"early_stop={args.early_stop_patience}@warmup{args.early_stop_warmup}"
        )
    print(f"  网络配置: {', '.join(flags)}")
    print(
        f"  容量: {args.hidden}×{args.n_layers}, lr={args.lr}, "
        f"iterations={args.n_iterations}, n_points={args.n_points}"
    )

    summary_rows: list[dict] = []
    show_pbar = not args.no_progress
    links_iter = links
    if show_pbar:
        links_iter = _tqdm_bar(
            links,
            desc="连杆",
            unit="link",
            dynamic_ncols=True,
        )

    for li in links_iter:
        print(f"\n{'=' * 64}")
        print(f"  连杆: {li.link_name}")
        print(f"{'=' * 64}")

        if not li.mesh_path.exists():
            print(f"  [跳过] mesh 文件不存在: {li.mesh_path}")
            continue

        # ---- 数据生成 ----
        print(f"  加载 mesh: {li.mesh_path.name}")
        dataset, meta = generate_pointcloud_data(
            li.mesh_path,
            n_surface=args.n_surface,
            grid_range=args.grid_range,
        )
        print(
            f"  数据集: {dataset.surface_points.shape[0]} 个表面点, "
            f"scale={dataset.scale:.6f}, "
            f"watertight={meta['is_watertight']}"
        )

        # ---- 训练 ----
        print(f"  训练中 ({args.n_iterations} iterations, n_points={args.n_points}) ...")
        result = train_link_sdf_unsupervised(
            dataset,
            n_iterations=args.n_iterations,
            n_points=args.n_points,
            lr=args.lr,
            device=device,
            loss_weights=tuple(args.loss_weights),
            loss_type=args.loss_type,
            div_decay=args.div_decay,
            div_decay_params=tuple(args.div_decay_params),
            div_type=args.div_type,
            eikonal_type=args.eikonal_type,
            hidden=args.hidden,
            n_layers=args.n_layers,
            n_frequencies=args.n_frequencies,
            use_fourier=use_fourier,
            quadratic=quadratic,
            activation=activation,
            omega_0=omega_0,
            init_type=init_type,
            sphere_init_params=sphere_init_params,
            grad_clip=grad_clip,
            grid_range=args.grid_range,
            show_progress=show_pbar,
            progress_desc=f"train:{li.link_name}",
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_tf32=use_tf32,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            lr_schedule=args.lr_schedule,
            lr_min_ratio=args.lr_min_ratio,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            plateau_threshold=args.plateau_threshold,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_warmup=args.early_stop_warmup,
            restore_best=not args.no_restore_best,
        )

        # 打印训练进度摘要
        print_interval = max(1, len(result.loss_history) // 5)
        for i, entry in enumerate(result.loss_history):
            if i % print_interval == 0 or i == len(result.loss_history) - 1:
                print(
                    f"    [Iter {entry['epoch']:5d}]  "
                    f"total={entry['loss_total']:.6f}  "
                    f"sdf={entry['loss_sdf']:.6f}  "
                    f"inter={entry['loss_inter']:.6f}  "
                    f"eik={entry['loss_eikonal']:.6f}  "
                    f"nrm={entry['loss_normal']:.6f}  "
                    f"div={entry['loss_dd']:.6f}"
                )
        print(
            f"  完成: {result.elapsed_seconds:.1f}s, "
            f"iter={result.iterations_run}/{args.n_iterations}, "
            f"final_loss={result.final_loss:.6f}, "
            f"best_loss={result.best_loss:.6f}, "
            f"stop={result.stop_reason}"
        )

        # ---- 保存 ----
        link_dir = output_base / li.link_name
        link_dir.mkdir(parents=True, exist_ok=True)

        # checkpoint
        ckpt = dict(
            model_state_dict=result.model.state_dict(),
            model_config=result.model.config,
            normalization=dict(
                center=dataset.center.tolist(),
                scale=dataset.scale,
            ),
            link_name=li.link_name,
            mesh_path=str(li.mesh_path),
        )
        torch.save(ckpt, link_dir / "checkpoint.pt")

        # 统计信息
        stats = dict(
            link_name=li.link_name,
            mesh_path=str(li.mesh_path),
            n_surface=args.n_surface,
            n_iterations=args.n_iterations,
            iterations_run=result.iterations_run,
            n_points=args.n_points,
            grid_range=args.grid_range,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_tf32=use_tf32,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            sphere_init_params=list(sphere_init_params),
            lr_schedule=args.lr_schedule,
            lr_min_ratio=args.lr_min_ratio,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            plateau_threshold=args.plateau_threshold,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_warmup=args.early_stop_warmup,
            restore_best=not args.no_restore_best,
            stop_reason=result.stop_reason,
            best_loss=result.best_loss,
            loss_weights=args.loss_weights,
            loss_type=args.loss_type,
            final_loss=result.final_loss,
            training_seconds=result.elapsed_seconds,
            center=dataset.center.tolist(),
            scale=dataset.scale,
            aabb_min=dataset.aabb_min.tolist(),
            aabb_max=dataset.aabb_max.tolist(),
            **meta,
        )
        with open(link_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # ---- 可视化 ----
        print(f"  生成可视化 ...")
        model_cpu = result.model.cpu().eval()
        _save_loss_curve(result.loss_history, link_dir / "loss_curve.png")
        _save_sdf_slices(model_cpu, dataset, link_dir / "sdf_slices.png",
                         resolution=args.viz_resolution, device="cpu")
        _save_surface_eval(model_cpu, dataset, link_dir / "surface_eval.png",
                           device="cpu")
        _save_isosurface(model_cpu, dataset,
                         png_path=link_dir / "isosurface.png",
                         obj_path=link_dir / "isosurface.obj",
                         resolution=args.iso_resolution, device="cpu")

        print(f"  输出: {link_dir}")
        summary_rows.append(stats)

    # ---- 汇总 ----
    if summary_rows:
        print(f"\n{'=' * 80}")
        print(f"  训练汇总")
        print(f"{'=' * 80}")
        header = f"  {'连杆':<22s} {'Mesh':<26s} {'样本数':>8s} {'最终损失':>12s} {'耗时':>7s}"
        print(header)
        print(f"  {'-' * 76}")
        for s in summary_rows:
            mesh_name = Path(s["mesh_path"]).name
            print(
                f"  {s['link_name']:<22s} {mesh_name:<26s} "
                f"{s['n_train_points']:>8d} "
                f"{s['final_loss']:>12.6f} "
                f"{s['training_seconds']:>6.1f}s"
            )
        print(f"\n  输出目录: {output_base.resolve()}")


if __name__ == "__main__":
    main()
