"""单模块开环测试：点云 → 3D 任务空间 IRIS

流程：
  1. 加载机器人 + 场景（沿用 visualization 模块）
  2. 按密度采样工件表面点云，每点膨胀为 VPolytope 微型立方体
  3. 读取机器人初始关节位置，计算 weld_point 末端位置（起点）
  4. 读取场景 l2 body 位置（目标点）
  5. 在起点→目标点区域运行 Drake Iris（3D 任务空间）
  6. MeshCat 可视化：点云障碍物 + IRIS 区域椭球 + 起/终点标记

不依赖控制器 / CBF / GCS。
"""
from __future__ import annotations

import re
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSETS         = ROOT / "assets"
ROBOT_PKG_DIR  = ASSETS / "robots" / "9_axis"
SCENE_PKG_DIR  = ASSETS / "cad_exports" / "model_CAD" / "scene"
ROBOT_URDF     = ROBOT_PKG_DIR / "urdf" / "9_axis.urdf"
SCENE_URDF     = SCENE_PKG_DIR / "urdf" / "中组立0725(1).stp.SLDASM.urdf"
ROBOT_PKG_NAME = "9_axis"
SCENE_PKG_NAME = "中组立0725(1).stp.SLDASM"


# ── 配置 ──────────────────────────────────────────────────────────────────────
@dataclass
class IrisTestConfig:
    # 初始关节配置（与主文件 DrakeConfig 保持一致）
    robot_initial_q: list[float] = field(
        default_factory=lambda: [10.0, 0.0, 0.0,
                                  0.0,  0.0, 0.0,
                                  0.0,  0.0, 0.0]
    )
    scene_translation: list[float] = field(default_factory=lambda: [0.0, 4.0, 0.0])

    # 目标：l2 body 在局部系中的 z 轴方向（用于 IRIS 目标偏移方向）
    target_z_in_l2_local: list[float] = field(default_factory=lambda: [0.0, 1.0, -1.0])
    # 预到位偏移距离（沿 -z_world 方向）
    pre_approach_dist: float = 0.2

    # 点云密度
    pointcloud_density: float = 50.0
    min_samples: int = 50
    max_samples: int = 20000
    # 每个点膨胀成立方体的半边长 (m)
    inflate_radius: float = 0.05

    # IRIS 参数
    iris_n_seeds: int = 15

    auto_open_browser: bool = True


# ── Drake 场景构建 ─────────────────────────────────────────────────────────────
def build_scene(cfg: IrisTestConfig):
    from pydrake.all import (
        AddMultibodyPlantSceneGraph, DiagramBuilder,
        MeshcatVisualizer, Parser, RigidTransform, StartMeshcat,
    )

    meshcat = StartMeshcat()
    print(f"[MeshCat] {meshcat.web_url()}", flush=True)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    def load_urdf(path: Path, pkg_name: str, pkg_dir: Path):
        text = re.sub(r'\.STL\b', '.obj', path.read_text(encoding="utf-8"),
                      flags=re.IGNORECASE)
        p = Parser(plant)
        p.package_map().Add(pkg_name, str(pkg_dir))
        return p.AddModelsFromString(text, "urdf")[0]

    robot_model = load_urdf(ROBOT_URDF, ROBOT_PKG_NAME, ROBOT_PKG_DIR)
    scene_model = load_urdf(SCENE_URDF, SCENE_PKG_NAME, SCENE_PKG_DIR)

    # 焊接机器人基座
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetBodyByName("base_link", robot_model).body_frame(),
        RigidTransform(np.zeros(3)),
    )
    # 焊接场景
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetBodyByName("base_link", scene_model).body_frame(),
        RigidTransform(np.array(cfg.scene_translation, dtype=float)),
    )

    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(context)

    # 设置初始关节
    q_init = np.array(cfg.robot_initial_q, dtype=float)
    plant.SetPositions(plant_ctx, robot_model, q_init)

    return plant, plant_ctx, robot_model, scene_model, meshcat, diagram, context


# ── 计算起/终点 ────────────────────────────────────────────────────────────────
def compute_ee_start(plant, plant_ctx, robot_model) -> np.ndarray:
    ee = plant.GetBodyByName("weld_point", robot_model)
    X_WE = plant.CalcRelativeTransform(plant_ctx, plant.world_frame(), ee.body_frame())
    pos = X_WE.translation().copy()
    print(f"[EE起点] weld_point 位置: {np.array2string(pos, precision=3)}", flush=True)
    return pos


def compute_target(plant, plant_ctx, scene_model,
                   z_local: np.ndarray,
                   pre_approach_dist: float) -> tuple[np.ndarray, np.ndarray]:
    """返回 (最终目标点, 预到位点)。"""
    l2 = plant.GetBodyByName("l2", scene_model)
    X_WL2 = plant.CalcRelativeTransform(plant_ctx, plant.world_frame(), l2.body_frame())
    goal = X_WL2.translation().copy()

    z_w = X_WL2.rotation().matrix() @ (z_local / np.linalg.norm(z_local))
    pre = goal - z_w * pre_approach_dist

    print(f"[目标点] l2 位置 (最终焊点): {np.array2string(goal, precision=3)}", flush=True)
    print(f"[预到位] 沿 -z_world×{pre_approach_dist:.1f}m: {np.array2string(pre, precision=3)}", flush=True)
    return goal, pre


# ── 点云 → VPolytope 障碍物 ────────────────────────────────────────────────────
def extract_obstacles(plant, plant_ctx, scene_model, cfg: IrisTestConfig):
    import trimesh
    from pydrake.geometry.optimization import VPolytope

    mesh_dir = SCENE_PKG_DIR / "meshes"
    r = cfg.inflate_radius
    box_offsets = np.array([
        [ r,  r,  r], [ r,  r, -r], [ r, -r,  r], [ r, -r, -r],
        [-r,  r,  r], [-r,  r, -r], [-r, -r,  r], [-r, -r, -r],
    ])

    obstacles: list = []
    all_centers: list[np.ndarray] = []

    for body_idx in plant.GetBodyIndices(scene_model):
        body = plant.get_body(body_idx)
        name = body.name()
        obj_path = mesh_dir / f"{name}.obj"
        if not obj_path.exists():
            print(f"  [跳过] {name}: OBJ 不存在", flush=True)
            continue

        mesh = trimesh.load(str(obj_path), force="mesh")
        X_WB = plant.CalcRelativeTransform(plant_ctx, plant.world_frame(), body.body_frame())
        R = X_WB.rotation().matrix()
        t = X_WB.translation()

        area = mesh.area if hasattr(mesh, 'area') else 0.0
        count = int(area * cfg.pointcloud_density) if area > 0 else cfg.min_samples
        count = max(cfg.min_samples, min(cfg.max_samples, count))

        pts_local, _ = trimesh.sample.sample_surface(mesh, count)
        pts_world = (R @ pts_local.T).T + t

        for pt in pts_world:
            obstacles.append(VPolytope((pt + box_offsets).T))
        all_centers.append(pts_world)

        print(f"  [{name}] 面积={area:.2f} m², 采样={len(pts_world)} 点 "
              f"→ {len(pts_world)} 个 VPolytope (边长={2*r}m)", flush=True)

    centers = np.vstack(all_centers) if all_centers else np.zeros((0, 3))
    print(f"  合计 {len(obstacles)} 个微型立方体障碍物", flush=True)
    return obstacles, centers


# ── 生成 IRIS 种子点 ────────────────────────────────────────────────────────────
def generate_seeds(start: np.ndarray, goal: np.ndarray, n: int) -> list[np.ndarray]:
    rng = np.random.RandomState(42)
    seeds = [start.copy(), goal.copy()]
    n_mid = min(n - 4, 8)
    for i in range(n_mid):
        alpha = (i + 1) / (n_mid + 1)
        seeds.append((1 - alpha) * start + alpha * goal + rng.randn(3) * 0.2)
    seeds.append(start + rng.randn(3) * 0.5)
    seeds.append(goal + rng.randn(3) * 0.5)
    return seeds


# ── 运行 3D 任务空间 IRIS ─────────────────────────────────────────────────────
def run_iris(obstacles: list, start: np.ndarray, goal: np.ndarray,
             n_seeds: int) -> list:
    from pydrake.geometry.optimization import Iris, IrisOptions, HPolyhedron

    if not obstacles:
        print("[IRIS] 无障碍物，跳过", flush=True)
        return []

    lb = np.minimum(start, goal) - 3.0
    ub = np.maximum(start, goal) + 3.0
    domain = HPolyhedron.MakeBox(lb, ub)

    options = IrisOptions()

    seeds = generate_seeds(start, goal, n_seeds)
    regions: list = []

    print(f"\n[IRIS] 域: {np.array2string(lb, precision=2)} → "
          f"{np.array2string(ub, precision=2)}", flush=True)
    print(f"[IRIS] 共 {len(seeds)} 个种子点，{len(obstacles)} 个障碍物", flush=True)

    for i, seed in enumerate(seeds):
        if not domain.PointInSet(seed):
            print(f"  种子 {i:02d}: 域外，跳过", flush=True)
            continue
        in_obs = any(obs.PointInSet(seed) for obs in obstacles)
        if in_obs:
            print(f"  种子 {i:02d}: 障碍物内，跳过", flush=True)
            continue
        try:
            region = Iris(obstacles, seed, domain, options)
            vol = region.MaximumVolumeInscribedEllipsoid().CalcVolume()
            if vol < 1e-6:
                print(f"  种子 {i:02d}: 区域体积过小 ({vol:.2e})，丢弃", flush=True)
                continue
            regions.append(region)
            contains_start = region.PointInSet(start)
            contains_goal  = region.PointInSet(goal)
            print(f"  种子 {i:02d}: ✓ 区域 {len(regions)-1}  "
                  f"vol={vol:.3e}  "
                  f"含起点={'是' if contains_start else '否'}  "
                  f"含终点={'是' if contains_goal else '否'}", flush=True)
        except Exception as e:
            print(f"  种子 {i:02d}: 失败 → {e}", flush=True)

    n_s = sum(1 for r in regions if r.PointInSet(start))
    n_g = sum(1 for r in regions if r.PointInSet(goal))
    print(f"\n[IRIS] 合计 {len(regions)} 个区域 | "
          f"含起点={n_s} | 含终点={n_g}", flush=True)
    return regions


# ── 可视化 ─────────────────────────────────────────────────────────────────────
def visualize_pointcloud(meshcat, centers: np.ndarray, point_size: float = 0.008):
    """点云整体渲染（绿色）。"""
    from pydrake.perception import PointCloud as DrakePC, Fields, BaseField

    n = len(centers)
    if n == 0:
        return
    cloud = DrakePC(n, Fields(BaseField.kXYZs | BaseField.kRGBs))
    cloud.mutable_xyzs()[:] = centers.astype(np.float32).T
    cloud.mutable_rgbs()[:] = np.tile(
        np.array([80, 220, 80], dtype=np.uint8).reshape(3, 1), (1, n))
    meshcat.SetObject("iris_test/pointcloud", cloud, point_size=point_size)
    print(f"[Viz] 点云 {n} 点 → MeshCat", flush=True)


def visualize_iris_regions(meshcat, regions: list):
    """将每个 IRIS 区域的最大内切椭球显示为半透明椭球体。"""
    from pydrake.all import Rgba
    from pydrake.math import RigidTransform, RotationMatrix

    colors = [
        Rgba(0.2, 0.5, 1.0, 0.25),
        Rgba(1.0, 0.5, 0.2, 0.25),
        Rgba(0.2, 1.0, 0.5, 0.25),
        Rgba(1.0, 0.2, 1.0, 0.25),
        Rgba(1.0, 1.0, 0.2, 0.25),
    ]
    meshcat.Delete("iris_test/regions")

    for i, region in enumerate(regions):
        try:
            ell = region.MaximumVolumeInscribedEllipsoid()
            center = ell.center()
            # A^T A = (shape matrix)^{-2}，用 SVD 分解半轴
            A = ell.A()          # (3,3)，椭球定义为 {x | ||A(x-c)||<=1}
            U, s, _ = np.linalg.svd(A)
            radii = 1.0 / s      # 半轴长度

            col = colors[i % len(colors)]
            from pydrake.geometry import Ellipsoid
            path = f"iris_test/regions/r{i:02d}"
            meshcat.SetObject(path, Ellipsoid(radii[0], radii[1], radii[2]), col)
            R_mat = U
            meshcat.SetTransform(path, RigidTransform(RotationMatrix(R_mat), center))
            print(f"  [Viz] 区域 {i}: center={np.array2string(center, precision=2)}  "
                  f"半轴=[{radii[0]:.2f},{radii[1]:.2f},{radii[2]:.2f}]", flush=True)
        except Exception as e:
            print(f"  [Viz] 区域 {i} 椭球绘制失败: {e}", flush=True)


def visualize_points(meshcat, points: dict[str, tuple[np.ndarray, tuple]]):
    """显示关键点（球体标记）。points = {name: (pos, rgba)}"""
    from pydrake.all import Rgba, Sphere
    from pydrake.math import RigidTransform

    for name, (pos, rgba) in points.items():
        path = f"iris_test/keypoints/{name}"
        meshcat.SetObject(path, Sphere(0.08), Rgba(*rgba))
        meshcat.SetTransform(path, RigidTransform(pos))
        print(f"  [Viz] 关键点 '{name}': {np.array2string(pos, precision=3)}", flush=True)


# ── 主流程 ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print("单模块测试：点云 → 3D 任务空间 IRIS", flush=True)
    print("=" * 60, flush=True)

    cfg = IrisTestConfig()

    # 1. 构建 Drake 场景
    print("\n[1/5] 构建 Drake 场景 ...", flush=True)
    plant, plant_ctx, robot_model, scene_model, meshcat, diagram, context = \
        build_scene(cfg)
    diagram.ForcedPublish(context)
    print("  场景就绪", flush=True)

    # 2. 计算起/终点
    print("\n[2/5] 计算起点和目标点 ...", flush=True)
    ee_start = compute_ee_start(plant, plant_ctx, robot_model)
    z_local  = np.array(cfg.target_z_in_l2_local, dtype=float)
    ee_goal, pre_approach = compute_target(
        plant, plant_ctx, scene_model, z_local, cfg.pre_approach_dist
    )

    # IRIS 从起点规划到预到位点（与主文件逻辑一致）
    iris_start = ee_start
    iris_goal  = pre_approach

    # 3. 点云采样 → VPolytope 障碍物
    print("\n[3/5] 工件点云采样 → VPolytope 障碍物 ...", flush=True)
    obstacles, centers = extract_obstacles(plant, plant_ctx, scene_model, cfg)

    # 4. 3D IRIS
    print("\n[4/5] 运行 3D 任务空间 IRIS ...", flush=True)
    t0 = time.time()
    regions = run_iris(obstacles, iris_start, iris_goal, cfg.iris_n_seeds)
    elapsed = time.time() - t0
    print(f"\n[IRIS] 耗时 {elapsed:.1f}s", flush=True)

    # 5. 可视化
    print("\n[5/5] MeshCat 可视化 ...", flush=True)
    visualize_pointcloud(meshcat, centers)
    print("[Viz] IRIS 区域椭球:", flush=True)
    visualize_iris_regions(meshcat, regions)
    print("[Viz] 关键点:", flush=True)
    visualize_points(meshcat, {
        "ee_start":    (ee_start,    (0.2, 0.8, 0.2, 1.0)),   # 绿
        "pre_approach":(pre_approach, (1.0, 0.8, 0.0, 1.0)),   # 黄
        "ee_goal":     (ee_goal,     (1.0, 0.2, 0.2, 1.0)),   # 红
    })

    # 6. 摘要
    n_s = sum(1 for r in regions if r.PointInSet(iris_start))
    n_g = sum(1 for r in regions if r.PointInSet(iris_goal))
    print(f"\n{'='*60}", flush=True)
    print(f"IRIS 结果摘要", flush=True)
    print(f"  区域总数:   {len(regions)}", flush=True)
    print(f"  含起点区域: {n_s}", flush=True)
    print(f"  含终点区域: {n_g}", flush=True)
    print(f"  IRIS 耗时:  {elapsed:.1f} s", flush=True)
    if n_s > 0 and n_g > 0:
        print(f"  ✓ 起点和终点均有区域覆盖，可进行 GCS 规划", flush=True)
    elif n_s == 0:
        print(f"  ✗ 起点无区域覆盖 — 检查初始位姿是否在障碍物内", flush=True)
    elif n_g == 0:
        print(f"  ✗ 终点无区域覆盖 — 尝试增大 pre_approach_dist", flush=True)
    print(f"{'='*60}", flush=True)

    url = meshcat.web_url()
    if cfg.auto_open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    print(f"\n可视化地址: {url}", flush=True)
    print("按 Ctrl+C 退出", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n已退出", flush=True)


if __name__ == "__main__":
    main()
