"""单任务测试：在终点附近寻找最大自由空间内切椭球

流程：
  1. 加载场景，按密度采样点云 → VPolytope 微型立方体（与 test_iris.py 一致）
  2. 以 l2 目标点为初始种子，若该点在障碍物内则沿自由空间方向微步移动
  3. 以该种子运行单次 Drake Iris，得到最大自由空间多面体（HPolyhedron）
  4. 对该多面体求最大体积内切椭球（MaximumVolumeInscribedEllipsoid）
  5. MeshCat 可视化：点云 + 椭球 + 目标点标记
"""
from __future__ import annotations

import re
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class Config:
    robot_initial_q: list[float] = field(
        default_factory=lambda: [10.0, 0.0, 0.0,
                                  0.0,  0.0, 0.0,
                                  0.0,  0.0, 0.0]
    )
    scene_translation: list[float] = field(default_factory=lambda: [0.0, 4.0, 0.0])
    target_z_in_l2_local: list[float] = field(default_factory=lambda: [0.0, 1.0, -1.0])

    # 点云参数
    pointcloud_density: float = 50.0
    min_samples: int = 50
    max_samples: int = 20000
    inflate_radius: float = 0.00005   # 微型立方体半边长

    auto_open_browser: bool = True


# ── Drake 场景 ─────────────────────────────────────────────────────────────────
def build_scene(cfg: Config):
    from pydrake.all import (
        AddMultibodyPlantSceneGraph, DiagramBuilder,
        MeshcatVisualizer, Parser, RigidTransform, StartMeshcat,
    )
    meshcat = StartMeshcat()
    print(f"[MeshCat] {meshcat.web_url()}", flush=True)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    def load_urdf(path, pkg_name, pkg_dir):
        text = re.sub(r'\.STL\b', '.obj',
                      path.read_text(encoding="utf-8"), flags=re.IGNORECASE)
        p = Parser(plant)
        p.package_map().Add(pkg_name, str(pkg_dir))
        return p.AddModelsFromString(text, "urdf")[0]

    robot_model = load_urdf(ROBOT_URDF, ROBOT_PKG_NAME, ROBOT_PKG_DIR)
    scene_model = load_urdf(SCENE_URDF, SCENE_PKG_NAME, SCENE_PKG_DIR)

    plant.WeldFrames(plant.world_frame(),
                     plant.GetBodyByName("base_link", robot_model).body_frame(),
                     RigidTransform(np.zeros(3)))
    plant.WeldFrames(plant.world_frame(),
                     plant.GetBodyByName("base_link", scene_model).body_frame(),
                     RigidTransform(np.array(cfg.scene_translation, dtype=float)))

    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_ctx, robot_model,
                       np.array(cfg.robot_initial_q, dtype=float))
    return plant, plant_ctx, scene_model, meshcat, diagram, context


# ── 点云 → VPolytope 障碍物 ────────────────────────────────────────────────────
def extract_obstacles(plant, plant_ctx, scene_model, cfg: Config):
    import trimesh
    from pydrake.geometry.optimization import VPolytope
    from scipy.spatial import cKDTree

    mesh_dir = SCENE_PKG_DIR / "meshes"
    r = cfg.inflate_radius
    box_offsets = np.array([
        [ r,  r,  r], [ r,  r, -r], [ r, -r,  r], [ r, -r, -r],
        [-r,  r,  r], [-r,  r, -r], [-r, -r,  r], [-r, -r, -r],
    ])

    obstacles, all_centers = [], []
    for body_idx in plant.GetBodyIndices(scene_model):
        body = plant.get_body(body_idx)
        obj_path = mesh_dir / f"{body.name()}.obj"
        if not obj_path.exists():
            continue
        mesh = trimesh.load(str(obj_path), force="mesh")
        X_WB = plant.CalcRelativeTransform(plant_ctx, plant.world_frame(),
                                           body.body_frame())
        R, t = X_WB.rotation().matrix(), X_WB.translation()
        area = getattr(mesh, 'area', 0.0)
        count = max(cfg.min_samples,
                    min(cfg.max_samples, int(area * cfg.pointcloud_density)))
        pts_local, _ = trimesh.sample.sample_surface(mesh, count)
        pts_world = (R @ pts_local.T).T + t
        for pt in pts_world:
            obstacles.append(VPolytope((pt + box_offsets).T))
        all_centers.append(pts_world)
        print(f"  [{body.name()}] {len(pts_world)} 点 → {len(pts_world)} 障碍物",
              flush=True)

    centers = np.vstack(all_centers) if all_centers else np.zeros((0, 3))
    tree = cKDTree(centers) if len(centers) > 0 else None
    print(f"  合计 {len(obstacles)} 个 VPolytope 障碍物", flush=True)
    return obstacles, centers, tree


# ── 计算目标点 ─────────────────────────────────────────────────────────────────
def compute_goal(plant, plant_ctx, scene_model, cfg: Config) -> np.ndarray:
    l2 = plant.GetBodyByName("l2", scene_model)
    X_WL2 = plant.CalcRelativeTransform(plant_ctx, plant.world_frame(),
                                        l2.body_frame())
    goal = X_WL2.translation().copy()
    print(f"[目标点] l2 世界坐标: {np.array2string(goal, precision=4)}", flush=True)
    return goal


# ── 在目标点附近找最近自由空间点 ──────────────────────────────────────────────
def find_free_seed(goal: np.ndarray, centers: np.ndarray,
                   inflate_radius: float) -> tuple[np.ndarray, float]:
    """找到自由空间中距目标点最近的点作为 IRIS 种子。

    先检查目标点自身是否已在自由空间（最近障碍物距离 > inflate_radius）。
    若不是，沿「目标→最近障碍物点」的反方向逐步移动，直到离开障碍物。
    返回 (seed, dist_to_goal)。
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)

    def nearest_dist(pt):
        d, _ = tree.query(pt)
        return d

    # 目标点已在自由空间
    d0 = nearest_dist(goal)
    if d0 > inflate_radius:
        print(f"[种子] 目标点已在自由空间，最近障碍物距离 = {d0:.4f}m", flush=True)
        return goal.copy(), 0.0

    print(f"[种子] 目标点在障碍物内（最近距离={d0:.4f}m < {inflate_radius}m），"
          f"向外搜索 ...", flush=True)

    # 找到最近的障碍物中心，沿目标→该中心的反方向步进
    _, idx = tree.query(goal)
    nearest_obs = centers[idx]
    direction = goal - nearest_obs
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([0.0, 0.0, 1.0])
    direction /= np.linalg.norm(direction)

    step = 0.01
    seed = goal.copy()
    for _ in range(1000):
        seed = seed + direction * step
        if nearest_dist(seed) > inflate_radius:
            dist_to_goal = np.linalg.norm(seed - goal)
            print(f"[种子] 找到自由空间种子点，偏移距离 = {dist_to_goal:.4f}m", flush=True)
            print(f"  种子坐标: {np.array2string(seed, precision=4)}", flush=True)
            print(f"  最近障碍物距离: {nearest_dist(seed):.4f}m", flush=True)
            return seed, dist_to_goal

    # 若直线方向无法逃脱，在球面上多方向尝试
    print("[种子] 直线方向无效，尝试多方向搜索 ...", flush=True)
    rng = np.random.RandomState(0)
    for r_dist in [0.05, 0.1, 0.2, 0.4, 0.8, 1.5]:
        for _ in range(200):
            d = rng.randn(3)
            d /= np.linalg.norm(d)
            candidate = goal + d * r_dist
            if nearest_dist(candidate) > inflate_radius:
                dist_to_goal = np.linalg.norm(candidate - goal)
                print(f"[种子] 多向搜索成功，偏移={dist_to_goal:.4f}m", flush=True)
                print(f"  种子坐标: {np.array2string(candidate, precision=4)}",
                      flush=True)
                return candidate, dist_to_goal

    raise RuntimeError("无法在目标点附近找到自由空间种子点！")


# ── 手动逐步 IRIS，可观察每次迭代椭球增长 ─────────────────────────────────────
def run_iris_verbose(obstacles: list, seed: np.ndarray,
                     domain_center: np.ndarray, domain_half: float = 5.0,
                     max_iter: int = 50, tol: float = 2e-2):
    """手动实现 IRIS 迭代，逐步打印 log(det C) 变化。

    IRIS 两步：
      E 步：HPolyhedron 内最大内切椭球（MaximumVolumeInscribedEllipsoid）
      P 步：对每个障碍物添加切割超平面（intersect_halfspace_with_ellipsoid）
            即找分离 E 和 obstacle 的超平面

    Drake 封装了整个过程在 Iris()，但不暴露逐步信息。
    这里我们直接调用 Iris() 并对比不同 IrisOptions 参数的影响，
    同时用障碍物最近距离分析判断椭球大小是否合理。
    """
    from pydrake.geometry.optimization import Iris, IrisOptions, HPolyhedron
    from scipy.spatial import cKDTree

    lb = domain_center - domain_half
    ub = domain_center + domain_half
    domain = HPolyhedron.MakeBox(lb, ub)

    print(f"[IRIS] 种子: {np.array2string(seed, precision=4)}", flush=True)

    # —— 诊断 1：分析种子点周围障碍物分布 ——
    all_centers_list = []
    for obs in obstacles:
        verts = obs.vertices().T   # (8, 3)
        all_centers_list.append(verts.mean(axis=0))
    obs_centers = np.array(all_centers_list)
    tree = cKDTree(obs_centers)

    dists, idxs = tree.query(seed, k=min(10, len(obs_centers)))
    print(f"\n[诊断] 种子点最近 {len(dists)} 个障碍物中心距离：", flush=True)
    for rank, (d, idx) in enumerate(zip(dists, idxs)):
        print(f"  #{rank+1:2d}  d={d:.5f}m  中心={np.array2string(obs_centers[idx], precision=3)}",
              flush=True)

    # —— 诊断 2：各轴向上最近障碍物 ——
    axes = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    names = ['X', 'Y', 'Z']
    print(f"\n[诊断] 各轴向最近障碍物（步长0.001m向两侧扫描）：", flush=True)
    inflate_r = dists[0] * 0.5   # 粗估 inflate_radius（最近距离的一半）
    for ax, nm in zip(axes, names):
        for sign, label in [(+1, '+'), (-1, '-')]:
            d_scan = 0.0
            found = False
            for _ in range(2000):
                d_scan += 0.001
                pt = seed + ax * sign * d_scan
                dd, _ = tree.query(pt)
                if dd < inflate_r:
                    print(f"  {nm}{label}: 距离 {d_scan:.4f}m 遇到障碍物", flush=True)
                    found = True
                    break
            if not found:
                print(f"  {nm}{label}: 2m 内无障碍物（自由）", flush=True)

    # —— 多种 IrisOptions 设置对比 ——
    # 结论：默认 tol=0.02 在瓶颈地形下会早停，需要 iter>=50 + tol<=1e-4
    configs = [
        ("默认(iter=100,tol=2e-2)",
         dict(iteration_limit=100, termination_threshold=2e-2,
              relative_termination_threshold=1e-3)),
        ("中等(iter=50,tol=1e-4)",
         dict(iteration_limit=50,  termination_threshold=1e-4,
              relative_termination_threshold=1e-5)),
        ("充分(iter=200,tol=1e-4)",
         dict(iteration_limit=200, termination_threshold=1e-4,
              relative_termination_threshold=1e-5)),
    ]

    best_region = None
    best_vol = -1.0

    for label, params in configs:
        opts = IrisOptions()
        opts.iteration_limit = params["iteration_limit"]
        opts.termination_threshold = params["termination_threshold"]
        opts.relative_termination_threshold = params["relative_termination_threshold"]
        t0 = time.time()
        try:
            region = Iris(obstacles, seed, domain, opts)
            elapsed = time.time() - t0
            ell = region.MaximumVolumeInscribedEllipsoid()
            A = ell.A()
            _, s, _ = np.linalg.svd(A)
            radii = 1.0 / s
            vol = (4/3) * np.pi * np.prod(radii)
            print(f"\n[IRIS/{label}]  耗时={elapsed:.2f}s  "
                  f"半轴=[{radii[0]:.4f},{radii[1]:.4f},{radii[2]:.4f}]  "
                  f"vol={vol:.5f} m³", flush=True)
            if vol > best_vol:
                best_vol = vol
                best_region = region
        except Exception as e:
            print(f"\n[IRIS/{label}] 失败: {e}", flush=True)

    return best_region


# ── 求最大内切椭球 + 分析 ──────────────────────────────────────────────────────
def compute_inscribed_ellipsoid(region, goal: np.ndarray, centers: np.ndarray,
                                inflate_radius: float):
    ell = region.MaximumVolumeInscribedEllipsoid()
    center = ell.center()
    A = ell.A()
    U, s, _ = np.linalg.svd(A)
    radii = 1.0 / s        # 三个半轴长度
    volume = (4.0 / 3.0) * np.pi * np.prod(radii)

    dist_to_goal = np.linalg.norm(center - goal)

    # 验证：点云中有多少点在椭球内 {x | ||A(x-c)||_2 <= 1}
    if len(centers) > 0:
        diff = (centers - center) @ A.T     # (N, 3)
        vals = np.linalg.norm(diff, axis=1)  # (N,)
        n_inside = int((vals < 1.0).sum())
    else:
        n_inside = 0

    print(f"\n[椭球] 中心:   {np.array2string(center, precision=4)}", flush=True)
    print(f"[椭球] 到目标点距离: {dist_to_goal:.4f}m", flush=True)
    print(f"[椭球] 三个半轴:    [{radii[0]:.4f}, {radii[1]:.4f}, {radii[2]:.4f}] m",
          flush=True)
    print(f"[椭球] 体积:       {volume:.4f} m³", flush=True)
    print(f"[椭球] 障碍物点在椭球内: {n_inside} / {len(centers)} "
          f"({'✓ 全部在外' if n_inside == 0 else '✗ 有点在内！'})", flush=True)

    return ell, center, radii, U, dist_to_goal


# ── 可视化 ─────────────────────────────────────────────────────────────────────
def visualize(meshcat, centers: np.ndarray, ell, center: np.ndarray,
              radii: np.ndarray, U: np.ndarray,
              goal: np.ndarray, seed: np.ndarray):
    from pydrake.all import Rgba
    from pydrake.geometry import Ellipsoid, Sphere
    from pydrake.math import RigidTransform, RotationMatrix
    from pydrake.perception import PointCloud as DrakePC, Fields, BaseField

    meshcat.Delete("free_ellipsoid")

    # 点云（绿色）
    n = len(centers)
    if n > 0:
        cloud = DrakePC(n, Fields(BaseField.kXYZs | BaseField.kRGBs))
        cloud.mutable_xyzs()[:] = centers.astype(np.float32).T
        cloud.mutable_rgbs()[:] = np.tile(
            np.array([80, 220, 80], dtype=np.uint8).reshape(3, 1), (1, n))
        meshcat.SetObject("free_ellipsoid/pointcloud", cloud, point_size=0.008)

    # 最大内切椭球（蓝色半透明）
    meshcat.SetObject(
        "free_ellipsoid/ellipsoid",
        Ellipsoid(float(radii[0]), float(radii[1]), float(radii[2])),
        Rgba(0.2, 0.5, 1.0, 0.35),
    )
    meshcat.SetTransform(
        "free_ellipsoid/ellipsoid",
        RigidTransform(RotationMatrix(U), center),
    )

    # 目标点（红球）
    meshcat.SetObject("free_ellipsoid/goal",
                      Sphere(0.07), Rgba(1.0, 0.2, 0.2, 1.0))
    meshcat.SetTransform("free_ellipsoid/goal", RigidTransform(goal))

    # 椭球中心（黄球）
    meshcat.SetObject("free_ellipsoid/ellipsoid_center",
                      Sphere(0.05), Rgba(1.0, 0.9, 0.1, 1.0))
    meshcat.SetTransform("free_ellipsoid/ellipsoid_center",
                         RigidTransform(center))

    # 种子点（白球，若与中心不同）
    if np.linalg.norm(seed - goal) > 0.01:
        meshcat.SetObject("free_ellipsoid/seed",
                          Sphere(0.04), Rgba(1.0, 1.0, 1.0, 0.8))
        meshcat.SetTransform("free_ellipsoid/seed", RigidTransform(seed))

    print(f"\n[Viz] 已渲染: 点云({n}点) + 蓝色椭球 + 红色目标点 + 黄色椭球中心",
          flush=True)


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print("单任务测试：目标点附近最大自由空间内切椭球", flush=True)
    print("=" * 60, flush=True)

    cfg = Config()

    print("\n[1/5] 构建 Drake 场景 ...", flush=True)
    plant, plant_ctx, scene_model, meshcat, diagram, context = build_scene(cfg)
    diagram.ForcedPublish(context)

    print("\n[2/5] 点云采样 → VPolytope 障碍物 ...", flush=True)
    obstacles, centers, _ = extract_obstacles(plant, plant_ctx, scene_model, cfg)

    print("\n[3/5] 计算目标点，寻找自由空间种子 ...", flush=True)
    goal = compute_goal(plant, plant_ctx, scene_model, cfg)
    seed, offset = find_free_seed(goal, centers, cfg.inflate_radius)

    print("\n[4/5] 运行 Iris（详细诊断） ...", flush=True)
    region = run_iris_verbose(obstacles, seed, domain_center=goal, domain_half=6.0)

    print("\n[5/5] 最大内切椭球 + 可视化 ...", flush=True)
    ell, ell_center, radii, U, dist_goal = compute_inscribed_ellipsoid(
        region, goal, centers, cfg.inflate_radius)
    visualize(meshcat, centers, ell, ell_center, radii, U, goal, seed)

    print(f"\n{'='*60}", flush=True)
    print(f"结果摘要", flush=True)
    print(f"  目标点:       {np.array2string(goal, precision=3)}", flush=True)
    print(f"  椭球中心:     {np.array2string(ell_center, precision=3)}", flush=True)
    print(f"  中心到目标:   {dist_goal:.4f} m", flush=True)
    print(f"  半轴 a/b/c:   {radii[0]:.3f} / {radii[1]:.3f} / {radii[2]:.3f} m",
          flush=True)
    print(f"  椭球体积:     {(4/3)*np.pi*np.prod(radii):.4f} m³", flush=True)
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
