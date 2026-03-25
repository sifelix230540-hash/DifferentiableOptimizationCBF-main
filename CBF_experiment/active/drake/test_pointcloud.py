"""单模块开环测试：工件 Mesh → 点云转换与可视化

功能：
  1. 加载场景 URDF（Drake），获取各 link 在世界坐标系的精确位姿
  2. 用 trimesh 加载 OBJ mesh，执行表面均匀采样
  3. 将采样点变换到世界坐标系，得到完整点云
  4. 在 MeshCat 中同时显示：原始 mesh（半透明）+ 点云（Drake PointCloud）
  5. 打印点云统计信息：数量、包围盒、密度

不依赖任何控制器 / IRIS / GCS / CBF 模块。
"""

from __future__ import annotations

import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSETS        = ROOT / "assets"
SCENE_PKG_DIR = ASSETS / "cad_exports" / "model_CAD" / "scene"
SCENE_URDF    = SCENE_PKG_DIR / "urdf" / "中组立0725(1).stp.SLDASM.urdf"
SCENE_PKG_NAME = "中组立0725(1).stp.SLDASM"


@dataclass
class PointCloudConfig:
    """点云采样参数。"""
    # 目标采样密度 (点/m²)，按表面积自动决定各 body 采样点数
    target_density: float = 50.0
    # 单个 body 点数上下限，防止极端情况
    min_samples: int = 50
    max_samples: int = 20000
    scene_translation: list[float] = field(default_factory=lambda: [0.0, 4.0, 0.0])
    point_size: float = 0.008
    auto_open_browser: bool = True


def build_drake_scene(cfg: PointCloudConfig):
    """最小化 Drake 环境：仅加载场景 URDF，获取各 body 世界位姿。"""
    import re
    from pydrake.all import (
        AddMultibodyPlantSceneGraph,
        DiagramBuilder,
        MeshcatVisualizer,
        Parser,
        RigidTransform,
        StartMeshcat,
    )

    meshcat = StartMeshcat()
    print(f"[MeshCat] {meshcat.web_url()}", flush=True)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    urdf_text = SCENE_URDF.read_text(encoding="utf-8")
    urdf_text = re.sub(r'\.STL\b', '.obj', urdf_text, flags=re.IGNORECASE)

    parser = Parser(plant)
    parser.package_map().Add(SCENE_PKG_NAME, str(SCENE_PKG_DIR))
    model = parser.AddModelsFromString(urdf_text, "urdf")[0]

    base_body = plant.GetBodyByName("base_link", model)
    t = np.array(cfg.scene_translation, dtype=float)
    plant.WeldFrames(plant.world_frame(), base_body.body_frame(), RigidTransform(t))

    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    return plant, plant_context, model, meshcat, diagram, context


def mesh_to_pointcloud(
    plant, plant_context, model_instance,
    target_density: float,
    min_samples: int,
    max_samples: int,
) -> dict[str, dict]:
    """将场景各 body 的 mesh 转换为世界坐标系点云。

    按表面积 × target_density 自动决定各 body 采样点数，
    保证整个点云在空间上分布均匀（每平方米相同数量的点）。

    Returns:
        {body_name: {"points": Nx3, "normals": Nx3, ...}} 字典
    """
    import trimesh

    mesh_dir = SCENE_PKG_DIR / "meshes"
    result: dict[str, dict] = {}

    for body_idx in plant.GetBodyIndices(model_instance):
        body = plant.get_body(body_idx)
        name = body.name()

        obj_path = mesh_dir / f"{name}.obj"
        if not obj_path.exists():
            print(f"  [跳过] {name}: {obj_path.name} 不存在", flush=True)
            continue

        mesh = trimesh.load(str(obj_path), force="mesh")

        X_WB = plant.CalcRelativeTransform(
            plant_context, plant.world_frame(), body.body_frame()
        )
        R = X_WB.rotation().matrix()
        t = X_WB.translation()

        n_verts = mesh.vertices.shape[0]
        n_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0
        area = mesh.area if hasattr(mesh, 'area') else 0.0

        # 按密度计算目标点数，并夹到 [min_samples, max_samples]
        count_by_density = int(area * target_density) if area > 0 else min_samples
        count = max(min_samples, min(max_samples, count_by_density))

        pts_local, face_indices = trimesh.sample.sample_surface(mesh, count)
        normals_local = mesh.face_normals[face_indices]

        pts_world = (R @ pts_local.T).T + t
        normals_world = (R @ normals_local.T).T

        result[name] = {
            "points": pts_world,
            "normals": normals_world,
            "n_vertices": n_verts,
            "n_faces": n_faces,
            "area": area,
            "actual_density": pts_world.shape[0] / area if area > 0 else 0.0,
        }

        bb_min = pts_world.min(axis=0)
        bb_max = pts_world.max(axis=0)
        bb_size = bb_max - bb_min

        print(f"  [{name}]", flush=True)
        print(f"    原始: {n_verts} 顶点, {n_faces} 面, 面积={area:.4f} m²", flush=True)
        print(f"    按密度计算: {count_by_density} 点 → 实际采样 {pts_world.shape[0]} 点  "
              f"(目标密度={target_density:.1f}, 实际={pts_world.shape[0]/area:.1f} 点/m²)" if area > 0
              else f"    采样: {pts_world.shape[0]} 点  (面积为0，使用最小值)", flush=True)
        print(f"    位姿: t={np.array2string(t, precision=3)}", flush=True)
        print(f"    包围盒 min={np.array2string(bb_min, precision=3)}", flush=True)
        print(f"    包围盒 max={np.array2string(bb_max, precision=3)}", flush=True)
        print(f"    包围盒 size={np.array2string(bb_size, precision=3)}", flush=True)

    return result


BODY_COLORS_RGB = {
    "base_link": (50,  150, 255),
    "l2":        (255,  80,  80),
    "l3":        (80,  255,  80),
}


def visualize_pointcloud(meshcat, pointclouds: dict, point_size: float):
    """在 MeshCat 中用 Drake PointCloud 高效可视化点云（一次性渲染）。"""
    from pydrake.perception import PointCloud as DrakePointCloud
    from pydrake.perception import Fields, BaseField

    total_points = 0

    for body_name, data in pointclouds.items():
        pts = data["points"].astype(np.float32)
        n = len(pts)
        rgb = BODY_COLORS_RGB.get(body_name, (200, 200, 50))

        cloud = DrakePointCloud(n, Fields(BaseField.kXYZs | BaseField.kRGBs))
        cloud.mutable_xyzs()[:] = pts.T
        colors = np.tile(np.array(rgb, dtype=np.uint8).reshape(3, 1), (1, n))
        cloud.mutable_rgbs()[:] = colors

        path = f"pointcloud/{body_name}"
        meshcat.SetObject(path, cloud, point_size=point_size)

        total_points += n
        print(f"  [{body_name}] {n} 个点 → MeshCat (颜色 RGB={rgb})", flush=True)

    print(f"  总计可视化 {total_points} 个点", flush=True)


def save_pointcloud(pointclouds: dict, output_dir: Path):
    """将点云保存为 .npy 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pts = []
    all_normals = []

    for body_name, data in pointclouds.items():
        pts = data["points"]
        normals = data["normals"]

        np.save(output_dir / f"{body_name}_points.npy", pts)
        np.save(output_dir / f"{body_name}_normals.npy", normals)
        print(f"  已保存: {body_name}_points.npy ({pts.shape})", flush=True)

        all_pts.append(pts)
        all_normals.append(normals)

    merged_pts = np.vstack(all_pts)
    merged_normals = np.vstack(all_normals)
    np.save(output_dir / "merged_points.npy", merged_pts)
    np.save(output_dir / "merged_normals.npy", merged_normals)
    print(f"  已保存: merged_points.npy ({merged_pts.shape})", flush=True)
    print(f"  保存目录: {output_dir}", flush=True)


def print_summary(pointclouds: dict):
    """打印点云总结。"""
    print("\n" + "=" * 60, flush=True)
    print("点云转换结果摘要", flush=True)
    print("=" * 60, flush=True)

    total_pts = 0
    all_pts = []

    for body_name, data in pointclouds.items():
        pts = data["points"]
        total_pts += len(pts)
        all_pts.append(pts)
        actual_d = data.get("actual_density", 0.0)
        print(f"  {body_name:>12s}: {len(pts):>5d} 点  "
              f"(原始 {data['n_vertices']} 顶点, {data['n_faces']} 面, "
              f"密度={actual_d:.1f} 点/m²)", flush=True)

    merged = np.vstack(all_pts)
    bb_min = merged.min(axis=0)
    bb_max = merged.max(axis=0)
    bb_size = bb_max - bb_min

    print(f"  {'合计':>10s}: {total_pts:>5d} 点", flush=True)
    print(f"\n  全局包围盒:", flush=True)
    print(f"    min  = [{bb_min[0]:.3f}, {bb_min[1]:.3f}, {bb_min[2]:.3f}]", flush=True)
    print(f"    max  = [{bb_max[0]:.3f}, {bb_max[1]:.3f}, {bb_max[2]:.3f}]", flush=True)
    print(f"    size = [{bb_size[0]:.3f}, {bb_size[1]:.3f}, {bb_size[2]:.3f}]", flush=True)
    print(f"    体积 = {np.prod(bb_size):.4f} m³", flush=True)

    from scipy.spatial import cKDTree
    tree = cKDTree(merged)
    dd, _ = tree.query(merged, k=2)
    nn_dist = dd[:, 1]
    print(f"\n  最近邻距离统计:", flush=True)
    print(f"    min={nn_dist.min():.5f}  "
          f"mean={nn_dist.mean():.5f}  max={nn_dist.max():.5f}", flush=True)
    print("=" * 60, flush=True)


def main():
    print("=" * 60, flush=True)
    print("单模块测试：工件 Mesh → 点云转换", flush=True)
    print("=" * 60, flush=True)

    cfg = PointCloudConfig()

    print(f"\n参数:", flush=True)
    print(f"  目标采样密度: {cfg.target_density} 点/m²", flush=True)
    print(f"  单体点数范围: [{cfg.min_samples}, {cfg.max_samples}]", flush=True)
    print(f"  场景平移: {cfg.scene_translation}", flush=True)
    print(f"  可视化点大小: {cfg.point_size} m", flush=True)

    print(f"\n[1/4] 构建 Drake 场景 ...", flush=True)
    plant, plant_ctx, model, meshcat, diagram, context = build_drake_scene(cfg)
    diagram.ForcedPublish(context)
    print("  Drake 场景就绪", flush=True)

    print(f"\n[2/4] Mesh → 点云采样 (密度={cfg.target_density} 点/m²) ...", flush=True)
    pointclouds = mesh_to_pointcloud(
        plant, plant_ctx, model,
        target_density=cfg.target_density,
        min_samples=cfg.min_samples,
        max_samples=cfg.max_samples,
    )

    if not pointclouds:
        print("错误：未生成任何点云！检查 OBJ 文件是否存在。", flush=True)
        return

    print(f"\n[3/4] MeshCat 点云可视化 ...", flush=True)
    visualize_pointcloud(meshcat, pointclouds, cfg.point_size)

    print(f"\n[4/4] 保存点云 ...", flush=True)
    output_dir = ROOT / "CBF_experiment" / "active" / "drake" / "pointcloud_output"
    save_pointcloud(pointclouds, output_dir)

    print_summary(pointclouds)

    url = meshcat.web_url()
    if cfg.auto_open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    print(f"\n可视化地址: {url}", flush=True)
    print("在 MeshCat 中检查:", flush=True)
    print("  - 半透明原始 mesh（Drake 自动渲染）", flush=True)
    print("  - 蓝色点云 = base_link", flush=True)
    print("  - 红色点云 = l2", flush=True)
    print("  - 绿色点云 = l3", flush=True)
    print("\n按 Ctrl+C 退出", flush=True)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n已退出", flush=True)


if __name__ == "__main__":
    main()
