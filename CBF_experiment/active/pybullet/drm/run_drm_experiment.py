"""DRM 实验入口：离线构建 → 在线规划 → PyBullet GUI 可视化 + 录像。

Usage
-----
# 构建（无 GUI，约需几分钟）
python -m CBF_experiment.active.pybullet.drm.run_drm_experiment --build --n-nodes 20000

# 从已有数据规划 + 可视化
python -m CBF_experiment.active.pybullet.drm.run_drm_experiment --plan --gui

# 一步到位（构建 + 规划 + 可视化）
python -m CBF_experiment.active.pybullet.drm.run_drm_experiment --build --plan --gui --n-nodes 10000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import (
    SimulationScene, Robot, load_config,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.robot_model import compose_full_q
from CBF_experiment.active.pybullet.self_collision.self_collision_cspace_hulls import (
    extract_revolute_metadata,
    extract_self_collision_monitor_metadata,
    build_monitored_link_pairs,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import RobotModelMetadata
from CBF_experiment.active.pybullet.drm.voxel_grid import VoxelGrid
from CBF_experiment.active.pybullet.drm.drm_builder import build_drm_pipeline, load_drm, save_drm
from CBF_experiment.active.pybullet.drm.drm_planner import (
    plan_point_to_point,
    voxelize_obstacle_mesh,
    prune_nodes,
)

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

DEFAULT_OUTPUT = str(REPO_ROOT / "artifacts" / "drm_data")


def _extract_metadata_from_robot(robot: Robot, config: RobotQueryConfig | None = None) -> RobotModelMetadata:
    """从已加载的 Robot 实例提取 metadata，不再二次加载 URDF。"""
    cfg = config or RobotQueryConfig()
    q_base, dq_base = robot.get_joint_state()
    revolute_ids, revolute_names, joint_limits, q_indices = extract_revolute_metadata(robot)
    monitored_link_ids, monitored_link_names = extract_self_collision_monitor_metadata(
        robot,
        include_welding_gun_base=bool(cfg.INCLUDE_WELDING_GUN_BASE),
        include_third_axis_chain=bool(cfg.INCLUDE_THIRD_AXIS_CHAIN),
    )
    monitored_pairs = build_monitored_link_pairs(monitored_link_ids, min_index_gap=int(cfg.MIN_INDEX_GAP))
    return RobotModelMetadata(
        revolute_ids=tuple(int(x) for x in revolute_ids),
        revolute_names=tuple(str(x) for x in revolute_names),
        joint_limits=tuple((float(lo), float(hi)) for lo, hi in joint_limits),
        q_indices=tuple(int(x) for x in q_indices),
        q_base=np.asarray(q_base, dtype=float),
        dq_base=np.asarray(dq_base, dtype=float),
        monitored_link_ids=tuple(int(x) for x in monitored_link_ids),
        monitored_link_names=tuple(str(x) for x in monitored_link_names),
        monitored_pairs=tuple((int(a), int(b)) for a, b in monitored_pairs),
    )


# ── 可视化工具 ───────────────────────────────────────


def draw_voxel_grid_boundary(vg: VoxelGrid, color=(0.5, 0.5, 0.5), width=1.0):
    """画出体素网格的 AABB 边界框。"""
    lo = vg.lo.tolist()
    hi = vg.hi.tolist()
    corners = [
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        p.addUserDebugLine(corners[a], corners[b], color, lineWidth=width)


def draw_occupied_voxels(vg: VoxelGrid, keys: set[int], color=(1, 0.3, 0.3, 0.3), max_draw=2000):
    """用半透明小方块标记被占据的体素（数量多时随机抽样）。"""
    keys_list = list(keys)
    if len(keys_list) > max_draw:
        rng = np.random.default_rng(0)
        keys_list = rng.choice(keys_list, max_draw, replace=False).tolist()

    r = vg.res * 0.45
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[r, r, r], rgbaColor=list(color))
    body_ids = []
    for k in keys_list:
        center = vg.key_to_center(int(k))
        bid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=center.tolist())
        body_ids.append(bid)
    return body_ids


def draw_ee_trajectory(robot, metadata, path_configs, color=(0, 0.6, 1), width=3.0):
    """在任务空间画出 EE 轨迹折线，并在起点终点画标记。"""
    ee_pts = []
    for q6 in path_configs:
        q_full = compose_full_q(metadata, q6)
        robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        pos, _ = robot.get_ee_pose()
        ee_pts.append(pos.tolist())

    for i in range(len(ee_pts) - 1):
        p.addUserDebugLine(ee_pts[i], ee_pts[i+1], color, lineWidth=width)

    if ee_pts:
        SimulationScene.create_marker(0.015, [0, 1, 0, 1], ee_pts[0])
        SimulationScene.create_marker(0.015, [1, 0, 0, 1], ee_pts[-1])


def animate_path(robot, metadata, path_configs, *, fps=30, video_path=None,
                 video_width=1280, video_height=720):
    """在 GUI 中逐帧播放路径，可选录像。"""
    frames = [] if video_path else None
    dt = 1.0 / fps
    n = len(path_configs)

    for step, q6 in enumerate(path_configs, 1):
        q_full = compose_full_q(metadata, q6)
        robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        ee_pos, ee_quat = robot.get_ee_pose()

        info_text = f"Step {step}/{n}  EE=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})"
        p.addUserDebugText(info_text, [0.0, -0.5, 1.2], [0, 0, 0],
                           textSize=1.5, lifeTime=dt * 2)

        if frames is not None:
            cam = p.getDebugVisualizerCamera()
            _, _, rgb, _, _ = p.getCameraImage(
                video_width, video_height,
                viewMatrix=cam[2], projectionMatrix=cam[3],
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            frames.append(np.array(rgb, dtype=np.uint8).reshape(video_height, video_width, 4)[:, :, :3])

        time.sleep(dt)

    if frames and video_path and imageio is not None:
        out = Path(video_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            imageio.mimsave(str(out), frames, fps=fps)
            print(f"[视频] 已保存: {out}")
        except Exception as e:
            gif_path = out.with_suffix(".gif")
            imageio.mimsave(str(gif_path), frames, fps=fps)
            print(f"[视频] MP4 失败({e}), 已保存 GIF: {gif_path}")


# ── 创建测试障碍物 ───────────────────────────────────


def create_test_obstacle(pos=(0.4, 0.0, 0.5), half_extents=(0.08, 0.08, 0.08)):
    """创建一个可视化的 box 障碍物并返回其顶点（世界坐标）。"""
    hx, hy, hz = half_extents
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                              rgbaColor=[1, 0.2, 0.2, 0.6])
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=list(pos))

    xs = np.linspace(-hx, hx, 5)
    ys = np.linspace(-hy, hy, 5)
    zs = np.linspace(-hz, hz, 5)
    grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)
    verts_world = grid + np.array(pos)
    return body, verts_world


# ── 主函数 ───────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DRM 实验")
    parser.add_argument("--build", action="store_true", help="执行离线 DRM 构建")
    parser.add_argument("--plan", action="store_true", help="执行在线规划")
    parser.add_argument("--gui", action="store_true", help="启动 PyBullet GUI 可视化")
    parser.add_argument("--n-nodes", type=int, default=10000, help="采样节点数")
    parser.add_argument("--k", type=int, default=15, help="k-NN 邻居数")
    parser.add_argument("--resolution", type=float, default=0.04, help="体素分辨率 (m)")
    parser.add_argument("--half-extents", type=float, nargs=3, default=[0.8, 0.8, 0.8])
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="DRM 存储目录")
    parser.add_argument("--obstacle-pos", type=float, nargs=3, default=None,
                        help="障碍物位置 (自动设为 robobase 前方)")
    parser.add_argument("--target-pos", type=float, nargs=3, default=None,
                        help="目标 EE 位置 (自动从 EE 位姿分布中选)")
    parser.add_argument("--video", type=str, default=None, help="录像输出路径")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.build and not args.plan:
        args.build = True
        args.plan = True

    # ── 离线构建 ──
    if args.build:
        print("=" * 60)
        print("离线 DRM 构建")
        print("=" * 60)
        t_total = time.time()
        build_drm_pipeline(
            n_nodes=args.n_nodes,
            k_neighbors=args.k,
            voxel_resolution=args.resolution,
            voxel_half_extents=tuple(args.half_extents),
            output_dir=args.output,
            seed=args.seed,
        )
        print(f"\n总构建耗时: {time.time()-t_total:.1f}s")

    # ── 在线规划 ──
    if args.plan:
        print("\n" + "=" * 60)
        print("在线 DRM 规划")
        print("=" * 60)

        nodes, adjacency, ee_poses, colrm, voxel_params = load_drm(args.output)
        vg = VoxelGrid(
            origin=np.array(voxel_params["origin"]),
            half_extents=np.array(voxel_params["half_extents"]),
            resolution=voxel_params["resolution"],
        )

        dim = nodes.shape[1]
        rng = np.random.default_rng(args.seed)

        # 起点：使用第一个节点
        q_start = nodes[0]

        # 目标 EE 位置：指定或从 roadmap 中选择一个合理目标
        if args.target_pos is not None:
            target_pos = np.array(args.target_pos)
        else:
            from CBF_experiment.active.pybullet.drm.drm_planner import find_connected_components
            from scipy.sparse import csr_matrix as _csr
            valid_mask = np.ones(len(nodes), dtype=bool)
            comp_labels, _ = find_connected_components(adjacency, valid_mask)
            start_dists = np.linalg.norm(nodes - q_start, axis=1)
            start_node = int(np.argmin(start_dists))
            start_comp = int(comp_labels[start_node])
            same_comp = (comp_labels == start_comp)
            same_comp_idx = np.where(same_comp)[0]
            ee_dists = np.linalg.norm(ee_poses[same_comp_idx, :3] - ee_poses[start_node, :3], axis=1)
            # 选择 EE 空间中距离适中的目标（75% 分位）
            p75 = np.percentile(ee_dists, 75)
            near_p75 = same_comp_idx[np.abs(ee_dists - p75) < p75 * 0.1]
            if len(near_p75) == 0:
                near_p75 = same_comp_idx[np.argsort(ee_dists)[-10:]]
            chosen = rng.choice(near_p75)
            target_pos = ee_poses[chosen, :3].copy()
            print(f"  自动选择目标: 节点 {chosen}, EE={target_pos.round(3)}", flush=True)

        # 障碍物：指定或在工作空间边缘放一个小 box
        obs_half = (0.05, 0.05, 0.05)
        if args.obstacle_pos is not None:
            obs_pos = tuple(args.obstacle_pos)
        else:
            ee_max = ee_poses[:, :3].max(axis=0)
            ee_min = ee_poses[:, :3].min(axis=0)
            obs_pos = tuple(((ee_max + ee_min) / 2 + (ee_max - ee_min) * 0.3).tolist())
            print(f"  自动放置障碍物 (工作空间边缘): {np.array(obs_pos).round(3)}")

        xs = np.linspace(-obs_half[0], obs_half[0], 5)
        ys = np.linspace(-obs_half[1], obs_half[1], 5)
        zs = np.linspace(-obs_half[2], obs_half[2], 5)
        obs_verts = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3) + np.array(obs_pos)

        t0 = time.time()
        result = plan_point_to_point(
            nodes, adjacency, ee_poses, colrm, vg,
            q_start_6d=q_start,
            target_pos=target_pos,
            obstacle_vertices=obs_verts,
            epsilon_pos=0.15,
        )
        t_plan = time.time() - t0
        print(f"\n规划耗时: {t_plan*1000:.1f}ms")

        if result["success"]:
            print(f"路径: {len(result['path_indices'])} 步, "
                  f"节点 {result['start_node']}→{result['goal_node']}")
        else:
            print("规划失败")

        # ── GUI 可视化 ──
        if args.gui and result["success"]:
            print("\n启动 PyBullet GUI ...")
            cfg = load_config()
            scene = SimulationScene(cfg)
            robot = Robot(cfg, scene)
            scene.enable_rendering()

            metadata = _extract_metadata_from_robot(robot)

            # 画 AABB
            draw_voxel_grid_boundary(vg, color=(0.7, 0.7, 0.7), width=1.5)

            # 画障碍物
            _, _ = create_test_obstacle(pos=obs_pos, half_extents=obs_half)

            # 标记目标点
            SimulationScene.create_marker(0.02, [1, 0, 0, 1], target_pos.tolist())

            # 画 EE 轨迹
            draw_ee_trajectory(robot, metadata, result["path_configs"],
                               color=(0, 0.6, 1), width=3)

            # 插值路径使动画更平滑
            raw_path = result["path_configs"]
            smooth_path = _interpolate_path(raw_path, n_interp=5)

            # 动画回放
            video_out = args.video or str(REPO_ROOT / "artifacts" / "drm_data" / "drm_path.mp4")
            print(f"动画回放 ({len(smooth_path)} 帧) ...")
            animate_path(robot, metadata, smooth_path, fps=30, video_path=video_out)

            # 保持窗口
            print("\n按 Ctrl+C 退出 GUI ...")
            try:
                while p.isConnected():
                    p.stepSimulation()
                    time.sleep(1.0 / 240.0)
            except KeyboardInterrupt:
                pass

        # 保存规划结果
        result_path = Path(args.output) / "plan_result.json"
        result_save = {
            "success": result["success"],
            "path_indices": result.get("path_indices", []),
            "path_configs": result["path_configs"].tolist() if result["success"] else [],
            "goal_node": result.get("goal_node"),
            "start_node": result.get("start_node"),
            "n_pruned": result.get("n_pruned"),
            "n_valid": result.get("n_valid"),
            "plan_time_ms": t_plan * 1000,
        }
        with open(result_path, "w") as f:
            json.dump(result_save, f, indent=2)
        print(f"规划结果已保存: {result_path}")


def _interpolate_path(path_configs: np.ndarray, n_interp: int = 5) -> np.ndarray:
    """在相邻路径点之间插入中间帧，使动画平滑。"""
    if len(path_configs) < 2:
        return path_configs
    segments = []
    for i in range(len(path_configs) - 1):
        for t in np.linspace(0, 1, n_interp, endpoint=False):
            q = (1 - t) * path_configs[i] + t * path_configs[i + 1]
            segments.append(q)
    segments.append(path_configs[-1])
    return np.array(segments)


if __name__ == "__main__":
    main()
