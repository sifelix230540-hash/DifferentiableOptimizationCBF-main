"""离线 DRM 构建：C-space 采样 → 自碰撞过滤 → k-NN 邻接 → COLRM → 序列化。

三大耗时步骤均支持 multiprocessing 并行：
  1) 采样 — 每个子进程独立 oracle 批量检测
  2) 边过滤 — 复用 visibility.py 的 oracle 工厂模式
  3) COLRM  — 每个子进程独立 PyBullet + FK
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, save_npz, load_npz

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.robot_model import (
    compose_full_q,
    sample_joint_box,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig

_log = lambda *a, **kw: print(*a, **kw, flush=True)


# ═══════════════════════════════════════════════════════
#  Oracle 工厂（子进程可序列化重建）
# ═══════════════════════════════════════════════════════

def _make_oracle_factory_spec(config: RobotQueryConfig) -> tuple:
    return (
        "CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.coal_oracle",
        "CoalSelfCollisionOracle",
        {"config": config},
    )


_worker_oracle = None
_worker_robot = None
_worker_metadata = None


def _worker_init(factory_spec: tuple):
    global _worker_oracle, _worker_robot, _worker_metadata
    mod = importlib.import_module(factory_spec[0])
    cls = getattr(mod, factory_spec[1])
    _worker_oracle = cls(**factory_spec[2])
    _worker_robot = _worker_oracle.robot
    _worker_metadata = _worker_oracle.metadata


# ═══════════════════════════════════════════════════════
#  Step 1: 并行采样 + 自碰撞过滤
# ═══════════════════════════════════════════════════════

def _worker_filter_batch(batch: np.ndarray) -> np.ndarray:
    """子进程：对一批配置做自碰撞过滤，返回通过的配置。"""
    free = [q for q in batch if not _worker_oracle.is_self_collision(q)]
    return np.array(free) if free else np.empty((0, batch.shape[1]))


def sample_free_nodes_parallel(
    config: RobotQueryConfig,
    n_target: int,
    n_workers: int,
    batch_per_worker: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """多进程并行采样无自碰撞节点。"""
    factory_spec = _make_oracle_factory_spec(config)

    # 先单进程获取 metadata 信息
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.coal_oracle import CoalSelfCollisionOracle
    oracle_tmp = CoalSelfCollisionOracle(config)
    metadata = oracle_tmp.metadata
    oracle_tmp.close()

    rng = np.random.default_rng(seed)
    collected: list[np.ndarray] = []
    total_sampled = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(factory_spec,)) as pool:
        while sum(len(c) for c in collected) < n_target:
            batches = [sample_joint_box(metadata, rng, batch_per_worker) for _ in range(n_workers)]
            total_sampled += batch_per_worker * n_workers
            results = pool.map(_worker_filter_batch, batches)
            for r in results:
                if len(r) > 0:
                    collected.append(r)
            n_got = sum(len(c) for c in collected)
            rate = n_got / total_sampled if total_sampled > 0 else 0
            _log(f"  采样进度: {n_got}/{n_target}  (自碰撞通过率 {rate:.1%})")

    nodes = np.vstack(collected)[:n_target]
    _log(f"  最终保留 {len(nodes)} 个无自碰撞节点")
    return nodes, metadata


# ═══════════════════════════════════════════════════════
#  Step 2: k-NN 邻接 + 并行边过滤
# ═══════════════════════════════════════════════════════

def _worker_check_edges(task: tuple) -> list[tuple[int, int, float]]:
    """子进程：检测一批边的自碰撞安全性。"""
    edge_list, nodes, interp_steps = task
    valid = []
    for i, j, dist in edge_list:
        if _worker_oracle.segment_is_collision_free(nodes[i], nodes[j], num_steps=interp_steps):
            valid.append((int(i), int(j), float(dist)))
    return valid


def build_knn_adjacency_parallel(
    nodes: np.ndarray,
    config: RobotQueryConfig,
    k: int = 15,
    edge_interp_steps: int = 8,
    n_workers: int = 4,
) -> csr_matrix:
    """并行 k-NN 邻接图构建 + 自碰撞边过滤。"""
    n = len(nodes)
    tree = KDTree(nodes)
    dists, indices = tree.query(nodes, k=k + 1)

    # 收集去重候选边
    candidate_edges: list[tuple[int, int, float]] = []
    seen = set()
    for i in range(n):
        for jj in range(1, k + 1):
            j = int(indices[i, jj])
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                candidate_edges.append((i, j, float(dists[i, jj])))
    _log(f"  候选边: {len(candidate_edges)}")

    # 分块并行
    factory_spec = _make_oracle_factory_spec(config)
    chunk_size = max(1, len(candidate_edges) // (n_workers * 4))
    chunks = []
    for start in range(0, len(candidate_edges), chunk_size):
        chunk = candidate_edges[start:start + chunk_size]
        chunks.append((chunk, nodes, edge_interp_steps))

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(factory_spec,)) as pool:
        results = pool.map(_worker_check_edges, chunks)

    rows, cols, weights = [], [], []
    n_valid = 0
    for chunk_result in results:
        for i, j, w in chunk_result:
            rows.extend([i, j])
            cols.extend([j, i])
            weights.extend([w, w])
            n_valid += 1

    adj = csr_matrix((weights, (rows, cols)), shape=(n, n))
    _log(f"  邻接图: {n} 节点, {n_valid}/{len(candidate_edges)} 边通过 ({n_valid/max(len(candidate_edges),1):.1%})")
    return adj


# ═══════════════════════════════════════════════════════
#  Step 3: 并行 COLRM 构建
# ═══════════════════════════════════════════════════════

def _worker_colrm_batch(task: tuple) -> list[tuple[int, list[int]]]:
    """子进程：对一批节点做 FK → mesh顶点 → 体素映射。"""
    node_ids, nodes, vg_lo, vg_res, vg_shape, link_indices = task
    import pybullet as p
    from CBF_experiment.active.pybullet.drm.voxel_grid import load_link_collision_vertices

    robot = _worker_robot
    metadata = _worker_metadata
    ny, nz = int(vg_shape[1]), int(vg_shape[2])

    # 在子进程中加载 mesh 顶点
    link_meshes = load_link_collision_vertices(robot.body_id, link_indices)

    results = []
    for node_id, q6 in zip(node_ids, nodes):
        q_full = compose_full_q(metadata, q6)
        robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        occupied_keys: list[int] = []
        for li, verts_local in link_meshes.items():
            state = p.getLinkState(robot.body_id, li, computeForwardKinematics=True)
            pos = np.array(state[4], dtype=float)
            R = np.array(p.getMatrixFromQuaternion(state[5]), dtype=float).reshape(3, 3)
            verts_world = (R @ verts_local.T).T + pos
            ijk = np.floor((verts_world - vg_lo) / vg_res).astype(np.int32)
            valid = np.all((ijk >= 0) & (ijk < vg_shape), axis=1)
            ijk_valid = ijk[valid]
            keys = ijk_valid[:, 0].astype(np.int64) * ny * nz + ijk_valid[:, 1] * nz + ijk_valid[:, 2]
            occupied_keys.extend(set(keys.tolist()))
        results.append((int(node_id), list(set(occupied_keys))))
    return results


def build_colrm_parallel(
    nodes: np.ndarray,
    config: RobotQueryConfig,
    vg_lo: np.ndarray,
    vg_res: float,
    vg_shape: np.ndarray,
    link_indices: list[int],
    n_workers: int = 4,
) -> dict[int, set[int]]:
    """并行构建 COLRM: voxel_key → set[node_id]。"""
    n = len(nodes)
    factory_spec = _make_oracle_factory_spec(config)

    # 分块
    chunk_size = max(1, n // (n_workers * 2))
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        ids = list(range(start, end))
        chunks.append((ids, nodes[start:end], vg_lo, vg_res, vg_shape, link_indices))

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(factory_spec,)) as pool:
        results = pool.map(_worker_colrm_batch, chunks)

    colrm: dict[int, set[int]] = defaultdict(set)
    for chunk_result in results:
        for node_id, keys in chunk_result:
            for vk in keys:
                colrm[vk].add(node_id)

    _log(f"  COLRM: {len(colrm)} 体素有映射, "
         f"平均每体素 {np.mean([len(v) for v in colrm.values()]):.1f} 个节点")
    return dict(colrm)


# ═══════════════════════════════════════════════════════
#  Step 4: EE 位姿（单进程，快速）
# ═══════════════════════════════════════════════════════

def build_ee_pose_index(robot, metadata, nodes: np.ndarray) -> np.ndarray:
    n = len(nodes)
    poses = np.zeros((n, 7), dtype=float)
    for i, q6 in enumerate(nodes):
        q_full = compose_full_q(metadata, q6)
        robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        pos, quat = robot.get_ee_pose()
        poses[i, :3] = pos
        poses[i, 3:] = quat
    _log(f"  EE 位姿索引: {n} 个")
    return poses


# ═══════════════════════════════════════════════════════
#  序列化 / 反序列化
# ═══════════════════════════════════════════════════════

def save_drm(path, nodes, adjacency, ee_poses, colrm, voxel_params):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "nodes.npz", nodes=nodes, ee_poses=ee_poses)
    save_npz(out / "adjacency.npz", adjacency)
    colrm_ser = {str(k): list(v) for k, v in colrm.items()}
    with open(out / "colrm.json", "w") as f:
        json.dump(colrm_ser, f)
    with open(out / "voxel_params.json", "w") as f:
        json.dump(voxel_params, f, indent=2)
    _log(f"  DRM 已保存到 {out}")


def load_drm(path):
    d = Path(path)
    data = np.load(d / "nodes.npz")
    nodes, ee_poses = data["nodes"], data["ee_poses"]
    adjacency = load_npz(d / "adjacency.npz")
    with open(d / "colrm.json", "r") as f:
        raw = json.load(f)
    colrm = {int(k): set(v) for k, v in raw.items()}
    with open(d / "voxel_params.json", "r") as f:
        voxel_params = json.load(f)
    _log(f"  DRM 已加载: {len(nodes)} 节点, {adjacency.nnz//2} 边, {len(colrm)} 体素映射")
    return nodes, adjacency, ee_poses, colrm, voxel_params


# ═══════════════════════════════════════════════════════
#  完整构建流水线
# ═══════════════════════════════════════════════════════

def build_drm_pipeline(
    n_nodes: int = 50000,
    k_neighbors: int = 20,
    edge_interp_steps: int = 8,
    voxel_resolution: float = 0.04,
    voxel_half_extents: tuple[float, float, float] = (0.8, 0.8, 0.8),
    output_dir: str = "artifacts/drm_data",
    robot_config: RobotQueryConfig | None = None,
    seed: int = 42,
    n_workers: int = 0,
):
    config = robot_config or RobotQueryConfig()
    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 1) - 1)
    _log(f"并行 workers: {n_workers}")

    # ── Step 1 ──
    _log("\n[Step 1/4] 并行采样无自碰撞节点 ...")
    t0 = time.time()
    nodes, metadata = sample_free_nodes_parallel(
        config, n_nodes, n_workers=n_workers,
        batch_per_worker=max(500, n_nodes // (n_workers * 5)),
        seed=seed,
    )
    _log(f"  Step 1 耗时 {time.time()-t0:.1f}s\n")

    # 获取 robobase 位置（需要一个 PyBullet 连接）
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.coal_oracle import CoalSelfCollisionOracle
    oracle_main = CoalSelfCollisionOracle(config)
    robot = oracle_main.robot
    robobase_pos, _ = robot.get_robobase_pose()

    from CBF_experiment.active.pybullet.drm.voxel_grid import VoxelGrid, load_link_collision_vertices
    vg = VoxelGrid(robobase_pos, np.array(voxel_half_extents), voxel_resolution)
    _log(f"体素网格: {vg}")

    # 收集需要监控的连杆索引
    monitored_links = list(metadata.monitored_link_ids)
    if robot.welding_gun_base_link_index >= 0:
        monitored_links.append(robot.welding_gun_base_link_index)
    if robot.ee_link_index >= 0 and robot.ee_link_index not in monitored_links:
        monitored_links.append(robot.ee_link_index)
    link_meshes = load_link_collision_vertices(robot.body_id, monitored_links)
    total_verts = sum(len(v) for v in link_meshes.values())
    _log(f"碰撞 mesh: {len(link_meshes)} 连杆, {total_verts} 顶点")

    # ── Step 2 ──
    _log("\n[Step 2/4] 并行 k-NN 邻接图 + 自碰撞边过滤 ...")
    t0 = time.time()
    adjacency = build_knn_adjacency_parallel(
        nodes, config, k=k_neighbors,
        edge_interp_steps=edge_interp_steps,
        n_workers=n_workers,
    )
    _log(f"  Step 2 耗时 {time.time()-t0:.1f}s\n")

    # ── Step 3 ──
    _log("[Step 3/4] 并行 COLRM 构建 ...")
    t0 = time.time()
    colrm = build_colrm_parallel(
        nodes, config,
        vg_lo=vg.lo, vg_res=vg.res, vg_shape=vg.shape,
        link_indices=monitored_links,
        n_workers=n_workers,
    )
    _log(f"  Step 3 耗时 {time.time()-t0:.1f}s\n")

    # ── Step 4 ──
    _log("[Step 4/4] 计算 EE 位姿索引 ...")
    t0 = time.time()
    ee_poses = build_ee_pose_index(robot, metadata, nodes)
    _log(f"  Step 4 耗时 {time.time()-t0:.1f}s\n")

    voxel_params = {
        "origin": robobase_pos.tolist(),
        "half_extents": list(voxel_half_extents),
        "resolution": voxel_resolution,
    }
    save_drm(output_dir, nodes, adjacency, ee_poses, colrm, voxel_params)
    oracle_main.close()
    _log("DRM 构建完成。")
    return nodes, adjacency, ee_poses, colrm, vg
