"""在线 DRM 规划：工件体素化 → 节点剪枝 → X_ε 目标邻域 → A* + lazy edge check → 重试。"""
from __future__ import annotations

import heapq
from collections import defaultdict, deque

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from CBF_experiment.active.pybullet.drm.voxel_grid import VoxelGrid, compute_node_occupied_voxels

_log = lambda *a, **kw: print(*a, **kw, flush=True)


# ── 1. 工件体素化 + 节点剪枝 ─────────────────────────


def voxelize_obstacle_mesh(
    obstacle_vertices: np.ndarray,
    voxel_grid: VoxelGrid,
    T_world_to_robobase: np.ndarray | None = None,
) -> set[int]:
    """将障碍物 mesh 顶点映射到 robobase 系体素。

    Parameters
    ----------
    obstacle_vertices : (M, 3) 障碍物世界坐标顶点
    voxel_grid : 体素网格（robobase 系）
    T_world_to_robobase : (4, 4) 从世界系到 robobase 系的齐次变换，None 表示恒等
    """
    pts = np.asarray(obstacle_vertices, dtype=float)
    if T_world_to_robobase is not None:
        R = T_world_to_robobase[:3, :3]
        t = T_world_to_robobase[:3, 3]
        pts = (R @ pts.T).T + t
    keys = voxel_grid.world_to_keys(pts)
    return set(k for k in keys.tolist() if k >= 0)


def prune_nodes(colrm: dict[int, set[int]], occupied_voxels: set[int]) -> set[int]:
    """用 COLRM 和障碍物体素快速剪枝节点。返回被剪除的节点 ID 集合。"""
    pruned: set[int] = set()
    for vk in occupied_voxels:
        if vk in colrm:
            pruned.update(colrm[vk])
    return pruned


# ── 2. X_ε 目标邻域搜索 ──────────────────────────────


def find_goal_candidates(
    ee_poses: np.ndarray,
    target_pos: np.ndarray,
    target_axis: np.ndarray | None,
    valid_mask: np.ndarray,
    epsilon_pos: float = 0.10,
    epsilon_pos_max: float = 0.30,
    axis_cos_thresh: float = 0.9,
    min_candidates: int = 3,
) -> list[int]:
    """在 EE 位姿索引中查找目标姿态邻域的候选节点。

    Parameters
    ----------
    ee_poses : (N, 7) [x,y,z, qx,qy,qz,qw]
    target_pos : (3,) 目标 TCP 位置
    target_axis : (3,) 目标枪轴方向（单位向量），None 则不约束方向
    valid_mask : (N,) bool, True = 未被剪枝
    epsilon_pos : 初始位置邻域半径 (m)
    axis_cos_thresh : 轴向夹角余弦阈值
    min_candidates : 至少找到这么多候选才返回
    """
    pos_tree = KDTree(ee_poses[:, :3])
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    candidates: list[int] = []

    eps = epsilon_pos
    while eps <= epsilon_pos_max:
        indices = pos_tree.query_ball_point(target_pos, eps)
        candidates = [i for i in indices if valid_mask[i]]

        if target_axis is not None and len(candidates) > 0:
            t_axis = np.asarray(target_axis, dtype=float).reshape(3)
            t_axis = t_axis / (np.linalg.norm(t_axis) + 1e-12)
            filtered = []
            for i in candidates:
                quat = ee_poses[i, 3:]
                ee_z = Rotation.from_quat(quat).apply([0, 0, 1])
                if abs(np.dot(ee_z, t_axis)) >= axis_cos_thresh:
                    filtered.append(i)
            candidates = filtered

        if len(candidates) >= min_candidates:
            return candidates
        eps *= 1.5

    return candidates


# ── 3. 连通分量分析 ──────────────────────────────────


def find_connected_components(adjacency: csr_matrix, valid_mask: np.ndarray) -> dict[int, int]:
    """BFS 寻找连通分量。返回 node_id → component_id 映射。"""
    n = adjacency.shape[0]
    comp = np.full(n, -1, dtype=int)
    comp_id = 0
    for seed in range(n):
        if not valid_mask[seed] or comp[seed] >= 0:
            continue
        queue = deque([seed])
        comp[seed] = comp_id
        while queue:
            u = queue.popleft()
            row_s, row_e = adjacency.indptr[u], adjacency.indptr[u + 1]
            for nb in adjacency.indices[row_s:row_e]:
                nb = int(nb)
                if valid_mask[nb] and comp[nb] < 0:
                    comp[nb] = comp_id
                    queue.append(nb)
        comp_id += 1
    return comp, comp_id


# ── 4. A* + lazy edge collision check ────────────────


def _reconstruct_path(came_from: dict[int, int], current: int) -> list[int]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_search(
    adjacency: csr_matrix,
    nodes: np.ndarray,
    ee_poses: np.ndarray,
    start_node: int,
    goal_node: int,
    pruned_nodes: set[int],
    lazy_check_fn=None,
) -> list[int] | None:
    """A* 最短路搜索，带 lazy edge collision check。

    Parameters
    ----------
    lazy_check_fn : callable(node_i, node_j) → bool
        返回 True 表示边安全。None 则不做 lazy check。
    """
    target_pos = ee_poses[goal_node, :3]

    open_set: list[tuple[float, int]] = [(0.0, start_node)]
    g_score: dict[int, float] = {start_node: 0.0}
    came_from: dict[int, int] = {}
    closed: set[int] = set()

    while open_set:
        f_val, current = heapq.heappop(open_set)
        if current == goal_node:
            return _reconstruct_path(came_from, current)
        if current in closed:
            continue
        closed.add(current)

        row_start = adjacency.indptr[current]
        row_end = adjacency.indptr[current + 1]
        neighbors = adjacency.indices[row_start:row_end]
        edge_weights = adjacency.data[row_start:row_end]

        for nb, w in zip(neighbors, edge_weights):
            nb = int(nb)
            if nb in closed or nb in pruned_nodes:
                continue
            if lazy_check_fn is not None and not lazy_check_fn(current, nb):
                continue
            tentative_g = g_score[current] + w
            if tentative_g < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative_g
                h = float(np.linalg.norm(ee_poses[nb, :3] - target_pos))
                heapq.heappush(open_set, (tentative_g + h, nb))

    return None


# ── 4. 完整在线规划 ──────────────────────────────────


def plan_point_to_point(
    nodes: np.ndarray,
    adjacency: csr_matrix,
    ee_poses: np.ndarray,
    colrm: dict[int, set[int]],
    voxel_grid: VoxelGrid,
    q_start_6d: np.ndarray,
    target_pos: np.ndarray,
    target_axis: np.ndarray | None = None,
    obstacle_vertices: np.ndarray | None = None,
    T_world_to_robobase: np.ndarray | None = None,
    lazy_check_fn=None,
    max_retries: int = 5,
    epsilon_pos: float = 0.10,
) -> dict:
    """完整在线规划：剪枝 → 目标搜索 → A* → 重试。

    Returns
    -------
    dict with keys:
        success: bool
        path_indices: list[int]  (roadmap 节点索引序列)
        path_configs: np.ndarray (N_path, dim)
        goal_node: int
        start_node: int
        n_pruned: int
        n_valid: int
    """
    n = len(nodes)
    q_start = np.asarray(q_start_6d, dtype=float).reshape(-1)

    # 体素剪枝
    pruned = set()
    if obstacle_vertices is not None:
        occupied = voxelize_obstacle_mesh(obstacle_vertices, voxel_grid, T_world_to_robobase)
        pruned = prune_nodes(colrm, occupied)
    valid_mask = np.ones(n, dtype=bool)
    valid_mask[list(pruned)] = False

    _log(f"  剪枝: {len(pruned)}/{n} 节点被排除, {n-len(pruned)} 有效")

    if not np.any(valid_mask):
        _log("  所有节点均被剪枝，规划失败")
        return _fail_result(nodes.shape[1])

    # 连通分量分析
    comp_labels, n_comp = find_connected_components(adjacency, valid_mask)
    comp_sizes = np.bincount(comp_labels[comp_labels >= 0])
    largest_comp = int(np.argmax(comp_sizes))
    _log(f"  连通分量: {n_comp} 个, 最大分量 #{largest_comp} 含 {comp_sizes[largest_comp]} 个节点")

    # 找起点最近有效节点（优先在最大连通分量内）
    large_comp_mask = valid_mask & (comp_labels == largest_comp)
    valid_indices = np.where(large_comp_mask)[0]
    if len(valid_indices) == 0:
        valid_indices = np.where(valid_mask)[0]
    dists_to_start = np.linalg.norm(nodes[valid_indices] - q_start, axis=1)
    start_node = int(valid_indices[np.argmin(dists_to_start)])
    start_comp = int(comp_labels[start_node])
    _log(f"  起点: 节点 {start_node} (分量 #{start_comp})")

    # 只在同一连通分量中搜索目标
    same_comp_mask = valid_mask & (comp_labels == start_comp)

    # 找目标邻域
    candidates = find_goal_candidates(
        ee_poses, target_pos, target_axis, same_comp_mask,
        epsilon_pos=epsilon_pos,
    )
    if not candidates:
        _log("  同一连通分量内无目标候选，尝试全局搜索...")
        candidates = find_goal_candidates(
            ee_poses, target_pos, target_axis, valid_mask,
            epsilon_pos=epsilon_pos, epsilon_pos_max=1.0,
        )
    if not candidates:
        _log("  无法找到目标邻域候选节点")
        return _fail_result(nodes.shape[1])

    # 按 EE 距离排序
    cand_dists = [(int(c), float(np.linalg.norm(ee_poses[c, :3] - target_pos))) for c in candidates]
    cand_dists.sort(key=lambda x: x[1])
    _log(f"  目标邻域: {len(candidates)} 个候选, 最近 EE 距离={cand_dists[0][1]:.4f}m")

    # A* 搜索 + 重试
    for attempt in range(max_retries):
        for goal_node, d in cand_dists:
            if goal_node in pruned:
                continue
            if comp_labels[goal_node] != start_comp:
                continue
            path = astar_search(
                adjacency, nodes, ee_poses,
                start_node, goal_node, pruned,
                lazy_check_fn=lazy_check_fn,
            )
            if path is not None:
                _log(f"  A* 成功 (attempt {attempt+1}): {len(path)} 步, "
                     f"start={start_node} → goal={goal_node}, EE误差={d:.4f}m")
                return {
                    "success": True,
                    "path_indices": path,
                    "path_configs": nodes[path],
                    "goal_node": goal_node,
                    "start_node": start_node,
                    "n_pruned": len(pruned),
                    "n_valid": n - len(pruned),
                    "ee_error": d,
                }

        _log(f"  attempt {attempt+1} 失败，扩大邻域重试...")
        epsilon_pos *= 1.5
        candidates = find_goal_candidates(
            ee_poses, target_pos, target_axis, same_comp_mask,
            epsilon_pos=epsilon_pos,
        )
        cand_dists = [(int(c), float(np.linalg.norm(ee_poses[c, :3] - target_pos))) for c in candidates]
        cand_dists.sort(key=lambda x: x[1])

    _log("  规划失败: 所有重试均未找到路径")
    return _fail_result(nodes.shape[1])


def _fail_result(dim: int) -> dict:
    return {"success": False, "path_indices": [], "path_configs": np.empty((0, dim))}
