"""GPU 批量球-球自碰撞检测（PyTorch 全向量化）。

接口设计要点：
- 一次性接收 (B, dof) 关节配置批，返回 (B,) bool: True=自碰撞。
- 把每个 link 的局部球链通过 batch FK 变换到世界系，再对 monitored_pairs 的
  每对 link 做 N_a × N_b 球-球距离检测（含 padding-mask）。
- 所有 link 的球被 pad 到统一长度 N_max，通过 valid_mask 屏蔽 padding 球。

保守性：球链是 mesh 的超集 → 球链不碰 ⇒ mesh 不碰（用作"放行 oracle"安全）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from CBF_experiment.active.pybullet.drm.gpu.batch_fk import GPUKinematics, batch_fk_link_world
from CBF_experiment.active.pybullet.drm.gpu.link_spheres import LinkSpheres


# ── 1. 把球链 dict 打包成 GPU 张量 ─────────────────────


@dataclass
class GPUSphereModel:
    """所有 link 的球链 + 监控对，全部驻留 GPU。

    link_indices  : (L,) int       PyBullet link id（与 batch_fk 输出顺序对应）
    centers_local : (L, N_max, 3)  各 link 局部球心（pad 到 N_max）
    radii         : (L, N_max)     各球半径，padding 处为 0
    valid_mask    : (L, N_max) bool padding 屏蔽
    pair_a, pair_b: (P,) int 索引到 (0..L-1)，表示要检测的 link 对
    pair_radii_sum: (P, N_max, N_max) 预算的 r_a + r_b（含 0 padding）
    q_base        : (dof_full,) float 默认全状态（含底盘等非 monitored 维度）
    q_indices     : (dof_mon,)  long  monitored q 在 q_full 中的位置
    """
    link_indices: list[int]
    centers_local: torch.Tensor
    radii: torch.Tensor
    valid_mask: torch.Tensor
    pair_a: torch.Tensor
    pair_b: torch.Tensor
    pair_radii_sum: torch.Tensor
    pair_valid: torch.Tensor          # (P, N_max, N_max) bool, 同时考虑两端 padding
    q_base: torch.Tensor
    q_indices: torch.Tensor
    device: torch.device

    def compose_full_q(self, q_mon: torch.Tensor) -> torch.Tensor:
        """(B, dof_mon) → (B, dof_full)，把 monitored q 嵌入 q_base。"""
        B = q_mon.shape[0]
        q_full = self.q_base.unsqueeze(0).expand(B, -1).clone()
        q_full[:, self.q_indices] = q_mon
        return q_full


def build_gpu_sphere_model(
    link_spheres: dict[int, LinkSpheres],
    monitored_link_ids: list[int],
    monitored_pairs: list[tuple[int, int]],
    q_base: np.ndarray,
    q_indices: list[int] | tuple[int, ...],
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    drop_pairs: set[tuple[int, int]] | None = None,
) -> GPUSphereModel:
    """把 dict 形式的球链整理成 GPU 友好张量。

    Parameters
    ----------
    monitored_link_ids : 与 batch FK 调用时的 link 索引顺序一致
    monitored_pairs    : [(li_a, li_b), ...] PyBullet link id 对
    drop_pairs         : 跳过 GPU 检测的 pair（这些 pair 由 CPU coal 单独负责）
    """
    drop_set: set[tuple[int, int]] = set()
    if drop_pairs is not None:
        for a, b in drop_pairs:
            drop_set.add((int(a), int(b)))
            drop_set.add((int(b), int(a)))
    device = torch.device(device)
    L = len(monitored_link_ids)
    n_max = max((link_spheres[li].n if li in link_spheres else 0)
                for li in monitored_link_ids)
    if n_max == 0:
        raise ValueError("no spheres at all")

    centers = np.zeros((L, n_max, 3), dtype=np.float32)
    radii = np.zeros((L, n_max), dtype=np.float32)
    valid = np.zeros((L, n_max), dtype=bool)

    li_to_idx: dict[int, int] = {}
    for idx, li in enumerate(monitored_link_ids):
        li_to_idx[int(li)] = idx
        ls = link_spheres.get(int(li))
        if ls is None or ls.n == 0:
            continue
        n = ls.n
        centers[idx, :n] = ls.centers
        radii[idx, :n] = ls.radii
        valid[idx, :n] = True

    pair_a, pair_b = [], []
    for a, b in monitored_pairs:
        if (int(a), int(b)) in drop_set:
            continue
        ia = li_to_idx.get(int(a), -1)
        ib = li_to_idx.get(int(b), -1)
        if ia < 0 or ib < 0:
            continue
        pair_a.append(ia)
        pair_b.append(ib)
    pair_a_t = torch.tensor(pair_a, dtype=torch.long, device=device)
    pair_b_t = torch.tensor(pair_b, dtype=torch.long, device=device)

    centers_t = torch.from_numpy(centers).to(device=device, dtype=dtype)
    radii_t = torch.from_numpy(radii).to(device=device, dtype=dtype)
    valid_t = torch.from_numpy(valid).to(device=device)

    # 预算 (P, N_max, N_max)
    ra = radii_t[pair_a_t]                      # (P, N_max)
    rb = radii_t[pair_b_t]                      # (P, N_max)
    pair_radii_sum = ra.unsqueeze(2) + rb.unsqueeze(1)            # (P, N_a, N_b)
    va = valid_t[pair_a_t].unsqueeze(2)                          # (P, N_a, 1)
    vb = valid_t[pair_b_t].unsqueeze(1)                          # (P, 1, N_b)
    pair_valid = va & vb                                          # (P, N_a, N_b)

    q_base_t = torch.tensor(np.asarray(q_base, dtype=np.float32), dtype=dtype, device=device)
    q_idx_t = torch.tensor([int(x) for x in q_indices], dtype=torch.long, device=device)

    return GPUSphereModel(
        link_indices=[int(x) for x in monitored_link_ids],
        centers_local=centers_t,
        radii=radii_t,
        valid_mask=valid_t,
        pair_a=pair_a_t,
        pair_b=pair_b_t,
        pair_radii_sum=pair_radii_sum,
        pair_valid=pair_valid,
        q_base=q_base_t,
        q_indices=q_idx_t,
        device=device,
    )


# ── 2. 球心世界变换（FK + transform） ─────────────────


def transform_spheres_world(
    pos_links: torch.Tensor,            # (B, L, 3)
    R_links: torch.Tensor,              # (B, L, 3, 3)
    centers_local: torch.Tensor,        # (L, N, 3)
) -> torch.Tensor:
    """把每个 link 的局部球心变换到世界系。

    Returns: (B, L, N, 3) 世界球心
    """
    # (B, L, N, 3) = einsum('blij, lnj -> blni')
    rotated = torch.einsum("blij,lnj->blni", R_links, centers_local)
    return rotated + pos_links.unsqueeze(2)


# ── 3. 自碰撞批量检测 ────────────────────────────────


def batch_self_collision(
    q_batch: torch.Tensor,              # (B, dof_mon)  monitored 子集
    gk: GPUKinematics,
    sm: GPUSphereModel,
    chunk_size: int = 1024,
    pair_chunk: int | None = None,
) -> torch.Tensor:
    """主接口：批量自碰撞检测。

    Parameters
    ----------
    q_batch    : (B, dof_mon) monitored joint 子集
    chunk_size : 配置批分块（控制显存）
    pair_chunk : pair 维度分块；None 时按显存预算自动估计

    Returns
    -------
    is_collision : (B,) bool, True 表示某对球链有相交（保守上界）
    """
    B = q_batch.shape[0]
    out = torch.zeros(B, dtype=torch.bool, device=sm.device)
    n_max = int(sm.centers_local.shape[1])

    # 自动估计 pair_chunk：单 chunk 张量 ≤ 1.5GB
    if pair_chunk is None:
        bytes_per_pair = chunk_size * n_max * n_max * 3 * 4  # diff 张量
        pair_chunk = max(1, int(1.5 * 1024**3 // max(1, bytes_per_pair)))
    pair_chunk = max(1, min(pair_chunk, int(sm.pair_a.shape[0])))

    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        q_full = sm.compose_full_q(q_batch[s:e])
        pos, R = batch_fk_link_world(gk, q_full, sm.link_indices)        # (b, L, 3), (b, L, 3, 3)
        pts = transform_spheres_world(pos, R, sm.centers_local)          # (b, L, N, 3)
        out[s:e] = _pair_collision_reduce(pts, sm, pair_chunk=pair_chunk)

    return out


def _pair_collision_reduce(
    pts_world: torch.Tensor,            # (B, L, N, 3)
    sm: GPUSphereModel,
    pair_chunk: int = 64,
) -> torch.Tensor:
    """对所有 monitored pair 检查 N_a × N_b 球对的距离（pair 分块以省显存）。"""
    B = pts_world.shape[0]
    P = int(sm.pair_a.shape[0])
    out = torch.zeros(B, dtype=torch.bool, device=pts_world.device)
    if P == 0:
        return out
    for s in range(0, P, pair_chunk):
        e = min(s + pair_chunk, P)
        pa = pts_world[:, sm.pair_a[s:e], :, :]               # (B, p, N_a, 3)
        pb = pts_world[:, sm.pair_b[s:e], :, :]
        diff = pa.unsqueeze(3) - pb.unsqueeze(2)              # (B, p, N_a, N_b, 3)
        dist2 = diff.pow(2).sum(-1)                           # (B, p, N_a, N_b)
        rsum2 = sm.pair_radii_sum[s:e].pow(2).unsqueeze(0)
        valid = sm.pair_valid[s:e].unsqueeze(0)
        overlap = (dist2 < rsum2) & valid
        out |= overlap.flatten(1).any(dim=1)
        if bool(out.all()):
            break
    return out


# ── 4. 批量边检测：分段插值 ──────────────────────────


def batch_segment_collision_free(
    q_a_batch: torch.Tensor,            # (M, dof)
    q_b_batch: torch.Tensor,            # (M, dof)
    gk: GPUKinematics,
    sm: GPUSphereModel,
    num_steps: int = 8,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """对 M 条边（每条用 num_steps+1 个插值点）做 GPU 自碰撞检测。

    Returns
    -------
    free_mask : (M,) bool, True 表示整段插值无碰撞（保守上界 → True 不一定真自由，
                需上层 coal 复检；False 说明球链已碰，可早剪）
    """
    M = q_a_batch.shape[0]
    free = torch.ones(M, dtype=torch.bool, device=sm.device)

    alphas = torch.linspace(0.0, 1.0, int(num_steps) + 1,
                            device=sm.device, dtype=q_a_batch.dtype)

    for s in range(0, M, chunk_size):
        e = min(s + chunk_size, M)
        qa = q_a_batch[s:e].unsqueeze(1)                     # (m, 1, dof_mon)
        qb = q_b_batch[s:e].unsqueeze(1)
        a = alphas.view(1, -1, 1)                            # (1, S, 1)
        q_interp = (1 - a) * qa + a * qb                     # (m, S, dof_mon)
        m, S, dof_mon = q_interp.shape
        q_flat = q_interp.reshape(m * S, dof_mon)
        q_full = sm.compose_full_q(q_flat)

        pos, R = batch_fk_link_world(gk, q_full, sm.link_indices)
        pts = transform_spheres_world(pos, R, sm.centers_local)
        col = _pair_collision_reduce(pts, sm)                # (m*S,)
        col = col.view(m, S).any(dim=1)
        free[s:e] = ~col

    return free
