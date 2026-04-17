"""Hybrid GPU+coal 自碰撞 oracle（带 pair calibration）。

设计：
  1. 球链是 mesh 超集 → 球链 free  ⇒  mesh free（保守安全）
  2. 但末端紧凑的 link 对（如焊枪与腕部），球链总是穿插造成 100% 假阳性
  3. 启动时做 calibration：抽 K 个 q，找出"球链碰概率 > drop_threshold"的 pair
     → 这些 pair 转给 coal 单独处理（dropped_pairs，CPU 单 q 仅检查少数 pair 很快）
  4. 单 q 检测流水：
       coal(dropped_pairs) -> 若任一碰: collide
       否则 GPU(gpu_pairs) batch -> GPU free: free
                                   GPU collide: coal(gpu_pairs) 复检
  5. 批量场景下 dropped 检查仍是单 q 的，但只检查 3~5 对，开销 ~50us/q
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from CBF_experiment.active.pybullet.drm.gpu.batch_fk import build_gpu_kinematics
from CBF_experiment.active.pybullet.drm.gpu.link_spheres import (
    LinkSpheres, fit_robot_spheres, save_link_spheres, load_link_spheres,
)
from CBF_experiment.active.pybullet.drm.gpu.sphere_collision import (
    GPUSphereModel, build_gpu_sphere_model, batch_self_collision,
    batch_segment_collision_free,
)
from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import _resolve, load_config
from CBF_experiment.active.pybullet.self_collision.self_collision_backend_coal import (
    is_any_pair_collision_fast,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.coal_oracle import CoalSelfCollisionOracle
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.robot_model import (
    compose_full_q, sample_joint_box,
)


class HybridGPUSelfCollisionOracle:
    """GPU 球链粗筛 + coal mesh 精检 的混合 oracle。"""

    def __init__(
        self,
        config: RobotQueryConfig | None = None,
        sphere_pitch: float = 0.04,
        max_spheres_per_link: int = 32,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        spheres_cache: str | Path | None = None,
        rebuild_spheres: bool = False,
        calibration_samples: int = 256,
        drop_threshold: float = 0.95,
        seed: int = 0,
        monitored_links: list[int] | None = None,
        min_index_gap: int = 2,
    ):
        self.config = config or RobotQueryConfig()
        self.device = torch.device(device)
        self.dtype = dtype

        # --- coal oracle 持有 PyBullet 连接 ---
        self.coal = CoalSelfCollisionOracle(self.config)
        self.robot = self.coal.robot
        self.metadata = self.coal.metadata
        self.link_models = self.coal.link_models

        # --- 监控 link 集合：默认 robobase + 后6轴 + welding_gun_base + weld_point ---
        if monitored_links is None:
            monitored = []
            if self.robot.robobase_link_index >= 0:
                monitored.append(int(self.robot.robobase_link_index))
            for j in self.robot.revolute_joints:        # link04..link09
                monitored.append(int(j))
            if self.robot.welding_gun_base_link_index >= 0:
                monitored.append(int(self.robot.welding_gun_base_link_index))
            for li in self.robot.welding_gun_links:     # 含 weld_point
                if li not in monitored:
                    monitored.append(int(li))
            monitored = sorted(set(monitored))
        else:
            monitored = sorted(set(int(x) for x in monitored_links))
        self.monitored_links = monitored

        # 用户要求 min_index_gap=2 重新生成 pair（避免相邻 link 必碰）
        self.all_pairs = [(a, b) for i, a in enumerate(monitored)
                          for b in monitored[i + 1:]
                          if abs(b - a) >= int(min_index_gap)]
        # 确保 coal backend 也用同样的 link 集合
        from CBF_experiment.active.pybullet.self_collision.self_collision_backend_coal import (
            build_coal_link_models,
        )
        self.link_models = build_coal_link_models(self.robot, self.monitored_links)

        # --- 球链拟合（带缓存） ---
        spheres = self._load_or_fit_spheres(spheres_cache, rebuild_spheres,
                                            sphere_pitch, max_spheres_per_link)
        self.link_spheres = spheres

        # --- GPU FK ---
        urdf_path = _resolve(load_config()["robot"]["urdf"])
        self.gk = build_gpu_kinematics(
            urdf_path, self.robot.body_id, self.robot.active_joints,
            device=device, dtype=dtype,
        )

        # --- 1) 先建 full GPU model 做 calibration ---
        sm_full = build_gpu_sphere_model(
            spheres, monitored, self.all_pairs,
            self.metadata.q_base, self.metadata.q_indices,
            device=device, dtype=dtype,
        )
        self.dropped_pairs = self._calibrate_pairs(
            sm_full, calibration_samples, drop_threshold, seed,
        )
        self.gpu_pairs = [pair for pair in self.all_pairs
                         if pair not in self.dropped_pairs and
                         (pair[1], pair[0]) not in self.dropped_pairs]
        print(f"  [HybridOracle] calibration: GPU pairs {len(self.gpu_pairs)}, "
              f"dropped (coal-only) pairs {len(self.dropped_pairs)}", flush=True)

        # --- 2) 重建只含 gpu_pairs 的精简 GPU model ---
        self.sm = build_gpu_sphere_model(
            spheres, monitored, self.all_pairs,
            self.metadata.q_base, self.metadata.q_indices,
            device=device, dtype=dtype,
            drop_pairs=self.dropped_pairs,
        )

    # ── 私有 helper ────────────────────────

    def _load_or_fit_spheres(self, cache, rebuild, pitch, max_per_link
                             ) -> dict[int, LinkSpheres]:
        spheres: dict[int, LinkSpheres] | None = None
        if cache is not None and not rebuild:
            cp = Path(cache)
            if cp.exists():
                spheres = load_link_spheres(cp)
                if not all(int(li) in spheres for li in self.monitored_links):
                    spheres = None
        if spheres is None:
            print(f"  [HybridOracle] 拟合球链 pitch={pitch:.3f} ...", flush=True)
            t0 = time.time()
            spheres = fit_robot_spheres(
                self.robot.body_id, self.monitored_links,
                pitch=pitch, max_spheres_per_link=max_per_link,
            )
            print(f"  [HybridOracle] 球链拟合 {time.time()-t0:.1f}s, "
                  f"共 {sum(s.n for s in spheres.values())} 球", flush=True)
            if cache is not None:
                save_link_spheres(spheres, cache)
        return spheres

    def _calibrate_pairs(
        self,
        sm_full: GPUSphereModel,
        n_samples: int,
        drop_threshold: float,
        seed: int,
    ) -> set[tuple[int, int]]:
        """统计每对 pair 的"球链碰概率"，超过阈值的 pair 标记为 dropped。

        分块策略：(配置块, pair 块) 双重 chunk，避免 N×P×M×M 显存爆掉。
        """
        from CBF_experiment.active.pybullet.drm.gpu.batch_fk import batch_fk_link_world
        from CBF_experiment.active.pybullet.drm.gpu.sphere_collision import transform_spheres_world

        rng = np.random.default_rng(seed)
        q = sample_joint_box(self.metadata, rng, n_samples).astype(np.float32)
        q_t = torch.tensor(q, device=self.device, dtype=self.dtype)

        P = int(sm_full.pair_a.shape[0])
        n_max = int(sm_full.centers_local.shape[1])
        per_pair_collide = torch.zeros(n_samples, P, dtype=torch.bool, device=self.device)

        # 自适应 chunk
        cfg_chunk = max(1, min(n_samples, 256))
        bytes_per_pair = cfg_chunk * n_max * n_max * 3 * 4
        pair_chunk = max(1, int(1.0 * 1024**3 // max(1, bytes_per_pair)))

        for s in range(0, n_samples, cfg_chunk):
            e = min(s + cfg_chunk, n_samples)
            q_full = sm_full.compose_full_q(q_t[s:e])
            pos, R = batch_fk_link_world(self.gk, q_full, sm_full.link_indices)
            pts = transform_spheres_world(pos, R, sm_full.centers_local)  # (b, L, M, 3)
            for ps in range(0, P, pair_chunk):
                pe = min(ps + pair_chunk, P)
                pa = pts[:, sm_full.pair_a[ps:pe], :, :]
                pb = pts[:, sm_full.pair_b[ps:pe], :, :]
                diff = pa.unsqueeze(3) - pb.unsqueeze(2)
                dist2 = diff.pow(2).sum(-1)
                rsum2 = sm_full.pair_radii_sum[ps:pe].pow(2).unsqueeze(0)
                valid = sm_full.pair_valid[ps:pe].unsqueeze(0)
                per_pair_collide[s:e, ps:pe] = ((dist2 < rsum2) & valid).flatten(2).any(dim=2)

        rate = per_pair_collide.float().mean(dim=0).cpu().numpy()  # (P,)

        dropped: set[tuple[int, int]] = set()
        pair_a_np = sm_full.pair_a.cpu().numpy()
        pair_b_np = sm_full.pair_b.cpu().numpy()
        for i, r in enumerate(rate):
            if r >= drop_threshold:
                la = sm_full.link_indices[int(pair_a_np[i])]
                lb = sm_full.link_indices[int(pair_b_np[i])]
                dropped.add((int(la), int(lb)))
        return dropped

    # ── 容量信息 ───────────────────────────
    @property
    def dim(self) -> int:
        return len(self.metadata.joint_limits)

    def close(self):
        self.coal.close()

    # ── coal 子集快速检测 ──────────────────

    def _coal_check_subset(self, q6: np.ndarray, pairs: list[tuple[int, int]]) -> bool:
        """让 coal 仅检查 pair 子集；空集返回 False。"""
        if not pairs:
            return False
        q_full = compose_full_q(self.metadata, q6)
        self.robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        return is_any_pair_collision_fast(
            self.robot, link_models=self.link_models, monitored_pairs=pairs,
        )

    # ── 单查询接口（与 coal 完全等价语义） ─

    def is_self_collision(self, q: np.ndarray) -> bool:
        # 1) coal 先检 dropped pair（少数对，很快）
        if self._coal_check_subset(q, list(self.dropped_pairs)):
            return True
        # 2) GPU 检 gpu pair
        q_t = torch.tensor(np.asarray(q, dtype=np.float32).reshape(1, -1),
                           device=self.device, dtype=self.dtype)
        gpu_col = bool(batch_self_collision(q_t, self.gk, self.sm).item())
        if not gpu_col:
            return False
        # 3) GPU 报警 → coal 复检 gpu pair
        return self._coal_check_subset(q, self.gpu_pairs)

    # ── 批量主接口 ─────────────────────────

    def is_self_collision_batch(self, q_batch: np.ndarray,
                                chunk_size: int = 8192) -> np.ndarray:
        q = np.asarray(q_batch, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        B = q.shape[0]
        result = np.zeros(B, dtype=bool)

        # 1) coal 串行检 dropped pair（每 q 仅检几对）
        dropped = list(self.dropped_pairs)
        if dropped:
            for i in range(B):
                if self._coal_check_subset(q[i], dropped):
                    result[i] = True
        remain = ~result
        idx_remain = np.where(remain)[0]
        if len(idx_remain) == 0:
            return result

        # 2) GPU 批量检 gpu pair
        q_t = torch.tensor(q[idx_remain], device=self.device, dtype=self.dtype)
        gpu_col = batch_self_collision(q_t, self.gk, self.sm,
                                       chunk_size=chunk_size).cpu().numpy()

        # GPU free → 整体 free
        # GPU collide → coal 复检 gpu pair
        gpu_pairs = list(self.gpu_pairs)
        for k, idx in enumerate(idx_remain):
            if gpu_col[k]:
                result[idx] = self._coal_check_subset(q[idx], gpu_pairs)
        return result

    def filter_free_batch(self, q_batch: np.ndarray,
                          chunk_size: int = 8192) -> tuple[np.ndarray, dict]:
        """筛选无自碰撞配置。返回 (q_free, stats)。"""
        q = np.asarray(q_batch, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        B = q.shape[0]
        col = np.zeros(B, dtype=bool)

        t0 = time.time()
        dropped = list(self.dropped_pairs)
        n_dropped_kill = 0
        if dropped:
            for i in range(B):
                if self._coal_check_subset(q[i], dropped):
                    col[i] = True
                    n_dropped_kill += 1
        t_dropped = time.time() - t0

        survivors = np.where(~col)[0]
        t0 = time.time()
        if len(survivors) > 0:
            q_t = torch.tensor(q[survivors], device=self.device, dtype=self.dtype)
            gpu_col = batch_self_collision(q_t, self.gk, self.sm,
                                           chunk_size=chunk_size).cpu().numpy()
        else:
            gpu_col = np.zeros(0, dtype=bool)
        t_gpu = time.time() - t0

        # GPU 报警的需要 coal 复检
        t0 = time.time()
        gpu_pairs = list(self.gpu_pairs)
        n_recover = 0
        for k, idx in enumerate(survivors):
            if gpu_col[k]:
                if self._coal_check_subset(q[idx], gpu_pairs):
                    col[idx] = True
                else:
                    n_recover += 1
        t_recheck = time.time() - t0

        free_mask = ~col
        stats = {
            "n_total": B,
            "n_dropped_kill": n_dropped_kill,
            "n_gpu_free_direct": int(np.sum(~gpu_col)),
            "n_gpu_collide": int(np.sum(gpu_col)),
            "n_coal_recovered": n_recover,
            "n_final_free": int(free_mask.sum()),
            "t_dropped_ms": t_dropped * 1000,
            "t_gpu_ms": t_gpu * 1000,
            "t_recheck_ms": t_recheck * 1000,
        }
        return q[free_mask], stats

    def segment_is_collision_free_batch(
        self,
        q_a_batch: np.ndarray,
        q_b_batch: np.ndarray,
        num_steps: int = 8,
        chunk_size: int = 1024,
    ) -> np.ndarray:
        """批量边检测：插值点上跑 hybrid is_self_collision。

        简化策略：先 GPU 跑全段 gpu_pairs；GPU free 段还要逐点 coal-dropped
        + coal-gpu_pairs 复检（保证语义与 coal 一致）。
        """
        qa = np.asarray(q_a_batch, dtype=np.float32)
        qb = np.asarray(q_b_batch, dtype=np.float32)
        M = qa.shape[0]

        # 1) GPU 段碰撞预筛（含 gpu_pairs 的所有插值点）
        qa_t = torch.tensor(qa, device=self.device, dtype=self.dtype)
        qb_t = torch.tensor(qb, device=self.device, dtype=self.dtype)
        gpu_seg_free = batch_segment_collision_free(
            qa_t, qb_t, self.gk, self.sm,
            num_steps=num_steps, chunk_size=chunk_size,
        ).cpu().numpy()

        # 2) GPU 报段碰 → coal 完整段复检
        # 2) GPU 段 free → 仍需 coal 检 dropped pair 段
        result = np.zeros(M, dtype=bool)
        dropped = list(self.dropped_pairs)
        gpu_pairs = list(self.gpu_pairs)
        alphas = np.linspace(0.0, 1.0, int(num_steps) + 1, dtype=np.float32)

        for i in range(M):
            ok = True
            if not gpu_seg_free[i]:
                # 完整 coal 串行复检（含 dropped + gpu pairs，等价 coal 全量）
                ok = self.coal.segment_is_collision_free(
                    qa[i], qb[i], num_steps=num_steps,
                )
            elif dropped:
                # GPU 段 free 但 dropped pair 未检 → 仅检 dropped pair
                for a in alphas:
                    qm = (1 - a) * qa[i] + a * qb[i]
                    if self._coal_check_subset(qm, dropped):
                        ok = False
                        break
            result[i] = ok
        return result

    def segment_is_collision_free(self, q_a: np.ndarray, q_b: np.ndarray, *,
                                  num_steps: int) -> bool:
        return bool(self.segment_is_collision_free_batch(
            q_a.reshape(1, -1), q_b.reshape(1, -1), num_steps=num_steps,
        )[0])
