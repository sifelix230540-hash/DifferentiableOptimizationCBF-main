"""采样估计多面体区域并集对 C-free 的覆盖率及置信半径。"""
from __future__ import annotations

import math

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import (
    CoverageEstimate,
    DualCoverageStats,
    FreeSample,
    IrisRegion,
)


def points_in_polytope(points: np.ndarray, A: np.ndarray, b: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.all(np.asarray(points, dtype=float) @ np.asarray(A, dtype=float).T <= np.asarray(b, dtype=float).reshape(1, -1) + float(tol), axis=1)


def estimate_region_coverage(samples: list[FreeSample], regions: list[IrisRegion] | tuple[IrisRegion, ...]) -> CoverageEstimate:
    points = np.asarray([np.asarray(sample.q, dtype=float) for sample in samples], dtype=float)
    covered = np.zeros(points.shape[0], dtype=bool)
    for region in regions:
        covered |= points_in_polytope(points, region.A, region.b, tol=1e-9)
    num_hits = int(np.sum(covered))
    num_samples = int(points.shape[0])
    ratio = float(num_hits / max(num_samples, 1))
    confidence_radius = float(1.96 * math.sqrt(max(ratio * (1.0 - ratio), 1e-12) / max(num_samples, 1)))
    return CoverageEstimate(
        num_hits=num_hits,
        num_samples=num_samples,
        ratio=ratio,
        confidence_radius=confidence_radius,
    )


# ────────────────────────── 对偶分解：5 指标统计 ──────────────────────────


def _union_membership(points: np.ndarray, regions) -> np.ndarray:
    """返回长度为 N 的 bool 数组：每个点是否落在任一 region 内。"""
    if points.size == 0 or not regions:
        return np.zeros((points.shape[0],), dtype=bool)
    covered = np.zeros(points.shape[0], dtype=bool)
    for region in regions:
        A = np.asarray(region.A, dtype=float) if hasattr(region, "A") else np.asarray(region[0], dtype=float)
        b = np.asarray(region.b, dtype=float).reshape(-1) if hasattr(region, "b") else np.asarray(region[1], dtype=float).reshape(-1)
        covered |= points_in_polytope(points, A, b, tol=1e-9)
    return covered


def estimate_dual_coverage(
    uniform_points: np.ndarray,
    pos_labels: np.ndarray,
    pos_regions,
    neg_regions,
) -> DualCoverageStats:
    """五指标对偶覆盖率统计。

    Parameters
    ----------
    uniform_points : (N, d) array
        在 joint box 上均匀采样的点（不区分 pos / neg）。
    pos_labels : (N,) bool array
        base oracle 对每个点的判定：True = 该点是 pos 样本（C-free）；False = neg 样本（C-obs）。
    pos_regions, neg_regions : list[IrisRegion]
    """
    points = np.asarray(uniform_points, dtype=float)
    pos_labels = np.asarray(pos_labels, dtype=bool).reshape(-1)
    n_total = int(points.shape[0])
    if n_total == 0:
        return DualCoverageStats(
            num_uniform_samples=0, num_pos_samples=0, num_neg_samples=0,
            num_in_pos_region=0, num_in_neg_region=0, num_in_overlap=0,
            cov_pos=0.0, cov_neg=0.0, cov_combined=0.0,
            balance=1.0,
            cov_uncov_in_Cfree=0.0, cov_uncov_in_Cobs=0.0,
            cov_pos_boosted=0.0, cov_neg_boosted=0.0,
            cov_combined_confidence_radius=0.0,
        )

    in_P = _union_membership(points, pos_regions)
    in_N = _union_membership(points, neg_regions)
    in_overlap = in_P & in_N

    n_pos = int(pos_labels.sum())
    n_neg = int((~pos_labels).sum())

    # cov_pos / cov_neg：传统单边覆盖
    cov_pos = float((in_P & pos_labels).sum() / max(n_pos, 1))
    cov_neg = float((in_N & ~pos_labels).sum() / max(n_neg, 1))

    # cov_combined：(|P| + |N|) / |Ω| —— 用均匀采样估计；零重叠时即 (|P| ∪ |N|) / |Ω|
    in_P_or_N = in_P | in_N
    cov_combined = float(in_P_or_N.sum() / n_total)
    p = max(min(cov_combined, 1.0 - 1e-12), 1e-12)
    cov_combined_radius = float(1.96 * math.sqrt(p * (1.0 - p) / n_total))

    # cov_pos_boosted：pos 样本中落在 P ∪ Ωc(N) 的比例
    # 等价于 1 - Pr[pos 样本 ∈ N]   (P 与 N 不相交时)
    boosted_pos_mask = (in_P | (~in_N)) & pos_labels
    cov_pos_boosted = float(boosted_pos_mask.sum() / max(n_pos, 1))

    # cov_neg_boosted：neg 样本中落在 N ∪ Ωc(P) 的比例
    boosted_neg_mask = (in_N | (~in_P)) & (~pos_labels)
    cov_neg_boosted = float(boosted_neg_mask.sum() / max(n_neg, 1))

    # ── Tier 1 新指标 ──
    # balance：两侧覆盖率的均衡度，触发自适应配额的依据
    balance = float(1.0 - abs(cov_pos - cov_neg))

    # uncov 拆解：cov_combined + cov_uncov_in_Cfree + cov_uncov_in_Cobs ≈ 1（零重叠下）
    cov_uncov_in_Cfree = float(((~in_P) & pos_labels).sum() / max(n_total, 1))
    cov_uncov_in_Cobs = float(((~in_N) & (~pos_labels)).sum() / max(n_total, 1))

    return DualCoverageStats(
        num_uniform_samples=n_total,
        num_pos_samples=n_pos,
        num_neg_samples=n_neg,
        num_in_pos_region=int(in_P.sum()),
        num_in_neg_region=int(in_N.sum()),
        num_in_overlap=int(in_overlap.sum()),
        cov_pos=cov_pos,
        cov_neg=cov_neg,
        cov_combined=cov_combined,
        balance=balance,
        cov_uncov_in_Cfree=cov_uncov_in_Cfree,
        cov_uncov_in_Cobs=cov_uncov_in_Cobs,
        cov_pos_boosted=cov_pos_boosted,
        cov_neg_boosted=cov_neg_boosted,
        cov_combined_confidence_radius=cov_combined_radius,
    )

