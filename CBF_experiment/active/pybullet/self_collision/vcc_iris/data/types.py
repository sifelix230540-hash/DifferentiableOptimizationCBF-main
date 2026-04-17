"""流水线各阶段共用的数据结构定义（dataclass）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RobotModelMetadata:
    revolute_ids: tuple[int, ...]
    revolute_names: tuple[str, ...]
    joint_limits: tuple[tuple[float, float], ...]
    q_indices: tuple[int, ...]
    q_base: np.ndarray
    dq_base: np.ndarray
    monitored_link_ids: tuple[int, ...]
    monitored_link_names: tuple[str, ...]
    monitored_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class FreeSample:
    q: np.ndarray
    clearance: float
    active_pair: tuple[int, int] | None


@dataclass(frozen=True)
class VisibilityGraph:
    vertices: np.ndarray
    adjacency: tuple[frozenset[int], ...]
    edges: tuple[tuple[int, int], ...]
    num_candidate_pairs: int
    num_visible_edges: int


@dataclass(frozen=True)
class Clique:
    vertex_indices: tuple[int, ...]
    score: float
    bad_vertex_count: int = 0


@dataclass(frozen=True)
class CliqueEllipsoid:
    vertex_indices: tuple[int, ...]
    center: np.ndarray
    C: np.ndarray
    clique_size: int


@dataclass(frozen=True)
class IrisRegion:
    region_id: int
    source_clique_indices: tuple[int, ...]
    A: np.ndarray
    b: np.ndarray
    center: np.ndarray
    C: np.ndarray
    log_det: float
    iterations: tuple[dict, ...]


@dataclass(frozen=True)
class CoverageEstimate:
    num_hits: int
    num_samples: int
    ratio: float
    confidence_radius: float


@dataclass(frozen=True)
class RoundStats:
    """每轮 VCC 迭代的统计信息。"""
    round_id: int
    num_samples: int
    num_pairs: int
    num_visible_edges: int
    num_cliques: int
    clique_sizes: tuple[int, ...]
    num_regions_grown: int
    coverage_after: float
    elapsed_seconds: float


@dataclass(frozen=True)
class ExperimentReport:
    regions: tuple[IrisRegion, ...]
    coverage: CoverageEstimate
    round_stats: tuple[RoundStats, ...] = ()
    visibility_stats: dict = field(default_factory=dict)
    clique_stats: dict = field(default_factory=dict)
    sample_stats: dict = field(default_factory=dict)
    curve_report: dict = field(default_factory=dict)


# ────────────────────────── 对偶分解：正负 region 同时增长 ──────────────────────────


@dataclass(frozen=True)
class DualCoverageStats:
    """对偶分解的覆盖率统计（基于一批均匀采样点）。

    设：
      * P = 所有正 region 的并  (≈ 落在 C-free 内)
      * N = 所有负 region 的并  (≈ 落在 C-obs 内)
      * Ω = 整个 C-space (joint box)

    给定 num_uniform_samples 个均匀采样点 q，由 base oracle 把每个 q 标为 pos / neg：
      * pos 样本 = oracle.is_self_collision(q) == False  (落在 C-free)
      * neg 样本 = oracle.is_self_collision(q) == True   (落在 C-obs)
    """
    num_uniform_samples: int                      # 总均匀采样点数
    num_pos_samples: int                          # 其中被标为 pos 的数量
    num_neg_samples: int                          # 其中被标为 neg 的数量

    # 显式落在 P / N 内的均匀样本数（由 region polytope 判定）
    num_in_pos_region: int                        # |q ∈ P|
    num_in_neg_region: int                        # |q ∈ N|
    num_in_overlap: int                           # |q ∈ P ∩ N|（应≈0，理论保证零重叠）

    # ── Tier 1 头牌指标 ─────────────────────────────────────────
    # 推荐看顺序：cov_pos / cov_neg / balance > cov_combined > uncov_breakdown
    cov_pos: float                                # # pos 样本 ∈ P / # pos 样本（每侧无加权）
    cov_neg: float                                # # neg 样本 ∈ N / # neg 样本（每侧无加权）
    cov_combined: float                           # (|P| + |N|) / |Ω|（混合体积比, 受 vol(F):vol(O) 隐式加权）
    balance: float                                # 1 - |cov_pos - cov_neg|, ∈[0,1], 高=两侧进度均衡

    # ── Tier 1 未覆盖拆解（替代旧 boosted 的信息量） ────────────────
    # 满足: cov_combined + uncov_in_Cfree + uncov_in_Cobs ≈ 1（零重叠下）
    cov_uncov_in_Cfree: float                     # # {pos 样本 ∉ P} / # 总均匀样本（C-free 内的碎片化空地）
    cov_uncov_in_Cobs: float                      # # {neg 样本 ∉ N} / # 总均匀样本（C-obs 内的碎片化空地）

    # ── 旧 boosted 指标（保留向后兼容；零重叠下 ≈ 1.0, 信息量低） ──
    cov_pos_boosted: float                        # # pos 样本 ∈ (P ∪ Ωc(N)) / # pos 样本
    cov_neg_boosted: float                        # # neg 样本 ∈ (N ∪ Ωc(P)) / # neg 样本

    # 置信半径（按 cov_combined 计算的 95% Wilson 半径）
    cov_combined_confidence_radius: float


@dataclass(frozen=True)
class DualRoundStats:
    """对偶分解每个 macro round 的统计：一个 pos sub-round + 一个 neg sub-round。"""
    macro_round_id: int
    side: str                                     # "pos" or "neg" (sub-round)
    sub_round_id: int                             # 该侧自启动以来的累计编号
    num_samples: int
    num_pairs: int
    num_visible_edges: int
    num_cliques: int
    clique_sizes: tuple[int, ...]
    num_regions_grown: int
    elapsed_seconds: float
    coverage_after: DualCoverageStats | None = None


@dataclass(frozen=True)
class DualExperimentReport:
    pos_regions: tuple[IrisRegion, ...]
    neg_regions: tuple[IrisRegion, ...]
    final_coverage: DualCoverageStats
    round_stats: tuple[DualRoundStats, ...] = ()
    sample_stats: dict = field(default_factory=dict)
    config_summary: dict = field(default_factory=dict)
