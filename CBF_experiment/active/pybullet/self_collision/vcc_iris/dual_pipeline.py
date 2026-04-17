"""对偶 VCC + IRIS-ZO 流水线：正负 region 对称交替增长。

每个 macro round 包含两个 sub-round：
    pos sub-round：DualOracle(pos_base, neg_regions, margin) → 增长一波 pos region
    neg sub-round：DualOracle(neg_base, pos_regions, margin) → 增长一波 neg region

对方 region 作为 mutual pseudo-obstacle 喂给 IRIS-ZO，
保证两侧 by construction 几乎不重叠（带 ε margin）。

覆盖率使用同一批均匀采样点统计五个指标（详见 DualCoverageStats）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import (
    CliqueCoverConfig,
    ExperimentConfig,
    IrisZoConfig,
    SamplingConfig,
    VisibilityConfig,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import (
    DualCoverageStats,
    DualExperimentReport,
    DualRoundStats,
    IrisRegion,
    RoundStats,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import FreeSample
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.dual_oracle import (
    DualOracle,
    _point_in_any_polytope,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.robot_model import sample_joint_box
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.clique_cover import truncated_clique_cover
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.coverage import estimate_dual_coverage
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.ellipsoids import summarize_cliques_with_ellipsoids
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.iris_zo import run_iris_zo
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.sampling import sample_free_configurations
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.visibility import build_visibility_graph
from CBF_experiment.active.pybullet.self_collision.vcc_iris.utils.progress import ProgressBar, stage_print


# ────────────────────────── 配置 ──────────────────────────


@dataclass(frozen=True)
class DualConfig:
    """对偶分解专属配置。"""
    SAMPLING: SamplingConfig = field(default_factory=SamplingConfig)
    VISIBILITY: VisibilityConfig = field(default_factory=VisibilityConfig)
    CLIQUE: CliqueCoverConfig = field(default_factory=CliqueCoverConfig)
    IRIS_ZO: IrisZoConfig = field(default_factory=IrisZoConfig)

    # ── 对偶专属参数 ──
    MUTUAL_MARGIN: float = 1e-3                  # 对方 region 收缩量（创造 ε 间隙）
    MAX_MACRO_ROUNDS: int = 10                   # 对偶 macro 轮数（每轮 = pos+neg 两个 sub-round）
    MAX_REGIONS_PER_SIDE: int = 64               # 每一侧最多生长的 region 数
    MAX_REGIONS_PER_SUBROUND: int = 6            # 每个 sub-round 的配额（防止某一侧贪光）
    COMBINED_COVERAGE_TARGET: float = 0.85       # 主停止条件：cov_combined >= 此值即停
    NUM_UNIFORM_COVERAGE_SAMPLES: int = 8000     # 均匀覆盖率采样点数（用于 5 指标）
    UNIFORM_COVERAGE_RNG_SEED: int = 99
    EARLY_STOP_NO_GROWTH_ROUNDS: int = 2         # 连续 N 个 macro round 没增长则停止

    # ── 共享采样开关 ──
    USE_SHARED_SAMPLING: bool = True             # True: 每 macro round 一次性均匀采样, base oracle 一次过标签, 正负各取自己那一半
    SHARED_SAMPLING_RNG_SEED: int = 31           # 共享采样的 RNG 种子


# ────────────────────────── 工具函数 ──────────────────────────


def _label_uniform_points(uniform_points: np.ndarray, base_oracle) -> np.ndarray:
    """对均匀采样点用 base_oracle.is_self_collision 打 pos/neg 标签（True = pos = C-free）。"""
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.utils.progress import ProgressBar
    n = uniform_points.shape[0]
    labels = np.zeros((n,), dtype=bool)
    pb = ProgressBar(n, prefix="[label-uniform]")
    for i in range(n):
        labels[i] = not bool(base_oracle.is_self_collision(uniform_points[i]))
        pb.update()
    pb.close()
    return labels


def _sample_uniform_points(metadata, num_samples: int, rng_seed: int) -> np.ndarray:
    """从 joint box 均匀采样（不做 oracle 过滤）。"""
    rng = np.random.default_rng(int(rng_seed))
    return sample_joint_box(metadata, rng, num_samples=int(num_samples))


def _shared_sample_round(
    *,
    base_oracle,
    pos_regions: list[IrisRegion],
    neg_regions: list[IrisRegion],
    cfg: DualConfig,
    rng: np.random.Generator,
) -> tuple[list[FreeSample], list[FreeSample], dict]:
    """共享采样：均匀采样 + base_oracle.is_self_collision 一次性打 pos/neg 标签，
    分别筛掉自己已覆盖的 region 和对方 region (带 margin) → 返回两侧的 sample 池。

    target = cfg.SAMPLING.NUM_SAMPLES_PER_ROUND（每侧目标）；
    持续画 batch 直到两侧都收满。
    每个 q 只调用 base_oracle.is_self_collision 一次（这是 oracle 调用主成本）。
    """
    metadata = base_oracle.metadata
    target = int(cfg.SAMPLING.NUM_SAMPLES_PER_ROUND)
    margin = float(cfg.MUTUAL_MARGIN)

    pos_arrays = [
        (np.asarray(r.A, dtype=float), np.asarray(r.b, dtype=float).reshape(-1))
        for r in pos_regions
    ]
    neg_arrays = [
        (np.asarray(r.A, dtype=float), np.asarray(r.b, dtype=float).reshape(-1))
        for r in neg_regions
    ]

    pos_pool: list[FreeSample] = []
    neg_pool: list[FreeSample] = []
    diag = {
        "total_drawn": 0,
        "pos_label_count": 0,
        "neg_label_count": 0,
        "pos_filtered_by_own": 0,
        "pos_filtered_by_opposite": 0,
        "neg_filtered_by_own": 0,
        "neg_filtered_by_opposite": 0,
    }

    pb = ProgressBar(2 * target, prefix="[shared-sampling]")
    while len(pos_pool) < target or len(neg_pool) < target:
        batch = sample_joint_box(metadata, rng, num_samples=max(8, int(cfg.SAMPLING.BATCH_SIZE)))
        for q in batch:
            diag["total_drawn"] += 1
            is_collision = bool(base_oracle.is_self_collision(q))
            q_arr = np.asarray(q, dtype=float)
            if not is_collision:
                # pos 候选
                diag["pos_label_count"] += 1
                if len(pos_pool) >= target:
                    continue
                if pos_arrays and _point_in_any_polytope(q_arr, pos_arrays, 0.0):
                    diag["pos_filtered_by_own"] += 1
                    continue
                if neg_arrays and _point_in_any_polytope(q_arr, neg_arrays, margin):
                    diag["pos_filtered_by_opposite"] += 1
                    continue
                pos_pool.append(FreeSample(q=q_arr, clearance=0.0, active_pair=None))
            else:
                # neg 候选
                diag["neg_label_count"] += 1
                if len(neg_pool) >= target:
                    continue
                if neg_arrays and _point_in_any_polytope(q_arr, neg_arrays, 0.0):
                    diag["neg_filtered_by_own"] += 1
                    continue
                if pos_arrays and _point_in_any_polytope(q_arr, pos_arrays, margin):
                    diag["neg_filtered_by_opposite"] += 1
                    continue
                neg_pool.append(FreeSample(q=q_arr, clearance=0.0, active_pair=None))

            pb.set(
                len(pos_pool) + len(neg_pool),
                suffix=f"pos={len(pos_pool)}/{target} neg={len(neg_pool)}/{target} drawn={diag['total_drawn']}",
            )
            if len(pos_pool) >= target and len(neg_pool) >= target:
                break
    pb.close(suffix=f"drawn={diag['total_drawn']} (pos {diag['pos_label_count']} / neg {diag['neg_label_count']})")
    return pos_pool, neg_pool, diag


def _print_dual_coverage(stats: DualCoverageStats, *, prefix: str = "  ", show_diag: bool = True) -> None:
    # 头牌：每侧无加权 + balance（两侧均衡度）
    stage_print(
        f"{prefix}cov_pos       = {stats.cov_pos:.4f}    cov_neg       = {stats.cov_neg:.4f}    "
        f"balance = {stats.balance:.4f}   <头牌>"
    )
    # 二线：混合体积比 + 未覆盖拆解（满足 cov_combined + uncov_in_Cfree + uncov_in_Cobs ≈ 1）
    stage_print(f"{prefix}cov_combined  = {stats.cov_combined:.4f} ± {stats.cov_combined_confidence_radius:.4f}   <混合>")
    stage_print(
        f"{prefix}uncov(Cfree)  = {stats.cov_uncov_in_Cfree:.4f}    "
        f"uncov(Cobs)   = {stats.cov_uncov_in_Cobs:.4f}    "
        f"check = {stats.cov_combined + stats.cov_uncov_in_Cfree + stats.cov_uncov_in_Cobs:.4f}"
    )
    # 旧 boosted（零重叠下接近 1, 仅作完整性保留）
    stage_print(f"{prefix}cov_pos_boost = {stats.cov_pos_boosted:.4f}    cov_neg_boost = {stats.cov_neg_boosted:.4f}   <旧·零重叠下≈1>")
    if show_diag:
        stage_print(
            f"{prefix}诊断: |P|/|Ω|={stats.num_in_pos_region / max(stats.num_uniform_samples, 1):.4f}  "
            f"|N|/|Ω|={stats.num_in_neg_region / max(stats.num_uniform_samples, 1):.4f}  "
            f"|P∩N|/|Ω|={stats.num_in_overlap / max(stats.num_uniform_samples, 1):.6f}  "
            f"(pos样本={stats.num_pos_samples}, neg样本={stats.num_neg_samples})"
        )


# ────────────────────────── 单个 sub-round ──────────────────────────


def _run_one_subround(
    *,
    side: str,                                   # "pos" or "neg"
    base_oracle,
    opposite_regions: list[IrisRegion],
    own_regions: list[IrisRegion],
    cfg: DualConfig,
    region_id_counter: int,
    region_quota: int,
    samples: list[FreeSample] | None = None,
) -> tuple[list[IrisRegion], int, dict]:
    """运行一个 sub-round：包装 DualOracle → 采样 → 可见性 → 团 → IRIS-ZO 增长。

    Parameters
    ----------
    samples : list[FreeSample] | None
        若提供，则直接使用（共享采样模式）；
        否则调用 sample_free_configurations 走 DualOracle 自行采（独立采样模式）。

    返回：(本轮新增的 region 列表, 新的 region_id_counter, sub-round 统计 dict)
    """
    t_round = time.perf_counter()
    stage_print("-" * 60)
    stage_print(f"  [{side.upper()}] sub-round | 已有 {len(own_regions)} 个 {side} region, "
                f"对方 region={len(opposite_regions)}, 本轮配额={region_quota}")
    stage_print("-" * 60)

    # 包装 DualOracle：base + 对方 polytope 作为伪障碍
    dual_oracle = DualOracle(
        base_oracle,
        opposite_polytopes=opposite_regions,
        margin=float(cfg.MUTUAL_MARGIN),
    )

    # ── 1. 采样未覆盖（自己的 region 做排除）的可行点 ──
    if samples is not None:
        round_samples = samples
        stage_print(f"    [1/4] 共享采样模式: 复用预先采好的 {len(round_samples)} 个 {side} 样本")
    else:
        stage_print(f"    [1/4] 独立采样模式: 采 {cfg.SAMPLING.NUM_SAMPLES_PER_ROUND} 个 {side}-可行点 (排除已覆盖)")
        try:
            round_samples = sample_free_configurations(
                dual_oracle,
                cfg.SAMPLING,
                existing_regions=own_regions if own_regions else None,
            )
        except Exception as exc:
            stage_print(f"    采样失败: {exc}")
            return [], region_id_counter, {
                "side": side, "num_samples": 0, "num_pairs": 0, "num_visible_edges": 0,
                "num_cliques": 0, "clique_sizes": (), "num_regions_grown": 0,
                "elapsed_seconds": time.perf_counter() - t_round,
            }
    if len(round_samples) < int(cfg.CLIQUE.MIN_CLIQUE_SIZE):
        stage_print(f"    仅采到 {len(round_samples)} 个未覆盖点 (< smin={cfg.CLIQUE.MIN_CLIQUE_SIZE}), 跳过")
        return [], region_id_counter, {
            "side": side, "num_samples": len(round_samples), "num_pairs": 0,
            "num_visible_edges": 0, "num_cliques": 0, "clique_sizes": (),
            "num_regions_grown": 0, "elapsed_seconds": time.perf_counter() - t_round,
        }

    # ── 2. 可见性图（dual oracle 会自动把对方 region 视为障碍）──
    n = len(round_samples)
    total_pairs = n * (n - 1) // 2
    stage_print(f"    [2/4] Visibility graph ({n} 节点, {total_pairs} pairs)")
    graph = build_visibility_graph(
        round_samples,
        dual_oracle,
        cfg.VISIBILITY,
        parallel_workers=int(cfg.VISIBILITY.PARALLEL_WORKERS),
    )
    stage_print(f"          visible: {graph.num_visible_edges}/{graph.num_candidate_pairs}")

    # ── 3. 团覆盖 + 椭球 ──
    stage_print(f"    [3/4] Truncated clique cover (strategy={cfg.CLIQUE.STRATEGY})")
    cliques = truncated_clique_cover(graph, cfg.CLIQUE)
    if not cliques:
        stage_print(f"          0 cliques (max_clique < smin={cfg.CLIQUE.MIN_CLIQUE_SIZE}), 跳过")
        return [], region_id_counter, {
            "side": side, "num_samples": n, "num_pairs": total_pairs,
            "num_visible_edges": graph.num_visible_edges, "num_cliques": 0,
            "clique_sizes": (), "num_regions_grown": 0,
            "elapsed_seconds": time.perf_counter() - t_round,
        }
    ellipsoids = summarize_cliques_with_ellipsoids(round_samples, cliques, oracle=dual_oracle)
    stage_print(f"          {len(cliques)} cliques → {len(ellipsoids)} 椭球 "
                f"sizes={[len(c.vertex_indices) for c in cliques]}")

    # ── 4. IRIS-ZO 区域膨胀（受 region_quota 限制）──
    new_regions: list[IrisRegion] = []
    for ellipsoid in ellipsoids[:region_quota]:
        t_region = time.perf_counter()
        try:
            region = run_iris_zo(ellipsoid, dual_oracle, cfg.IRIS_ZO, region_id=region_id_counter)
        except ValueError as exc:
            stage_print(f"          R{region_id_counter} 跳过: {exc}")
            region_id_counter += 1
            continue
        new_regions.append(region)
        region_id_counter += 1
        stage_print(
            f"          {side.upper()}-R{region.region_id}: log_det={region.log_det:+.4f}  "
            f"planes={region.A.shape[0]}  耗时={time.perf_counter()-t_region:.1f}s"
        )

    dt_round = time.perf_counter() - t_round
    stage_print(f"    {side.upper()} sub-round 完成: 新增 {len(new_regions)} regions, 耗时={dt_round:.1f}s")
    return new_regions, region_id_counter, {
        "side": side,
        "num_samples": n,
        "num_pairs": total_pairs,
        "num_visible_edges": graph.num_visible_edges,
        "num_cliques": len(cliques),
        "clique_sizes": tuple(len(c.vertex_indices) for c in cliques),
        "num_regions_grown": len(new_regions),
        "elapsed_seconds": dt_round,
    }


# ────────────────────────── 主入口 ──────────────────────────


def run_dual_decomposition_pipeline(
    *,
    pos_base_oracle,
    neg_base_oracle,
    cfg: DualConfig = None,
    progress_callback: Callable[[int, DualCoverageStats], None] | None = None,
) -> DualExperimentReport:
    """对偶分解主入口。

    Parameters
    ----------
    pos_base_oracle : oracle
        正侧基础 oracle（如 CoalSelfCollisionOracle）。要求 is_self_collision(q) = True ⇔ q 不可作为 pos region 内点。
    neg_base_oracle : oracle
        负侧基础 oracle（如 NegationOracle(CoalSelfCollisionOracle)）。
        与 pos_base_oracle 在同一个 metadata / pybullet 连接上。
        若两个 oracle 共享同一个 pybullet 物理状态，本函数会调度避免冲突。
    cfg : DualConfig
    """
    if cfg is None:
        cfg = DualConfig()

    t_pipeline = time.perf_counter()
    stage_print("=" * 60)
    stage_print("Dual VCC + IRIS-ZO  (对称交替, mutual pseudo-obstacle)")
    stage_print("=" * 60)
    stage_print(f"  MAX_MACRO_ROUNDS={cfg.MAX_MACRO_ROUNDS}  MAX_REGIONS/side={cfg.MAX_REGIONS_PER_SIDE}  "
                f"配额/sub-round={cfg.MAX_REGIONS_PER_SUBROUND}  margin={cfg.MUTUAL_MARGIN}")
    stage_print(f"  COVERAGE_TARGET (combined) = {cfg.COMBINED_COVERAGE_TARGET}  "
                f"均匀采样 = {cfg.NUM_UNIFORM_COVERAGE_SAMPLES}")

    # ── 准备均匀覆盖率采样点 + 标签 ──
    stage_print("准备均匀覆盖率采样点 ...")
    uniform_pts = _sample_uniform_points(
        pos_base_oracle.metadata,
        num_samples=int(cfg.NUM_UNIFORM_COVERAGE_SAMPLES),
        rng_seed=int(cfg.UNIFORM_COVERAGE_RNG_SEED),
    )
    stage_print(f"  {uniform_pts.shape[0]} 个均匀点采集完成，开始用 base oracle 打 pos/neg 标签 ...")
    pos_labels = _label_uniform_points(uniform_pts, pos_base_oracle)
    n_pos_uniform = int(pos_labels.sum())
    n_neg_uniform = int((~pos_labels).sum())
    stage_print(f"  标签分布: pos={n_pos_uniform} ({n_pos_uniform/uniform_pts.shape[0]:.3f}), "
                f"neg={n_neg_uniform} ({n_neg_uniform/uniform_pts.shape[0]:.3f})")

    pos_regions: list[IrisRegion] = []
    neg_regions: list[IrisRegion] = []
    pos_id_counter = 0
    neg_id_counter = 0
    round_stats_list: list[DualRoundStats] = []
    no_growth_rounds = 0
    last_combined = 0.0
    last_coverage_stats = estimate_dual_coverage(uniform_pts, pos_labels, pos_regions, neg_regions)
    pos_subround_id = 0
    neg_subround_id = 0

    shared_rng = np.random.default_rng(int(cfg.SHARED_SAMPLING_RNG_SEED)) if cfg.USE_SHARED_SAMPLING else None
    if cfg.USE_SHARED_SAMPLING:
        stage_print("已启用共享采样：每 macro round 用 base oracle 一次性打 pos/neg 标签，正负各取 NUM_SAMPLES_PER_ROUND 个。")
    else:
        stage_print("未启用共享采样：正负 sub-round 各自走 DualOracle 独立采样。")

    for macro_round in range(1, int(cfg.MAX_MACRO_ROUNDS) + 1):
        stage_print("=" * 60)
        stage_print(f"Macro Round {macro_round}/{cfg.MAX_MACRO_ROUNDS}")
        stage_print("=" * 60)

        round_pos_grown = 0
        round_neg_grown = 0

        # ── 共享采样：一次性画 base oracle 标签 → 正负样本池 ──
        pos_samples_for_round: list[FreeSample] | None = None
        neg_samples_for_round: list[FreeSample] | None = None
        if cfg.USE_SHARED_SAMPLING:
            stage_print("-" * 60)
            stage_print(f"  [shared sampling] 准备本 macro round 的共享样本池 (target/side={cfg.SAMPLING.NUM_SAMPLES_PER_ROUND})")
            pos_samples_for_round, neg_samples_for_round, sample_diag = _shared_sample_round(
                base_oracle=pos_base_oracle,   # base 用 pos 侧（is_self_collision=True 即 C-obs）
                pos_regions=pos_regions,
                neg_regions=neg_regions,
                cfg=cfg,
                rng=shared_rng,
            )
            stage_print(
                f"  [shared sampling] 共采 {sample_diag['total_drawn']} 点 → "
                f"label pos={sample_diag['pos_label_count']} neg={sample_diag['neg_label_count']} | "
                f"pool pos={len(pos_samples_for_round)} neg={len(neg_samples_for_round)}"
            )
            stage_print(
                f"  [shared sampling] 过滤明细: "
                f"pos|own={sample_diag['pos_filtered_by_own']} pos|opp={sample_diag['pos_filtered_by_opposite']}  "
                f"neg|own={sample_diag['neg_filtered_by_own']} neg|opp={sample_diag['neg_filtered_by_opposite']}"
            )

        # ── POS sub-round ──
        if len(pos_regions) < int(cfg.MAX_REGIONS_PER_SIDE):
            quota = min(
                int(cfg.MAX_REGIONS_PER_SUBROUND),
                int(cfg.MAX_REGIONS_PER_SIDE) - len(pos_regions),
            )
            new_pos, pos_id_counter, pos_subround_stats = _run_one_subround(
                side="pos",
                base_oracle=pos_base_oracle,
                opposite_regions=neg_regions,
                own_regions=pos_regions,
                cfg=cfg,
                region_id_counter=pos_id_counter,
                region_quota=quota,
                samples=pos_samples_for_round,
            )
            pos_regions.extend(new_pos)
            round_pos_grown = len(new_pos)
            pos_subround_id += 1
            round_stats_list.append(DualRoundStats(
                macro_round_id=macro_round, side="pos", sub_round_id=pos_subround_id,
                num_samples=pos_subround_stats["num_samples"],
                num_pairs=pos_subround_stats["num_pairs"],
                num_visible_edges=pos_subround_stats["num_visible_edges"],
                num_cliques=pos_subround_stats["num_cliques"],
                clique_sizes=pos_subround_stats["clique_sizes"],
                num_regions_grown=pos_subround_stats["num_regions_grown"],
                elapsed_seconds=pos_subround_stats["elapsed_seconds"],
                coverage_after=None,
            ))
        else:
            stage_print(f"  POS sub-round 跳过：已达 MAX_REGIONS_PER_SIDE={cfg.MAX_REGIONS_PER_SIDE}")

        # ── NEG sub-round ──
        if len(neg_regions) < int(cfg.MAX_REGIONS_PER_SIDE):
            quota = min(
                int(cfg.MAX_REGIONS_PER_SUBROUND),
                int(cfg.MAX_REGIONS_PER_SIDE) - len(neg_regions),
            )
            new_neg, neg_id_counter, neg_subround_stats = _run_one_subround(
                side="neg",
                base_oracle=neg_base_oracle,
                opposite_regions=pos_regions,
                own_regions=neg_regions,
                cfg=cfg,
                region_id_counter=neg_id_counter,
                region_quota=quota,
                samples=neg_samples_for_round,
            )
            neg_regions.extend(new_neg)
            round_neg_grown = len(new_neg)
            neg_subround_id += 1
            round_stats_list.append(DualRoundStats(
                macro_round_id=macro_round, side="neg", sub_round_id=neg_subround_id,
                num_samples=neg_subround_stats["num_samples"],
                num_pairs=neg_subround_stats["num_pairs"],
                num_visible_edges=neg_subround_stats["num_visible_edges"],
                num_cliques=neg_subround_stats["num_cliques"],
                clique_sizes=neg_subround_stats["clique_sizes"],
                num_regions_grown=neg_subround_stats["num_regions_grown"],
                elapsed_seconds=neg_subround_stats["elapsed_seconds"],
                coverage_after=None,
            ))
        else:
            stage_print(f"  NEG sub-round 跳过：已达 MAX_REGIONS_PER_SIDE={cfg.MAX_REGIONS_PER_SIDE}")

        # ── 评估 5 指标 ──
        stage_print("-" * 60)
        stage_print(f"Macro Round {macro_round} 评估 5 个覆盖率指标 ...")
        last_coverage_stats = estimate_dual_coverage(uniform_pts, pos_labels, pos_regions, neg_regions)
        # 在最后一个 sub-round 的 stats 上挂覆盖率
        if round_stats_list:
            rs = round_stats_list[-1]
            round_stats_list[-1] = DualRoundStats(
                macro_round_id=rs.macro_round_id, side=rs.side, sub_round_id=rs.sub_round_id,
                num_samples=rs.num_samples, num_pairs=rs.num_pairs,
                num_visible_edges=rs.num_visible_edges, num_cliques=rs.num_cliques,
                clique_sizes=rs.clique_sizes, num_regions_grown=rs.num_regions_grown,
                elapsed_seconds=rs.elapsed_seconds, coverage_after=last_coverage_stats,
            )
        _print_dual_coverage(last_coverage_stats, prefix="  ")
        stage_print(f"  当前累计：pos_regions={len(pos_regions)}  neg_regions={len(neg_regions)}")

        if progress_callback is not None:
            progress_callback(macro_round, last_coverage_stats)

        # ── 早停判定 ──
        if last_coverage_stats.cov_combined >= float(cfg.COMBINED_COVERAGE_TARGET):
            stage_print(f"  cov_combined {last_coverage_stats.cov_combined:.4f} >= 目标 "
                        f"{cfg.COMBINED_COVERAGE_TARGET}, 停止。")
            break

        if round_pos_grown == 0 and round_neg_grown == 0:
            no_growth_rounds += 1
            stage_print(f"  本轮 pos/neg 均未新增 region (连续 {no_growth_rounds}/{cfg.EARLY_STOP_NO_GROWTH_ROUNDS} 轮)")
            if no_growth_rounds >= int(cfg.EARLY_STOP_NO_GROWTH_ROUNDS):
                stage_print(f"  连续 {no_growth_rounds} 轮无增长，提前停止。")
                break
        else:
            no_growth_rounds = 0

        if (
            len(pos_regions) >= int(cfg.MAX_REGIONS_PER_SIDE)
            and len(neg_regions) >= int(cfg.MAX_REGIONS_PER_SIDE)
        ):
            stage_print("  pos / neg 均达到 MAX_REGIONS_PER_SIDE, 停止。")
            break

        last_combined = last_coverage_stats.cov_combined

    dt_total = time.perf_counter() - t_pipeline
    stage_print("=" * 60)
    stage_print(f"对偶分解完成。 pos={len(pos_regions)}  neg={len(neg_regions)}  "
                f"cov_combined={last_coverage_stats.cov_combined:.4f}  总耗时={dt_total:.1f}s")
    stage_print("=" * 60)
    _print_dual_coverage(last_coverage_stats, prefix="  ")

    return DualExperimentReport(
        pos_regions=tuple(pos_regions),
        neg_regions=tuple(neg_regions),
        final_coverage=last_coverage_stats,
        round_stats=tuple(round_stats_list),
        sample_stats={
            "num_uniform_samples": int(uniform_pts.shape[0]),
            "num_pos_samples": n_pos_uniform,
            "num_neg_samples": n_neg_uniform,
            "samples_per_subround": int(cfg.SAMPLING.NUM_SAMPLES_PER_ROUND),
            "total_macro_rounds_run": int(macro_round),
            "total_elapsed_seconds": float(dt_total),
        },
        config_summary={
            "MUTUAL_MARGIN": float(cfg.MUTUAL_MARGIN),
            "MAX_MACRO_ROUNDS": int(cfg.MAX_MACRO_ROUNDS),
            "MAX_REGIONS_PER_SIDE": int(cfg.MAX_REGIONS_PER_SIDE),
            "MAX_REGIONS_PER_SUBROUND": int(cfg.MAX_REGIONS_PER_SUBROUND),
            "COMBINED_COVERAGE_TARGET": float(cfg.COMBINED_COVERAGE_TARGET),
            "VISIBILITY.PARALLEL_WORKERS": int(cfg.VISIBILITY.PARALLEL_WORKERS),
            "IRIS_ZO.NUM_PARTICLES": int(cfg.IRIS_ZO.NUM_PARTICLES),
            "IRIS_ZO.MAX_OUTER_ITERATIONS": int(cfg.IRIS_ZO.MAX_OUTER_ITERATIONS),
            "IRIS_ZO.MAX_INNER_ITERATIONS": int(cfg.IRIS_ZO.MAX_INNER_ITERATIONS),
        },
    )
