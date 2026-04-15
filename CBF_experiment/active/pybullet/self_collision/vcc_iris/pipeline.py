"""VCC + IRIS-ZO 多轮迭代流水线（论文 Algorithm 1）。

每轮：
  1. 从未覆盖的 C-free 区域采 K 个 free samples
  2. 全连接构造 visibility graph
  3. 截断团覆盖 (Truncated Clique Cover)
  4. MVEE 椭球初始化 → IRIS-ZO 区域膨胀
  5. 估计覆盖率，不满足则进入下一轮
"""
from __future__ import annotations

import time

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.clique_cover import truncated_clique_cover
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.coal_oracle import CoalSelfCollisionOracle
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import ExperimentConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.coverage import estimate_region_coverage
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.ellipsoids import summarize_cliques_with_ellipsoids
from CBF_experiment.active.pybullet.self_collision.vcc_iris.io.gui import evaluate_curve, playback_curve_gui, sample_curve_in_region
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.iris_zo import run_iris_zo
from CBF_experiment.active.pybullet.self_collision.vcc_iris.utils.progress import stage_print
from CBF_experiment.active.pybullet.self_collision.vcc_iris.io.reporting import write_experiment_report
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.sampling import sample_free_configurations, sample_coverage_test_points
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import CoverageEstimate, ExperimentReport, RoundStats
from CBF_experiment.active.pybullet.self_collision.vcc_iris.stages.visibility import build_visibility_graph


def run_vcc_iris_pipeline(cfg: ExperimentConfig = ExperimentConfig()) -> ExperimentReport:
    t_pipeline = time.perf_counter()
    stage_print("=" * 56)
    stage_print("VCC + IRIS-ZO iterative pipeline (Algorithm 1)")
    stage_print("=" * 56)
    stage_print(f"  K={cfg.SAMPLING.NUM_SAMPLES_PER_ROUND}  smin={cfg.CLIQUE.MIN_CLIQUE_SIZE}  "
                f"target={cfg.COVERAGE_TARGET}  max_rounds={cfg.MAX_VCC_ROUNDS}  "
                f"max_regions={cfg.MAX_TOTAL_REGIONS}")

    oracle = CoalSelfCollisionOracle(cfg.ROBOT)
    try:
        all_regions: list = []
        round_stats_list: list[RoundStats] = []
        region_counter = 0

        stage_print("采样覆盖率评估点 ...")
        coverage_test_pts = sample_coverage_test_points(
            oracle,
            num_samples=int(cfg.SAMPLING.NUM_COVERAGE_SAMPLES),
            rng_seed=99,
        )

        for vcc_round in range(1, int(cfg.MAX_VCC_ROUNDS) + 1):
            t_round = time.perf_counter()
            stage_print("-" * 56)
            stage_print(f"VCC 第 {vcc_round}/{cfg.MAX_VCC_ROUNDS} 轮")
            stage_print("-" * 56)

            # ── 1. 从未覆盖区域采样 ──
            stage_print(f"  [1/4] 采样 {cfg.SAMPLING.NUM_SAMPLES_PER_ROUND} 个 free samples (排除已覆盖)")
            round_samples = sample_free_configurations(
                oracle,
                cfg.SAMPLING,
                existing_regions=all_regions if all_regions else None,
            )
            if len(round_samples) < int(cfg.CLIQUE.MIN_CLIQUE_SIZE):
                stage_print(f"  仅采到 {len(round_samples)} 个未覆盖点, 停止迭代")
                break

            # ── 2. 全连接 visibility graph ──
            n = len(round_samples)
            total_pairs = n * (n - 1) // 2
            stage_print(f"  [2/4] Visibility graph ({n} 节点, {total_pairs} pairs)")
            graph = build_visibility_graph(
                round_samples,
                oracle,
                cfg.VISIBILITY,
                parallel_workers=int(cfg.VISIBILITY.PARALLEL_WORKERS),
            )
            stage_print(f"         visible: {graph.num_visible_edges}/{graph.num_candidate_pairs}")

            # ── 3. 截断团覆盖 + 椭球初始化 ──
            stage_print(f"  [3/4] Truncated clique cover (strategy={cfg.CLIQUE.STRATEGY})")
            cliques = truncated_clique_cover(graph, cfg.CLIQUE)
            if not cliques:
                stage_print(f"         0 cliques (max_clique < smin={cfg.CLIQUE.MIN_CLIQUE_SIZE}), 跳过本轮")
                round_stats_list.append(RoundStats(
                    round_id=vcc_round, num_samples=n, num_pairs=total_pairs,
                    num_visible_edges=graph.num_visible_edges, num_cliques=0,
                    clique_sizes=(), num_regions_grown=0,
                    coverage_after=estimate_region_coverage(coverage_test_pts, all_regions).ratio if all_regions else 0.0,
                    elapsed_seconds=time.perf_counter() - t_round,
                ))
                continue

            ellipsoids = summarize_cliques_with_ellipsoids(round_samples, cliques, oracle=oracle)
            stage_print(f"         {len(cliques)} cliques → {len(ellipsoids)} 初始椭球  "
                        f"sizes={[len(c.vertex_indices) for c in cliques]}")

            # ── 4. IRIS-ZO 区域膨胀 ──
            regions_this_round = 0
            budget_left = int(cfg.MAX_TOTAL_REGIONS) - len(all_regions)
            for ellipsoid in ellipsoids[:budget_left]:
                t_region = time.perf_counter()
                try:
                    region = run_iris_zo(ellipsoid, oracle, cfg.IRIS_ZO, region_id=region_counter)
                except ValueError as exc:
                    stage_print(f"         R{region_counter} 跳过: {exc}")
                    region_counter += 1
                    continue
                all_regions.append(region)
                regions_this_round += 1
                region_counter += 1
                stage_print(
                    f"         R{region.region_id}: log_det={region.log_det:+.4f}  "
                    f"planes={region.A.shape[0]}  耗时={time.perf_counter()-t_region:.1f}s"
                )

            # ── 覆盖率检查 ──
            coverage = estimate_region_coverage(coverage_test_pts, all_regions)
            dt_round = time.perf_counter() - t_round
            stage_print(f"  覆盖率={coverage.ratio:.4f}±{coverage.confidence_radius:.4f}  "
                        f"regions={len(all_regions)}  本轮耗时={dt_round:.1f}s")

            round_stats_list.append(RoundStats(
                round_id=vcc_round, num_samples=n, num_pairs=total_pairs,
                num_visible_edges=graph.num_visible_edges,
                num_cliques=len(cliques),
                clique_sizes=tuple(len(c.vertex_indices) for c in cliques),
                num_regions_grown=regions_this_round,
                coverage_after=coverage.ratio,
                elapsed_seconds=dt_round,
            ))

            if coverage.ratio >= float(cfg.COVERAGE_TARGET):
                stage_print(f"  覆盖率 {coverage.ratio:.4f} >= 目标 {cfg.COVERAGE_TARGET}, 停止迭代")
                break
            if len(all_regions) >= int(cfg.MAX_TOTAL_REGIONS):
                stage_print(f"  达到 region 上限 {cfg.MAX_TOTAL_REGIONS}, 停止迭代")
                break

        # ── 最终覆盖率 & 曲线采样 ──
        final_coverage = estimate_region_coverage(coverage_test_pts, all_regions) if all_regions else \
            CoverageEstimate(num_hits=0, num_samples=0, ratio=0.0, confidence_radius=0.0)

        rng = np.random.default_rng(int(cfg.IRIS_ZO.RNG_SEED))
        curve_report: dict = {}
        if all_regions:
            best_region = max(all_regions, key=lambda r: float(r.log_det))
            for _ in range(int(cfg.CURVE_MAX_ATTEMPTS)):
                curve = sample_curve_in_region(best_region, num_points=int(cfg.CURVE_NUM_POINTS), rng=rng)
                curve_report = evaluate_curve(oracle, curve)
                if not bool(curve_report["any_collision"]):
                    break

        report = ExperimentReport(
            regions=tuple(all_regions),
            coverage=final_coverage,
            round_stats=tuple(round_stats_list),
            visibility_stats={},
            clique_stats={
                "total_cliques": sum(rs.num_cliques for rs in round_stats_list),
            },
            sample_stats={
                "coverage_test_samples": int(cfg.SAMPLING.NUM_COVERAGE_SAMPLES),
                "samples_per_round": int(cfg.SAMPLING.NUM_SAMPLES_PER_ROUND),
                "total_rounds": len(round_stats_list),
            },
            curve_report=curve_report,
        )
        write_experiment_report(
            report,
            cover_json_path=cfg.REPORTING.COVER_JSON,
            experiment_json_path=cfg.REPORTING.EXPERIMENT_JSON,
        )

    finally:
        oracle.close()

    dt = time.perf_counter() - t_pipeline
    stage_print("=" * 56)
    stage_print(
        f"pipeline 完成: {len(report.regions)} regions  "
        f"覆盖率={report.coverage.ratio:.4f}  "
        f"曲线碰撞={report.curve_report.get('any_collision')}  "
        f"总耗时={dt:.1f}s  共 {len(round_stats_list)} 轮"
    )
    stage_print("=" * 56)

    if cfg.PLAYBACK_GUI and report.curve_report:
        playback_curve_gui(cfg.ROBOT, report.curve_report, hold_seconds=float(cfg.GUI_HOLD_SECONDS))
    return report
