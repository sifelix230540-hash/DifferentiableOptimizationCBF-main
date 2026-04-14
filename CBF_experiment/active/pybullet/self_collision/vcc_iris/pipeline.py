from __future__ import annotations

import time

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.clique_cover import greedy_clique_cover
from CBF_experiment.active.pybullet.self_collision.vcc_iris.coal_oracle import CoalSelfCollisionOracle
from CBF_experiment.active.pybullet.self_collision.vcc_iris.config import ExperimentConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.coverage import estimate_region_coverage
from CBF_experiment.active.pybullet.self_collision.vcc_iris.ellipsoids import summarize_cliques_with_ellipsoids
from CBF_experiment.active.pybullet.self_collision.vcc_iris.gui import evaluate_curve, playback_curve_gui, sample_curve_in_region
from CBF_experiment.active.pybullet.self_collision.vcc_iris.iris_zo import run_iris_zo
from CBF_experiment.active.pybullet.self_collision.vcc_iris.progress import stage_print
from CBF_experiment.active.pybullet.self_collision.vcc_iris.reporting import write_experiment_report
from CBF_experiment.active.pybullet.self_collision.vcc_iris.sampling import load_free_samples, sample_free_configurations, save_free_samples
from CBF_experiment.active.pybullet.self_collision.vcc_iris.types import ExperimentReport
from CBF_experiment.active.pybullet.self_collision.vcc_iris.visibility import build_visibility_graph


def run_vcc_iris_pipeline(cfg: ExperimentConfig = ExperimentConfig()) -> ExperimentReport:
    t_pipeline = time.perf_counter()
    stage_print("═" * 50)
    stage_print("VCC + IRIS-ZO pipeline 启动")
    stage_print("═" * 50)
    oracle = CoalSelfCollisionOracle(cfg.ROBOT)
    try:
        stage_print(f"阶段 1/5: 采样 (目标 {cfg.SAMPLING.NUM_SEED_SAMPLES} 个 free samples)")
        free_samples = load_free_samples(cfg.SAMPLING.SAMPLE_CACHE_PATH)
        if len(free_samples) < int(cfg.SAMPLING.NUM_SEED_SAMPLES):
            free_samples = sample_free_configurations(oracle, cfg.SAMPLING)
            save_free_samples(free_samples, cfg.SAMPLING.SAMPLE_CACHE_PATH)
        else:
            stage_print(f"  从缓存加载 {len(free_samples)} 个 samples")

        stage_print(f"阶段 2/5: Visibility graph ({len(free_samples)} 节点)")
        graph = build_visibility_graph(
            free_samples,
            oracle,
            cfg.VISIBILITY,
            parallel_workers=int(cfg.VISIBILITY.PARALLEL_WORKERS),
        )
        stage_print(f"  边: {graph.num_visible_edges}/{graph.num_candidate_pairs} visible")

        stage_print("阶段 3/5: Clique cover + 椭球初始化")
        cliques = greedy_clique_cover(graph, cfg.CLIQUE)
        ellipsoids = summarize_cliques_with_ellipsoids(free_samples, cliques, oracle=oracle)
        stage_print(f"  {len(cliques)} cliques → {len(ellipsoids)} 初始椭球")

        stage_print(f"阶段 4/5: IRIS-ZO region growth (最多 {cfg.IRIS_ZO.MAX_REGIONS} regions)")
        regions = []
        for region_id, ellipsoid in enumerate(ellipsoids[: int(cfg.IRIS_ZO.MAX_REGIONS)]):
            t_region = time.perf_counter()
            region = run_iris_zo(ellipsoid, oracle, cfg.IRIS_ZO, region_id=int(region_id))
            regions.append(region)
            coverage_est = estimate_region_coverage(free_samples, regions)
            stage_print(
                f"  region {region_id} 完成: log_det={region.log_det:+.4f}  "
                f"覆盖率={coverage_est.ratio:.4f}  耗时={time.perf_counter()-t_region:.1f}s"
            )
            if coverage_est.ratio >= float(cfg.COVERAGE_TARGET):
                stage_print(f"  覆盖率 {coverage_est.ratio:.4f} ≥ 目标 {cfg.COVERAGE_TARGET}, 提前停止")
                break

        coverage = estimate_region_coverage(free_samples, regions)
        stage_print(f"阶段 5/5: 曲线采样 & 碰撞评估")
        rng = np.random.default_rng(int(cfg.IRIS_ZO.RNG_SEED))
        curve_report = {}
        if regions:
            best_region = max(regions, key=lambda region: float(region.log_det))
            for _ in range(int(cfg.CURVE_MAX_ATTEMPTS)):
                curve = sample_curve_in_region(best_region, num_points=int(cfg.CURVE_NUM_POINTS), rng=rng)
                curve_report = evaluate_curve(oracle, curve)
                if not bool(curve_report["any_collision"]):
                    break
        report = ExperimentReport(
            regions=tuple(regions),
            coverage=coverage,
            visibility_stats={
                "num_vertices": int(len(graph.vertices)),
                "num_candidate_pairs": int(graph.num_candidate_pairs),
                "num_visible_edges": int(graph.num_visible_edges),
            },
            clique_stats={
                "num_cliques": int(len(cliques)),
                "clique_sizes": [int(len(clique.vertex_indices)) for clique in cliques],
            },
            sample_stats={
                "num_free_samples": int(len(free_samples)),
                "cache_path": str(cfg.SAMPLING.SAMPLE_CACHE_PATH),
                "iris_mode": "IRIS-ZO",
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
    stage_print("═" * 50)
    stage_print(
        f"pipeline 完成: {len(report.regions)} regions  "
        f"覆盖率={report.coverage.ratio:.4f}  "
        f"曲线碰撞={report.curve_report.get('any_collision')}  "
        f"总耗时={dt:.1f}s"
    )
    stage_print("═" * 50)

    if cfg.PLAYBACK_GUI and report.curve_report:
        playback_curve_gui(cfg.ROBOT, report.curve_report, hold_seconds=float(cfg.GUI_HOLD_SECONDS))
    return report

