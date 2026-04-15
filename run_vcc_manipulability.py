"""VCC + IRIS-ZO 用于高可操作度 C-space 凸区域分解。

将"可操作度低于阈值"视为"碰撞"，复用全部 VCC+IRIS-ZO 流水线，
输出覆盖高可操作度区域的凸多面体集合。

标定数据（JAKA 6-DOF, 5000 uniform samples）:
  3×6 平移 Jacobian:
    w >= 0.05: 74.1%  visibility 16.0%  ← 推荐
    w >= 0.10: 45.5%  visibility  3.9%
  6×6 全 Jacobian:
    w >= 0.01: 56.0%  visibility  2.5%  ← 碎片化严重
    w >= 0.02: 39.7%  visibility  1.8%
"""
import time


def main():
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import (
        ExperimentConfig,
        SamplingConfig,
        VisibilityConfig,
        CliqueCoverConfig,
        IrisZoConfig,
        ReportingConfig,
    )
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.pipeline import run_vcc_iris_pipeline
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.manipulability_oracle import ManipulabilityOracle

    # ── 可操作度阈值 ──────────────────────────────────
    # 3×6 平移 Jacobian, w>=0.05: 74% feasible, 16% visibility
    MANIP_THRESHOLD = 0.05
    COND_THRESHOLD = None
    USE_POSITION_ONLY = True

    # ── VCC + IRIS-ZO 参数（中等规模 + GUI） ─────────
    cfg = ExperimentConfig(
        SAMPLING=SamplingConfig(
            NUM_SAMPLES_PER_ROUND=600,
            BATCH_SIZE=512,
            NUM_COVERAGE_SAMPLES=5000,
        ),
        VISIBILITY=VisibilityConfig(
            PARALLEL_WORKERS=8,
            SEGMENT_INTERPOLATION_STEPS=12,
        ),
        CLIQUE=CliqueCoverConfig(
            MIN_CLIQUE_SIZE=5,
            MAX_CLIQUES_PER_ROUND=16,
            STRATEGY="igraph_exact",
        ),
        IRIS_ZO=IrisZoConfig(
            NUM_PARTICLES=100,
            MAX_OUTER_ITERATIONS=5,
            MAX_INNER_ITERATIONS=10,
            HIT_AND_RUN_MIXING_STEPS=20,
        ),
        REPORTING=ReportingConfig(
            COVER_JSON="artifacts/sdf_exp/vcc_manipulability_cover.json",
            EXPERIMENT_JSON="artifacts/sdf_exp/vcc_manipulability_experiment.json",
        ),
        MAX_VCC_ROUNDS=8,
        MAX_TOTAL_REGIONS=80,
        COVERAGE_TARGET=0.70,
        PLAYBACK_GUI=True,
        GUI_HOLD_SECONDS=15.0,
    )

    print("=" * 60)
    print("VCC + IRIS-ZO 高可操作度区域凸分解")
    print("=" * 60)
    print(f"  Yoshikawa 可操作度阈值:  {MANIP_THRESHOLD}")
    print(f"  条件数阈值:              {COND_THRESHOLD}")
    print(f"  仅平移 Jacobian (3×n):   {USE_POSITION_ONLY}")
    print(f"  每轮采样:   K = {cfg.SAMPLING.NUM_SAMPLES_PER_ROUND}")
    print(f"  覆盖率评估: M = {cfg.SAMPLING.NUM_COVERAGE_SAMPLES}")
    print(f"  可见性并行 workers:  {cfg.VISIBILITY.PARALLEL_WORKERS}")
    print(f"  smin:               {cfg.CLIQUE.MIN_CLIQUE_SIZE}")
    print(f"  每轮最大 cliques:   {cfg.CLIQUE.MAX_CLIQUES_PER_ROUND}")
    print(f"  IRIS-ZO particles:  {cfg.IRIS_ZO.NUM_PARTICLES}")
    print(f"  IRIS-ZO outer/inner: {cfg.IRIS_ZO.MAX_OUTER_ITERATIONS}/{cfg.IRIS_ZO.MAX_INNER_ITERATIONS}")
    print(f"  最大 VCC 轮次:      {cfg.MAX_VCC_ROUNDS}")
    print(f"  最大 regions:       {cfg.MAX_TOTAL_REGIONS}")
    print(f"  覆盖率目标:         {cfg.COVERAGE_TARGET}")
    print(f"  GUI 回放:           {cfg.PLAYBACK_GUI}")
    print()

    oracle = ManipulabilityOracle(
        cfg.ROBOT,
        manipulability_threshold=MANIP_THRESHOLD,
        condition_number_threshold=COND_THRESHOLD,
        use_position_only=USE_POSITION_ONLY,
    )

    try:
        t0 = time.perf_counter()
        report = run_vcc_iris_pipeline(cfg, oracle=oracle)
        dt = time.perf_counter() - t0
    finally:
        oracle.close()

    print()
    print("=" * 60)
    print("结果摘要")
    print("=" * 60)
    print(f"  总耗时:     {dt:.1f}s")
    print(f"  VCC 轮数:   {len(report.round_stats)}")
    print(f"  Regions:    {len(report.regions)}")
    print(f"  覆盖率:     {report.coverage.ratio:.4f} ± {report.coverage.confidence_radius:.4f}")
    print()
    for rs in report.round_stats:
        print(f"  Round {rs.round_id}: samples={rs.num_samples}  visible={rs.num_visible_edges}/{rs.num_pairs}  "
              f"cliques={rs.num_cliques}  regions={rs.num_regions_grown}  "
              f"coverage={rs.coverage_after:.4f}  time={rs.elapsed_seconds:.1f}s")
    print()
    for i, r in enumerate(report.regions):
        print(f"  Region {i}: log_det={r.log_det:+.4f}  planes={r.A.shape[0]}  iters={len(r.iterations)}")


if __name__ == "__main__":
    main()
