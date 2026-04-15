"""VCC + IRIS-ZO 用于低操作度 C-space 凸区域分解。

思路：
  将“可操作度高于阈值”视为“障碍”，把低操作度区域当作可行域，
  从而对奇异附近/低灵巧度区域做凸分解。

标定摘要（JAKA 6-DOF, 3000 uniform samples, 6x6 full Jacobian）:
  w <= 0.003: frac=22.8%, visibility=9.1%
  w <= 0.005: frac=30.2%, visibility=9.3%
  w <= 0.008: frac=38.6%, visibility=12.3%   <- 推荐
  w <= 0.010: frac=42.7%, visibility=12.2%
"""
import time


def main():
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import (
        CliqueCoverConfig,
        ExperimentConfig,
        IrisZoConfig,
        ReportingConfig,
        SamplingConfig,
        VisibilityConfig,
    )
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.pipeline import run_vcc_iris_pipeline
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.manipulability_oracle import ManipulabilityOracle

    MANIP_THRESHOLD = 0.008
    COND_THRESHOLD = None
    USE_POSITION_ONLY = False
    ACCEPT_BELOW_THRESHOLD = True

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
            COVER_JSON="artifacts/sdf_exp/vcc_low_manipulability_cover.json",
            EXPERIMENT_JSON="artifacts/sdf_exp/vcc_low_manipulability_experiment.json",
        ),
        MAX_VCC_ROUNDS=6,
        MAX_TOTAL_REGIONS=64,
        COVERAGE_TARGET=0.50,
        PLAYBACK_GUI=True,
        GUI_HOLD_SECONDS=15.0,
    )

    print("=" * 60)
    print("VCC + IRIS-ZO 低操作度区域凸分解")
    print("=" * 60)
    print(f"  Jacobian 模式:            {'3xN position' if USE_POSITION_ONLY else '6xN full'}")
    print(f"  低操作度阈值:            w <= {MANIP_THRESHOLD}")
    print(f"  accept_below_threshold:  {ACCEPT_BELOW_THRESHOLD}")
    print(f"  可见性并行 workers:      {cfg.VISIBILITY.PARALLEL_WORKERS}")
    print(f"  每轮采样:                {cfg.SAMPLING.NUM_SAMPLES_PER_ROUND}")
    print(f"  覆盖率样本:              {cfg.SAMPLING.NUM_COVERAGE_SAMPLES}")
    print(f"  覆盖率目标:              {cfg.COVERAGE_TARGET}")
    print(f"  最大轮数 / region:       {cfg.MAX_VCC_ROUNDS} / {cfg.MAX_TOTAL_REGIONS}")
    print(f"  GUI 回放:                {cfg.PLAYBACK_GUI}")
    print()

    oracle = ManipulabilityOracle(
        cfg.ROBOT,
        manipulability_threshold=MANIP_THRESHOLD,
        condition_number_threshold=COND_THRESHOLD,
        use_position_only=USE_POSITION_ONLY,
        accept_below_threshold=ACCEPT_BELOW_THRESHOLD,
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
        print(
            f"  Round {rs.round_id}: samples={rs.num_samples}  "
            f"visible={rs.num_visible_edges}/{rs.num_pairs}  "
            f"cliques={rs.num_cliques}  regions={rs.num_regions_grown}  "
            f"coverage={rs.coverage_after:.4f}  time={rs.elapsed_seconds:.1f}s"
        )


if __name__ == "__main__":
    main()
