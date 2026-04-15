"""VCC + IRIS-ZO 多轮迭代流水线运行入口。"""
import time


def main():
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import (
        ExperimentConfig,
        SamplingConfig,
        VisibilityConfig,
        CliqueCoverConfig,
        IrisZoConfig,
    )
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.pipeline import run_vcc_iris_pipeline

    # ── 在这里修改参数 ──────────────────────────────────────
    # 本地小规模测试默认值；服务器参数见注释
    cfg = ExperimentConfig(
        SAMPLING=SamplingConfig(
            NUM_SAMPLES_PER_ROUND=200,          # 服务器: 1500
            BATCH_SIZE=256,                     # 服务器: 1024
            NUM_COVERAGE_SAMPLES=3000,          # 服务器: 10_000
        ),
        VISIBILITY=VisibilityConfig(
            PARALLEL_WORKERS=1,                 # 服务器: 20
            SEGMENT_INTERPOLATION_STEPS=18,     # 服务器: 24
        ),
        CLIQUE=CliqueCoverConfig(
            MIN_CLIQUE_SIZE=6,                  # 服务器: 10
            MAX_CLIQUES_PER_ROUND=12,           # 服务器: 24
            STRATEGY="igraph_exact",            # 或 "greedy"
        ),
        IRIS_ZO=IrisZoConfig(
            NUM_PARTICLES=64,                   # 服务器: 300
            MAX_OUTER_ITERATIONS=5,             # 服务器: 8
            MAX_INNER_ITERATIONS=8,             # 服务器: 12
            HIT_AND_RUN_MIXING_STEPS=12,        # 服务器: 20
        ),
        MAX_VCC_ROUNDS=10,                      # 服务器: 20
        MAX_TOTAL_REGIONS=32,                   # 服务器: 64
        COVERAGE_TARGET=0.7,                    # 服务器: 0.85
        PLAYBACK_GUI=True,                      # 服务器: False
        GUI_HOLD_SECONDS=8.0,
    )

    print("=" * 56)
    print("VCC + IRIS-ZO iterative pipeline 参数摘要")
    print("=" * 56)
    print(f"  每轮采样: K = {cfg.SAMPLING.NUM_SAMPLES_PER_ROUND}")
    print(f"  覆盖率检测样本: M = {cfg.SAMPLING.NUM_COVERAGE_SAMPLES}")
    print(f"  可见性插值步数: {cfg.VISIBILITY.SEGMENT_INTERPOLATION_STEPS}")
    print(f"  并行 workers:  {cfg.VISIBILITY.PARALLEL_WORKERS}")
    print(f"  clique 策略:   {cfg.CLIQUE.STRATEGY}")
    print(f"  smin:          {cfg.CLIQUE.MIN_CLIQUE_SIZE}")
    print(f"  IRIS-ZO particles: {cfg.IRIS_ZO.NUM_PARTICLES}")
    print(f"  最大 VCC 轮次: {cfg.MAX_VCC_ROUNDS}")
    print(f"  最大 regions:  {cfg.MAX_TOTAL_REGIONS}")
    print(f"  覆盖率目标:    {cfg.COVERAGE_TARGET}")
    print(f"  GUI 回放:      {cfg.PLAYBACK_GUI}")
    print()

    t0 = time.perf_counter()
    report = run_vcc_iris_pipeline(cfg)
    dt = time.perf_counter() - t0

    print()
    print("=" * 56)
    print("结果摘要")
    print("=" * 56)
    print(f"  总耗时:     {dt:.1f}s")
    print(f"  VCC 轮数:   {len(report.round_stats)}")
    print(f"  Regions:    {len(report.regions)}")
    print(f"  覆盖率:     {report.coverage.ratio:.4f} ± {report.coverage.confidence_radius:.4f}")
    any_col = report.curve_report.get("any_collision") if report.curve_report else None
    print(f"  曲线碰撞:   {any_col}")
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
