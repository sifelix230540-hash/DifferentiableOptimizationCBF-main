"""对偶 VCC + IRIS-ZO 流水线入口（自碰撞场景）。

正侧 = C-free（用 CoalSelfCollisionOracle）
负侧 = C-obs（用 NegationOracle 包装同一个 CoalSelfCollisionOracle）

正负对称交替增长，对方 region 作为 mutual pseudo-obstacle。
五指标统计：cov_pos / cov_neg / cov_combined / cov_pos_boosted / cov_neg_boosted。

运行：
    本地 smoke test（少样本 / 单进程）：
        python run_vcc_dual_decomposition.py --smoke

    服务器全量：
        python run_vcc_dual_decomposition.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def build_smoke_config():
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import (
        CliqueCoverConfig, IrisZoConfig, SamplingConfig, VisibilityConfig,
    )
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.dual_pipeline import DualConfig

    return DualConfig(
        SAMPLING=SamplingConfig(
            NUM_SAMPLES_PER_ROUND=80,
            BATCH_SIZE=128,
            NUM_COVERAGE_SAMPLES=500,
        ),
        VISIBILITY=VisibilityConfig(
            PARALLEL_WORKERS=1,                    # smoke 单进程，避免并行启动开销
            SEGMENT_INTERPOLATION_STEPS=10,
        ),
        CLIQUE=CliqueCoverConfig(
            MIN_CLIQUE_SIZE=4,
            MAX_CLIQUES_PER_ROUND=4,
            STRATEGY="igraph_exact",
        ),
        IRIS_ZO=IrisZoConfig(
            NUM_PARTICLES=40,
            MAX_OUTER_ITERATIONS=2,
            MAX_INNER_ITERATIONS=4,
            HIT_AND_RUN_MIXING_STEPS=15,
        ),
        MUTUAL_MARGIN=1e-3,
        MAX_MACRO_ROUNDS=2,
        MAX_REGIONS_PER_SIDE=4,
        MAX_REGIONS_PER_SUBROUND=2,
        COMBINED_COVERAGE_TARGET=0.95,
        NUM_UNIFORM_COVERAGE_SAMPLES=500,
    )


def build_server_config():
    """服务器推荐参数（参考自碰撞 70% pos-only 配置等比放大）。

    估算运行时长（20 workers, 16 macro round, USE_SHARED_SAMPLING=True）：
      * label-uniform 8000 点：≈ 5 min
      * 共享采样：每 macro round 一次画 ~3000 base oracle 调用，
        填满正负各 1500 池 ≈ 1 min
      * 每 sub-round visibility (~1.1M pair, 20 workers) + IRIS-ZO ≈ 10 min
      * 32 sub-round 共 ≈ 5.5 h
      * 总计 ≈ 5~7 h（共享采样比独立采样省 ~30~45 min）

    提前停止：cov_combined >= 0.85 或连续 2 macro round 无增长。
    """
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import (
        CliqueCoverConfig, IrisZoConfig, SamplingConfig, VisibilityConfig,
    )
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.dual_pipeline import DualConfig

    return DualConfig(
        SAMPLING=SamplingConfig(
            NUM_SAMPLES_PER_ROUND=1500,           # 每 sub-round 采样
            BATCH_SIZE=1024,
            NUM_COVERAGE_SAMPLES=10000,           # （此项被对偶 pipeline 内部覆盖率忽略，留给 base pipeline）
        ),
        VISIBILITY=VisibilityConfig(
            PARALLEL_WORKERS=20,                  # 服务器 CPU 数
            SEGMENT_INTERPOLATION_STEPS=24,
        ),
        CLIQUE=CliqueCoverConfig(
            MIN_CLIQUE_SIZE=10,
            MAX_CLIQUES_PER_ROUND=24,
            STRATEGY="igraph_exact",
        ),
        IRIS_ZO=IrisZoConfig(
            NUM_PARTICLES=300,
            MAX_OUTER_ITERATIONS=8,
            MAX_INNER_ITERATIONS=12,
            HIT_AND_RUN_MIXING_STEPS=20,
        ),
        MUTUAL_MARGIN=1e-3,
        MAX_MACRO_ROUNDS=16,                      # 上限，结合早停
        MAX_REGIONS_PER_SIDE=80,                  # 每侧 region 总数上限
        MAX_REGIONS_PER_SUBROUND=6,               # 每 sub-round 配额
        COMBINED_COVERAGE_TARGET=0.85,            # 主停止：cov_combined >= 0.85
        NUM_UNIFORM_COVERAGE_SAMPLES=8000,        # 5 指标估计精度（~1.1% 95% Wilson）
        UNIFORM_COVERAGE_RNG_SEED=99,
        EARLY_STOP_NO_GROWTH_ROUNDS=2,
    )


def main():
    parser = argparse.ArgumentParser(description="Dual VCC + IRIS-ZO (self-collision)")
    parser.add_argument("--smoke", action="store_true", help="本地 smoke test：少量样本快速跑通")
    parser.add_argument("--cover-json", type=str, default="artifacts/sdf_exp/vcc_dual_cover.json")
    parser.add_argument("--experiment-json", type=str, default="artifacts/sdf_exp/vcc_dual_experiment.json")
    args = parser.parse_args()

    from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.dual_pipeline import (
        run_dual_decomposition_pipeline,
    )
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.io.reporting import (
        write_dual_experiment_report,
    )
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.coal_oracle import CoalSelfCollisionOracle
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.dual_oracle import NegationOracle

    cfg = build_smoke_config() if args.smoke else build_server_config()

    print("=" * 60)
    print("Dual VCC + IRIS-ZO 自碰撞对偶分解 — 参数摘要")
    print("=" * 60)
    print(f"  模式:                    {'SMOKE (本地)' if args.smoke else 'SERVER (全量)'}")
    print(f"  共享采样:                 {'ON  (每 macro round 一次画 base oracle)' if cfg.USE_SHARED_SAMPLING else 'OFF (正负各自独立采样)'}")
    print(f"  每 sub-round 采样池目标:    {cfg.SAMPLING.NUM_SAMPLES_PER_ROUND}")
    print(f"  可见性 workers:           {cfg.VISIBILITY.PARALLEL_WORKERS}")
    print(f"  IRIS-ZO particles:       {cfg.IRIS_ZO.NUM_PARTICLES}")
    print(f"  IRIS-ZO outer/inner:     {cfg.IRIS_ZO.MAX_OUTER_ITERATIONS}/{cfg.IRIS_ZO.MAX_INNER_ITERATIONS}")
    print(f"  Mutual margin:           {cfg.MUTUAL_MARGIN}")
    print(f"  Max macro rounds:        {cfg.MAX_MACRO_ROUNDS}")
    print(f"  Max regions / side:      {cfg.MAX_REGIONS_PER_SIDE}")
    print(f"  配额 / sub-round:         {cfg.MAX_REGIONS_PER_SUBROUND}")
    print(f"  cov_combined 目标:        {cfg.COMBINED_COVERAGE_TARGET}")
    print(f"  均匀采样 (5 指标):        {cfg.NUM_UNIFORM_COVERAGE_SAMPLES}")
    print(f"  早停无增长轮:             {cfg.EARLY_STOP_NO_GROWTH_ROUNDS}")
    print(f"  cover JSON  → {args.cover_json}")
    print(f"  expt  JSON  → {args.experiment_json}")
    print()

    robot_cfg = RobotQueryConfig()
    pos_oracle = CoalSelfCollisionOracle(robot_cfg)
    neg_oracle = NegationOracle(pos_oracle)

    try:
        t0 = time.perf_counter()
        report = run_dual_decomposition_pipeline(
            pos_base_oracle=pos_oracle,
            neg_base_oracle=neg_oracle,
            cfg=cfg,
        )
        dt = time.perf_counter() - t0
    finally:
        pos_oracle.close()

    write_dual_experiment_report(
        report,
        cover_json_path=args.cover_json,
        experiment_json_path=args.experiment_json,
    )

    print()
    print("=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"  总耗时:                  {dt:.1f}s ({dt/3600:.2f}h)")
    print(f"  Macro rounds 实际跑:      {len(set(rs.macro_round_id for rs in report.round_stats))}")
    print(f"  Sub-rounds 总数:          {len(report.round_stats)}")
    print(f"  Pos regions:             {len(report.pos_regions)}")
    print(f"  Neg regions:             {len(report.neg_regions)}")
    fc = report.final_coverage
    print(f"  cov_pos:                 {fc.cov_pos:.4f}                <头牌·正侧>")
    print(f"  cov_neg:                 {fc.cov_neg:.4f}                <头牌·负侧>")
    print(f"  balance:                 {fc.balance:.4f}                <头牌·均衡>")
    print(f"  cov_combined:            {fc.cov_combined:.4f} ± {fc.cov_combined_confidence_radius:.4f}   <混合体积比>")
    print(f"  uncov(Cfree):            {fc.cov_uncov_in_Cfree:.4f}                <Cfree 内剩余>")
    print(f"  uncov(Cobs) :            {fc.cov_uncov_in_Cobs:.4f}                <Cobs  内剩余>")
    print(f"  check (cov+un_f+un_o):   {fc.cov_combined + fc.cov_uncov_in_Cfree + fc.cov_uncov_in_Cobs:.4f}  <应≈1>")
    print(f"  cov_pos_boosted:         {fc.cov_pos_boosted:.4f}                <旧·零重叠下≈1>")
    print(f"  cov_neg_boosted:         {fc.cov_neg_boosted:.4f}                <旧·零重叠下≈1>")
    print(f"  |P∩N| / |Ω| (重叠率):    {fc.num_in_overlap/max(fc.num_uniform_samples,1):.6f}")
    print()
    print("  每 sub-round 摘要 (按时间序):")
    for rs in report.round_stats:
        cov_str = ""
        if rs.coverage_after is not None:
            cov_str = f"  cov_combined→{rs.coverage_after.cov_combined:.4f}"
        print(
            f"    M{rs.macro_round_id} {rs.side.upper()}#{rs.sub_round_id}: "
            f"samples={rs.num_samples}  visible={rs.num_visible_edges}/{rs.num_pairs}  "
            f"cliques={rs.num_cliques}  +regions={rs.num_regions_grown}  "
            f"{rs.elapsed_seconds:.1f}s{cov_str}"
        )


if __name__ == "__main__":
    main()
