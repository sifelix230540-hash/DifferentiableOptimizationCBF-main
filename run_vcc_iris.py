"""Full VCC + IRIS-ZO benchmark with GUI visualization."""
import time


def main():
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.config import (
        ExperimentConfig,
        SamplingConfig,
        VisibilityConfig,
        CliqueCoverConfig,
        IrisZoConfig,
    )
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.pipeline import run_vcc_iris_pipeline
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.statistical_test import (
        required_trials,
        union_bound_delta,
    )

    cfg = ExperimentConfig(
        SAMPLING=SamplingConfig(
            NUM_SEED_SAMPLES=300,
            SAMPLE_CACHE_PATH="__benchmark_scratch_samples.json",
        ),
        VISIBILITY=VisibilityConfig(
            MAX_CANDIDATE_PAIRS=None,
            PARALLEL_WORKERS=1,
        ),
        CLIQUE=CliqueCoverConfig(MIN_CLIQUE_SIZE=4),
        IRIS_ZO=IrisZoConfig(MAX_REGIONS=8),
        PLAYBACK_GUI=True,
        GUI_HOLD_SECONDS=8.0,
    )
    print("=== Config summary ===")
    print(f"NUM_SEED_SAMPLES: {cfg.SAMPLING.NUM_SEED_SAMPLES}")
    print(f"IRIS_ZO.EPSILON: {cfg.IRIS_ZO.EPSILON}")
    print(f"IRIS_ZO.DELTA: {cfg.IRIS_ZO.DELTA}")
    print(f"IRIS_ZO.TAU: {cfg.IRIS_ZO.TAU}")
    print(f"IRIS_ZO.NUM_PARTICLES: {cfg.IRIS_ZO.NUM_PARTICLES}")
    print(f"IRIS_ZO.MAX_OUTER_ITERATIONS: {cfg.IRIS_ZO.MAX_OUTER_ITERATIONS}")
    print(f"IRIS_ZO.MAX_INNER_ITERATIONS: {cfg.IRIS_ZO.MAX_INNER_ITERATIONS}")
    print(f"IRIS_ZO.NUM_BISECTION_STEPS: {cfg.IRIS_ZO.NUM_BISECTION_STEPS}")
    print(f"IRIS_ZO.MAX_REGIONS: {cfg.IRIS_ZO.MAX_REGIONS}")
    print(f"VISIBILITY.MAX_CANDIDATE_PAIRS: {cfg.VISIBILITY.MAX_CANDIDATE_PAIRS}")
    print(f"CLIQUE.MIN_CLIQUE_SIZE: {cfg.CLIQUE.MIN_CLIQUE_SIZE}")
    print(f"PLAYBACK_GUI: {cfg.PLAYBACK_GUI}")
    print()

    d_ik = union_bound_delta(total_delta=cfg.IRIS_ZO.DELTA, outer_iter=1, inner_iter=1)
    M = required_trials(epsilon=cfg.IRIS_ZO.EPSILON, delta=d_ik, tau=cfg.IRIS_ZO.TAU)
    print(f"delta_ik (outer=1,inner=1): {d_ik:.6f}")
    print(f"required_trials (Chernoff): {M}")
    print(f"sample_budget = max({cfg.IRIS_ZO.NUM_PARTICLES}, {M}) = {max(cfg.IRIS_ZO.NUM_PARTICLES, M)}")
    print()

    t0 = time.perf_counter()
    report = run_vcc_iris_pipeline(cfg)
    dt = time.perf_counter() - t0

    print()
    print("=== RESULT ===")
    print(f"Total time: {dt:.1f}s")
    print(f"Regions: {len(report.regions)}")
    print(f"Coverage: {report.coverage.ratio:.4f}")
    any_col = report.curve_report.get("any_collision") if report.curve_report else None
    print(f"Curve collision: {any_col}")
    for i, r in enumerate(report.regions):
        print(f"  Region {i}: log_det={r.log_det:+.4f}, planes={r.A.shape[0]}, iters={len(r.iterations)}")


if __name__ == "__main__":
    main()
