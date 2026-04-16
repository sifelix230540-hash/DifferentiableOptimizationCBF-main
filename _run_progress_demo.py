"""Quick demo to verify progress bars work with real coal oracle."""
from CBF_experiment.active.pybullet.self_collision.vcc_iris import (
    ExperimentConfig, RobotQueryConfig, SamplingConfig,
    VisibilityConfig, CliqueCoverConfig, IrisZoConfig,
    ReportingConfig, run_vcc_iris_pipeline,
)

cfg = ExperimentConfig(
    ROBOT=RobotQueryConfig(),
    SAMPLING=SamplingConfig(
        NUM_SEED_SAMPLES=24, BATCH_SIZE=24,
        SAMPLE_CACHE_PATH="artifacts/sdf_exp/vcc_iriszo_pg_samples.json",
        NUM_COVERAGE_SAMPLES=120,
    ),
    VISIBILITY=VisibilityConfig(
        MAX_CANDIDATE_PAIRS=200, SEGMENT_INTERPOLATION_STEPS=8,
        BISECTION_STEPS=6, RANDOM_SEED=31, PARALLEL_WORKERS=1,
    ),
    CLIQUE=CliqueCoverConfig(MIN_CLIQUE_SIZE=4, MAX_CLIQUES=4),
    IRIS_ZO=IrisZoConfig(
        MAX_OUTER_ITERATIONS=2, MAX_INNER_ITERATIONS=3,
        NUM_PARTICLES=24, NUM_BISECTION_STEPS=6,
        MAX_NEW_FACES_PER_INNER_ITER=4, HIT_AND_RUN_MIXING_STEPS=6,
        MAX_REGIONS=2, RNG_SEED=29,
    ),
    REPORTING=ReportingConfig(
        OUTPUT_DIR="artifacts/sdf_exp",
        COVER_JSON="artifacts/sdf_exp/vcc_iriszo_pg_cover.json",
        EXPERIMENT_JSON="artifacts/sdf_exp/vcc_iriszo_pg_exp.json",
    ),
    CURVE_NUM_POINTS=20, CURVE_MAX_ATTEMPTS=3,
    COVERAGE_TARGET=0.2, PLAYBACK_GUI=False,
)

run_vcc_iris_pipeline(cfg)
