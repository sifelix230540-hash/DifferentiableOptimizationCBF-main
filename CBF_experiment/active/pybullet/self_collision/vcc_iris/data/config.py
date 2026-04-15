"""实验参数集中配置：采样、可见性、团覆盖、IRIS-ZO、报告路径。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import _resolve  # noqa: E402


@dataclass(frozen=True)
class RobotQueryConfig:
    CFG_PATH: str | None = None
    MIN_INDEX_GAP: int = 2
    PENETRATION_THRESH: float = -0.001
    INCLUDE_WELDING_GUN_BASE: bool = True
    INCLUDE_THIRD_AXIS_CHAIN: bool = True


@dataclass(frozen=True)
class SamplingConfig:
    NUM_SAMPLES_PER_ROUND: int = 1000
    BATCH_SIZE: int = 256
    RNG_SEED: int = 11
    SAMPLE_CACHE_PATH: str = str(Path(_resolve("artifacts/sdf_exp/vcc_iris_free_samples.json")))
    NUM_COVERAGE_SAMPLES: int = 5000


@dataclass(frozen=True)
class VisibilityConfig:
    SEGMENT_INTERPOLATION_STEPS: int = 18
    RANDOM_SEED: int = 17
    PARALLEL_WORKERS: int = 1


@dataclass(frozen=True)
class CliqueCoverConfig:
    MIN_CLIQUE_SIZE: int = 10
    MAX_CLIQUES_PER_ROUND: int = 24
    STRATEGY: str = "igraph_exact"


@dataclass(frozen=True)
class IrisZoConfig:
    EPSILON: float = 0.10
    DELTA: float = 0.10
    TAU: float = 0.50
    STARTING_BALL_RADIUS: float = 0.05
    MAX_OUTER_ITERATIONS: int = 5
    MAX_INNER_ITERATIONS: int = 8
    NUM_PARTICLES: int = 64
    NUM_BISECTION_STEPS: int = 10
    MAX_NEW_FACES_PER_INNER_ITER: int = 12
    HIT_AND_RUN_MIXING_STEPS: int = 12
    CONVERGENCE_TOL: float = 1e-3
    STEPBACK_MARGIN: float = 0.01
    RNG_SEED: int = 23


@dataclass(frozen=True)
class ReportingConfig:
    OUTPUT_DIR: str = str(Path(_resolve("artifacts/sdf_exp")))
    COVER_JSON: str = str(Path(_resolve("artifacts/sdf_exp/vcc_iris_cover.json")))
    EXPERIMENT_JSON: str = str(Path(_resolve("artifacts/sdf_exp/vcc_iris_experiment.json")))


@dataclass(frozen=True)
class ExperimentConfig:
    ROBOT: RobotQueryConfig = field(default_factory=RobotQueryConfig)
    SAMPLING: SamplingConfig = field(default_factory=SamplingConfig)
    VISIBILITY: VisibilityConfig = field(default_factory=VisibilityConfig)
    CLIQUE: CliqueCoverConfig = field(default_factory=CliqueCoverConfig)
    IRIS_ZO: IrisZoConfig = field(default_factory=IrisZoConfig)
    REPORTING: ReportingConfig = field(default_factory=ReportingConfig)

    MAX_VCC_ROUNDS: int = 20
    MAX_TOTAL_REGIONS: int = 64
    COVERAGE_TARGET: float = 0.7

    CURVE_NUM_POINTS: int = 60
    CURVE_MAX_ATTEMPTS: int = 16
    PLAYBACK_GUI: bool = False
    GUI_HOLD_SECONDS: float = 3.0
