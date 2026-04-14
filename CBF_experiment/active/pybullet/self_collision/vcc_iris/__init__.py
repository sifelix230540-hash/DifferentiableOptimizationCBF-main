"""VCC + IRIS-ZO: 基于可见性团覆盖与零阶 IRIS 的配置空间安全区域生成。

模块结构：
  数据层:   types, config
  工具层:   progress, polytope_sampling, statistical_test
  机器人层: robot_model, coal_oracle
  流水线:   sampling → visibility → clique_cover → ellipsoids → iris_zo → coverage
  I/O 层:   gui, reporting
  编排层:   pipeline  (串联上述所有阶段)
"""

from .robot.coal_oracle import CoalSelfCollisionOracle
from .data.config import (
    CliqueCoverConfig,
    ExperimentConfig,
    IrisZoConfig,
    ReportingConfig,
    RobotQueryConfig,
    SamplingConfig,
    VisibilityConfig,
)
from .stages.coverage import estimate_region_coverage
from .pipeline import run_vcc_iris_pipeline

