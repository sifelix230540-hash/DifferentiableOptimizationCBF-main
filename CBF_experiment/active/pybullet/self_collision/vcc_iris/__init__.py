"""VCC + IRIS-ZO: 基于可见性团覆盖与零阶 IRIS 的配置空间安全区域生成。

模块结构：
  数据层:   data.types, data.config
  工具层:   utils.progress, utils.polytope_sampling, utils.statistical_test
  机器人层: robot.robot_model, robot.coal_oracle
  流水线:   stages.sampling → stages.visibility → stages.clique_cover
            → stages.ellipsoids → stages.iris_zo → stages.coverage
  I/O 层:   io.gui, io.reporting
  编排层:   pipeline  (多轮迭代串联上述所有阶段)
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
