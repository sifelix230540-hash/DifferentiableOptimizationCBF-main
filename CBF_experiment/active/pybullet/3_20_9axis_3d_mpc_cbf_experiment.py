"""精简后的 3_20 主入口。

当前文件只保留 9 轴焊接实验的主链路导出，具体实现已拆分到同目录的
`welding_320_*` 模块中，便于继续维护和定位问题。
"""

from pathlib import Path
import sys

from scipy.spatial.transform import Rotation


# 仓库根目录：本文件位于 .../active/pybullet/，parents[3] 为含 CBF_experiment 的仓库根
ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBF_experiment.active.pybullet.welding_320_common import (  # noqa: E402
    ExperimentConfig,
    SimulationScene,
    build_weld_reference_quat,
    quaternion_error_rotvec,
)
from CBF_experiment.active.pybullet.welding_320_control import (  # noqa: E402
    CartesianRRTNominalPlanner,
    MPCDCBFController,
    create_controller,
)
from CBF_experiment.active.pybullet.welding_320_experiment import (  # noqa: E402
    AvoidanceExperiment,
    main,
)
from CBF_experiment.active.pybullet.welding_320_robot import (  # noqa: E402
    JakaRobot,
    URDFObstacle,
    WorkpieceModel,
)
from CBF_experiment.active.pybullet.welding_320_trajectory import (  # noqa: E402
    JointWaypointTrajectory,
    LineSlerpTrajectory,
    PathProgressTrajectory,
    PiecewiseLineSlerpTrajectory,
)


if __name__ == "__main__":
    main()
