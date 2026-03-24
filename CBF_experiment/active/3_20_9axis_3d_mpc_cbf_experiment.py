"""精简后的 3_20 主入口。

当前文件只保留 9 轴焊接实验的主链路导出，具体实现已拆分到同目录的
`welding_320_*` 模块中，便于继续维护和定位问题。
"""

from pathlib import Path
import sys

from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBF_experiment.active.welding_320_common import (  # noqa: E402
    ExperimentConfig,
    SimulationScene,
    build_weld_reference_quat,
    quaternion_error_rotvec,
)
from CBF_experiment.active.welding_320_control import (  # noqa: E402
    CartesianRRTNominalPlanner,
    JointSpaceNominalPlanner,
    MPCDCBFController,
    PlannerDiagnostics,
    create_controller,
)
from CBF_experiment.active.welding_320_corridor import (  # noqa: E402
    JointSpaceCorridor,
    PyBulletModelSnapshot,
)
from CBF_experiment.active.welding_320_experiment import (  # noqa: E402
    AvoidanceExperiment,
    build_pre_approach_pose,
    main,
)
from CBF_experiment.active.welding_320_ik import (  # noqa: E402
    IKCandidate,
    MultiSeedIKSolver,
)
from CBF_experiment.active.welding_320_robot import (  # noqa: E402
    JakaRobot,
    URDFObstacle,
    WorkpieceModel,
)
from CBF_experiment.active.welding_320_trajectory import (  # noqa: E402
    JointWaypointTrajectory,
    LineSlerpTrajectory,
    PathProgressTrajectory,
    PiecewiseLineSlerpTrajectory,
)


if __name__ == "__main__":
    main()
