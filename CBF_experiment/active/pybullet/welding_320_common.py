import math
import os as _os
import shutil as _shutil
import tempfile as _tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

DEFAULT_GRAVITY = (0.0, 0.0, -9.81)


@dataclass
class ExperimentConfig:
    """9 轴焊接主实验当前保留的可调参数。"""

    # 场景
    urdf_path: str = (
        r"C:\Users\12049\OneDrive\Desktop\科研相关\博一春季\免示教焊接轨迹规划"
        r"\相关资料\CBF_grad_optim_on_trajPlanning"
        r"\DifferentiableOptimizationCBF-main\assets\robots\9_axis\urdf\9_axis.urdf"
    )
    workpiece_urdf_path: str = (
        r"C:\Users\12049\OneDrive\Desktop\科研相关\博一春季\免示教焊接轨迹规划"
        r"\相关资料\CBF_grad_optim_on_trajPlanning"
        r"\DifferentiableOptimizationCBF-main\assets\cad_exports\model_CAD\scene\urdf"
        r"\中组立0725(1).stp.SLDASM.urdf"
    )
    workpiece_position: tuple[float, float, float] = (-3, 5.0, 0.10)
    workpiece_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0)
    ignore_all_collisions: bool = False
    start_link_name: str = "l2"
    goal_link_name: str = "l3"
    weld_local_direction: tuple[float, float, float] = (0.0, 1.0, -1.0)
    dt: float = 1.0 / 240.0
    gantry_initial_q: tuple[float, float, float] = (12.0, -7.0, 0.0)
    camera_distance: float = 1.4
    camera_yaw: float = -215.0
    camera_pitch: float = -26.0
    camera_target: tuple[float, float, float] = (0.30, 0.20, 0.60)
    print_every: int = 120

    # 轨迹
    approach_duration: float = 6.0
    weld_duration: float = 7.0
    return_duration: float = 6.0
    hold_duration: float = 3.0
    dq_limit: float = 1.0
    base_vel_limit: float = 0.4
    position_gain: float = 8.0
    orientation_gain: float = 3.0
    use_mesh_cbf: bool = True
    safety_margin: float = 0.02
    q_nominal_tracking: float = 0.02
    use_dynamic_nominal_reference: bool = False
    dynamic_nominal_history_size: int = 15
    dynamic_nominal_progress_epsilon: float = 0.03
    dynamic_nominal_exec_motion_trigger: float = 1e-4
    dynamic_nominal_tracking_error_trigger: float = 0.05
    dynamic_nominal_escape_distance: float = 0.10
    dynamic_nominal_normal_gain: float = 0.35
    dynamic_nominal_max_weight: float = 0.55
    dynamic_nominal_release_progress: float = 0.06
    progress_end_tolerance: float = 0.02

    # MPC
    N_mpc: int = 5
    mpc_dt: float = 0.04
    gamma_dcbf: float = 0.15
    mpc_tracking_weight: float = 5.0
    mpc_orientation_tracking_weight: float = 0.2
    mpc_control_weight: float = 0.2
    mpc_smooth_weight: float = 0.2
    mpc_replan_steps: int = 6
    mpc_progress_step_min: float = 0.01

    # RRT
    rrt_max_iterations: int = 2400
    rrt_restarts: int = 2
    rrt_smooth: int = 16
    rrt_max_time: float = 3.0
    rrt_cartesian_resolution: float = 0.04
    rrt_cartesian_margin: float = 0.35
    rrt_ik_tolerance: float = 0.05


class SimulationScene:
    """负责 PyBullet 场景初始化与基础可视化。"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client_id = p.connect(p.GUI, options="--width=1920 --height=1080")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*DEFAULT_GRAVITY)
        p.setTimeStep(config.dt)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(rgbBackground=[1, 1, 1])
        p.resetDebugVisualizerCamera(
            cameraDistance=config.camera_distance,
            cameraYaw=config.camera_yaw,
            cameraPitch=config.camera_pitch,
            cameraTargetPosition=config.camera_target,
        )
        self.reference_height = self._build_environment()
        self._draw_axes()
        self.status_text_id = None

    def _build_environment(self) -> float:
        plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.95, 0.95, 0.95, 1.0])
        return 0.0

    def _draw_axes(self):
        axis_len = 0.12
        origin = [0, 0, self.reference_height + 0.001]
        p.addUserDebugLine(origin, [axis_len, 0, origin[2]], [1, 0, 0], lineWidth=2)
        p.addUserDebugLine(origin, [0, axis_len, origin[2]], [0, 0.8, 0], lineWidth=2)
        p.addUserDebugLine(origin, [0, 0, origin[2] + axis_len], [0, 0, 1], lineWidth=2)

    def create_marker(self, radius, color, pos):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos)

    def update_marker(self, body_id, pos):
        p.resetBasePositionAndOrientation(body_id, pos, [0, 0, 0, 1])

    def draw_polyline(self, pts, color, width=1.5):
        for i in range(len(pts) - 1):
            p.addUserDebugLine(pts[i].tolist(), pts[i + 1].tolist(), color, lineWidth=width)

    def draw_direction(self, origin, direction, color, length=0.10, width=2.0):
        d = _normalize(np.array(direction, dtype=float))
        p.addUserDebugLine(origin.tolist(), (origin + length * d).tolist(), color, lineWidth=width)

    def update_status(self, text):
        self.status_text_id = p.addUserDebugText(
            text,
            [0.02, -0.26, 0.52],
            [0.1] * 3,
            textSize=1.2,
            replaceItemUniqueId=self.status_text_id if self.status_text_id else -1,
        )

    def enable_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


def _prepare_package_urdf(
    urdf_path: str,
    package_name: str | None = None,
    package_alias: str | None = None,
    remove_collision: bool = False,
) -> tuple[str, str]:
    urdf_path = _os.path.abspath(urdf_path)
    source_root = _os.path.dirname(_os.path.dirname(urdf_path))
    source_pkg_name = _os.path.basename(source_root)
    package_name = package_name or source_pkg_name
    package_alias = package_alias or package_name

    copy_required = package_name != source_pkg_name or package_alias != source_pkg_name
    try:
        urdf_path.encode("ascii")
        source_root.encode("ascii")
        package_name.encode("ascii")
        package_alias.encode("ascii")
    except UnicodeEncodeError:
        copy_required = True

    if not copy_required and not remove_collision:
        return urdf_path, source_root

    tmp_root = _os.path.join(_tempfile.gettempdir(), "pybullet_urdf")
    tmp_pkg = _os.path.join(tmp_root, package_alias)
    if _os.path.exists(tmp_pkg):
        _shutil.rmtree(tmp_pkg, ignore_errors=True)
        for _ in range(30):
            if not _os.path.exists(tmp_pkg):
                break
            time.sleep(0.1)

    _shutil.copytree(source_root, tmp_pkg, dirs_exist_ok=True)

    source_urdf_dir = _os.path.dirname(urdf_path)
    rel_urdf_dir = _os.path.relpath(source_urdf_dir, source_root)
    new_urdf_dir = _os.path.join(tmp_pkg, rel_urdf_dir)
    _os.makedirs(new_urdf_dir, exist_ok=True)
    new_urdf = _os.path.join(new_urdf_dir, "model.urdf")

    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_text = f.read()

    urdf_text = urdf_text.replace(f"package://{package_name}/", f"package://{package_alias}/")
    urdf_text = urdf_text.replace(f"package://{source_pkg_name}/", f"package://{package_alias}/")

    if remove_collision:
        root = ET.fromstring(urdf_text)
        for link_elem in root.findall("link"):
            for collision_elem in list(link_elem.findall("collision")):
                link_elem.remove(collision_elem)
        urdf_text = ET.tostring(root, encoding="unicode")

    with open(new_urdf, "w", encoding="utf-8", newline="\n") as f:
        f.write(urdf_text)

    print(f"[info] URDF 已复制到临时目录: {tmp_pkg}")
    return new_urdf, tmp_root


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def _project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return v - float(np.dot(v, normal)) * normal


def quaternion_error_rotvec(current_quat, target_quat) -> np.ndarray:
    return (Rotation.from_quat(target_quat) * Rotation.from_quat(current_quat).inv()).as_rotvec()


def build_weld_reference_quat(frame_quat, weld_local_direction, prev_quat=None):
    frame_rot = Rotation.from_quat(frame_quat)
    z_world = _normalize(frame_rot.apply(np.array(weld_local_direction, dtype=float)))

    candidates = []
    if prev_quat is not None:
        candidates.append(Rotation.from_quat(prev_quat).apply([1.0, 0.0, 0.0]))
    candidates.append(frame_rot.apply([1.0, 0.0, 0.0]))
    candidates.append(np.array([1.0, 0.0, 0.0]))
    candidates.append(np.array([0.0, 1.0, 0.0]))

    x_world = None
    for cand in candidates:
        proj = _project_to_plane(np.array(cand, dtype=float), z_world)
        if np.linalg.norm(proj) > 1e-6:
            x_world = _normalize(proj)
            break
    if x_world is None:
        x_world = np.array([1.0, 0.0, 0.0])

    y_world = _normalize(np.cross(z_world, x_world))
    x_world = _normalize(np.cross(y_world, z_world))
    rot_mat = np.column_stack([x_world, y_world, z_world])
    quat = Rotation.from_matrix(rot_mat).as_quat()
    if prev_quat is not None and float(np.dot(quat, prev_quat)) < 0.0:
        quat = -quat
    return quat
