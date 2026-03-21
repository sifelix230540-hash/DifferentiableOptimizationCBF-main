"""3_20 版本：9 轴焊接三段轨迹 + 中组立焊点坐标系 + MPC-DCBF。

在 3_14 的整体结构上扩展：
1. 加载中组立 URDF，并读取 `l2` / `l3` 两个焊点坐标系；
2. 生成三段轨迹：初始点 -> 起点、起点 -> 终点、终点 -> 初始点；
3. 焊接参考姿态由焊点局部方向 `(0, 1, -1)` 生成，要求末端尖点 z 轴对齐；
4. 控制器继续使用原有 `mpc_dcbf` / `cbf_qp` 主线。
"""

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os as _os
import shutil as _shutil
import tempfile as _tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import pybullet_data
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

try:
    import imageio
except ImportError:
    imageio = None


# ============================================================================
#  配置
# ============================================================================

@dataclass
class ExperimentConfig:

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
    workpiece_package_name: str = "中组立0725(1).stp.SLDASM"
    workpiece_package_alias: str = "workpiece_scene"
    workpiece_position: tuple[float, float, float] = (0.30, 5.0, 0.10)
    workpiece_orientation_deg: tuple[float, float, float] = (0.0, 0.0, -90)
    disable_workpiece_collision: bool = False
    start_link_name: str = "l2"
    goal_link_name: str = "l3"
    weld_local_direction: tuple[float, float, float] = (0.0, 1.0, -1.0)

    dt: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # ---- 龙门吊初始关节位置 [pris01, pris02, pris03] ----
    gantry_initial_q: tuple[float, float, float] = (12.0, -7.0, 0.0)

    camera_distance: float = 1.4
    camera_yaw: float = -215.0
    camera_pitch: float = -26.0
    camera_target: tuple[float, float, float] = (0.30, 0.20, 0.60)

    record_video: bool = False
    video_output_path: str = "artifacts/videos/cbf_welding_experiment.mp4"
    video_fps: int = 30
    video_width: int = 960
    video_height: int = 720

    approach_duration: float = 6.0
    weld_duration: float = 7.0
    return_duration: float = 6.0
    hold_duration: float = 3.0

    obstacle_type: str = "none"  # "none" | "sphere" | "plate"
    sphere_radius: float = 0.06
    sphere_rgba: tuple[float, float, float, float] = (1.0, 0.35, 0.2, 0.75)
    sphere_initial_offset: tuple[float, float, float] = (0.0, -0.5, -0.2)
    plate_half_extents: tuple[float, float, float] = (0.12, 0.08, 0.004)
    plate_rgba: tuple[float, float, float, float] = (0.30, 0.55, 0.85, 0.80)
    plate_initial_offset: tuple[float, float, float] = (0.0, -0.15, -0.05)
    obstacle_slider_range: float = 0.5

    ee_force_limit: float = 250.0
    dq_limit: float = 1.0
    dq_nominal_gain: float = 0.25
    base_vel_limit: float = 0.4

    position_gain: float = 8.0
    orientation_gain: float = 0.5
    nullspace_weight: float = 0.001
    slack_weight: float = 200000.0
    use_slack: bool = False
    use_mesh_cbf: bool = True
    cbf_alpha: float = 2.0
    safety_margin: float = 0.02
    q_nominal_tracking: float = 0.02

    controller_type: str = "mpc_dcbf"
    N_mpc: int = 20
    mpc_dt: float = 0.04
    gamma_dcbf: float = 0.15
    mpc_tracking_weight: float = 5.0
    mpc_control_weight: float = 0.2
    mpc_smooth_weight: float = 0.2
    mpc_replan_steps: int = 6

    segment_switch_threshold: float = 0.03

    print_every: int = 120
    reference_samples: int = 80


# ============================================================================
#  仿真场景
# ============================================================================

class SimulationScene:

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client_id = p.connect(p.GUI, options="--width=1920 --height=1080")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*config.gravity)
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

    def capture_frame(self, width, height):
        cam = p.getDebugVisualizerCamera()
        view_matrix = cam[2]
        proj_matrix = cam[3]
        _, _, rgb, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def enable_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


def _prepare_package_urdf(
    urdf_path: str,
    package_name: str | None = None,
    package_alias: str | None = None,
    remove_collision: bool = False,
) -> tuple[str, str]:
    """PyBullet 不支持中文路径，且 package 名可能与目录名不一致。"""
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


# ============================================================================
#  机器人
# ============================================================================

class JakaRobot:

    def __init__(self, config: ExperimentConfig, scene: SimulationScene):
        self.config = config
        self.scene = scene

        urdf_path, search_root = _prepare_package_urdf(config.urdf_path)
        p.setAdditionalSearchPath(search_root)

        self.body_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )

        self.num_joints = p.getNumJoints(self.body_id)
        self.active_joints = []
        self.prismatic_joints = []
        self.revolute_joints = []
        for joint_index in range(self.num_joints):
            joint_info = p.getJointInfo(self.body_id, joint_index)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_PRISMATIC:
                self.active_joints.append(joint_index)
                self.prismatic_joints.append(joint_index)
            elif joint_type == p.JOINT_REVOLUTE:
                self.active_joints.append(joint_index)
                self.revolute_joints.append(joint_index)

        self.n_pris = len(self.prismatic_joints)
        self.n_revo = len(self.revolute_joints)
        self.dof = len(self.active_joints)
        self.total_dof = self.dof

        self.ee_link_index = self.revolute_joints[-1]
        self.welding_gun_links = []
        for joint_index in range(self.num_joints):
            link_name = p.getJointInfo(self.body_id, joint_index)[12].decode()
            if link_name == "weld_point":
                self.ee_link_index = joint_index
            if link_name in ("welding_gun_base", "weld_point"):
                self.welding_gun_links.append(joint_index)

        self.cbf_link_indices = self.prismatic_joints[2:] + list(self.revolute_joints) + self.welding_gun_links

        for ji in self.active_joints:
            p.changeDynamics(self.body_id, ji, linearDamping=0, angularDamping=0)
            p.setJointMotorControl2(self.body_id, ji, p.VELOCITY_CONTROL, force=0)

        self._ik_lower, self._ik_upper = [], []
        self._ik_ranges, self._ik_rest = [], []
        for joint_index in range(self.num_joints):
            info = p.getJointInfo(self.body_id, joint_index)
            lo, hi = float(info[8]), float(info[9])
            if hi < lo:
                self._ik_lower.append(0.0)
                self._ik_upper.append(0.0)
                self._ik_ranges.append(0.0)
            else:
                self._ik_lower.append(lo)
                self._ik_upper.append(hi)
                self._ik_ranges.append(hi - lo if hi > lo else 12.56)
            self._ik_rest.append(0.0)

        for axis_id, joint_index in enumerate(self.prismatic_joints):
            if axis_id < len(config.gantry_initial_q):
                p.resetJointState(self.body_id, joint_index, config.gantry_initial_q[axis_id])
                self._ik_rest[joint_index] = config.gantry_initial_q[axis_id]

        self.q_nominal = np.zeros(self.dof)

    def get_joint_state(self):
        states = p.getJointStates(self.body_id, self.active_joints)
        return np.array([s[0] for s in states]), np.array([s[1] for s in states])

    def get_ee_pose(self):
        state = p.getLinkState(self.body_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(state[4], dtype=float), np.array(state[5], dtype=float)

    def get_link_origin(self, link_index):
        state = p.getLinkState(self.body_id, link_index, computeForwardKinematics=True)
        return np.array(state[4], dtype=float)

    def calculate_ik(self, target_pos, target_quat):
        ik = p.calculateInverseKinematics(
            self.body_id,
            self.ee_link_index,
            target_pos,
            target_quat,
            maxNumIterations=500,
            residualThreshold=1e-6,
        )
        return np.array(ik[: self.dof], dtype=float)

    def reset_to_pose(self, target_pos, target_quat):
        q_target = self.calculate_ik(target_pos, target_quat)
        for i, joint_index in enumerate(self.active_joints):
            p.resetJointState(self.body_id, joint_index, q_target[i])
        self.q_nominal = q_target.copy()

    def get_link_jacobian(self, link_index, q, dq):
        zeros = np.zeros_like(q)
        jt, jr = p.calculateJacobian(
            self.body_id,
            link_index,
            [0, 0, 0],
            q.tolist(),
            dq.tolist(),
            zeros.tolist(),
        )
        return np.array(jt, dtype=float), np.array(jr, dtype=float)

    def get_ee_jacobian(self, q, dq):
        jt, jr = self.get_link_jacobian(self.ee_link_index, q, dq)
        return np.vstack([jt, jr])

    def get_link_cbf_row(self, link_index, normal, q, dq):
        jt, _ = self.get_link_jacobian(link_index, q, dq)
        return normal @ jt

    def get_closest_point_to_obstacle(self, link_index, obs_body_id, max_dist=1.0):
        contacts = p.getClosestPoints(self.body_id, obs_body_id, max_dist, linkIndexA=link_index)
        if not contacts:
            return None
        best = min(contacts, key=lambda c: c[8])
        pos = np.array(best[5], dtype=float)
        dist = float(best[8])
        normal = np.array(best[7], dtype=float)
        nl = np.linalg.norm(normal)
        return pos, dist, (normal / nl if nl > 1e-9 else np.array([1, 0, 0], dtype=float))

    def get_link_cbf_row_at_point(self, link_index, world_point, normal, q, dq):
        link_state = p.getLinkState(self.body_id, link_index, computeForwardKinematics=True)
        inv_pos, inv_quat = p.invertTransform(link_state[4], link_state[5])
        local_point, _ = p.multiplyTransforms(inv_pos, inv_quat, world_point.tolist(), [0, 0, 0, 1])
        zeros = np.zeros_like(q)
        jt, _ = p.calculateJacobian(
            self.body_id,
            link_index,
            list(local_point),
            q.tolist(),
            dq.tolist(),
            zeros.tolist(),
        )
        return normal @ np.array(jt, dtype=float)

    def command_velocities(self, u_cmd):
        lb = np.concatenate([
            np.full(self.n_pris, -self.config.base_vel_limit),
            np.full(self.n_revo, -self.config.dq_limit),
        ])
        u_clip = np.clip(u_cmd, lb, -lb)
        q, _ = self.get_joint_state()
        q_new = q + u_clip * self.config.dt
        for i, joint_index in enumerate(self.active_joints):
            p.resetJointState(self.body_id, joint_index, float(q_new[i]), float(u_clip[i]))

    def get_gantry_pos(self):
        states = p.getJointStates(self.body_id, self.prismatic_joints)
        return np.array([s[0] for s in states])


# ============================================================================
#  工件 / 焊点坐标系
# ============================================================================

class WorkpieceModel:

    def __init__(self, config: ExperimentConfig):
        self.config = config
        urdf_path, search_root = _prepare_package_urdf(
            config.workpiece_urdf_path,
            package_name=config.workpiece_package_name,
            package_alias=config.workpiece_package_alias,
            remove_collision=config.disable_workpiece_collision,
        )
        p.setAdditionalSearchPath(search_root)

        quat = p.getQuaternionFromEuler([math.radians(v) for v in config.workpiece_orientation_deg])
        self.body_id = p.loadURDF(
            urdf_path,
            basePosition=config.workpiece_position,
            baseOrientation=quat,
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )

        self.link_name_to_index = {}
        for joint_index in range(p.getNumJoints(self.body_id)):
            link_name = p.getJointInfo(self.body_id, joint_index)[12].decode()
            self.link_name_to_index[link_name] = joint_index

        if config.disable_workpiece_collision:
            for link_index in range(-1, p.getNumJoints(self.body_id)):
                p.setCollisionFilterGroupMask(self.body_id, link_index, 0, 0)

    def get_frame_pose(self, link_name: str):
        if link_name == "base_link":
            pos, quat = p.getBasePositionAndOrientation(self.body_id)
            return np.array(pos, dtype=float), np.array(quat, dtype=float)
        if link_name not in self.link_name_to_index:
            raise KeyError(f"未找到工件 link: {link_name}")
        link_index = self.link_name_to_index[link_name]
        state = p.getLinkState(self.body_id, link_index, computeForwardKinematics=True)
        return np.array(state[4], dtype=float), np.array(state[5], dtype=float)


# ============================================================================
#  障碍物
# ============================================================================

class Obstacle(ABC):
    @property
    @abstractmethod
    def body_id(self) -> int: ...

    @abstractmethod
    def update_from_slider(self) -> np.ndarray: ...

    @abstractmethod
    def get_position(self) -> np.ndarray: ...

    @abstractmethod
    def compute_distance(self, point: np.ndarray) -> tuple[float, np.ndarray]: ...

    def disable_collision_with(self, robot_body_id, num_joints):
        num_obs_links = p.getNumJoints(self.body_id)
        for obs_link in range(-1, num_obs_links):
            p.setCollisionFilterPair(robot_body_id, self.body_id, -1, obs_link, enableCollision=0)
            for robot_link in range(num_joints):
                p.setCollisionFilterPair(robot_body_id, self.body_id, robot_link, obs_link, enableCollision=0)


class SphereObstacle(Obstacle):
    def __init__(self, config, scene, ee_pos):
        self._r = config.sphere_radius
        dx, dy, dz = config.sphere_initial_offset
        self._init = np.array([ee_pos[0] + dx, ee_pos[1] + dy, ee_pos[2] + dz])
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self._r, rgbaColor=config.sphere_rgba)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self._r)
        self._bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=self._init.tolist(),
        )
        sr = config.obstacle_slider_range
        self.sliders = {
            "x": p.addUserDebugParameter("obs_x", ee_pos[0] - sr, ee_pos[0] + sr, float(self._init[0])),
            "y": p.addUserDebugParameter("obs_y", ee_pos[1] - sr, ee_pos[1] + sr, float(self._init[1])),
            "z": p.addUserDebugParameter("obs_z", ee_pos[2] - sr, ee_pos[2] + sr, float(self._init[2])),
        }

    @property
    def body_id(self):
        return self._bid

    def update_from_slider(self):
        pos = np.array([p.readUserDebugParameter(self.sliders[k]) for k in "xyz"])
        p.resetBasePositionAndOrientation(self._bid, pos.tolist(), [0, 0, 0, 1])
        return pos

    def get_position(self):
        return np.array(p.getBasePositionAndOrientation(self._bid)[0], dtype=float)

    def compute_distance(self, point):
        diff = point - self.get_position()
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return -self._r, np.array([1, 0, 0.0])
        return dist - self._r, diff / dist


class PlateObstacle(Obstacle):
    def __init__(self, config, scene, ee_pos):
        self._half = np.array(config.plate_half_extents, dtype=float)
        dx, dy, dz = config.plate_initial_offset
        self._init = np.array([ee_pos[0] + dx, ee_pos[1] + dy, ee_pos[2] + dz])
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=self._half.tolist(),
            rgbaColor=config.plate_rgba,
            specularColor=[0.5] * 3,
        )
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=self._half.tolist())
        self._bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=self._init.tolist(),
        )
        sr = config.obstacle_slider_range
        self.sliders = {
            "x": p.addUserDebugParameter("plate_x", ee_pos[0] - sr, ee_pos[0] + sr, float(self._init[0])),
            "y": p.addUserDebugParameter("plate_y", ee_pos[1] - sr, ee_pos[1] + sr, float(self._init[1])),
            "z": p.addUserDebugParameter("plate_z", ee_pos[2] - sr, ee_pos[2] + sr, float(self._init[2])),
        }

    @property
    def body_id(self):
        return self._bid

    def update_from_slider(self):
        pos = np.array([p.readUserDebugParameter(self.sliders[k]) for k in "xyz"])
        p.resetBasePositionAndOrientation(self._bid, pos.tolist(), [0, 0, 0, 1])
        return pos

    def get_position(self):
        return np.array(p.getBasePositionAndOrientation(self._bid)[0], dtype=float)

    def compute_distance(self, point):
        ctr, orn = p.getBasePositionAndOrientation(self._bid)
        inv_pos, inv_quat = p.invertTransform(ctr, orn)
        local_point_raw, _ = p.multiplyTransforms(inv_pos, inv_quat, point.tolist(), [0, 0, 0, 1])
        local_point = np.array(local_point_raw, dtype=float)
        d = np.abs(local_point) - self._half
        if np.any(d > 0):
            clipped = np.clip(local_point, -self._half, self._half)
            diff = local_point - clipped
            dist = np.linalg.norm(diff)
            normal_local = diff / dist if dist > 1e-9 else np.array([1, 0, 0], dtype=float)
            signed_dist = dist
        else:
            face_dist = self._half - np.abs(local_point)
            min_axis = int(np.argmin(face_dist))
            signed_dist = -face_dist[min_axis]
            normal_local = np.zeros(3)
            normal_local[min_axis] = 1.0 if local_point[min_axis] >= 0 else -1.0
        rot_mat = np.array(p.getMatrixFromQuaternion(orn), dtype=float).reshape(3, 3)
        return float(signed_dist), rot_mat @ normal_local


class URDFObstacle(Obstacle):
    """将任意 PyBullet body (如 URDF 工件) 包装为 CBF 障碍物。

    Parameters
    ----------
    body_id : int
        PyBullet body ID.
    cbf_link_indices : list[int] | None
        仅对这些机器人 link 做 CBF 距离检查。
        ``None`` 表示使用机器人默认的全部 ``cbf_link_indices``。
    """

    def __init__(self, body_id: int, cbf_link_indices: list[int] | None = None):
        self._bid = body_id
        self._cbf_links = cbf_link_indices

    @property
    def body_id(self):
        return self._bid

    @property
    def cbf_link_indices(self):
        return self._cbf_links

    def update_from_slider(self):
        return self.get_position()

    def get_position(self):
        return np.array(p.getBasePositionAndOrientation(self._bid)[0], dtype=float)

    def compute_distance(self, point):
        diff = point - self.get_position()
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return 0.0, np.array([1.0, 0.0, 0.0])
        return dist, diff / dist


def create_obstacle(config, scene, ee_pos):
    if config.obstacle_type == "none":
        return None
    if config.obstacle_type == "sphere":
        return SphereObstacle(config, scene, ee_pos)
    if config.obstacle_type == "plate":
        return PlateObstacle(config, scene, ee_pos)
    raise ValueError(f"未知障碍物类型: {config.obstacle_type}")


# ============================================================================
#  轨迹与姿态构造
# ============================================================================

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def _project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return v - float(np.dot(v, normal)) * normal


def build_weld_reference_quat(frame_quat, weld_local_direction, prev_quat=None):
    """构造参考姿态：仅约束工具 z 轴，滚转按连续性选择。"""
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


class LineSlerpTrajectory:

    def __init__(self, start_pos, start_quat, goal_pos, goal_quat, duration, dt):
        self.start_pos = np.array(start_pos, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.start_quat = np.array(start_quat, dtype=float)
        self.goal_quat = np.array(goal_quat, dtype=float)
        self.duration = float(duration)
        self.dt = float(dt)
        self.linear_velocity = (self.goal_pos - self.start_pos) / max(self.duration, 1e-6)
        self.slerp = Slerp(
            np.array([0.0, self.duration]),
            Rotation.from_quat(np.vstack([self.start_quat, self.goal_quat])),
        )

    def sample(self, t):
        tau = float(np.clip(t, 0, self.duration))
        blend = tau / max(self.duration, 1e-6)
        pos = (1 - blend) * self.start_pos + blend * self.goal_pos
        quat = self.slerp([tau]).as_quat()[0]
        if tau >= self.duration:
            return pos, quat, np.zeros(3), np.zeros(3)
        next_t = min(self.duration, tau + self.dt)
        next_quat = self.slerp([next_t]).as_quat()[0]
        ang_vel = (
            Rotation.from_quat(next_quat) * Rotation.from_quat(quat).inv()
        ).as_rotvec() / max(next_t - tau, 1e-6)
        return pos, quat, self.linear_velocity.copy(), ang_vel

    def reference_points(self, n):
        return [self.sample(t)[0] for t in np.linspace(0, self.duration, n, endpoint=True)]


class PiecewiseLineSlerpTrajectory:

    def __init__(self, segments: list[LineSlerpTrajectory]):
        self.segments = segments
        self.durations = [seg.duration for seg in segments]
        self.cumulative = np.cumsum(self.durations)
        self.duration = float(self.cumulative[-1]) if len(self.cumulative) else 0.0
        self.dt = segments[0].dt if segments else 0.0

    def current_segment_index(self, t: float) -> int:
        if not self.segments:
            return 0
        tau = float(np.clip(t, 0, self.duration))
        return int(np.searchsorted(self.cumulative, tau, side="right"))

    def sample(self, t):
        if not self.segments:
            raise RuntimeError("空轨迹无法采样")
        tau = float(np.clip(t, 0, self.duration))
        segment_index = min(self.current_segment_index(tau), len(self.segments) - 1)
        prev_end = 0.0 if segment_index == 0 else self.cumulative[segment_index - 1]
        local_t = tau - prev_end
        return self.segments[segment_index].sample(local_t)

    def reference_points(self, n):
        return [self.sample(t)[0] for t in np.linspace(0, self.duration, n, endpoint=True)]

    def segment_reference_points(self, n_per_segment):
        return [seg.reference_points(n_per_segment) for seg in self.segments]


# ============================================================================
#  控制器
# ============================================================================

class Controller(ABC):
    @abstractmethod
    def solve(
        self,
        q,
        dq,
        ee_pos,
        ee_quat,
        ref_pos,
        ref_quat,
        ref_lin_vel,
        ref_ang_vel,
        obstacles: list[Obstacle],
        current_time: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        ...


class CBFQPController(Controller):

    def __init__(self, robot: JakaRobot, config: ExperimentConfig, **_kwargs):
        self.robot = robot
        self.config = config
        self.n = robot.total_dof

    def _build_cbf_data(self, q, dq, obstacles):
        a_rows, h_vals = [], []
        for obs in obstacles:
            use_mesh = self.config.use_mesh_cbf and obs.body_id >= 0
            obs_links = getattr(obs, 'cbf_link_indices', None)
            check_links = obs_links if obs_links is not None else self.robot.cbf_link_indices
            for li in check_links:
                if use_mesh:
                    cp = self.robot.get_closest_point_to_obstacle(li, obs.body_id)
                    if cp is None:
                        continue
                    support_point, signed_dist, normal = cp
                    h_vals.append(signed_dist - self.config.safety_margin)
                    a_rows.append(self.robot.get_link_cbf_row_at_point(li, support_point, normal, q, dq))
                else:
                    link_pos = self.robot.get_link_origin(li)
                    signed_dist, normal = obs.compute_distance(link_pos)
                    h_vals.append(signed_dist - self.config.safety_margin)
                    a_rows.append(self.robot.get_link_cbf_row(li, normal, q, dq))
        return a_rows, h_vals

    def solve(self, q, dq, ee_pos, ee_quat, ref_pos, ref_quat, ref_lin_vel, ref_ang_vel, obstacles, current_time=0.0):
        pos_err = ref_pos - ee_pos
        rot_err = (Rotation.from_quat(ref_quat) * Rotation.from_quat(ee_quat).inv()).as_rotvec()
        xdot_ref = np.concatenate([
            ref_lin_vel + self.config.position_gain * pos_err,
            ref_ang_vel + self.config.orientation_gain * rot_err,
        ])
        dq_nom = self.config.dq_nominal_gain * (self.robot.q_nominal - q)
        j_ee = self.robot.get_ee_jacobian(q, dq)
        a_rows, h_vals = self._build_cbf_data(q, dq, obstacles)
        min_h = float(np.min(h_vals)) if h_vals else 1.0
        m = len(a_rows)
        a_mat = np.array(a_rows) if a_rows else np.zeros((0, self.n))
        b_vec = np.array([self.config.cbf_alpha * hv for hv in h_vals[:m]]) if m else np.zeros(0)

        def objective(x):
            u = x[: self.n]
            return np.sum((j_ee @ u - xdot_ref) ** 2) + self.config.nullspace_weight * np.sum((u - dq_nom) ** 2)

        constraints = []
        for row_idx in range(m):
            row = a_mat[row_idx].copy()
            rhs = b_vec[row_idx]
            constraints.append({"type": "ineq", "fun": lambda x, r=row, b=rhs: r @ x[: self.n] + b})

        bounds = (
            [(-self.config.base_vel_limit, self.config.base_vel_limit)] * self.robot.n_pris
            + [(-self.config.dq_limit, self.config.dq_limit)] * self.robot.n_revo
        )

        u_pinv = np.linalg.lstsq(j_ee, xdot_ref, rcond=None)[0]
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        x0 = np.clip(u_pinv, lb, ub)
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6},
        )
        if res.success:
            u_cmd = res.x[: self.n]
            status = "cbf_optimal"
        else:
            u_cmd = np.clip(dq_nom, lb, ub)
            status = "cbf_fallback"
        return u_cmd, {
            "min_h": min_h,
            "max_slack": 0.0,
            "status": status,
            "tracking_error": float(np.linalg.norm(pos_err)),
        }


class MPCDCBFController(Controller):

    def __init__(self, robot: JakaRobot, config: ExperimentConfig, trajectory):
        self.robot = robot
        self.config = config
        self.n = robot.total_dof
        self.N = config.N_mpc
        self.trajectory = trajectory
        self._prev_sol: np.ndarray | None = None
        self._cached_u: np.ndarray | None = None
        self._cached_info: dict | None = None
        self._step_count = 0

        single_bounds = (
            [(-config.base_vel_limit, config.base_vel_limit)] * robot.n_pris
            + [(-config.dq_limit, config.dq_limit)] * robot.n_revo
        )
        self._bounds = single_bounds * self.N
        self._lb = np.array([b[0] for b in single_bounds])
        self._ub = np.array([b[1] for b in single_bounds])

    def _build_cbf_data(self, q, dq, obstacles):
        grad_rows, h_vals = [], []
        for obs in obstacles:
            use_mesh = self.config.use_mesh_cbf and obs.body_id >= 0
            obs_links = getattr(obs, 'cbf_link_indices', None)
            check_links = obs_links if obs_links is not None else self.robot.cbf_link_indices
            for li in check_links:
                if use_mesh:
                    cp = self.robot.get_closest_point_to_obstacle(li, obs.body_id)
                    if cp is None:
                        continue
                    support_point, signed_dist, normal = cp
                    h_vals.append(signed_dist - self.config.safety_margin)
                    grad_rows.append(self.robot.get_link_cbf_row_at_point(li, support_point, normal, q, dq))
                else:
                    link_pos = self.robot.get_link_origin(li)
                    signed_dist, normal = obs.compute_distance(link_pos)
                    h_vals.append(signed_dist - self.config.safety_margin)
                    grad_rows.append(self.robot.get_link_cbf_row(li, normal, q, dq))
        return grad_rows, h_vals

    def _build_qp(self, ee_pos, j_pos, ref_positions, grad_rows, h_vals):
        n, N = self.n, self.N
        mdt = self.config.mpc_dt
        cfg = self.config
        dim = n * N

        jtj = mdt ** 2 * (j_pos.T @ j_pos)
        idx = np.arange(N)
        weight_mat = (N - np.maximum(idx[:, None], idx[None, :])).astype(float)
        h_mat = 2.0 * cfg.mpc_tracking_weight * np.kron(weight_mat, jtj)

        c_vecs = np.array([ee_pos - ref_positions[k] for k in range(N)])
        c_suffix = np.cumsum(c_vecs[::-1], axis=0)[::-1]
        f_vec = 2.0 * cfg.mpc_tracking_weight * mdt * (c_suffix @ j_pos).ravel()

        h_mat += 2.0 * cfg.mpc_control_weight * np.eye(dim)
        if N > 1:
            diff = np.zeros((N - 1, N))
            for k in range(N - 1):
                diff[k, k] = -1.0
                diff[k, k + 1] = 1.0
            h_mat += 2.0 * cfg.mpc_smooth_weight * np.kron(diff.T @ diff, np.eye(n))

        n_cbf = len(grad_rows)
        gamma = cfg.gamma_dcbf
        tril = np.tril(gamma * np.ones((N, N)))
        np.fill_diagonal(tril, 1.0)
        tril_mdt = mdt * tril

        a_cbf = np.zeros((n_cbf * N, dim))
        b_cbf = np.zeros(n_cbf * N)
        for cbf_idx in range(n_cbf):
            grad = grad_rows[cbf_idx]
            a_cbf[cbf_idx * N : (cbf_idx + 1) * N, :] = np.kron(tril_mdt, grad.reshape(1, -1))
            b_cbf[cbf_idx * N : (cbf_idx + 1) * N] = gamma * h_vals[cbf_idx]

        return h_mat, f_vec, a_cbf, b_cbf

    def solve(self, q, dq, ee_pos, ee_quat, ref_pos, ref_quat, ref_lin_vel, ref_ang_vel, obstacles, current_time=0.0):
        self._step_count += 1
        if self._cached_u is not None and self._step_count % self.config.mpc_replan_steps != 0:
            return self._cached_u, self._cached_info

        n, N = self.n, self.N
        cfg = self.config

        j_full = self.robot.get_ee_jacobian(q, dq)
        j_pos = j_full[:3]

        ref_positions = []
        for k in range(1, N + 1):
            pk, _, _, _ = self.trajectory.sample(min(current_time + k * cfg.mpc_dt, self.trajectory.duration))
            ref_positions.append(pk)

        grad_rows, h_vals = self._build_cbf_data(q, dq, obstacles)
        min_h = float(np.min(h_vals)) if h_vals else 1.0
        h_mat, f_vec, a_cbf, b_cbf = self._build_qp(ee_pos, j_pos, ref_positions, grad_rows, h_vals)

        constraints = []
        if len(grad_rows) > 0:
            constraints.append({
                "type": "ineq",
                "fun": lambda x: a_cbf @ x + b_cbf,
                "jac": lambda x: a_cbf,
            })

        if self._prev_sol is not None:
            x0 = np.empty(n * N)
            x0[: (N - 1) * n] = self._prev_sol[n:]
            x0[(N - 1) * n :] = self._prev_sol[(N - 1) * n :]
        else:
            pos_err = ref_pos - ee_pos
            rot_err = (Rotation.from_quat(ref_quat) * Rotation.from_quat(ee_quat).inv()).as_rotvec()
            xdot = np.concatenate([
                ref_lin_vel + cfg.position_gain * pos_err,
                ref_ang_vel + cfg.orientation_gain * rot_err,
            ])
            u0 = np.clip(np.linalg.lstsq(j_full, xdot, rcond=None)[0], self._lb, self._ub)
            x0 = np.tile(u0, N)

        res = minimize(
            lambda x: 0.5 * x @ h_mat @ x + f_vec @ x,
            x0,
            method="SLSQP",
            jac=lambda x: h_mat @ x + f_vec,
            bounds=self._bounds,
            constraints=constraints,
            options={"maxiter": 50, "ftol": 1e-6},
        )

        if res.success:
            self._prev_sol = res.x.copy()
            u_cmd = res.x[:n]
            status = "mpc_optimal"
        else:
            u_cmd = np.clip(x0[:n], self._lb, self._ub)
            status = "mpc_fallback"

        pos_err = ref_pos - ee_pos
        info = {
            "min_h": min_h,
            "max_slack": 0.0,
            "status": status,
            "tracking_error": float(np.linalg.norm(pos_err)),
        }
        self._cached_u = u_cmd
        self._cached_info = info
        return u_cmd, info


def create_controller(robot, config, trajectory) -> Controller:
    if config.controller_type == "cbf_qp":
        return CBFQPController(robot, config)
    if config.controller_type == "mpc_dcbf":
        return MPCDCBFController(robot, config, trajectory)
    raise ValueError(f"未知控制器类型: {config.controller_type}")


# ============================================================================
#  实验主类
# ============================================================================

class AvoidanceExperiment:

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.scene = SimulationScene(config)
        self.robot = JakaRobot(config, self.scene)
        self.workpiece = WorkpieceModel(config)

        ee_pos_init, ee_quat_init = self.robot.get_ee_pose()
        self.initial_pos = ee_pos_init.copy()
        self.initial_quat = ee_quat_init.copy()

        obstacle = create_obstacle(config, self.scene, ee_pos_init)
        self.obstacles: list[Obstacle] = []
        if obstacle is not None:
            obstacle.disable_collision_with(self.robot.body_id, self.robot.num_joints)
            self.obstacles.append(obstacle)

        wp_obs = URDFObstacle(
            self.workpiece.body_id,
            cbf_link_indices=[li for li in self.robot.cbf_link_indices if li != self.robot.ee_link_index],
        )
        wp_obs.disable_collision_with(self.robot.body_id, self.robot.num_joints)
        self.obstacles.append(wp_obs)

        start_pos, start_frame_quat = self.workpiece.get_frame_pose(config.start_link_name)
        goal_pos, goal_frame_quat = self.workpiece.get_frame_pose(config.goal_link_name)
        start_quat = build_weld_reference_quat(
            start_frame_quat,
            config.weld_local_direction,
            prev_quat=ee_quat_init,
        )
        goal_quat = build_weld_reference_quat(
            goal_frame_quat,
            config.weld_local_direction,
            prev_quat=start_quat,
        )

        self.start_frame = (start_pos, np.array(start_frame_quat, dtype=float))
        self.goal_frame = (goal_pos, np.array(goal_frame_quat, dtype=float))
        self.start_ref = (start_pos, np.array(start_quat, dtype=float))
        self.goal_ref = (goal_pos, np.array(goal_quat, dtype=float))

        self.trajectory = PiecewiseLineSlerpTrajectory([
            LineSlerpTrajectory(self.initial_pos, self.initial_quat, start_pos, start_quat, config.approach_duration, config.dt),
            LineSlerpTrajectory(start_pos, start_quat, goal_pos, goal_quat, config.weld_duration, config.dt),
            LineSlerpTrajectory(goal_pos, goal_quat, self.initial_pos, self.initial_quat, config.return_duration, config.dt),
        ])

        colors = ([0.85, 0.35, 0.15], [0.95, 0.15, 0.15], [0.20, 0.45, 0.90])
        for pts, color in zip(self.trajectory.segment_reference_points(config.reference_samples), colors):
            self.scene.draw_polyline(pts, color=color, width=1.8)

        weld_dir_start = Rotation.from_quat(start_frame_quat).apply(_normalize(np.array(config.weld_local_direction, dtype=float)))
        weld_dir_goal = Rotation.from_quat(goal_frame_quat).apply(_normalize(np.array(config.weld_local_direction, dtype=float)))
        self.scene.draw_direction(start_pos, weld_dir_start, [0.90, 0.20, 0.20], length=0.12)
        self.scene.draw_direction(goal_pos, weld_dir_goal, [0.20, 0.20, 0.90], length=0.12)

        focus = 0.5 * (start_pos + goal_pos)
        p.resetDebugVisualizerCamera(
            cameraDistance=config.camera_distance,
            cameraYaw=config.camera_yaw,
            cameraPitch=config.camera_pitch,
            cameraTargetPosition=focus.tolist(),
        )

        self.ee_marker = self.scene.create_marker(0.012, (0.1, 0.9, 0.2, 0.9), ee_pos_init.tolist())
        self.ref_marker = self.scene.create_marker(0.012, (0.95, 0.2, 0.2, 0.9), self.initial_pos.tolist())
        self.start_marker = self.scene.create_marker(0.014, (0.95, 0.15, 0.15, 0.85), start_pos.tolist())
        self.goal_marker = self.scene.create_marker(0.014, (0.15, 0.25, 0.95, 0.85), goal_pos.tolist())

        self.controller = create_controller(self.robot, config, self.trajectory)
        self.prev_ee = ee_pos_init.copy()
        self.sim_step = 0
        self.scene.enable_rendering()
        print(f"===== 焊接实验 (控制器: {config.controller_type}, 障碍: {config.obstacle_type}) =====")

    def _update_visuals(self, ee_pos, ref_pos, info, t):
        self.scene.update_marker(self.ee_marker, ee_pos.tolist())
        self.scene.update_marker(self.ref_marker, ref_pos.tolist())
        if np.linalg.norm(ee_pos - self.prev_ee) > 1e-3:
            p.addUserDebugLine(self.prev_ee.tolist(), ee_pos.tolist(), [0.1, 0.8, 0.2], lineWidth=1.5)
            self.prev_ee = ee_pos.copy()
        seg_idx = min(self.trajectory.current_segment_index(t) + 1, len(self.trajectory.segments))
        self.scene.update_status(
            f"seg={seg_idx}  step={self.sim_step}  "
            f"err={info['tracking_error']*1000:.1f}mm  "
            f"h={info['min_h']*1000:.1f}mm  "
            f"{info['status']}"
        )

    def _solve_step(self, q, dq, ee_pos, ee_quat, ref_pos, ref_quat, ref_lv, ref_av, t):
        for obs in self.obstacles:
            obs.update_from_slider()
        return self.controller.solve(
            q,
            dq,
            ee_pos,
            ee_quat,
            ref_pos,
            ref_quat,
            ref_lv,
            ref_av,
            self.obstacles,
            current_time=t,
        )

    def run(self):
        video_frames = []
        rec_every = max(1, int(round((1 / self.config.dt) / self.config.video_fps))) if self.config.record_video else 0

        traj_time = 0.0
        gate_seg = 0
        n_segs = len(self.trajectory.segments)
        hold_steps = int(self.config.hold_duration / self.config.dt)
        hold_counter = 0

        try:
            while p.isConnected():
                q, dq = self.robot.get_joint_state()
                ee_pos, ee_quat = self.robot.get_ee_pose()

                if traj_time < self.trajectory.duration:
                    next_t = traj_time + self.config.dt
                    next_seg = self.trajectory.current_segment_index(
                        min(next_t, self.trajectory.duration))
                    if next_seg > gate_seg and gate_seg < n_segs - 1:
                        seg_goal = self.trajectory.segments[gate_seg].goal_pos
                        if np.linalg.norm(ee_pos - seg_goal) < self.config.segment_switch_threshold:
                            gate_seg = next_seg
                            traj_time = min(next_t, self.trajectory.duration)
                            print(f"[gate] 段 {gate_seg} 已解锁 (step {self.sim_step})")
                        else:
                            seg_end = float(self.trajectory.cumulative[gate_seg])
                            traj_time = min(next_t, seg_end - 1e-6)
                    else:
                        traj_time = min(next_t, self.trajectory.duration)
                else:
                    hold_counter += 1
                    if hold_counter >= hold_steps:
                        break

                ref = self.trajectory.sample(traj_time)
                u_cmd, info = self._solve_step(q, dq, ee_pos, ee_quat, *ref, traj_time)

                alpha_q = self.config.q_nominal_tracking
                self.robot.q_nominal = (1 - alpha_q) * self.robot.q_nominal + alpha_q * q
                self.robot.command_velocities(u_cmd)

                p.stepSimulation()
                self._update_visuals(ee_pos, ref[0], info, traj_time)
                if self.config.record_video and self.sim_step % rec_every == 0:
                    video_frames.append(self.scene.capture_frame(self.config.video_width, self.config.video_height))
                if self.sim_step % self.config.print_every == 0:
                    gp = self.robot.get_gantry_pos()
                    print(
                        f"[step {self.sim_step:4d}] "
                        f"seg={gate_seg + 1} "
                        f"gantry=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                        f"err={info['tracking_error']*1000:.1f}mm "
                        f"h={info['min_h']*1000:.1f}mm "
                        f"{info['status']}"
                    )
                self.sim_step += 1
                time.sleep(self.config.dt)

            self.robot.command_velocities(np.zeros(self.robot.dof))
            if self.config.record_video and video_frames and imageio:
                imageio.mimsave(self.config.video_output_path, video_frames, fps=self.config.video_fps)
                print(f"录像已保存: {self.config.video_output_path}")

            print("===== 轨迹结束，保持窗口 (Ctrl+C 退出) =====")
            while p.isConnected():
                q, dq = self.robot.get_joint_state()
                ee_pos, ee_quat = self.robot.get_ee_pose()
                ref = self.trajectory.sample(self.trajectory.duration)
                u_cmd, info = self._solve_step(q, dq, ee_pos, ee_quat, *ref, self.trajectory.duration)
                self.robot.command_velocities(u_cmd)
                p.stepSimulation()
                self._update_visuals(ee_pos, ref[0], info, self.trajectory.duration)
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            print("\n用户中断。")
        finally:
            if p.isConnected():
                p.disconnect()
            print(f"仿真结束，共 {self.sim_step} 步。")


def main():
    AvoidanceExperiment(ExperimentConfig()).run()


if __name__ == "__main__":
    main()
