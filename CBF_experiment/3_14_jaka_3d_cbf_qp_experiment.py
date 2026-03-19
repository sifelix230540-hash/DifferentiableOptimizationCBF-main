"""3_14 版本：9 自由度龙门吊 + 倒挂机械臂 (URDF 一体化)。

与 3_13 的差异：
1. 使用 9_axis URDF，内含 3 个移动副 (龙门吊 X/Y/Z) + 6 个转动副 (机械臂)；
2. 机械臂倒挂安装，底座通过 fixed joint 接在第三轴末端平台下方；
3. 所有 9 个自由度均通过 PyBullet 关节电机控制，无需手动移动 base；
4. Jacobian 由 PyBullet 直接计算 (6×9)，无需手动拼接增广矩阵。

MPC-DCBF 数学模型
─────────────────
  状态: q̃ = [gantry_xyz, q_joints] ∈ ℝ⁹
  控制: u = [v_gantry, dq] ∈ ℝ⁹  (速度)
  动力学: q̃_{k+1} = q̃_k + u_k · Δt  (积分器)
  末端位置线性化: p_ee(q̃_k) ≈ p_ee_0 + J_pos · (q̃_k − q̃_0)
  离散 CBF: h_{k+1} ≥ (1 − γ) · h_k

  min  Σ w_track ‖p_ee_pred_k − p_ref_k‖²
     + Σ w_u ‖u_k‖²
     + Σ w_smooth ‖u_k − u_{k−1}‖²
     + w_slack ‖s‖²
  s.t. 线性化离散 CBF 约束, 速度 bounds
"""

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
        r"\DifferentiableOptimizationCBF-main\9_axis\urdf\9_axis.urdf"
    )
    dt: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # ---- 龙门吊初始关节位置 [pris01, pris02, pris03] ----
    # pris01 沿 +X, pris02 沿 +Y, pris03 沿 +Z
    # 与 3_18_import 保持一致：X前移12, Y偏移-7, Z=0
    gantry_initial_q: tuple[float, float, float] = (12.0, -7.0, 0.0)

    camera_distance: float = 1.8
    camera_yaw: float = -225.0
    camera_pitch: float = -30
    camera_target: tuple[float, float, float] = (0.15, 0.20, 0.60)

    record_video: bool = False
    video_output_path: str = "cbf_experiment.mp4"
    video_fps: int = 30
    video_width: int = 960
    video_height: int = 720

    line_half_span: float = 0.14
    line_bias_y: float = 0.08
    line_bias_z: float = 0.0
    trajectory_duration: float = 7.0
    hold_duration: float = 6

    obstacle_type: str = "sphere"
    sphere_radius: float = 0.06
    sphere_rgba: tuple[float, float, float, float] = (1.0, 0.35, 0.2, 0.75)
    # 障碍物初始偏移（相对于末端位置 ee_pos 的偏移量）
    sphere_initial_offset: tuple[float, float, float] = (0.0, -0.5, -0.2)
    plate_half_extents: tuple[float, float, float] = (0.12, 0.08, 0.004)
    plate_rgba: tuple[float, float, float, float] = (0.30, 0.55, 0.85, 0.80)
    plate_initial_offset: tuple[float, float, float] = (0.0, -0.15, -0.05)
    workpiece_urdf_path: str = "workpiece_T.urdf"
    workpiece_position: tuple[float, float, float] = (0.30, 0.35, 0.0)
    workpiece_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    obstacle_slider_range: float = 0.5

    start_euler_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)
    goal_euler_deg: tuple[float, float, float] = (140.0, 0, 0)

    ee_force_limit: float = 250.0
    dq_limit: float = 1.0
    dq_nominal_gain: float = 0.25
    base_vel_limit: float = 0.4  # 龙门吊各轴速度限制

    # ---- 通用 QP / CBF ----
    position_gain: float = 8
    orientation_gain: float = 0.5
    nullspace_weight: float = 0.001
    slack_weight: float = 200000.0
    use_slack: bool = False
    use_mesh_cbf: bool = True
    cbf_alpha: float = 2
    safety_margin: float = 0.02
    q_nominal_tracking: float = 0.02

    # ---- 控制器选择 ----
    controller_type: str = "mpc_dcbf"  # "cbf_qp" | "mpc_dcbf"

    # ---- MPC-DCBF 专有参数 ----
    N_mpc: int = 5
    mpc_dt: float = 0.04
    gamma_dcbf: float = 0.15
    mpc_tracking_weight: float = 5
    mpc_control_weight: float = 0.02
    mpc_smooth_weight: float = 0.02
    mpc_replan_steps: int = 6

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
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(rgbBackground=[1, 1, 1])
        p.resetDebugVisualizerCamera(
            cameraDistance=config.camera_distance, cameraYaw=config.camera_yaw,
            cameraPitch=config.camera_pitch, cameraTargetPosition=config.camera_target)
        self.reference_height = self._build_environment()
        self._draw_axes()
        self.status_text_id = None

    def _build_environment(self) -> float:
        plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.95, 0.95, 0.95, 1.0])
        return 0.0

    def _draw_axes(self):
        al = 0.12; o = [0, 0, self.reference_height+0.001]
        p.addUserDebugLine(o, [al, 0, o[2]], [1, 0, 0], lineWidth=2)
        p.addUserDebugLine(o, [0, al, o[2]], [0, 0.8, 0], lineWidth=2)
        p.addUserDebugLine(o, [0, 0, o[2]+al], [0, 0, 1], lineWidth=2)

    def create_marker(self, radius, color, pos):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos)

    def update_marker(self, bid, pos):
        p.resetBasePositionAndOrientation(bid, pos, [0, 0, 0, 1])

    def draw_polyline(self, pts, color, width=1.5):
        for i in range(len(pts)-1):
            p.addUserDebugLine(pts[i].tolist(), pts[i+1].tolist(), color, lineWidth=width)

    def update_status(self, text):
        self.status_text_id = p.addUserDebugText(
            text, [0.02, -0.26, 0.52], [0.1]*3, textSize=1.2,
            replaceItemUniqueId=self.status_text_id if self.status_text_id else -1)

    def capture_frame(self, w, h):
        _, _, rgb, _, _ = p.getCameraImage(w, h)
        return np.array(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]

import os as _os
import shutil as _shutil
import tempfile as _tempfile


def _prepare_urdf(urdf_path: str) -> tuple[str, str]:
    """PyBullet 不支持中文路径，必要时复制到临时目录。

    返回 (可用的 urdf 路径, package 搜索根目录)。
    """
    pkg_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(urdf_path)))
    try:
        pkg_dir.encode("ascii")
        return _os.path.abspath(urdf_path), pkg_dir
    except UnicodeEncodeError:
        pass

    pkg_name = _os.path.basename(pkg_dir)
    urdf_rel = _os.path.relpath(_os.path.abspath(urdf_path), pkg_dir)
    tmp_root = _os.path.join(_tempfile.gettempdir(), "pybullet_urdf")
    tmp_pkg = _os.path.join(tmp_root, pkg_name)
    if _os.path.exists(tmp_pkg):
        _shutil.rmtree(tmp_pkg, ignore_errors=True)
        for _ in range(30):
            if not _os.path.exists(tmp_pkg):
                break
            time.sleep(0.1)
    _shutil.copytree(pkg_dir, tmp_pkg, dirs_exist_ok=True)
    new_urdf = _os.path.join(tmp_pkg, urdf_rel)
    print(f"[info] URDF 已复制到临时目录: {tmp_pkg}")
    return new_urdf, tmp_root


# ============================================================================
#  机器人 — 9-DOF (3 移动副 + 6 转动副，全部由 URDF 关节驱动)
# ============================================================================

class JakaRobot:

    def __init__(self, config: ExperimentConfig, scene: SimulationScene):
        self.config = config
        self.scene = scene

        urdf_path, search_root = _prepare_urdf(config.urdf_path)
        p.setAdditionalSearchPath(search_root)

        self.body_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

        self.num_joints = p.getNumJoints(self.body_id)

        # 收集所有可动关节 (prismatic + revolute)
        self.active_joints = []
        self.prismatic_joints = []
        self.revolute_joints = []
        for i in range(self.num_joints):
            jinfo = p.getJointInfo(self.body_id, i)
            jtype = jinfo[2]
            if jtype == p.JOINT_PRISMATIC:
                self.active_joints.append(i)
                self.prismatic_joints.append(i)
            elif jtype == p.JOINT_REVOLUTE:
                self.active_joints.append(i)
                self.revolute_joints.append(i)

        self.n_pris = len(self.prismatic_joints)   # 3
        self.n_revo = len(self.revolute_joints)     # 6
        self.dof = len(self.active_joints)          # 9
        self.total_dof = self.dof
        self.ee_link_index = self.revolute_joints[-1]
        self.cbf_link_indices = list(self.revolute_joints)  # CBF 只检查臂杆

        # 初始化关节
        for ji in self.active_joints:
            p.changeDynamics(self.body_id, ji, linearDamping=0, angularDamping=0)
            p.setJointMotorControl2(self.body_id, ji, p.VELOCITY_CONTROL, force=0)

        # 从 URDF 读取关节限位 (IK 用)
        # PyBullet calculateInverseKinematics 要求 limits 长度 == getNumJoints()
        # 必须遍历所有关节（含 FIXED），对 FIXED joint 填占位值
        self._ik_lower, self._ik_upper = [], []
        self._ik_ranges, self._ik_rest = [], []
        for ji in range(self.num_joints):
            info = p.getJointInfo(self.body_id, ji)
            lo, hi = float(info[8]), float(info[9])
            if hi < lo:
                # FIXED joint (PyBullet 用 lo=0,hi=-1 表示无约束)，填占位 0
                self._ik_lower.append(0.0)
                self._ik_upper.append(0.0)
                self._ik_ranges.append(0.0)
            else:
                self._ik_lower.append(lo)
                self._ik_upper.append(hi)
                self._ik_ranges.append(hi - lo if hi > lo else 12.56)
            self._ik_rest.append(0.0)

        # 设置龙门吊初始位置，并写入 rest poses
        for k, ji in enumerate(self.prismatic_joints):
            if k < len(config.gantry_initial_q):
                p.resetJointState(self.body_id, ji, config.gantry_initial_q[k])
                self._ik_rest[ji] = config.gantry_initial_q[k]  # ji 即关节 index，与 _ik_rest 下标对齐

        self.q_nominal = np.zeros(self.dof)

    def get_joint_state(self):
        st = p.getJointStates(self.body_id, self.active_joints)
        return np.array([s[0] for s in st]), np.array([s[1] for s in st])

    def get_ee_pose(self):
        s = p.getLinkState(self.body_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(s[4], dtype=float), np.array(s[5], dtype=float)

    def get_link_origin(self, li):
        s = p.getLinkState(self.body_id, li, computeForwardKinematics=True)
        return np.array(s[4], dtype=float)

    def calculate_ik(self, tpos, tquat):
        # 不传 limits，避免触发逆动力学（龙门架惯性张量精度问题）
        a = p.calculateInverseKinematics(
            self.body_id, self.ee_link_index, tpos, tquat,
            maxNumIterations=500, residualThreshold=1e-6)
        return np.array(a[:self.dof], dtype=float)

    def reset_to_pose(self, tpos, tquat):
        qt = self.calculate_ik(tpos, tquat)
        for i, ji in enumerate(self.active_joints):
            p.resetJointState(self.body_id, ji, qt[i])
        self.q_nominal = qt.copy()

    def get_link_jacobian(self, li, q, dq):
        z = np.zeros_like(q)
        jt, jr = p.calculateJacobian(self.body_id, li, [0, 0, 0],
                                      q.tolist(), dq.tolist(), z.tolist())
        return np.array(jt, dtype=float), np.array(jr, dtype=float)

    def get_ee_jacobian(self, q, dq):
        """返回 6×dof 的末端 Jacobian (PyBullet 已包含所有关节列)。"""
        jt, jr = self.get_link_jacobian(self.ee_link_index, q, dq)
        return np.vstack([jt, jr])

    def get_link_cbf_row(self, li, normal, q, dq):
        jt, _ = self.get_link_jacobian(li, q, dq)
        return normal @ jt  # dof 维向量

    def get_closest_point_to_obstacle(self, li, obs_bid, max_dist=1.0):
        contacts = p.getClosestPoints(self.body_id, obs_bid, max_dist, linkIndexA=li)
        if not contacts:
            return None
        best = min(contacts, key=lambda c: c[8])
        pos = np.array(best[5], dtype=float)
        d = float(best[8])
        n = np.array(best[7], dtype=float)
        nl = np.linalg.norm(n)
        return pos, d, (n/nl if nl > 1e-9 else np.array([1, 0, 0], dtype=float))

    def get_link_cbf_row_at_point(self, li, wpt, normal, q, dq):
        ls = p.getLinkState(self.body_id, li, computeForwardKinematics=True)
        ip, io = p.invertTransform(ls[4], ls[5])
        lp, _ = p.multiplyTransforms(ip, io, wpt.tolist(), [0, 0, 0, 1])
        z = np.zeros_like(q)
        jt, _ = p.calculateJacobian(self.body_id, li, list(lp),
                                     q.tolist(), dq.tolist(), z.tolist())
        jt = np.array(jt, dtype=float)
        return normal @ jt  # dof 维向量

    def command_velocities(self, u_cmd):
        """纯运动学模式：速度积分 → resetJointState，不走动力学。"""
        lb = np.concatenate([np.full(self.n_pris, -self.config.base_vel_limit),
                             np.full(self.n_revo, -self.config.dq_limit)])
        u_clip = np.clip(u_cmd, lb, -lb)
        q, _ = self.get_joint_state()
        q_new = q + u_clip * self.config.dt
        for i, ji in enumerate(self.active_joints):
            p.resetJointState(self.body_id, ji, float(q_new[i]), float(u_clip[i]))

    def get_gantry_pos(self):
        """读取龙门吊 3 个移动副当前位置。"""
        st = p.getJointStates(self.body_id, self.prismatic_joints)
        return np.array([s[0] for s in st])


# ============================================================================
#  障碍物 (同 3_12，保留 Sphere / Plate / URDF)
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

    def disable_collision_with(self, robot_bid, nj):
        p.setCollisionFilterPair(robot_bid, self.body_id, -1, -1, enableCollision=0)
        for li in range(nj):
            p.setCollisionFilterPair(robot_bid, self.body_id, li, -1, enableCollision=0)


class SphereObstacle(Obstacle):
    def __init__(self, cfg, scene, ee_pos):
        self._r = cfg.sphere_radius
        dx, dy, dz = cfg.sphere_initial_offset
        self._init = np.array([ee_pos[0]+dx, ee_pos[1]+dy, ee_pos[2]+dz])
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self._r, rgbaColor=cfg.sphere_rgba)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self._r)
        self._bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis, basePosition=self._init.tolist())
        sr = cfg.obstacle_slider_range
        self.sliders = {
            "x": p.addUserDebugParameter("obs_x", ee_pos[0]-sr, ee_pos[0]+sr, float(self._init[0])),
            "y": p.addUserDebugParameter("obs_y", ee_pos[1]-sr, ee_pos[1]+sr, float(self._init[1])),
            "z": p.addUserDebugParameter("obs_z", ee_pos[2]-sr, ee_pos[2]+sr, float(self._init[2])),
        }
    @property
    def body_id(self): return self._bid
    def update_from_slider(self):
        pos = np.array([p.readUserDebugParameter(self.sliders[k]) for k in "xyz"])
        p.resetBasePositionAndOrientation(self._bid, pos.tolist(), [0, 0, 0, 1]); return pos
    def get_position(self):
        return np.array(p.getBasePositionAndOrientation(self._bid)[0], dtype=float)
    def compute_distance(self, pt):
        d = pt - self.get_position(); dist = np.linalg.norm(d)
        if dist < 1e-9: return -self._r, np.array([1, 0, 0.])
        return dist - self._r, d / dist


class PlateObstacle(Obstacle):
    def __init__(self, cfg, scene, ee_pos):
        self._half = np.array(cfg.plate_half_extents, dtype=float)
        dx, dy, dz = cfg.plate_initial_offset
        self._init = np.array([ee_pos[0]+dx, ee_pos[1]+dy, ee_pos[2]+dz])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=self._half.tolist(),
                                  rgbaColor=cfg.plate_rgba, specularColor=[.5]*3)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=self._half.tolist())
        self._bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis, basePosition=self._init.tolist())
        sr = cfg.obstacle_slider_range
        self.sliders = {
            "x": p.addUserDebugParameter("plate_x", ee_pos[0]-sr, ee_pos[0]+sr, float(self._init[0])),
            "y": p.addUserDebugParameter("plate_y", ee_pos[1]-sr, ee_pos[1]+sr, float(self._init[1])),
            "z": p.addUserDebugParameter("plate_z", ee_pos[2]-sr, ee_pos[2]+sr, float(self._init[2])),
        }
    @property
    def body_id(self): return self._bid
    def update_from_slider(self):
        pos = np.array([p.readUserDebugParameter(self.sliders[k]) for k in "xyz"])
        p.resetBasePositionAndOrientation(self._bid, pos.tolist(), [0, 0, 0, 1]); return pos
    def get_position(self):
        return np.array(p.getBasePositionAndOrientation(self._bid)[0], dtype=float)
    def compute_distance(self, pt):
        ctr, orn = p.getBasePositionAndOrientation(self._bid)
        ip, io = p.invertTransform(ctr, orn)
        lp_raw, _ = p.multiplyTransforms(ip, io, pt.tolist(), [0, 0, 0, 1])
        lp = np.array(lp_raw, dtype=float); h = self._half; d = np.abs(lp) - h
        if np.any(d > 0):
            cl = np.clip(lp, -h, h); diff = lp - cl; dist = np.linalg.norm(diff)
            nl = diff/dist if dist > 1e-9 else np.array([1, 0, 0.]); sd = dist
        else:
            fd = h - np.abs(lp); mi = int(np.argmin(fd)); sd = -fd[mi]
            nl = np.zeros(3); nl[mi] = 1.0 if lp[mi] >= 0 else -1.0
        rm = np.array(p.getMatrixFromQuaternion(orn), dtype=float).reshape(3, 3)
        return float(sd), rm @ nl


def create_obstacle(cfg, scene, ee_pos):
    if cfg.obstacle_type == "sphere": return SphereObstacle(cfg, scene, ee_pos)
    elif cfg.obstacle_type == "plate": return PlateObstacle(cfg, scene, ee_pos)
    else: raise ValueError(f"未知障碍物类型: {cfg.obstacle_type}")


# ============================================================================
#  参考轨迹 (同 3_12)
# ============================================================================

class LineSlerpTrajectory:

    def __init__(self, start_pos, start_quat, goal_pos, goal_quat, duration, dt):
        self.start_pos = np.array(start_pos, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.duration = duration; self.dt = dt
        self.linear_velocity = (self.goal_pos - self.start_pos) / max(duration, 1e-6)
        self.slerp = Slerp(np.array([0.0, duration]),
                            Rotation.from_quat(np.vstack([start_quat, goal_quat])))

    def sample(self, t):
        tau = float(np.clip(t, 0, self.duration))
        blend = tau / max(self.duration, 1e-6)
        pos = (1-blend)*self.start_pos + blend*self.goal_pos
        quat = self.slerp([tau]).as_quat()[0]
        if tau >= self.duration:
            return pos, quat, np.zeros(3), np.zeros(3)
        nt = min(self.duration, tau + self.dt)
        nq = self.slerp([nt]).as_quat()[0]
        av = (Rotation.from_quat(nq) * Rotation.from_quat(quat).inv()).as_rotvec() / max(nt-tau, 1e-6)
        return pos, quat, self.linear_velocity.copy(), av

    def reference_points(self, n):
        return [self.sample(t)[0] for t in np.linspace(0, self.duration, n, endpoint=True)]


# ============================================================================
#  控制器抽象基类
# ============================================================================

class Controller(ABC):
    """所有控制器的统一接口。"""

    @abstractmethod
    def solve(
        self, q, dq, ee_pos, ee_quat,
        ref_pos, ref_quat, ref_lin_vel, ref_ang_vel,
        obstacles: list[Obstacle],
        current_time: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        """返回 9 维控制命令 u 和诊断信息 dict。"""
        ...


# ============================================================================
#  控制器 1: 单步 CBF-QP (继承自 3_12)
# ============================================================================

class CBFQPController(Controller):

    def __init__(self, robot: JakaRobot, config: ExperimentConfig, **_kwargs):
        self.robot = robot
        self.config = config
        self.n = robot.total_dof

    def _build_cbf_data(self, q, dq, obstacles):
        A_rows, h_vals = [], []
        for obs in obstacles:
            use_mesh = self.config.use_mesh_cbf and obs.body_id >= 0
            for li in self.robot.cbf_link_indices:
                if use_mesh:
                    cp = self.robot.get_closest_point_to_obstacle(li, obs.body_id)
                    if cp is None: continue
                    spt, sd, n = cp
                    h_vals.append(sd - self.config.safety_margin)
                    A_rows.append(self.robot.get_link_cbf_row_at_point(li, spt, n, q, dq))
                else:
                    lp = self.robot.get_link_origin(li)
                    sd, n = obs.compute_distance(lp)
                    h_vals.append(sd - self.config.safety_margin)
                    A_rows.append(self.robot.get_link_cbf_row(li, n, q, dq))
        return A_rows, h_vals

    def solve(self, q, dq, ee_pos, ee_quat,
              ref_pos, ref_quat, ref_lin_vel, ref_ang_vel,
              obstacles, current_time=0.0):
        pos_err = ref_pos - ee_pos
        rot_err = (Rotation.from_quat(ref_quat)*Rotation.from_quat(ee_quat).inv()).as_rotvec()
        xdot_ref = np.concatenate([
            ref_lin_vel + self.config.position_gain * pos_err,
            ref_ang_vel + self.config.orientation_gain * rot_err])
        dq_nom = self.config.dq_nominal_gain * (self.robot.q_nominal - q)
        J_ee = self.robot.get_ee_jacobian(q, dq)
        A_rows, h_vals = self._build_cbf_data(q, dq, obstacles)
        if not h_vals: h_vals = [1.0]
        m = len(A_rows)
        A = np.array(A_rows) if A_rows else np.zeros((0, self.n))
        b = np.array([self.config.cbf_alpha * hv for hv in h_vals[:m]])

        def objective(x):
            u = x[:self.n]
            return (np.sum((J_ee @ u - xdot_ref)**2)
                    + self.config.nullspace_weight * np.sum((u - dq_nom)**2))
        constraints = []
        for ri in range(m):
            r, rhs = A[ri].copy(), b[ri]
            def ineq(x, r=r, rhs=rhs): return r @ x[:self.n] + rhs
            constraints.append({"type": "ineq", "fun": ineq})
        bounds = ([(-self.config.base_vel_limit, self.config.base_vel_limit)] * self.robot.n_pris
                  + [(-self.config.dq_limit, self.config.dq_limit)] * self.robot.n_revo)

        u_pinv = np.linalg.lstsq(J_ee, xdot_ref, rcond=None)[0]
        lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
        x0 = np.clip(u_pinv, lb, ub)
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 100, "ftol": 1e-6})
        if res.success:
            u_cmd = res.x[:self.n]; status = "cbf_optimal"
        else:
            u_cmd = np.clip(dq_nom, lb, ub); status = f"cbf_fallback"
        return u_cmd, {"min_h": float(np.min(h_vals)), "max_slack": 0.0,
                       "status": status, "tracking_error": float(np.linalg.norm(pos_err))}


# ============================================================================
#  控制器 2: MPC-DCBF (多步预测 + 离散 CBF)
# ============================================================================

class MPCDCBFController(Controller):
    """矩阵化 MPC-DCBF：预构建 QP 矩阵 + 解析梯度 + 热启动 + 重规划间隔。

    加速手段（相比朴素循环版本）：
    1. 将跟踪/控制/平滑代价全部预构建为 H, f 矩阵 → obj = 0.5 xᵀHx + fᵀx
    2. 将离散 CBF 约束预构建为 A_cbf, b_cbf → A_cbf·x + b_cbf ≥ 0
    3. 解析梯度直接传给 SLSQP（消除 40 次有限差分）
    4. 热启动：将上次 MPC 解左移一步作为初始猜测
    5. 重规划间隔：仅每 mpc_replan_steps 步重新求解
    """

    def __init__(self, robot: JakaRobot, config: ExperimentConfig,
                 trajectory: LineSlerpTrajectory):
        self.robot = robot
        self.config = config
        self.n = robot.total_dof  # 9
        self.N = config.N_mpc
        self.trajectory = trajectory
        self._prev_sol: np.ndarray | None = None
        self._cached_u: np.ndarray | None = None
        self._cached_info: dict | None = None
        self._step_count = 0

        single_bnd = ([(-config.base_vel_limit, config.base_vel_limit)] * robot.n_pris
                      + [(-config.dq_limit, config.dq_limit)] * robot.n_revo)
        self._bounds = single_bnd * self.N
        self._lb = np.array([b[0] for b in single_bnd])
        self._ub = np.array([b[1] for b in single_bnd])

    def _build_cbf_data(self, q, dq, obstacles):
        grad_rows, h_vals = [], []
        for obs in obstacles:
            use_mesh = self.config.use_mesh_cbf and obs.body_id >= 0
            for li in self.robot.cbf_link_indices:
                if use_mesh:
                    cp = self.robot.get_closest_point_to_obstacle(li, obs.body_id)
                    if cp is None: continue
                    spt, sd, normal = cp
                    h_vals.append(sd - self.config.safety_margin)
                    grad_rows.append(self.robot.get_link_cbf_row_at_point(
                        li, spt, normal, q, dq))
                else:
                    lp = self.robot.get_link_origin(li)
                    sd, normal = obs.compute_distance(lp)
                    h_vals.append(sd - self.config.safety_margin)
                    grad_rows.append(self.robot.get_link_cbf_row(li, normal, q, dq))
        return grad_rows, h_vals

    def _build_qp(self, ee_pos, J_pos, ref_positions, grad_rows, h_vals):
        """预构建 QP 矩阵，所有代价/约束均为解析形式。"""
        n, N = self.n, self.N
        mdt = self.config.mpc_dt
        cfg = self.config
        dim = n * N

        # ---- 代价 H, f ----
        JtJ = mdt**2 * (J_pos.T @ J_pos)  # n×n

        # 跟踪代价: H_track block(i,j) = 2·w_t·(N-max(i,j))·JᵀJ
        idx = np.arange(N)
        W = (N - np.maximum(idx[:, None], idx[None, :])).astype(float)
        H = 2.0 * cfg.mpc_tracking_weight * np.kron(W, JtJ)

        # 跟踪 f: 每个块 j 是 2·w_t·mdt·Jᵀ·Σ_{k≥j} (p_ee₀ − p_ref_k)
        c_vecs = np.array([ee_pos - ref_positions[k] for k in range(N)])  # N×3
        c_suffix = np.cumsum(c_vecs[::-1], axis=0)[::-1]  # N×3, suffix sums
        f = 2.0 * cfg.mpc_tracking_weight * mdt * (c_suffix @ J_pos).ravel()

        # 控制量代价
        H += 2.0 * cfg.mpc_control_weight * np.eye(dim)

        # 平滑代价: ||u_k − u_{k-1}||²  → D^T D 三对角块
        if N > 1:
            D = np.zeros((N - 1, N))
            for k in range(N - 1):
                D[k, k] = -1.0; D[k, k + 1] = 1.0
            H += 2.0 * cfg.mpc_smooth_weight * np.kron(D.T @ D, np.eye(n))

        # ---- CBF 约束: A_cbf · x + b_cbf ≥ 0 ----
        n_cbf = len(grad_rows)
        gamma = cfg.gamma_dcbf
        T = np.tril(gamma * np.ones((N, N)))
        np.fill_diagonal(T, 1.0)
        T_mdt = mdt * T

        A_cbf = np.zeros((n_cbf * N, dim))
        b_cbf = np.zeros(n_cbf * N)
        for ci in range(n_cbf):
            g = grad_rows[ci]  # (n,)
            A_cbf[ci * N:(ci + 1) * N, :] = np.kron(T_mdt, g.reshape(1, -1))
            b_cbf[ci * N:(ci + 1) * N] = gamma * h_vals[ci]

        return H, f, A_cbf, b_cbf

    def solve(self, q, dq, ee_pos, ee_quat,
              ref_pos, ref_quat, ref_lin_vel, ref_ang_vel,
              obstacles, current_time=0.0):

        self._step_count += 1
        if (self._cached_u is not None
                and self._step_count % self.config.mpc_replan_steps != 0):
            return self._cached_u, self._cached_info

        n, N = self.n, self.N
        cfg = self.config

        J_full = self.robot.get_ee_jacobian(q, dq)
        J_pos = J_full[:3]

        ref_positions = []
        for k in range(1, N + 1):
            pk, _, _, _ = self.trajectory.sample(
                min(current_time + k * cfg.mpc_dt, cfg.trajectory_duration))
            ref_positions.append(pk)

        grad_rows, h_vals = self._build_cbf_data(q, dq, obstacles)
        if not h_vals:
            h_vals = [1.0]

        H, f_vec, A_cbf, b_cbf = self._build_qp(
            ee_pos, J_pos, ref_positions, grad_rows, h_vals)

        constraints = [{"type": "ineq",
                        "fun": lambda x: A_cbf @ x + b_cbf,
                        "jac": lambda x: A_cbf}]

        # 热启动
        if self._prev_sol is not None:
            x0 = np.empty(n * N)
            x0[:(N - 1) * n] = self._prev_sol[n:]
            x0[(N - 1) * n:] = self._prev_sol[(N - 1) * n:]
        else:
            pos_err = ref_pos - ee_pos
            rot_err = (Rotation.from_quat(ref_quat)
                       * Rotation.from_quat(ee_quat).inv()).as_rotvec()
            xdot = np.concatenate([ref_lin_vel + cfg.position_gain * pos_err,
                                    ref_ang_vel + cfg.orientation_gain * rot_err])
            u0 = np.clip(np.linalg.lstsq(J_full, xdot, rcond=None)[0],
                         self._lb, self._ub)
            x0 = np.tile(u0, N)

        res = minimize(lambda x: 0.5 * x @ H @ x + f_vec @ x,
                       x0, method="SLSQP",
                       jac=lambda x: H @ x + f_vec,
                       bounds=self._bounds, constraints=constraints,
                       options={"maxiter": 50, "ftol": 1e-6})

        if res.success:
            self._prev_sol = res.x.copy()
            u_cmd = res.x[:n]
            status = "mpc_optimal"
        else:
            u_cmd = np.clip(x0[:n], self._lb, self._ub)
            status = "mpc_fallback"

        pos_err = ref_pos - ee_pos
        info = {"min_h": float(np.min(h_vals)), "max_slack": 0.0,
                "status": status,
                "tracking_error": float(np.linalg.norm(pos_err))}
        self._cached_u = u_cmd
        self._cached_info = info
        return u_cmd, info


# ============================================================================
#  控制器工厂
# ============================================================================

def create_controller(robot, config, trajectory) -> Controller:
    if config.controller_type == "cbf_qp":
        return CBFQPController(robot, config)
    elif config.controller_type == "mpc_dcbf":
        return MPCDCBFController(robot, config, trajectory)
    else:
        raise ValueError(f"未知控制器类型: {config.controller_type}")


# ============================================================================
#  实验主类
# ============================================================================

class AvoidanceExperiment:

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.scene = SimulationScene(config)
        self.robot = JakaRobot(config, self.scene)

        ee_pos_init, _ = self.robot.get_ee_pose()
        obs = create_obstacle(config, self.scene, ee_pos_init)
        obs.disable_collision_with(self.robot.body_id, self.robot.num_joints)
        self.obstacles: list[Obstacle] = [obs]

        obs_center = obs.update_from_slider()
        start_pos = np.array([obs_center[0]-config.line_half_span,
                              obs_center[1]+config.line_bias_y,
                              obs_center[2]+config.line_bias_z])
        goal_pos = np.array([obs_center[0]+config.line_half_span,
                             obs_center[1]+config.line_bias_y,
                             obs_center[2]+config.line_bias_z])
        sq = p.getQuaternionFromEuler([math.radians(v) for v in config.start_euler_deg])
        gq = p.getQuaternionFromEuler([math.radians(v) for v in config.goal_euler_deg])

        self.robot.reset_to_pose(start_pos, sq)
        ee_pos, _ = self.robot.get_ee_pose()
        self.ee_marker = self.scene.create_marker(0.012, (0.1, 0.9, 0.2, 0.9), ee_pos)
        self.ref_marker = self.scene.create_marker(0.012, (0.95, 0.2, 0.2, 0.9), start_pos.tolist())

        self.trajectory = LineSlerpTrajectory(
            start_pos, np.array(sq, dtype=float),
            goal_pos, np.array(gq, dtype=float),
            config.trajectory_duration, config.dt)
        self.scene.draw_polyline(
            self.trajectory.reference_points(config.reference_samples),
            color=[0.85, 0.1, 0.1], width=1.2)

        self.controller = create_controller(self.robot, config, self.trajectory)
        self.prev_ee = ee_pos.copy()
        self.sim_step = 0
        print(f"===== 避障实验 (控制器: {config.controller_type}, "
              f"障碍: {config.obstacle_type}) =====")

    def _update_visuals(self, ee_pos, ref_pos, info):
        self.scene.update_marker(self.ee_marker, ee_pos.tolist())
        self.scene.update_marker(self.ref_marker, ref_pos.tolist())
        if np.linalg.norm(ee_pos - self.prev_ee) > 1e-3:
            p.addUserDebugLine(self.prev_ee.tolist(), ee_pos.tolist(), [0.1, 0.8, 0.2], lineWidth=1.5)
            self.prev_ee = ee_pos.copy()
        self.scene.update_status(
            f"step={self.sim_step}  "
            f"err={info['tracking_error']*1000:.1f}mm  "
            f"h={info['min_h']*1000:.1f}mm  "
            f"{info['status']}")

    def _solve_step(self, q, dq, ee_pos, ee_quat, ref_pos, ref_quat, ref_lv, ref_av, t):
        for obs in self.obstacles:
            obs.update_from_slider()
        return self.controller.solve(
            q, dq, ee_pos, ee_quat,
            ref_pos, ref_quat, ref_lv, ref_av,
            self.obstacles, current_time=t)

    def run(self):
        total_steps = int((self.config.trajectory_duration + self.config.hold_duration) / self.config.dt)
        video_frames = []
        rec_every = (max(1, int(round((1/self.config.dt)/self.config.video_fps)))
                     if self.config.record_video else 0)
        try:
            while p.isConnected() and self.sim_step < total_steps:
                t = self.sim_step * self.config.dt
                q, dq = self.robot.get_joint_state()
                ee_pos, ee_quat = self.robot.get_ee_pose()
                ref = self.trajectory.sample(min(t, self.config.trajectory_duration))
                u_cmd, info = self._solve_step(q, dq, ee_pos, ee_quat, *ref, t)

                aq = self.config.q_nominal_tracking
                self.robot.q_nominal = (1-aq)*self.robot.q_nominal + aq*q
                self.robot.command_velocities(u_cmd)

                p.stepSimulation()
                self._update_visuals(ee_pos, ref[0], info)
                if self.config.record_video and self.sim_step % rec_every == 0:
                    video_frames.append(self.scene.capture_frame(
                        self.config.video_width, self.config.video_height))
                if self.sim_step % self.config.print_every == 0:
                    gp = self.robot.get_gantry_pos()
                    print(f"[step {self.sim_step:4d}] "
                          f"gantry=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                          f"err={info['tracking_error']*1000:.1f}mm "
                          f"h={info['min_h']*1000:.1f}mm "
                          f"{info['status']}")
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
                ref = self.trajectory.sample(self.config.trajectory_duration)
                u_cmd, info = self._solve_step(q, dq, ee_pos, ee_quat, *ref,
                                               self.config.trajectory_duration)
                self.robot.command_velocities(u_cmd)
                p.stepSimulation()
                self._update_visuals(ee_pos, ref[0], info)
                time.sleep(1/60)
        except KeyboardInterrupt:
            print("\n用户中断。")
        finally:
            if p.isConnected(): p.disconnect()
            print(f"仿真结束，共 {self.sim_step} 步。")


def main():
    AvoidanceExperiment(ExperimentConfig()).run()


if __name__ == "__main__":
    main()
