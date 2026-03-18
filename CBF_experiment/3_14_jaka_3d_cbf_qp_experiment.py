"""3_14 版本：9 自由度龙门吊 + 倒挂机械臂。

与 3_13 的差异：
1. 底盘从 2-DOF 移动底座改为 3-DOF 龙门吊 (X/Y/Z 平移)；
2. 机械臂倒挂安装，底座接在龙门吊第三轴末端平台下方；
3. 总自由度 = 3 (龙门吊) + 6 (机械臂) = 9。

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
        r"C:\Users\12049\OneDrive\Desktop\Zu 7.SLDASM\urdf\Zu 7.SLDASM.urdf"
    )
    dt: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # ---- 龙门吊参数 ----
    gantry_initial_pos: tuple[float, float, float] = (0.0, 0.0, 1.2)
    gantry_platform_half: tuple[float, float, float] = (0.15, 0.15, 0.015)
    gantry_rail_height: float = 1.5          # 龙门吊横梁高度
    gantry_pillar_half: tuple[float, float, float] = (0.03, 0.03, 0.75)
    gantry_span_x: float = 0.8              # 两排立柱 X 方向间距
    gantry_span_y: float = 0.8              # 两排立柱 Y 方向间距

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
    sphere_initial_offset: tuple[float, float, float] = (0.15, 0.15, 0.55)
    plate_half_extents: tuple[float, float, float] = (0.12, 0.08, 0.004)
    plate_rgba: tuple[float, float, float, float] = (0.30, 0.55, 0.85, 0.80)
    plate_initial_offset: tuple[float, float, float] = (0.15, 0.15, 0.45)
    workpiece_urdf_path: str = "workpiece_T.urdf"
    workpiece_position: tuple[float, float, float] = (0.30, 0.35, 0.0)
    workpiece_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    obstacle_slider_y_min: float = -0.30
    obstacle_slider_y_max: float = 0.80

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
    mpc_control_weight: float = 4
    mpc_smooth_weight: float = 3
    mpc_replan_steps: int = 6

    print_every: int = 120
    reference_samples: int = 80


# ============================================================================
#  仿真场景 — 龙门吊
# ============================================================================

class SimulationScene:

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*config.gravity)
        p.setTimeStep(config.dt)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(rgbBackground=[0.82, 0.87, 0.92])
        p.configureDebugVisualizer(lightPosition=[1.5, -1.5, 2.5])
        p.resetDebugVisualizerCamera(
            cameraDistance=config.camera_distance, cameraYaw=config.camera_yaw,
            cameraPitch=config.camera_pitch, cameraTargetPosition=config.camera_target)
        self.gantry_parts: list[tuple[int, np.ndarray, tuple]] = []
        self.reference_height = self._build_environment()
        self._draw_axes()
        self.status_text_id = None

    def _build_environment(self) -> float:
        plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.6, 0.6, 0.6, 1.0])

        cfg = self.config
        rh = cfg.gantry_rail_height
        ph = cfg.gantry_pillar_half
        sx, sy = cfg.gantry_span_x / 2, cfg.gantry_span_y / 2

        # 四根立柱
        pillar_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=list(ph),
            rgbaColor=[0.35, 0.35, 0.38, 1], specularColor=[0.3]*3)
        for cx, cy in [(-sx, -sy), (-sx, sy), (sx, -sy), (sx, sy)]:
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=pillar_vis,
                              basePosition=[cx, cy, ph[2]])

        # 顶部横梁 (X 方向两根)
        beam_hx = cfg.gantry_span_x / 2 + ph[0]
        beam_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[beam_hx, ph[1], ph[1]],
            rgbaColor=[0.45, 0.45, 0.48, 1], specularColor=[0.3]*3)
        for cy in [-sy, sy]:
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=beam_vis,
                              basePosition=[0, cy, rh])

        # 龙门吊移动平台 (跟随机器人运动)
        plat_h = cfg.gantry_platform_half
        plat_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=list(plat_h),
            rgbaColor=[0.82, 0.84, 0.88, 1], specularColor=[0.5]*3)
        gx, gy, gz = cfg.gantry_initial_pos
        plat_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=plat_vis,
            basePosition=[gx, gy, gz])
        self.gantry_parts.append(
            (plat_id, np.array([0, 0, 0], dtype=float), (0, 0, 0, 1)))

        return 0.0  # 地面参考高度

    def update_gantry_pose(self, gantry_xyz):
        pos = np.asarray(gantry_xyz, dtype=float).flatten()[:3]
        for bid, offset, lq in self.gantry_parts:
            p.resetBasePositionAndOrientation(
                bid, (pos + offset).tolist(), lq)

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


# ============================================================================
#  机器人 — 龙门吊 3-DOF + 倒挂 6-DOF 机械臂 = 9-DOF
# ============================================================================

class JakaRobot:

    INVERTED_QUAT = tuple(p.getQuaternionFromEuler([math.pi, 0, 0]))

    def __init__(self, config: ExperimentConfig, scene: SimulationScene):
        self.config = config
        self.scene = scene
        gx, gy, gz = config.gantry_initial_pos
        self.body_id = p.loadURDF(
            config.urdf_path,
            basePosition=[gx, gy, gz],
            baseOrientation=self.INVERTED_QUAT,
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self._set_link_colors()
        self.num_joints = p.getNumJoints(self.body_id)
        self.active_joints = [i for i in range(self.num_joints)
                              if p.getJointInfo(self.body_id, i)[2] == p.JOINT_REVOLUTE]
        self.dof = len(self.active_joints)
        self.ee_link_index = self.active_joints[-1]
        self.cbf_link_indices = list(self.active_joints)
        self.base_pos = np.array([gx, gy, gz], dtype=float)
        self.total_dof = 3 + self.dof  # 9
        for ji in self.active_joints:
            p.changeDynamics(self.body_id, ji, linearDamping=0, angularDamping=0)
            p.setJointMotorControl2(self.body_id, ji, p.VELOCITY_CONTROL, force=0)
        self.q_nominal = np.zeros(self.dof)

    def move_base(self, v_gantry, dt):
        """移动龙门吊 3 轴 (x, y, z)。"""
        v = np.clip(np.asarray(v_gantry, dtype=float).flatten()[:3],
                     -self.config.base_vel_limit, self.config.base_vel_limit)
        self.base_pos += v * dt
        p.resetBasePositionAndOrientation(
            self.body_id,
            self.base_pos.tolist(),
            self.INVERTED_QUAT)
        self.scene.update_gantry_pose(self.base_pos)

    def _set_link_colors(self):
        for li, rgba in {-1:[.15]*3+[1], 0:[.92]*3+[1], 1:[.12,.46,.70,1],
                          2:[.92]*3+[1], 3:[.12,.46,.70,1], 4:[.92]*3+[1], 5:[.15]*3+[1]}.items():
            p.changeVisualShape(self.body_id, li, rgbaColor=rgba, specularColor=[.6]*3)

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
        a = p.calculateInverseKinematics(
            self.body_id, self.ee_link_index, tpos, tquat,
            lowerLimits=[-6.28]*self.dof, upperLimits=[6.28]*self.dof,
            jointRanges=[12.56]*self.dof, restPoses=[0.0]*self.dof,
            maxNumIterations=200, residualThreshold=1e-6)
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

    def get_augmented_ee_jacobian(self, q, dq):
        jt, jr = self.get_link_jacobian(self.ee_link_index, q, dq)
        base = np.vstack([np.eye(3, dtype=float), np.zeros((3, 3))])
        return np.hstack([base, np.vstack([jt, jr])])

    def get_link_cbf_row_aug(self, li, normal, q, dq):
        jt, _ = self.get_link_jacobian(li, q, dq)
        return np.concatenate([normal[:3], normal @ jt])

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

    def get_link_cbf_row_aug_at_point(self, li, wpt, normal, q, dq):
        ls = p.getLinkState(self.body_id, li, computeForwardKinematics=True)
        ip, io = p.invertTransform(ls[4], ls[5])
        lp, _ = p.multiplyTransforms(ip, io, wpt.tolist(), [0, 0, 0, 1])
        z = np.zeros_like(q)
        jt, _ = p.calculateJacobian(self.body_id, li, list(lp),
                                     q.tolist(), dq.tolist(), z.tolist())
        jt = np.array(jt, dtype=float)
        return np.concatenate([normal[:3], normal @ jt])

    def command_joint_velocities(self, dq_cmd):
        dq_clip = np.clip(dq_cmd, -self.config.dq_limit, self.config.dq_limit)
        p.setJointMotorControlArray(self.body_id, self.active_joints, p.VELOCITY_CONTROL,
                                    targetVelocities=dq_clip.tolist(),
                                    forces=[self.config.ee_force_limit]*self.dof)


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
    def __init__(self, cfg, scene):
        self._r = cfg.sphere_radius
        x, y, z = cfg.sphere_initial_offset
        self._init = np.array([x, y, scene.reference_height+z])
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self._r, rgbaColor=cfg.sphere_rgba)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self._r)
        self._bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis, basePosition=self._init.tolist())
        th = scene.reference_height
        self.sliders = {
            "x": p.addUserDebugParameter("obs_x", -0.10, 0.60, float(self._init[0])),
            "y": p.addUserDebugParameter("obs_y", cfg.obstacle_slider_y_min,
                                         cfg.obstacle_slider_y_max, float(self._init[1])),
            "z": p.addUserDebugParameter("obs_z", th+0.08, th+0.42, float(self._init[2])),
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
    def __init__(self, cfg, scene):
        self._half = np.array(cfg.plate_half_extents, dtype=float)
        x, y, z = cfg.plate_initial_offset
        self._init = np.array([x, y, scene.reference_height+z])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=self._half.tolist(),
                                  rgbaColor=cfg.plate_rgba, specularColor=[.5]*3)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=self._half.tolist())
        self._bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis, basePosition=self._init.tolist())
        th = scene.reference_height
        self.sliders = {
            "x": p.addUserDebugParameter("plate_x", -0.10, 0.60, float(self._init[0])),
            "y": p.addUserDebugParameter("plate_y", cfg.obstacle_slider_y_min,
                                         cfg.obstacle_slider_y_max, float(self._init[1])),
            "z": p.addUserDebugParameter("plate_z", th+0.08, th+0.42, float(self._init[2])),
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


def create_obstacle(cfg, scene):
    if cfg.obstacle_type == "sphere": return SphereObstacle(cfg, scene)
    elif cfg.obstacle_type == "plate": return PlateObstacle(cfg, scene)
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
        """返回 8 维控制命令 u 和诊断信息 dict。"""
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
                    A_rows.append(self.robot.get_link_cbf_row_aug_at_point(li, spt, n, q, dq))
                else:
                    lp = self.robot.get_link_origin(li)
                    sd, n = obs.compute_distance(lp)
                    h_vals.append(sd - self.config.safety_margin)
                    A_rows.append(self.robot.get_link_cbf_row_aug(li, n, q, dq))
        return A_rows, h_vals

    def solve(self, q, dq, ee_pos, ee_quat,
              ref_pos, ref_quat, ref_lin_vel, ref_ang_vel,
              obstacles, current_time=0.0):
        pos_err = ref_pos - ee_pos
        rot_err = (Rotation.from_quat(ref_quat)*Rotation.from_quat(ee_quat).inv()).as_rotvec()
        xdot_ref = np.concatenate([
            ref_lin_vel + self.config.position_gain * pos_err,
            ref_ang_vel + self.config.orientation_gain * rot_err])
        dq_nom = np.concatenate([np.zeros(3),
                                  self.config.dq_nominal_gain * (self.robot.q_nominal - q)])
        J_ee = self.robot.get_augmented_ee_jacobian(q, dq)
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
        bounds = ([(-self.config.base_vel_limit, self.config.base_vel_limit)]*3
                  + [(-self.config.dq_limit, self.config.dq_limit)]*self.robot.dof)

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
        self.n = robot.total_dof  # 8
        self.N = config.N_mpc
        self.trajectory = trajectory
        self._prev_sol: np.ndarray | None = None
        self._cached_u: np.ndarray | None = None
        self._cached_info: dict | None = None
        self._step_count = 0

        single_bnd = ([(-config.base_vel_limit, config.base_vel_limit)] * 3
                      + [(-config.dq_limit, config.dq_limit)] * robot.dof)
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
                    grad_rows.append(self.robot.get_link_cbf_row_aug_at_point(
                        li, spt, normal, q, dq))
                else:
                    lp = self.robot.get_link_origin(li)
                    sd, normal = obs.compute_distance(lp)
                    h_vals.append(sd - self.config.safety_margin)
                    grad_rows.append(self.robot.get_link_cbf_row_aug(li, normal, q, dq))
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

        J_full = self.robot.get_augmented_ee_jacobian(q, dq)
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

        obs = create_obstacle(config, self.scene)
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
                self.robot.move_base(u_cmd[:3], self.config.dt)
                self.robot.command_joint_velocities(u_cmd[3:])

                p.stepSimulation()
                self._update_visuals(ee_pos, ref[0], info)
                if self.config.record_video and self.sim_step % rec_every == 0:
                    video_frames.append(self.scene.capture_frame(
                        self.config.video_width, self.config.video_height))
                if self.sim_step % self.config.print_every == 0:
                    print(f"[step {self.sim_step:4d}] "
                          f"gantry=({self.robot.base_pos[0]:.3f},{self.robot.base_pos[1]:.3f},{self.robot.base_pos[2]:.3f}) "
                          f"err={info['tracking_error']*1000:.1f}mm "
                          f"h={info['min_h']*1000:.1f}mm "
                          f"{info['status']}")
                self.sim_step += 1
                time.sleep(self.config.dt)

            self.robot.move_base(np.zeros(3), self.config.dt)
            self.robot.command_joint_velocities(np.zeros(self.robot.dof))
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
                self.robot.move_base(u_cmd[:3], 1/60)
                self.robot.command_joint_velocities(u_cmd[3:])
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
