import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
    """集中管理实验配置，避免控制逻辑里散落魔法数字。"""

    urdf_path: str = (
        r"C:\Users\12049\OneDrive\Desktop\Zu 7.SLDASM\urdf\Zu 7.SLDASM.urdf"
    )
    dt: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # 移动小车外观尺寸。
    cart_body_half: tuple[float, float, float] = (0.24, 0.18, 0.05)
    cart_deck_half: tuple[float, float, float] = (0.18, 0.14, 0.015)
    wheel_radius: float = 0.055
    wheel_width: float = 0.03

    # 相机。
    camera_distance: float = 1.0
    camera_yaw: float = -225.0
    camera_pitch: float = -15
    camera_target: tuple[float, float, float] = (0.15, 0.20, 0.45)

    # 录像。
    record_video: bool = False
    video_output_path: str = "cbf_experiment.mp4"
    video_fps: int = 30
    video_width: int = 960
    video_height: int = 720

    # 直线参考轨迹。
    line_half_span: float = 0.14
    line_bias_y: float = 0.08
    line_bias_z: float = 0.0
    trajectory_duration: float = 7.0
    hold_duration: float = 6

    # ---- 障碍物选择 ----
    # "sphere" / "plate"，决定使用哪个子类。
    obstacle_type: str = "plate"

    # 球障碍参数。
    sphere_radius: float = 0.06
    sphere_rgba: tuple[float, float, float, float] = (1.0, 0.35, 0.2, 0.75)
    sphere_initial_offset: tuple[float, float, float] = (0.32, 0.4, 0.3)

    # 平板障碍参数。
    plate_half_extents: tuple[float, float, float] = (0.12, 0.08, 0.004)
    plate_rgba: tuple[float, float, float, float] = (0.30, 0.55, 0.85, 0.80)
    plate_initial_offset: tuple[float, float, float] = (0.32, 0.4, 0.25)

    # 障碍物通用滑块范围。
    obstacle_slider_y_min: float = -0.30
    obstacle_slider_y_max: float = 0.80

    # 起终点姿态。
    start_euler_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)
    goal_euler_deg: tuple[float, float, float] = (140.0, 0, 0)

    # 执行器。
    ee_force_limit: float = 250.0
    dq_limit: float = 1.0
    dq_nominal_gain: float = 0.25
    base_vel_limit: float = 0.4

    # QP / CBF。
    position_gain: float = 8
    orientation_gain: float = 0
    nullspace_weight: float = 0.1
    slack_weight: float = 200000.0
    use_slack: bool = False
    use_mesh_cbf: bool = True
    cbf_alpha: float = 2
    safety_margin: float = 0.02

    print_every: int = 120
    reference_samples: int = 80


# ============================================================================
#  仿真场景
# ============================================================================

class SimulationScene:
    """负责 PyBullet GUI、移动小车环境与调试可视化。"""

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
            cameraDistance=config.camera_distance,
            cameraYaw=config.camera_yaw,
            cameraPitch=config.camera_pitch,
            cameraTargetPosition=config.camera_target,
        )

        self.cart_parts: list[tuple[int, np.ndarray, tuple]] = []
        self.table_height = self._build_environment()
        self._draw_axes()
        self.status_text_id = None

    def _build_environment(self) -> float:
        plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.6, 0.6, 0.6, 1.0])

        bh = self.config.cart_body_half
        dh = self.config.cart_deck_half
        wr = self.config.wheel_radius
        ww = self.config.wheel_width

        body_z = wr + bh[2]
        deck_z = wr + 2.0 * bh[2] + dh[2]
        top_height = wr + 2.0 * bh[2] + 2.0 * dh[2]

        body_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=bh,
                                       rgbaColor=[0.18, 0.22, 0.28, 1.0],
                                       specularColor=[0.6, 0.6, 0.6])
        bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                baseVisualShapeIndex=body_vis,
                                basePosition=[0.0, 0.0, body_z])
        self.cart_parts.append((bid, np.array([0, 0, body_z], dtype=float), (0, 0, 0, 1)))

        deck_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=dh,
                                       rgbaColor=[0.82, 0.84, 0.88, 1.0],
                                       specularColor=[0.7, 0.7, 0.7])
        did = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                baseVisualShapeIndex=deck_vis,
                                basePosition=[0.0, 0.0, deck_z])
        self.cart_parts.append((did, np.array([0, 0, deck_z], dtype=float), (0, 0, 0, 1)))

        wheel_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=wr, length=ww,
                                        rgbaColor=[0.08, 0.08, 0.08, 1.0],
                                        specularColor=[0.2, 0.2, 0.2])
        wq = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
        wx = bh[0] * 0.72
        wy = bh[1] + ww * 0.35
        for lp in (np.array([wx, wy, wr]), np.array([wx, -wy, wr]),
                    np.array([-wx, wy, wr]), np.array([-wx, -wy, wr])):
            wid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                    baseVisualShapeIndex=wheel_vis,
                                    basePosition=lp.tolist(), baseOrientation=wq)
            self.cart_parts.append((wid, lp.astype(float), wq))
        return top_height

    def update_cart_pose(self, base_xy: np.ndarray) -> None:
        bxy = np.asarray(base_xy, dtype=float).flatten()[:2]
        for bid, lp, lq in self.cart_parts:
            p.resetBasePositionAndOrientation(
                bid, [bxy[0] + lp[0], bxy[1] + lp[1], lp[2]], lq)

    def _draw_axes(self) -> None:
        al = 0.12
        o = [0.0, 0.0, self.table_height + 0.001]
        p.addUserDebugLine(o, [al, 0, o[2]], [1, 0, 0], lineWidth=2)
        p.addUserDebugLine(o, [0, al, o[2]], [0, 0.8, 0], lineWidth=2)
        p.addUserDebugLine(o, [0, 0, o[2] + al], [0, 0, 1], lineWidth=2)
        p.addUserDebugText("X", [al + 0.02, 0, o[2]], [1, 0, 0], textSize=1)
        p.addUserDebugText("Y", [0, al + 0.02, o[2]], [0, 0.8, 0], textSize=1)
        p.addUserDebugText("Z", [0, 0, o[2] + al + 0.02], [0, 0, 1], textSize=1)

    def create_marker(self, radius, color, pos):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos)

    def update_marker(self, body_id, pos):
        p.resetBasePositionAndOrientation(body_id, pos, [0, 0, 0, 1])

    def draw_polyline(self, points, color, width=1.5):
        for i in range(len(points) - 1):
            p.addUserDebugLine(points[i].tolist(), points[i + 1].tolist(), color, lineWidth=width)

    def update_status(self, text):
        self.status_text_id = p.addUserDebugText(
            text, [0.02, -0.26, self.table_height + 0.52], [0.1, 0.1, 0.1],
            textSize=1.2,
            replaceItemUniqueId=self.status_text_id if self.status_text_id else -1)

    def capture_frame(self, width, height):
        _, _, rgb, _, _ = p.getCameraImage(width, height)
        return np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


# ============================================================================
#  机器人
# ============================================================================

class JakaRobot:
    """封装 JAKA 机械臂的模型加载、状态读取、FK/IK 和雅可比计算。"""

    def __init__(self, config: ExperimentConfig, scene: SimulationScene):
        self.config = config
        self.scene = scene
        self.body_id = p.loadURDF(
            config.urdf_path,
            basePosition=[0.0, 0.0, scene.table_height],
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )
        self._set_link_colors()

        self.num_joints = p.getNumJoints(self.body_id)
        self.active_joints = [
            i for i in range(self.num_joints)
            if p.getJointInfo(self.body_id, i)[2] == p.JOINT_REVOLUTE
        ]
        self.dof = len(self.active_joints)
        self.ee_link_index = self.active_joints[-1]
        self.cbf_link_indices = list(self.active_joints)
        self.base_pos = np.array([0.0, 0.0], dtype=float)
        self.total_dof = 2 + self.dof

        print(f"活动关节索引: {self.active_joints}")
        print(f"末端执行器连杆索引: {self.ee_link_index}")

        for ji in self.active_joints:
            p.changeDynamics(self.body_id, ji, linearDamping=0, angularDamping=0)
            p.setJointMotorControl2(self.body_id, ji, p.VELOCITY_CONTROL, force=0)

        self.q_nominal = np.zeros(self.dof)

    def move_base(self, v_base, dt):
        v = np.clip(np.asarray(v_base, dtype=float).flatten()[:2],
                     -self.config.base_vel_limit, self.config.base_vel_limit)
        self.base_pos += v * dt
        bz = p.getBasePositionAndOrientation(self.body_id)[0][2]
        p.resetBasePositionAndOrientation(
            self.body_id, [self.base_pos[0], self.base_pos[1], bz], [0, 0, 0, 1])
        self.scene.update_cart_pose(self.base_pos)

    def _set_link_colors(self):
        colors = {-1: [0.15]*3+[1], 0: [0.92]*3+[1], 1: [0.12,0.46,0.70,1],
                  2: [0.92]*3+[1], 3: [0.12,0.46,0.70,1], 4: [0.92]*3+[1],
                  5: [0.15]*3+[1]}
        for li, rgba in colors.items():
            p.changeVisualShape(self.body_id, li, rgbaColor=rgba, specularColor=[0.6]*3)

    def get_joint_state(self):
        st = p.getJointStates(self.body_id, self.active_joints)
        return np.array([s[0] for s in st]), np.array([s[1] for s in st])

    def get_ee_pose(self):
        s = p.getLinkState(self.body_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(s[4], dtype=float), np.array(s[5], dtype=float)

    def get_link_origin(self, link_idx):
        s = p.getLinkState(self.body_id, link_idx, computeForwardKinematics=True)
        return np.array(s[4], dtype=float)

    def calculate_ik(self, target_pos, target_quat):
        ll = [-6.28] * self.dof
        ul = [6.28] * self.dof
        jr = [12.56] * self.dof
        rp = [0.0] * self.dof
        a = p.calculateInverseKinematics(
            self.body_id, self.ee_link_index, target_pos, target_quat,
            lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp,
            maxNumIterations=200, residualThreshold=1e-6)
        return np.array(a[:self.dof], dtype=float)

    def reset_to_pose(self, target_pos, target_quat):
        qt = self.calculate_ik(target_pos, target_quat)
        for i, ji in enumerate(self.active_joints):
            p.resetJointState(self.body_id, ji, qt[i])
        self.q_nominal = qt.copy()

    def get_link_jacobian(self, link_idx, q, dq):
        z = np.zeros_like(q)
        jt, jr = p.calculateJacobian(self.body_id, link_idx, [0, 0, 0],
                                      q.tolist(), dq.tolist(), z.tolist())
        return np.array(jt, dtype=float), np.array(jr, dtype=float)

    def get_ee_jacobian(self, q, dq):
        jt, jr = self.get_link_jacobian(self.ee_link_index, q, dq)
        return np.vstack([jt, jr])

    def get_augmented_ee_jacobian(self, q, dq):
        jt, jr = self.get_link_jacobian(self.ee_link_index, q, dq)
        base_block = np.vstack([
            np.array([[1, 0], [0, 1], [0, 0]], dtype=float),
            np.zeros((3, 2), dtype=float),
        ])
        return np.hstack([base_block, np.vstack([jt, jr])])

    def get_link_cbf_row_aug(self, link_idx, normal, q, dq):
        jt, _ = self.get_link_jacobian(link_idx, q, dq)
        return np.concatenate([normal[:2], normal @ jt])

    def get_closest_point_to_obstacle(self, link_idx, obstacle_body_id, max_dist=1.0):
        contacts = p.getClosestPoints(self.body_id, obstacle_body_id, max_dist, linkIndexA=link_idx)
        if not contacts:
            return None
        best = min(contacts, key=lambda c: c[8])
        pos = np.array(best[5], dtype=float)
        dist = float(best[8])
        n = np.array(best[7], dtype=float)
        nl = np.linalg.norm(n)
        n = n / nl if nl > 1e-9 else np.array([1, 0, 0], dtype=float)
        return pos, dist, n

    def get_link_cbf_row_aug_at_point(self, link_idx, world_point, normal, q, dq):
        ls = p.getLinkState(self.body_id, link_idx, computeForwardKinematics=True)
        inv_p, inv_o = p.invertTransform(ls[4], ls[5])
        local_pt, _ = p.multiplyTransforms(inv_p, inv_o, world_point.tolist(), [0, 0, 0, 1])
        z = np.zeros_like(q)
        jt, _ = p.calculateJacobian(self.body_id, link_idx, list(local_pt),
                                     q.tolist(), dq.tolist(), z.tolist())
        jt = np.array(jt, dtype=float)
        return np.concatenate([normal[:2], normal @ jt])

    def command_joint_velocities(self, dq_cmd):
        dq_clip = np.clip(dq_cmd, -self.config.dq_limit, self.config.dq_limit)
        p.setJointMotorControlArray(
            self.body_id, self.active_joints, p.VELOCITY_CONTROL,
            targetVelocities=dq_clip.tolist(),
            forces=[self.config.ee_force_limit] * self.dof)


# ============================================================================
#  障碍物体系：抽象基类 + 具体子类
# ============================================================================

class Obstacle(ABC):
    """障碍物统一接口。

    所有子类必须:
    1. 创建带碰撞形状的 PyBullet body（供 getClosestPoints 使用）
    2. 提供 GUI 滑块来实时调参
    3. 提供 body_id / 位置 等 CBF-QP 所需信息
    4. 实现 compute_distance —— 用各自几何形状的解析公式精确计算
       点到障碍物表面的有符号距离与外法向量
    """

    @property
    @abstractmethod
    def body_id(self) -> int:
        """PyBullet body ID。"""
        ...

    @abstractmethod
    def update_from_slider(self) -> np.ndarray:
        """从 GUI 滑块读取参数并更新位姿，返回当前中心位置。"""
        ...

    @abstractmethod
    def get_position(self) -> np.ndarray:
        """返回障碍物当前世界坐标中心。"""
        ...

    @abstractmethod
    def compute_distance(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        """计算空间中一点到障碍物表面的有符号距离与外法向量。

        返回:
            (signed_distance, outward_normal)
            signed_distance > 0: 点在障碍物外部（安全）
            signed_distance < 0: 点在障碍物内部（穿透）
            outward_normal: 单位向量，从障碍物表面指向查询点方向
        """
        ...

    def disable_collision_with(self, robot_body_id: int, num_joints: int) -> None:
        """禁用与机器人之间的物理碰撞响应，避障完全由 CBF-QP 负责。"""
        p.setCollisionFilterPair(robot_body_id, self.body_id, -1, -1, enableCollision=0)
        for li in range(num_joints):
            p.setCollisionFilterPair(robot_body_id, self.body_id, li, -1, enableCollision=0)


class SphereObstacle(Obstacle):
    """球形障碍物。"""

    def __init__(self, config: ExperimentConfig, scene: SimulationScene):
        self.config = config
        self._radius = config.sphere_radius
        x, y, z = config.sphere_initial_offset
        self._init_pos = np.array([x, y, scene.table_height + z], dtype=float)

        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self._radius, rgbaColor=config.sphere_rgba)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self._radius)
        self._body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                          baseVisualShapeIndex=vis,
                                          basePosition=self._init_pos.tolist())
        th = scene.table_height
        self.sliders = {
            "x": p.addUserDebugParameter("obs_x", -0.10, 0.60, float(self._init_pos[0])),
            "y": p.addUserDebugParameter("obs_y", config.obstacle_slider_y_min,
                                         config.obstacle_slider_y_max, float(self._init_pos[1])),
            "z": p.addUserDebugParameter("obs_z", th + 0.08, th + 0.42, float(self._init_pos[2])),
        }

    @property
    def body_id(self) -> int:
        return self._body_id

    def update_from_slider(self) -> np.ndarray:
        pos = np.array([p.readUserDebugParameter(self.sliders[k]) for k in ("x", "y", "z")])
        p.resetBasePositionAndOrientation(self._body_id, pos.tolist(), [0, 0, 0, 1])
        return pos

    def get_position(self) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(self._body_id)[0], dtype=float)

    def compute_distance(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        """球的解析距离：h = ||p - c|| - r。"""
        center = self.get_position()
        delta = point - center
        dist = np.linalg.norm(delta)
        if dist < 1e-9:
            return -self._radius, np.array([1.0, 0.0, 0.0])
        normal = delta / dist
        return dist - self._radius, normal


class PlateObstacle(Obstacle):
    """薄平板障碍物（BOX 形状，thickness 很小）。"""

    def __init__(self, config: ExperimentConfig, scene: SimulationScene):
        self.config = config
        self._half = np.array(config.plate_half_extents, dtype=float)
        x, y, z = config.plate_initial_offset
        self._init_pos = np.array([x, y, scene.table_height + z], dtype=float)

        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=self._half.tolist(),
                                  rgbaColor=config.plate_rgba,
                                  specularColor=[0.5, 0.5, 0.5])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=self._half.tolist())
        self._body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                          baseVisualShapeIndex=vis,
                                          basePosition=self._init_pos.tolist())
        th = scene.table_height
        self.sliders = {
            "x": p.addUserDebugParameter("plate_x", -0.10, 0.60, float(self._init_pos[0])),
            "y": p.addUserDebugParameter("plate_y", config.obstacle_slider_y_min,
                                         config.obstacle_slider_y_max, float(self._init_pos[1])),
            "z": p.addUserDebugParameter("plate_z", th + 0.08, th + 0.42, float(self._init_pos[2])),
        }

    @property
    def body_id(self) -> int:
        return self._body_id

    def update_from_slider(self) -> np.ndarray:
        pos = np.array([p.readUserDebugParameter(self.sliders[k]) for k in ("x", "y", "z")])
        p.resetBasePositionAndOrientation(self._body_id, pos.tolist(), [0, 0, 0, 1])
        return pos

    def get_position(self) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(self._body_id)[0], dtype=float)

    def compute_distance(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        """点到轴对齐长方体（BOX）表面的精确有符号距离。

        算法:
        1. 将查询点变换到 BOX 局部坐标系
        2. 计算局部坐标下到 BOX 的有符号距离:
           - 外部: d_i = max(|p_i| - h_i, 0)，距离 = sqrt(sum(d_i^2))
           - 内部: 找到离最近面的深度 = min(h_i - |p_i|)，距离取负
        3. 法向量变换回世界系
        """
        center, orn = p.getBasePositionAndOrientation(self._body_id)
        inv_pos, inv_orn = p.invertTransform(center, orn)
        local_pt, _ = p.multiplyTransforms(inv_pos, inv_orn, point.tolist(), [0, 0, 0, 1])
        lp = np.array(local_pt, dtype=float)
        h = self._half

        d = np.abs(lp) - h  # 各轴到面的有符号距离分量

        if np.any(d > 0):
            # 点在 BOX 外部
            clamped = np.clip(lp, -h, h)
            diff = lp - clamped
            dist = np.linalg.norm(diff)
            normal_local = diff / dist if dist > 1e-9 else np.array([1.0, 0.0, 0.0])
            signed_dist = dist
        else:
            # 点在 BOX 内部 —— 找穿透最浅的那个面
            face_depths = h - np.abs(lp)  # 全部 > 0
            min_axis = int(np.argmin(face_depths))
            signed_dist = -face_depths[min_axis]
            normal_local = np.zeros(3)
            normal_local[min_axis] = 1.0 if lp[min_axis] >= 0 else -1.0

        rot_mat = np.array(p.getMatrixFromQuaternion(orn), dtype=float).reshape(3, 3)
        normal_world = rot_mat @ normal_local
        return float(signed_dist), normal_world


def create_obstacle(config: ExperimentConfig, scene: SimulationScene) -> Obstacle:
    """工厂函数：根据 config.obstacle_type 创建对应子类。"""
    if config.obstacle_type == "sphere":
        return SphereObstacle(config, scene)
    elif config.obstacle_type == "plate":
        return PlateObstacle(config, scene)
    else:
        raise ValueError(f"未知障碍物类型: {config.obstacle_type}")


# ============================================================================
#  参考轨迹
# ============================================================================

class LineSlerpTrajectory:
    """位置直线 + 姿态四元数 Slerp 插值的连续参考轨迹。"""

    def __init__(self, start_pos, start_quat, goal_pos, goal_quat, duration, dt):
        self.start_pos = np.array(start_pos, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.duration = duration
        self.dt = dt
        self.linear_velocity = (self.goal_pos - self.start_pos) / max(duration, 1e-6)
        key_times = np.array([0.0, duration])
        key_rots = Rotation.from_quat(np.vstack([start_quat, goal_quat]))
        self.slerp = Slerp(key_times, key_rots)

    def sample(self, t):
        tau = float(np.clip(t, 0, self.duration))
        blend = tau / max(self.duration, 1e-6)
        pos = (1 - blend) * self.start_pos + blend * self.goal_pos
        quat = self.slerp([tau]).as_quat()[0]
        if tau >= self.duration:
            return pos, quat, np.zeros(3), np.zeros(3)
        nt = min(self.duration, tau + self.dt)
        nq = self.slerp([nt]).as_quat()[0]
        ang_vel = (Rotation.from_quat(nq) * Rotation.from_quat(quat).inv()).as_rotvec() / max(nt - tau, 1e-6)
        return pos, quat, self.linear_velocity.copy(), ang_vel

    def reference_points(self, n):
        return [self.sample(t)[0] for t in np.linspace(0, self.duration, n, endpoint=True)]


# ============================================================================
#  CBF-QP 控制器（支持多障碍物列表）
# ============================================================================

class CBFQPController:
    """8 自由度（底座 x/y + 6 关节）CBF-QP 控制器。

    接受 obstacles: list[Obstacle]，对每个障碍物的每个连杆建立 CBF 约束。
    """

    def __init__(self, robot: JakaRobot, config: ExperimentConfig):
        self.robot = robot
        self.config = config
        self.n = robot.total_dof
        self.m = len(robot.cbf_link_indices)
        self.use_slack = config.use_slack
        slack_dim = self.m if self.use_slack else 0
        self.prev_solution = np.zeros(self.n + slack_dim, dtype=float)

    def solve(
        self,
        q, dq, current_pos, current_quat,
        ref_pos, ref_quat, ref_lin_vel, ref_ang_vel,
        obstacles: list[Obstacle],
    ) -> tuple[np.ndarray, dict]:

        pos_err = ref_pos - current_pos
        rot_err = (Rotation.from_quat(ref_quat) * Rotation.from_quat(current_quat).inv()).as_rotvec()
        xdot_ref = np.concatenate([
            ref_lin_vel + self.config.position_gain * pos_err,
            ref_ang_vel + self.config.orientation_gain * rot_err,
        ])
        dq_nom = np.concatenate([
            np.zeros(2),
            self.config.dq_nominal_gain * (self.robot.q_nominal - q),
        ])

        J_ee = self.robot.get_augmented_ee_jacobian(q, dq)
        A_rows: list[np.ndarray] = []
        b_vals: list[float] = []
        h_vals: list[float] = []

        for obs in obstacles:
            use_mesh = self.config.use_mesh_cbf and obs.body_id >= 0

            for link_idx in self.robot.cbf_link_indices:
                if use_mesh:
                    cp = self.robot.get_closest_point_to_obstacle(link_idx, obs.body_id)
                    if cp is None:
                        continue
                    surface_pt, signed_dist, normal = cp
                    h_val = signed_dist - self.config.safety_margin
                    A_rows.append(self.robot.get_link_cbf_row_aug_at_point(
                        link_idx, surface_pt, normal, q, dq))
                else:
                    link_pos = self.robot.get_link_origin(link_idx)
                    signed_dist, normal = obs.compute_distance(link_pos)
                    h_val = signed_dist - self.config.safety_margin
                    A_rows.append(self.robot.get_link_cbf_row_aug(link_idx, normal, q, dq))
                b_vals.append(self.config.cbf_alpha * h_val)
                h_vals.append(h_val)

        if not h_vals:
            h_vals = [1.0]
        self.m = len(A_rows)
        A = np.array(A_rows, dtype=float) if A_rows else np.zeros((0, self.n))
        b = np.array(b_vals, dtype=float) if b_vals else np.zeros(0)

        # ---- 构建 QP ----
        if self.use_slack:
            def objective(x):
                u = x[:self.n]; s = x[self.n:]
                return (np.sum((J_ee @ u - xdot_ref)**2)
                        + self.config.nullspace_weight * np.sum((u - dq_nom)**2)
                        + self.config.slack_weight * np.sum(s**2))

            constraints = []
            for ri in range(self.m):
                r, rhs = A[ri].copy(), b[ri]
                def ineq(x, r=r, rhs=rhs, ri=ri):
                    return r @ x[:self.n] + rhs + x[self.n + ri]
                constraints.append({"type": "ineq", "fun": ineq})

            bounds = ([(-self.config.base_vel_limit, self.config.base_vel_limit)] * 2
                      + [(-self.config.dq_limit, self.config.dq_limit)] * self.robot.dof
                      + [(0, None)] * self.m)
        else:
            def objective(x):
                u = x[:self.n]
                return (np.sum((J_ee @ u - xdot_ref)**2)
                        + self.config.nullspace_weight * np.sum((u - dq_nom)**2))

            constraints = []
            for ri in range(self.m):
                r, rhs = A[ri].copy(), b[ri]
                def ineq(x, r=r, rhs=rhs):
                    return r @ x[:self.n] + rhs
                constraints.append({"type": "ineq", "fun": ineq})

            bounds = ([(-self.config.base_vel_limit, self.config.base_vel_limit)] * 2
                      + [(-self.config.dq_limit, self.config.dq_limit)] * self.robot.dof)

        # 初始猜测维度与 bounds 一致
        x0_len = len(bounds)
        if len(self.prev_solution) != x0_len:
            self.prev_solution = np.zeros(x0_len, dtype=float)

        result = minimize(objective, self.prev_solution, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 100, "ftol": 1e-6, "disp": False})

        if result.success:
            sol = np.asarray(result.x, dtype=float)
            self.prev_solution = sol
            u_cmd = sol[:self.n]
            slack = sol[self.n:] if self.use_slack else np.zeros(self.m)
            status = "optimal"
        else:
            un = dq_nom.copy()
            un[:2] = np.clip(un[:2], -self.config.base_vel_limit, self.config.base_vel_limit)
            un[2:] = np.clip(un[2:], -self.config.dq_limit, self.config.dq_limit)
            u_cmd = un
            slack = np.full(self.m, np.nan)
            status = f"fallback:{result.message}"

        return u_cmd, {
            "min_h": float(np.min(h_vals)),
            "max_slack": float(np.nanmax(slack)) if slack.size else 0.0,
            "status": status,
            "tracking_error": float(np.linalg.norm(pos_err)),
        }


# ============================================================================
#  实验主类
# ============================================================================

class AvoidanceExperiment:
    """把场景、机器人、障碍物列表、轨迹和控制器串起来。"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.scene = SimulationScene(config)
        self.robot = JakaRobot(config, self.scene)

        obs = create_obstacle(config, self.scene)
        obs.disable_collision_with(self.robot.body_id, self.robot.num_joints)
        self.obstacles: list[Obstacle] = [obs]

        obs_center = obs.update_from_slider()
        start_pos = np.array([
            obs_center[0] - config.line_half_span,
            obs_center[1] + config.line_bias_y,
            obs_center[2] + config.line_bias_z,
        ])
        goal_pos = np.array([
            obs_center[0] + config.line_half_span,
            obs_center[1] + config.line_bias_y,
            obs_center[2] + config.line_bias_z,
        ])
        start_quat = p.getQuaternionFromEuler([math.radians(v) for v in config.start_euler_deg])
        goal_quat = p.getQuaternionFromEuler([math.radians(v) for v in config.goal_euler_deg])

        self.robot.reset_to_pose(start_pos, start_quat)
        ee_pos, _ = self.robot.get_ee_pose()
        self.ee_marker = self.scene.create_marker(0.012, (0.1, 0.9, 0.2, 0.9), ee_pos)
        self.ref_marker = self.scene.create_marker(0.012, (0.95, 0.2, 0.2, 0.9), start_pos.tolist())

        self.trajectory = LineSlerpTrajectory(
            start_pos, np.array(start_quat, dtype=float),
            goal_pos, np.array(goal_quat, dtype=float),
            config.trajectory_duration, config.dt)
        self.scene.draw_polyline(
            self.trajectory.reference_points(config.reference_samples),
            color=[0.85, 0.1, 0.1], width=1.2)

        self.controller = CBFQPController(self.robot, config)
        self.prev_ee = ee_pos.copy()
        self.sim_step = 0

        print(f"===== CBF-QP 避障实验 (障碍类型: {config.obstacle_type}) =====")
        print("右侧滑块可调障碍物位置。")

    def _update_visuals(self, ee_pos, ref_pos, info):
        self.scene.update_marker(self.ee_marker, ee_pos.tolist())
        self.scene.update_marker(self.ref_marker, ref_pos.tolist())
        if np.linalg.norm(ee_pos - self.prev_ee) > 1e-3:
            p.addUserDebugLine(self.prev_ee.tolist(), ee_pos.tolist(), [0.1, 0.8, 0.2], lineWidth=1.5)
            self.prev_ee = ee_pos.copy()
        self.scene.update_status(
            f"step={self.sim_step}  "
            f"track_err={info['tracking_error']*1000:.1f} mm  "
            f"min_h={info['min_h']*1000:.1f} mm  "
            f"slack={info['max_slack']:.3f}  "
            f"{info['status']}")

    def _solve_step(self, q, dq, ee_pos, ee_quat, ref_pos, ref_quat, ref_lv, ref_av):
        for obs in self.obstacles:
            obs.update_from_slider()
        return self.controller.solve(
            q, dq, ee_pos, ee_quat,
            ref_pos, ref_quat, ref_lv, ref_av,
            obstacles=self.obstacles)

    def run(self):
        total_steps = int((self.config.trajectory_duration + self.config.hold_duration) / self.config.dt)
        video_frames = []
        rec_every = max(1, int(round((1 / self.config.dt) / self.config.video_fps))) if self.config.record_video else 0

        try:
            while p.isConnected() and self.sim_step < total_steps:
                t = self.sim_step * self.config.dt
                q, dq = self.robot.get_joint_state()
                ee_pos, ee_quat = self.robot.get_ee_pose()
                ref = self.trajectory.sample(min(t, self.config.trajectory_duration))

                u_cmd, info = self._solve_step(q, dq, ee_pos, ee_quat, *ref)
                self.robot.move_base(u_cmd[:2], self.config.dt)
                self.robot.command_joint_velocities(u_cmd[2:])

                p.stepSimulation()
                self._update_visuals(ee_pos, ref[0], info)

                if self.config.record_video and self.sim_step % rec_every == 0:
                    video_frames.append(self.scene.capture_frame(
                        self.config.video_width, self.config.video_height))

                if self.sim_step % self.config.print_every == 0:
                    print(f"[step {self.sim_step:4d}] "
                          f"base=({self.robot.base_pos[0]:.3f}, {self.robot.base_pos[1]:.3f}) "
                          f"ee=({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}) "
                          f"track_err={info['tracking_error']*1000:.1f}mm "
                          f"min_h={info['min_h']*1000:.1f}mm")
                self.sim_step += 1
                time.sleep(self.config.dt)

            self.robot.move_base(np.zeros(2), self.config.dt)
            self.robot.command_joint_velocities(np.zeros(self.robot.dof))

            if self.config.record_video and video_frames:
                if imageio is not None:
                    imageio.mimsave(self.config.video_output_path, video_frames,
                                   fps=self.config.video_fps)
                    print(f"录像已保存: {self.config.video_output_path}")
                else:
                    print("未安装 imageio，可执行: pip install imageio imageio-ffmpeg")

            print("===== 轨迹执行结束，保持窗口 (Ctrl+C 退出) =====")
            while p.isConnected():
                q, dq = self.robot.get_joint_state()
                ee_pos, ee_quat = self.robot.get_ee_pose()
                ref = self.trajectory.sample(self.config.trajectory_duration)
                u_cmd, info = self._solve_step(q, dq, ee_pos, ee_quat, *ref)
                self.robot.move_base(u_cmd[:2], 1 / 60)
                self.robot.command_joint_velocities(u_cmd[2:])
                p.stepSimulation()
                self._update_visuals(ee_pos, ref[0], info)
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            print("\n用户中断。")
        finally:
            if p.isConnected():
                p.disconnect()
            print(f"仿真结束，共 {self.sim_step} 步。")


def main():
    config = ExperimentConfig()
    AvoidanceExperiment(config).run()


if __name__ == "__main__":
    main()
