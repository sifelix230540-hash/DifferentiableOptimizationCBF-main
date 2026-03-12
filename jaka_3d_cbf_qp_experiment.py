import math
import time
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


@dataclass
class ExperimentConfig:
    """集中管理实验配置，避免控制逻辑里散落魔法数字。"""

    # 机器人模型与仿真基础设置。
    urdf_path: str = (
        r"C:\Users\12049\OneDrive\Desktop\Zu 7.SLDASM\urdf\Zu 7.SLDASM.urdf"
    )
    dt: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # 移动小车外观尺寸。
    # 机械臂底座固定在小车上甲板上方，控制时让小车与机械臂底座同步平移。
    cart_body_half: tuple[float, float, float] = (0.24, 0.18, 0.05)
    cart_deck_half: tuple[float, float, float] = (0.18, 0.14, 0.015)
    wheel_radius: float = 0.055
    wheel_width: float = 0.03

    # 相机与可视化视角（俯视偏侧，便于看到机械臂、小车与障碍球）。
    # camera_target 是相机注视的世界坐标点，不是相机自身位置。
    # camera_distance/yaw/pitch 决定相机围绕 target 的球坐标位置。
    camera_distance: float = 1.0
    camera_yaw: float = -225.0
    camera_pitch: float = -15    
    camera_target: tuple[float, float, float] = (0.15, 0.20, 0.45)

    # 录像：轨迹执行阶段每帧采样后写入 MP4。
    record_video: bool = True
    video_output_path: str = "cbf_experiment.mp4"
    video_fps: int = 30
    video_width: int = 960
    video_height: int = 720

    # 直线参考轨迹参数。
    # line_half_span: 直线在 x 方向上的半长度；
    # line_bias_y/z: 给参考直线施加偏置，使其“穿球附近但不过球心”。
    line_half_span: float = 0.14
    line_bias_y: float = 0.08
    line_bias_z: float = 0.0
    trajectory_duration: float = 7.0
    hold_duration: float = 6

    # 单障碍球参数。obstacle_initial_offset 的 z 是相对小车安装面的高度偏置。
    obstacle_radius: float = 0.06
    obstacle_rgba: tuple[float, float, float, float] = (1.0, 0.35, 0.2, 0.75)
    obstacle_initial_offset: tuple[float, float, float] = (0.32, 0.4, 0.3)
    obstacle_slider_y_min: float = -0.30
    obstacle_slider_y_max: float = 0.80

    # 起终点姿态采用完整欧拉角配置（度）。
    # 之前只改 yaw，会导致末端大部分时间仍接近“竖直朝下”。
    # 现在同时改变 roll / pitch / yaw，让四元数插值过程中的姿态变化更明显。
    start_euler_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)
    goal_euler_deg: tuple[float, float, float] = (140.0, 0, 0)

    # 执行器速度/力上限。
    ee_force_limit: float = 250.0
    dq_limit: float = 1.0
    dq_nominal_gain: float = 0.25
    # 底座平面移动速度上限（x/y，与 6 关节一起构成 8 自由度）。
    base_vel_limit: float = 0.4

    # QP 中的跟踪/正则/CBF 权重。
    position_gain: float = 8
    orientation_gain: float = 2.5
    nullspace_weight: float = 0.1
    slack_weight: float = 200000.0
    use_slack: bool = False
    # True: 用 getClosestPoints 查询 mesh 表面最近点构造 CBF；
    # False: 仅用各连杆坐标系原点（单点）构造 CBF。
    use_mesh_cbf: bool = True
    cbf_alpha: float = 2
    safety_margin: float = 0.02

    # 打印频率与参考线绘制分辨率。
    print_every: int = 120
    reference_samples: int = 80


class SimulationScene:
    """负责 PyBullet GUI、移动小车环境与调试可视化。"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*config.gravity)
        p.setTimeStep(config.dt)

        # 保留侧边调试栏，便于用滑块调节障碍球位置；
        # 关闭其余预览面板，避免界面过于杂乱。
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

        self.cart_parts: list[tuple[int, np.ndarray, tuple[float, float, float, float]]] = []
        self.table_height = self._build_environment()
        self._draw_axes()
        self.status_text_id = None

    def _build_environment(self) -> float:
        """创建地面与移动小车外观，并返回机械臂安装面的世界高度。"""
        plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.6, 0.6, 0.6, 1.0])

        body_half = self.config.cart_body_half
        deck_half = self.config.cart_deck_half
        wheel_radius = self.config.wheel_radius
        wheel_width = self.config.wheel_width

        body_z = wheel_radius + body_half[2]
        deck_z = wheel_radius + 2.0 * body_half[2] + deck_half[2]
        top_height = wheel_radius + 2.0 * body_half[2] + 2.0 * deck_half[2]

        body_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=body_half,
            rgbaColor=[0.18, 0.22, 0.28, 1.0],
            specularColor=[0.6, 0.6, 0.6],
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=body_vis,
            basePosition=[0.0, 0.0, body_z],
        )
        self.cart_parts.append(
            (body_id, np.array([0.0, 0.0, body_z], dtype=float), (0.0, 0.0, 0.0, 1.0))
        )

        deck_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=deck_half,
            rgbaColor=[0.82, 0.84, 0.88, 1.0],
            specularColor=[0.7, 0.7, 0.7],
        )
        deck_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=deck_vis,
            basePosition=[0.0, 0.0, deck_z],
        )
        self.cart_parts.append(
            (deck_id, np.array([0.0, 0.0, deck_z], dtype=float), (0.0, 0.0, 0.0, 1.0))
        )

        wheel_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=wheel_radius,
            length=wheel_width,
            rgbaColor=[0.08, 0.08, 0.08, 1.0],
            specularColor=[0.2, 0.2, 0.2],
        )
        wheel_quat = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
        wheel_x = body_half[0] * 0.72
        wheel_y = body_half[1] + wheel_width * 0.35
        for local_pos in (
            np.array([wheel_x, wheel_y, wheel_radius], dtype=float),
            np.array([wheel_x, -wheel_y, wheel_radius], dtype=float),
            np.array([-wheel_x, wheel_y, wheel_radius], dtype=float),
            np.array([-wheel_x, -wheel_y, wheel_radius], dtype=float),
        ):
            wheel_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=wheel_vis,
                basePosition=local_pos.tolist(),
                baseOrientation=wheel_quat,
            )
            self.cart_parts.append((wheel_id, local_pos, wheel_quat))

        return top_height

    def update_cart_pose(self, base_xy: np.ndarray) -> None:
        """让小车外观与机械臂底座同步平移。"""
        base_xy = np.asarray(base_xy, dtype=float).flatten()[:2]
        for body_id, local_pos, local_quat in self.cart_parts:
            world_pos = [
                base_xy[0] + local_pos[0],
                base_xy[1] + local_pos[1],
                local_pos[2],
            ]
            p.resetBasePositionAndOrientation(body_id, world_pos, local_quat)

    def _draw_axes(self) -> None:
        """在机器人底座附近画出世界坐标轴，方便观察运动方向。"""
        axis_len = 0.12
        origin = [0.0, 0.0, self.table_height + 0.001]
        p.addUserDebugLine(origin, [axis_len, 0, origin[2]], [1, 0, 0], lineWidth=2)
        p.addUserDebugLine(origin, [0, axis_len, origin[2]], [0, 0.8, 0], lineWidth=2)
        p.addUserDebugLine(origin, [0, 0, origin[2] + axis_len], [0, 0, 1], lineWidth=2)
        p.addUserDebugText("X", [axis_len + 0.02, 0, origin[2]], [1, 0, 0], textSize=1)
        p.addUserDebugText("Y", [0, axis_len + 0.02, origin[2]], [0, 0.8, 0], textSize=1)
        p.addUserDebugText(
            "Z", [0, 0, origin[2] + axis_len + 0.02], [0, 0, 1], textSize=1
        )

    def create_marker(self, radius: float, color: tuple[float, float, float, float], pos):
        """创建无碰撞的纯可视化标记球。"""
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=pos)

    def update_marker(self, body_id: int, pos) -> None:
        """更新标记球位置。"""
        p.resetBasePositionAndOrientation(body_id, pos, [0, 0, 0, 1])

    def draw_polyline(self, points: list[np.ndarray], color, width: float = 1.5) -> None:
        """把参考路径画成折线，便于和真实末端轨迹对比。"""
        for idx in range(len(points) - 1):
            p.addUserDebugLine(
                points[idx].tolist(),
                points[idx + 1].tolist(),
                color,
                lineWidth=width,
            )

    def update_status(self, text: str) -> None:
        """在场景里实时显示当前误差、最小 h 值和求解器状态。"""
        self.status_text_id = p.addUserDebugText(
            text,
            [0.02, -0.26, self.table_height + 0.52],
            [0.1, 0.1, 0.1],
            textSize=1.2,
            replaceItemUniqueId=self.status_text_id if self.status_text_id else -1,
        )

    def capture_frame(self, width: int, height: int) -> np.ndarray:
        """截取当前视角一帧，返回 (H, W, 3) RGB uint8。"""
        _, _, rgb, _, _ = p.getCameraImage(width, height)
        frame = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
        return frame[:, :, :3]


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
            idx
            for idx in range(self.num_joints)
            if p.getJointInfo(self.body_id, idx)[2] == p.JOINT_REVOLUTE
        ]
        self.dof = len(self.active_joints)
        self.ee_link_index = self.active_joints[-1]
        self.cbf_link_indices = list(self.active_joints)
        # 底座平面 x/y 位置（世界系），与 6 关节共同构成 8 自由度。
        self.base_pos = np.array([0.0, 0.0], dtype=float)
        self.total_dof = 2 + self.dof

        print(f"活动关节索引: {self.active_joints}")
        print(f"末端执行器连杆索引: {self.ee_link_index}")

        for joint_idx in self.active_joints:
            p.changeDynamics(self.body_id, joint_idx, linearDamping=0, angularDamping=0)
            p.setJointMotorControl2(self.body_id, joint_idx, p.VELOCITY_CONTROL, force=0)

        self.q_nominal = np.zeros(self.dof)

    def move_base(self, v_base: np.ndarray, dt: float) -> None:
        """根据底座平面速度积分更新底座位置，并重置 PyBullet 基座位姿。"""
        v_clipped = np.clip(
            np.asarray(v_base, dtype=float).flatten()[:2],
            -self.config.base_vel_limit,
            self.config.base_vel_limit,
        )
        self.base_pos += v_clipped * dt
        base_pos_3d = p.getBasePositionAndOrientation(self.body_id)[0]
        p.resetBasePositionAndOrientation(
            self.body_id,
            [self.base_pos[0], self.base_pos[1], base_pos_3d[2]],
            [0.0, 0.0, 0.0, 1.0],
        )
        self.scene.update_cart_pose(self.base_pos)

    def _set_link_colors(self) -> None:
        """给各连杆设置更容易区分的配色。"""
        link_colors = {
            -1: [0.15, 0.15, 0.15, 1.0],
            0: [0.92, 0.92, 0.94, 1.0],
            1: [0.12, 0.46, 0.70, 1.0],
            2: [0.92, 0.92, 0.94, 1.0],
            3: [0.12, 0.46, 0.70, 1.0],
            4: [0.92, 0.92, 0.94, 1.0],
            5: [0.15, 0.15, 0.15, 1.0],
        }
        for link_idx, rgba in link_colors.items():
            p.changeVisualShape(
                self.body_id, link_idx, rgbaColor=rgba, specularColor=[0.6, 0.6, 0.6]
            )

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """返回所有主动关节的位置 q 与速度 dq。"""
        states = p.getJointStates(self.body_id, self.active_joints)
        q = np.array([state[0] for state in states], dtype=float)
        dq = np.array([state[1] for state in states], dtype=float)
        return q, dq

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """读取末端连杆坐标系原点的世界位置和姿态四元数。"""
        state = p.getLinkState(self.body_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(state[4], dtype=float), np.array(state[5], dtype=float)

    def get_link_origin(self, link_idx: int) -> np.ndarray:
        """读取指定连杆坐标系原点的位置，用于构造多点 CBF 约束。"""
        state = p.getLinkState(self.body_id, link_idx, computeForwardKinematics=True)
        return np.array(state[4], dtype=float)

    def calculate_ik(self, target_pos, target_quat) -> np.ndarray:
        """仅用于实验初始化，把机器人快速摆到起始位姿。"""
        lower_limits = [-6.28] * self.dof
        upper_limits = [6.28] * self.dof
        joint_ranges = [12.56] * self.dof
        rest_poses = [0.0] * self.dof
        angles = p.calculateInverseKinematics(
            self.body_id,
            self.ee_link_index,
            target_pos,
            target_quat,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=200,
            residualThreshold=1.0e-6,
        )
        return np.array(angles[: self.dof], dtype=float)

    def reset_to_pose(self, target_pos, target_quat) -> None:
        """用 IK 把机器人直接放到初始位姿，并记录名义关节位形。"""
        q_target = self.calculate_ik(target_pos, target_quat)
        for i, joint_idx in enumerate(self.active_joints):
            p.resetJointState(self.body_id, joint_idx, q_target[i])
        self.q_nominal = q_target.copy()

    def get_link_jacobian(self, link_idx: int, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回指定连杆原点的平动/转动雅可比。"""
        zeros = np.zeros_like(q)
        jac_t, jac_r = p.calculateJacobian(
            self.body_id,
            link_idx,
            [0.0, 0.0, 0.0],
            q.tolist(),
            dq.tolist(),
            zeros.tolist(),
        )
        return np.array(jac_t, dtype=float), np.array(jac_r, dtype=float)

    def get_ee_jacobian(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """把末端平动和转动雅可比拼成 6 x n，用于末端 twist 跟踪。"""
        jac_t, jac_r = self.get_link_jacobian(self.ee_link_index, q, dq)
        return np.vstack([jac_t, jac_r])

    def get_augmented_ee_jacobian(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """返回 6×8 增广雅可比：前两列对应底座 x/y 速度对末端 twist 的贡献，后 6 列为关节速度贡献。"""
        jac_t, jac_r = self.get_link_jacobian(self.ee_link_index, q, dq)
        # 底座平移：dx 只影响末端 x 线速度，dy 只影响末端 y 线速度；底座不旋转。
        base_linear = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)
        base_angular = np.zeros((3, 2), dtype=float)
        base_block = np.vstack([base_linear, base_angular])
        joint_block = np.vstack([jac_t, jac_r])
        return np.hstack([base_block, joint_block])

    def get_link_cbf_row_aug(
        self,
        link_idx: int,
        normal: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> np.ndarray:
        """返回 1×8 的 CBF 梯度行：前 2 维为底座 x/y 对距离变化率，后 6 维为关节速度对距离变化率。"""
        jac_t, _ = self.get_link_jacobian(link_idx, q, dq)
        # 底座移动：d(dist)/d(base_vel) = normal 在 x,y 上的分量。
        base_part = np.array([normal[0], normal[1]], dtype=float)
        joint_part = normal @ jac_t
        return np.concatenate([base_part, joint_part])

    # ---- mesh-based CBF 辅助方法 ----

    def get_closest_point_to_obstacle(
        self, link_idx: int, obstacle_body_id: int, max_dist: float = 1.0,
    ) -> tuple[np.ndarray, float, np.ndarray] | None:
        """用 getClosestPoints 查询指定连杆碰撞 mesh 到障碍物的最近点。

        返回 (closest_point_world, signed_distance, normal_from_obs_to_link) 或 None。
        signed_distance > 0 表示分离，< 0 表示穿透。
        """
        contacts = p.getClosestPoints(
            self.body_id, obstacle_body_id, max_dist, linkIndexA=link_idx,
        )
        if not contacts:
            return None
        best = min(contacts, key=lambda c: c[8])
        pos_on_robot = np.array(best[5], dtype=float)
        distance = float(best[8])
        normal = np.array(best[7], dtype=float)
        if np.linalg.norm(normal) < 1e-9:
            normal = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            normal = normal / np.linalg.norm(normal)
        return pos_on_robot, distance, normal

    def get_link_cbf_row_aug_at_point(
        self,
        link_idx: int,
        world_point: np.ndarray,
        normal: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> np.ndarray:
        """在连杆 mesh 表面最近点处计算 1×8 增广 CBF 梯度行。

        与 get_link_cbf_row_aug 的区别：雅可比不在连杆原点计算，
        而是在 world_point 对应的局部坐标处计算，更精确地反映
        表面点速度对安全函数的影响。
        """
        link_state = p.getLinkState(
            self.body_id, link_idx, computeForwardKinematics=True,
        )
        link_pos = link_state[4]
        link_orn = link_state[5]
        inv_pos, inv_orn = p.invertTransform(link_pos, link_orn)
        local_point, _ = p.multiplyTransforms(
            inv_pos, inv_orn, world_point.tolist(), [0, 0, 0, 1],
        )
        zeros = np.zeros_like(q)
        jac_t, _ = p.calculateJacobian(
            self.body_id, link_idx, list(local_point),
            q.tolist(), dq.tolist(), zeros.tolist(),
        )
        jac_t = np.array(jac_t, dtype=float)
        base_part = np.array([normal[0], normal[1]], dtype=float)
        joint_part = normal @ jac_t
        return np.concatenate([base_part, joint_part])

    def command_joint_velocities(self, dq_cmd: np.ndarray) -> None:
        """向速度控制器发送关节速度命令，并进行限幅。"""
        dq_clip = np.clip(dq_cmd, -self.config.dq_limit, self.config.dq_limit)
        p.setJointMotorControlArray(
            bodyUniqueId=self.body_id,
            jointIndices=self.active_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=dq_clip.tolist(),
            forces=[self.config.ee_force_limit] * self.dof,
        )


class SphereObstacle:
    """单个可调球障碍物。

    这里故意只保留视觉形状，不启用 PyBullet 物理碰撞，
    这样实验中的“绕球”只来自 CBF-QP，而不是接触力把机械臂挡开。
    """

    def __init__(self, config: ExperimentConfig, scene: SimulationScene):
        self.config = config
        self.scene = scene
        initial_pos = self.default_world_position(scene.table_height)
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=config.obstacle_radius,
            rgbaColor=config.obstacle_rgba,
        )
        collision = p.createCollisionShape(
            p.GEOM_SPHERE, radius=config.obstacle_radius,
        )
        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=initial_pos.tolist(),
        )
        self.sliders = {
            "x": p.addUserDebugParameter("obs_x", -0.10, 0.60, float(initial_pos[0])),
            "y": p.addUserDebugParameter(
                "obs_y",
                config.obstacle_slider_y_min,
                config.obstacle_slider_y_max,
                float(initial_pos[1]),
            ),
            "z": p.addUserDebugParameter(
                "obs_z",
                scene.table_height + 0.08,
                scene.table_height + 0.42,
                float(initial_pos[2]),
            ),
        }

    def default_world_position(self, table_height: float) -> np.ndarray:
        """根据桌面高度计算障碍球初始世界坐标。"""
        x, y, z = self.config.obstacle_initial_offset
        return np.array([x, y, table_height + z], dtype=float)

    def update_from_slider(self) -> np.ndarray:
        """读取侧边栏滑块，并实时更新障碍球位置。"""
        pos = np.array(
            [
                p.readUserDebugParameter(self.sliders["x"]),
                p.readUserDebugParameter(self.sliders["y"]),
                p.readUserDebugParameter(self.sliders["z"]),
            ],
            dtype=float,
        )
        p.resetBasePositionAndOrientation(self.body_id, pos.tolist(), [0, 0, 0, 1])
        return pos

    def get_position(self) -> np.ndarray:
        """返回当前障碍球位置。"""
        return np.array(p.getBasePositionAndOrientation(self.body_id)[0], dtype=float)

    @property
    def radius(self) -> float:
        return self.config.obstacle_radius


class LineSlerpTrajectory:
    """生成“位置直线 + 姿态四元数插值”的连续参考轨迹。"""

    def __init__(
        self,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        duration: float,
        dt: float,
    ):
        self.start_pos = np.array(start_pos, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.duration = duration
        self.dt = dt
        self.linear_velocity = (self.goal_pos - self.start_pos) / max(duration, 1.0e-6)

        key_times = np.array([0.0, duration], dtype=float)
        key_rots = Rotation.from_quat(np.vstack([start_quat, goal_quat]))
        self.slerp = Slerp(key_times, key_rots)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """采样时刻 t 的参考位姿与参考 twist。

        返回:
        - pos: 参考位置
        - quat: 参考姿态四元数
        - linear_velocity: 参考线速度
        - ang_vel: 由相邻四元数差分得到的参考角速度
        """
        tau = float(np.clip(t, 0.0, self.duration))
        blend = tau / max(self.duration, 1.0e-6)
        pos = (1.0 - blend) * self.start_pos + blend * self.goal_pos
        quat = self.slerp([tau]).as_quat()[0]

        if tau >= self.duration:
            return pos, quat, np.zeros(3), np.zeros(3)

        next_tau = min(self.duration, tau + self.dt)
        next_quat = self.slerp([next_tau]).as_quat()[0]
        rot_now = Rotation.from_quat(quat)
        rot_next = Rotation.from_quat(next_quat)
        ang_vel = (rot_next * rot_now.inv()).as_rotvec() / max(next_tau - tau, 1.0e-6)
        return pos, quat, self.linear_velocity.copy(), ang_vel

    def reference_points(self, sample_count: int) -> list[np.ndarray]:
        """用于绘图，均匀采样若干位置点形成参考折线。"""
        return [
            self.sample(t)[0]
            for t in np.linspace(0.0, self.duration, sample_count, endpoint=True)
        ]


class CBFQPController:
    """关节速度空间 + 底座平面速度的 CBF-QP 控制器（8 自由度：底座 x/y + 6 关节）。

    QP 变量为:
    - u: [v_base_x, v_base_y, dq1..dq6]，即底座平面速度与关节速度
    - slack: 每条 CBF 约束的松弛变量

    目标函数由三部分组成:
    1. 末端 6D twist 跟踪误差（含底座与关节对末端的贡献）
    2. 向名义关节位形回拉的正则项（仅关节部分）
    3. 松弛变量惩罚

    约束形式:
        grad_h * u + alpha * h(q) + slack >= 0，其中 grad_h 为 1×8 增广梯度。
    """

    def __init__(self, robot: JakaRobot, config: ExperimentConfig):
        self.robot = robot
        self.config = config
        self.n = robot.total_dof  # 8
        self.m = len(robot.cbf_link_indices)
        self.use_slack = config.use_slack
        slack_dim = self.m if self.use_slack else 0
        self.prev_solution = np.zeros(self.n + slack_dim, dtype=float)

    def solve(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        ref_pos: np.ndarray,
        ref_quat: np.ndarray,
        ref_lin_vel: np.ndarray,
        ref_ang_vel: np.ndarray,
        obstacle_center: np.ndarray,
        obstacle_radius: float,
        obstacle_body_id: int = -1,
    ) -> tuple[np.ndarray, dict]:
        """在当前状态下求解一次 CBF-QP，并返回 8 维控制命令。"""

        # 位置误差和姿态误差共同组成末端 6 维跟踪误差。
        # 姿态误差采用旋转向量形式，便于和角速度放到同一 3 维空间。
        pos_err = ref_pos - current_pos
        rot_err = (
            Rotation.from_quat(ref_quat) * Rotation.from_quat(current_quat).inv()
        ).as_rotvec()

        # 构造期望末端 twist。
        # 前 3 维是线速度目标，后 3 维是角速度目标。
        xdot_ref = np.concatenate(
            [
                ref_lin_vel + self.config.position_gain * pos_err,
                ref_ang_vel + self.config.orientation_gain * rot_err,
            ]
        )

        # 名义控制：底座速度为零，关节向名义位形轻微回拉。
        dq_nom = np.concatenate(
            [
                np.zeros(2),
                self.config.dq_nominal_gain * (self.robot.q_nominal - q),
            ]
        )

        J_ee = self.robot.get_augmented_ee_jacobian(q, dq)
        A_rows = []
        b_vals = []
        h_vals = []

        # 对多个关键连杆点同时建立“点-球”安全函数（梯度为 1×8 增广行）。
        use_mesh = self.config.use_mesh_cbf and obstacle_body_id >= 0

        for link_idx in self.robot.cbf_link_indices:
            if use_mesh:
                cp_result = self.robot.get_closest_point_to_obstacle(
                    link_idx, obstacle_body_id,
                )
                if cp_result is None:
                    continue
                surface_pt, signed_dist, normal = cp_result
                h_val = signed_dist - self.config.safety_margin
                A_rows.append(
                    self.robot.get_link_cbf_row_aug_at_point(
                        link_idx, surface_pt, normal, q, dq,
                    )
                )
            else:
                link_pos = self.robot.get_link_origin(link_idx)
                delta = link_pos - obstacle_center
                dist = np.linalg.norm(delta)
                if dist < 1.0e-9:
                    normal = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    normal = delta / dist
                h_val = dist - (obstacle_radius + self.config.safety_margin)
                A_rows.append(
                    self.robot.get_link_cbf_row_aug(link_idx, normal, q, dq)
                )
            b_vals.append(self.config.cbf_alpha * h_val)
            h_vals.append(h_val)

        if not h_vals:
            h_vals = [1.0]
        self.m = len(A_rows)
        A = np.array(A_rows, dtype=float) if A_rows else np.zeros((0, self.n))
        b = np.array(b_vals, dtype=float) if b_vals else np.zeros(0)

        if self.use_slack:
            def objective(x):
                u_var = x[: self.n]
                slack_var = x[self.n :]
                track_cost = np.sum((J_ee @ u_var - xdot_ref) ** 2)
                nominal_cost = self.config.nullspace_weight * np.sum((u_var - dq_nom) ** 2)
                slack_cost = self.config.slack_weight * np.sum(slack_var**2)
                return track_cost + nominal_cost + slack_cost

            constraints = []
            for row_idx in range(self.m):
                row = A[row_idx].copy()
                rhs = b[row_idx]

                def ineq(x, row=row, rhs=rhs, row_idx=row_idx):
                    u_var = x[: self.n]
                    slack_var = x[self.n :]
                    return row @ u_var + rhs + slack_var[row_idx]

                constraints.append({"type": "ineq", "fun": ineq})

            bounds = (
                [(-self.config.base_vel_limit, self.config.base_vel_limit)] * 2
                + [(-self.config.dq_limit, self.config.dq_limit)] * self.robot.dof
                + [(0.0, None)] * self.m
            )
        else:
            def objective(x):
                u_var = x[: self.n]
                track_cost = np.sum((J_ee @ u_var - xdot_ref) ** 2)
                nominal_cost = self.config.nullspace_weight * np.sum((u_var - dq_nom) ** 2)
                return track_cost + nominal_cost

            constraints = []
            for row_idx in range(self.m):
                row = A[row_idx].copy()
                rhs = b[row_idx]

                def ineq(x, row=row, rhs=rhs):
                    u_var = x[: self.n]
                    return row @ u_var + rhs

                constraints.append({"type": "ineq", "fun": ineq})

            bounds = (
                [(-self.config.base_vel_limit, self.config.base_vel_limit)] * 2
                + [(-self.config.dq_limit, self.config.dq_limit)] * self.robot.dof
            )

        result = minimize(
            objective,
            self.prev_solution,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1.0e-6, "disp": False},
        )

        if result.success:
            solution = np.asarray(result.x, dtype=float)
            self.prev_solution = solution
            u_cmd = solution[: self.n]
            slack = solution[self.n :] if self.use_slack else np.zeros(self.m)
            status = "optimal"
        else:
            u_nom = dq_nom.copy()
            u_nom[:2] = np.clip(u_nom[:2], -self.config.base_vel_limit, self.config.base_vel_limit)
            u_nom[2:] = np.clip(u_nom[2:], -self.config.dq_limit, self.config.dq_limit)
            u_cmd = u_nom
            slack = np.full(self.m, np.nan)
            status = f"fallback:{result.message}"

        info = {
            "min_h": float(np.min(h_vals)),
            "max_slack": float(np.nanmax(slack)) if slack.size else 0.0,
            "status": status,
            "tracking_error": float(np.linalg.norm(pos_err)),
        }
        return u_cmd, info


class AvoidanceExperiment:
    """把场景、机器人、障碍、轨迹和控制器串起来的实验主类。"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.scene = SimulationScene(config)
        self.robot = JakaRobot(config, self.scene)
        self.obstacle = SphereObstacle(config, self.scene)

        # 禁用机器人与障碍球之间的物理碰撞响应，
        # 避障行为完全由 CBF-QP 控制。
        p.setCollisionFilterPair(
            self.robot.body_id, self.obstacle.body_id, -1, -1, enableCollision=0,
        )
        for link_idx in range(self.robot.num_joints):
            p.setCollisionFilterPair(
                self.robot.body_id, self.obstacle.body_id, link_idx, -1, enableCollision=0,
            )

        # 用障碍球当前位置定义参考直线。
        # 这里的直线在 x 方向穿过障碍球附近，但在 y/z 上留固定偏置，
        # 因而实验目标是“从球旁边掠过”，而不是穿过球心。
        obstacle_center = self.obstacle.update_from_slider()
        start_pos = np.array(
            [
                obstacle_center[0] - config.line_half_span,
                obstacle_center[1] + config.line_bias_y,
                obstacle_center[2] + config.line_bias_z,
            ],
            dtype=float,
        )
        goal_pos = np.array(
            [
                obstacle_center[0] + config.line_half_span,
                obstacle_center[1] + config.line_bias_y,
                obstacle_center[2] + config.line_bias_z,
            ],
            dtype=float,
        )
        start_quat = p.getQuaternionFromEuler(
            [math.radians(v) for v in config.start_euler_deg]
        )
        goal_quat = p.getQuaternionFromEuler(
            [math.radians(v) for v in config.goal_euler_deg]
        )

        # 启动实验前，先把机械臂直接摆到参考起点。
        self.robot.reset_to_pose(start_pos, start_quat)
        ee_pos, _ = self.robot.get_ee_pose()
        self.ee_marker = self.scene.create_marker(0.012, (0.1, 0.9, 0.2, 0.9), ee_pos)
        self.ref_marker = self.scene.create_marker(
            0.012, (0.95, 0.2, 0.2, 0.9), start_pos.tolist()
        )

        # 先把理论参考轨迹画出来，运行中再画真实末端轨迹。
        self.trajectory = LineSlerpTrajectory(
            start_pos,
            np.array(start_quat, dtype=float),
            goal_pos,
            np.array(goal_quat, dtype=float),
            duration=config.trajectory_duration,
            dt=config.dt,
        )
        self.scene.draw_polyline(
            self.trajectory.reference_points(config.reference_samples),
            color=[0.85, 0.1, 0.1],
            width=1.2,
        )
        self.controller = CBFQPController(self.robot, config)
        self.prev_ee = ee_pos.copy()
        self.sim_step = 0

        print("===== 单障碍球 CBF-QP 避障实验 =====")
        print("右侧滑块可调障碍球位置。")
        print("末端参考为带固定偏置的直线，姿态使用四元数插值。")

    def _update_visuals(self, ee_pos: np.ndarray, ref_pos: np.ndarray, info: dict) -> None:
        """更新参考球、末端球、真实轨迹线和状态文字。"""
        self.scene.update_marker(self.ee_marker, ee_pos.tolist())
        self.scene.update_marker(self.ref_marker, ref_pos.tolist())
        if np.linalg.norm(ee_pos - self.prev_ee) > 1.0e-3:
            p.addUserDebugLine(
                self.prev_ee.tolist(), ee_pos.tolist(), [0.1, 0.8, 0.2], lineWidth=1.5
            )
            self.prev_ee = ee_pos.copy()

        status = (
            f"step={self.sim_step}  "
            f"track_err={info['tracking_error'] * 1000:.1f} mm  "
            f"min_h={info['min_h'] * 1000:.1f} mm  "
            f"slack={info['max_slack']:.3f}  "
            f"{info['status']}"
        )
        self.scene.update_status(status)

    def run(self) -> None:
        """运行整段轨迹，再在终点附近持续保持。"""
        total_time = self.config.trajectory_duration + self.config.hold_duration
        total_steps = int(total_time / self.config.dt)

        video_frames: list[np.ndarray] = []
        record_every = max(
            1, int(round((1.0 / self.config.dt) / self.config.video_fps))
        ) if self.config.record_video else 0

        try:
            while p.isConnected() and self.sim_step < total_steps:
                current_time = self.sim_step * self.config.dt

                # 1. 读取机器人当前状态。
                q, dq = self.robot.get_joint_state()
                ee_pos, ee_quat = self.robot.get_ee_pose()

                # 2. 读取障碍球位置和当前时刻参考轨迹。
                obstacle_center = self.obstacle.update_from_slider()
                ref_pos, ref_quat, ref_lin_vel, ref_ang_vel = self.trajectory.sample(
                    min(current_time, self.config.trajectory_duration)
                )

                # 3. 基于当前状态、参考轨迹和障碍位置求解一次 CBF-QP（8 维：底座 + 关节）。
                u_cmd, info = self.controller.solve(
                    q=q,
                    dq=dq,
                    current_pos=ee_pos,
                    current_quat=ee_quat,
                    ref_pos=ref_pos,
                    ref_quat=ref_quat,
                    ref_lin_vel=ref_lin_vel,
                    ref_ang_vel=ref_ang_vel,
                    obstacle_center=obstacle_center,
                    obstacle_radius=self.obstacle.radius,
                    obstacle_body_id=self.obstacle.body_id,
                )

                # 4. 拆分 8 维命令：前 2 维驱动底座平面移动，后 6 维驱动关节。
                self.robot.move_base(u_cmd[:2], self.config.dt)
                self.robot.command_joint_velocities(u_cmd[2:])

                p.stepSimulation()
                self._update_visuals(ee_pos, ref_pos, info)

                if self.config.record_video and self.sim_step % record_every == 0:
                    video_frames.append(
                        self.scene.capture_frame(
                            self.config.video_width, self.config.video_height,
                        )
                    )

                if self.sim_step % self.config.print_every == 0:
                    print(
                        f"[step {self.sim_step:4d}] "
                        f"base=({self.robot.base_pos[0]:.3f}, {self.robot.base_pos[1]:.3f}) "
                        f"ee=({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}) "
                        f"ref=({ref_pos[0]:.3f}, {ref_pos[1]:.3f}, {ref_pos[2]:.3f}) "
                        f"track_err={info['tracking_error'] * 1000:.1f}mm "
                        f"min_h={info['min_h'] * 1000:.1f}mm"
                    )

                self.sim_step += 1
                time.sleep(self.config.dt)

            # 轨迹时间结束后，底座与关节均置零，并在终点附近闭环保持。
            self.robot.move_base(np.zeros(2), self.config.dt)
            self.robot.command_joint_velocities(np.zeros(self.robot.dof))

            if self.config.record_video and video_frames:
                if imageio is not None:
                    out_path = self.config.video_output_path
                    imageio.mimsave(out_path, video_frames, fps=self.config.video_fps)
                    print(f"录像已保存: {out_path}")
                else:
                    print("未安装 imageio，无法输出录像。可执行: pip install imageio imageio-ffmpeg")

            print("===== 轨迹执行结束，保持窗口 (Ctrl+C 退出) =====")
            while p.isConnected():
                obstacle_center = self.obstacle.update_from_slider()
                q, dq = self.robot.get_joint_state()
                ee_pos, ee_quat = self.robot.get_ee_pose()
                ref_pos, ref_quat, ref_lin_vel, ref_ang_vel = self.trajectory.sample(
                    self.config.trajectory_duration
                )
                u_cmd, info = self.controller.solve(
                    q=q,
                    dq=dq,
                    current_pos=ee_pos,
                    current_quat=ee_quat,
                    ref_pos=ref_pos,
                    ref_quat=ref_quat,
                    ref_lin_vel=ref_lin_vel,
                    ref_ang_vel=ref_ang_vel,
                    obstacle_center=obstacle_center,
                    obstacle_radius=self.obstacle.radius,
                    obstacle_body_id=self.obstacle.body_id,
                )
                self.robot.move_base(u_cmd[:2], 1.0 / 60.0)
                self.robot.command_joint_velocities(u_cmd[2:])
                p.stepSimulation()
                self._update_visuals(ee_pos, ref_pos, info)
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            print("\n用户中断。")
        finally:
            if p.isConnected():
                p.disconnect()
            print(f"仿真结束，共 {self.sim_step} 步。")


def main() -> None:
    config = ExperimentConfig()
    experiment = AvoidanceExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
