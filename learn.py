import math 
import time 
from abc import ABC, abstractclassmethod
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

## 配置

@dataclass
class ExperimentConfig:

    urdf_path: str = (r"C:\Users\12049\OneDrive\Desktop\Zu 7.SLDASM\urdf\Zu 7.SLDASM.urdf")
    dt: float = 1.0/240.0 #工作频率是240Hz
    gravity: tuple[float,float,float] = (0,0,-9.81)

    #移动小车的外观参数
class SimulationScene:
    def __init__(self,config:ExperimentConfig):
        self.config = config
        self.client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*config.gravity)
        # 设置仿真时间步长（以秒为单位），决定 physics 引擎每次 advance 的时间间隔，常见值为 1/240
        p.setTimeStep(config.dt)

        # 注意：如果在同一进程中存在多个 physics client，建议在下面的调用中显式传入
        # physicsClientId=self.client_id 以确保配置作用在正确的客户端上。

        # 是否显示 PyBullet 的内置 GUI 控件面板（右上角的调试面板），1：显示，0：隐藏
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

        # 是否启用渲染阴影（提高视觉真实感，但会增加渲染开销）
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # 是否显示 RGB 渲染缓冲预览小窗口（调试用），0：关闭
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        # 是否显示深度缓冲预览小窗口（调试用深度图），0：关闭
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)

        # 是否显示分割/实例 ID 预览（调试用），0：关闭
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # 设置渲染背景色，RGB 值范围 0~1（这里是浅蓝灰色，便于截图）
        p.configureDebugVisualizer(rgbBackground=[0.82, 0.87, 0.92])

        # 设置场景光源位置（影响光照和阴影效果）
        p.configureDebugVisualizer(lightPosition=[1.5, -1.5, 2.5])

        # 重置调试相机视角：距离、偏航（yaw）、俯仰（pitch）和目标点
        # 确保 config 中存在 camera_distance/camera_yaw/camera_pitch/camera_target
        p.resetDebugVisualizerCamera(
            cameraDistance=config.camera_distance,
            cameraYaw=config.camera_yaw,
            cameraPitch=config.camera_pitch,
            cameraTargetPosition=config.camera_target,
        )

        self.cart_parts : list[tuple[int,np.ndarray,tuple]] = []
        self.table_height = self._build_environment()
        self._draw_axes()
        self.status_text_id = None

    def _build_environment(self) -> float :
        # 加载地面 URDF（使用 pybullet_data 提供的 plane.urdf），返回 body id
        # 常用参数说明：
        #   fileName (str)：URDF 文件名或路径（这里使用相对名，依赖 setAdditionalSearchPath）
        #   basePosition (list[3])：初始位置 [x,y,z]
        #   baseOrientation (quat)：四元数 [x,y,z,w]
        #   useFixedBase (bool)：是否固定基座（True 表示不参与动力学）
        plane_id = p.loadURDF("plane.urdf")
        # 修改地面外观颜色（RGBA）以避免完全白色
        # changeVisualShape(objectUniqueId, linkIndex, rgbaColor=[r,g,b,a], specularColor=[r,g,b])
        #  - objectUniqueId: loadURDF 返回的 id
        #  - linkIndex: -1 表示 base
        #  - rgbaColor: [r,g,b,a] 取值 0~1
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.6, 0.6, 0.6, 1.0])

        # 从配置读取小车外观参数：车身半尺寸、甲板半尺寸、车轮半径与宽度
        bh = self.config.cart_body_half
        dh = self.config.cart_deck_half
        wr = self.config.wheel_radius
        ww = self.config.wheel_width

        # 计算各部件在 z 方向上的高度（基于半尺寸与轮子高度）
        body_z = wr + bh[2]
        deck_z = wr + 2.0 * bh[2] + dh[2]
        top_height = wr + 2.0 * bh[2] + 2.0 * dh[2]

        # 创建车身的可视形状（不创建碰撞形状），并生成一个静态多体作为视觉占位
        # createVisualShape(shapeType, halfExtents=..., radius=..., length=..., rgbaColor=..., specularColor=...)
        #  - shapeType: p.GEOM_BOX/p.GEOM_CYLINDER 等
        #  - halfExtents: 盒子半尺寸 [hx,hy,hz]
        #  - radius/length: 圆柱或球体参数
        body_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=bh,
            rgbaColor=[0.18, 0.22, 0.28, 1.0],
            specularColor=[0.6, 0.6, 0.6]
        )
        # createMultiBody(baseMass, baseCollisionShapeIndex, baseVisualShapeIndex, basePosition, baseOrientation, ...)
        #  - baseMass: 质量，0 表示静态（不参与动力学）
        #  - baseCollisionShapeIndex: 碰撞形状索引（-1 表示无碰撞，仅视觉）
        #  - baseVisualShapeIndex: 上面 createVisualShape 返回的视觉形状索引
        bid = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=body_vis,
            basePosition=[0.0, 0.0, body_z]
        )
        # 记录该部件 id、在世界坐标系中的位置（np.array）和姿态（四元数）
        self.cart_parts.append((bid, np.array([0, 0, body_z], dtype=float), (0, 0, 0, 1)))

        # 创建甲板的可视形状并放置（同上，视觉占位）
        deck_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=dh,
            rgbaColor=[0.82, 0.84, 0.88, 1.0],
            specularColor=[0.7, 0.7, 0.7]
        )
        did = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=deck_vis,
            basePosition=[0.0, 0.0, deck_z]
        )
        self.cart_parts.append((did, np.array([0, 0, deck_z], dtype=float), (0, 0, 0, 1)))

        # 创建车轮的可视形状（圆柱），并旋转四元数使其轴向正确
        # 圆柱参数：radius (米)，length (米，沿局部 z 轴长度)
        wheel_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=wr, length=ww,
            rgbaColor=[0.08, 0.08, 0.08, 1.0],
            specularColor=[0.2, 0.2, 0.2]
        )
        # getQuaternionFromEuler([roll,pitch,yaw])：参数以弧度为单位，返回四元数 [x,y,z,w]
        # 这里将圆柱绕 x 轴旋转 90°（math.pi/2）使其轴线与期望方向一致
        wq = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])

        # 计算四个车轮的 xy 偏移量（基于车身半尺寸与车轮宽度微调）
        wx = bh[0] * 0.72
        wy = bh[1] + ww * 0.35

        # 在四个角位置放置车轮（只做可视多体，不添加碰撞或动力学约束）
        for lp in (
            np.array([wx, wy, wr]), np.array([wx, -wy, wr]),
            np.array([-wx, wy, wr]), np.array([-wx, -wy, wr])
        ):
            # createMultiBody 中的 baseOrientation 接受四元数（[x,y,z,w]）
            # 如果希望轮子参与动力学，请将 baseMass 设置为非零并传入 collisionShapeIndex（通过 createCollisionShape 创建）
            wid = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=wheel_vis,
                basePosition=lp.tolist(), baseOrientation=wq
            )
            self.cart_parts.append((wid, lp.astype(float), wq))

        # 返回桌面/顶面高度，供上层使用
        return top_height

    def update_cart_pose(self, base_xy: np.ndarray) -> None:
        # 更新小车各视觉部件的位置：
        # 参数说明：
        #  - base_xy: 可迭代对象，包含小车基座在 XY 平面的目标位置，可为 list/tuple/np.ndarray
        #  - bxy = np.asarray(...).flatten()[:2]：将输入转为 1D float 数组并取前两个元素作为 x,y
        #  - self.cart_parts 中每项为 (body_id, local_pos(np.array([x,y,z])), local_quat)
        # p.resetBasePositionAndOrientation(objectUniqueId, posList, ornList, physicsClientId=...)
        #  - objectUniqueId: body id
        #  - posList: 世界坐标系下的位置 [x,y,z]
        #  - ornList: 四元数 [x,y,z,w]
        # 注意：若在多 client 场景推荐传入 physicsClientId=self.client_id
        bxy = np.asarray(base_xy, dtype=float).flatten()[:2]
        for bid, lp, lq in self.cart_parts:
            p.resetBasePositionAndOrientation(
                bid, [bxy[0] + lp[0], bxy[1] + lp[1], lp[2]], lq)
            
    def _draw_axes(self) -> None:
        # 在场景中绘制 XYZ 轴的调试线和文本，用于可视化坐标系基准
        # al: 轴长度（米）
        # o: 轴原点位置，使用 table_height 稍微抬高以避免与地面重合
        # addUserDebugLine(fromXYZ, toXYZ, colorRGB, lineWidth=...)
        # addUserDebugText(text, textPosition, textColor, textSize=...)
        al = 0.12
        o = [0.0, 0.0, self.table_height + 0.001]
        p.addUserDebugLine(o, [al, 0, o[2]], [1, 0, 0], lineWidth=2)
        p.addUserDebugLine(o, [0, al, o[2]], [0, 0.8, 0], lineWidth=2)
        p.addUserDebugLine(o, [0, 0, o[2] + al], [0, 0, 1], lineWidth=2)
        p.addUserDebugText("X", [al + 0.02, 0, o[2]], [1, 0, 0], textSize=1)
        p.addUserDebugText("Y", [0, al + 0.02, o[2]], [0, 0.8, 0], textSize=1)
        p.addUserDebugText("Z", [0, 0, o[2] + al + 0.02], [0, 0, 1], textSize=1)

    def create_marker(self, radius, color, pos):
        # 创建一个视觉标记（球形），常用于标注目标点或参考点
        # 参数：
        #  - radius: 球半径（米）
        #  - color: RGBA 列表或元组，如 [r,g,b,a]（0~1）
        #  - pos: 三元列表/数组 [x,y,z] 表示世界坐标位置
        # 返回：body id（整型），该 body 为静态视觉占位（baseMass=0）
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos)
    
    def update_marker(self, body_id, pos):
        # 更新标记体的位置
        # resetBasePositionAndOrientation(objectUniqueId, posList, ornList)
        #  - pos: [x,y,z]
        #  - orn: 四元数 [x,y,z,w]，这里用单位四元数表示不旋转
        p.resetBasePositionAndOrientation(body_id, pos, [0, 0, 0, 1])

    def draw_polyline(self, points, color, width=1.5):
        # 用若干条短线连接 points 中的点以绘制折线
        # points: 可迭代的点序列（每个点为可转为 list 的长度为3 的向量，如 np.array）
        # color: RGB 或 RGBA 列表（颜色），width: 线宽
        for i in range(len(points) - 1):
            p.addUserDebugLine(points[i].tolist(), points[i + 1].tolist(), color, lineWidth=width)

    def update_status(self, text):
        # 在视图中添加或更新一段状态文本（位于场景左下角）
        # addUserDebugText(text, textPosition, textColor, textSize, replaceItemUniqueId=...)
        # replaceItemUniqueId: 如果传入已有文本的 id，会替换该文本；-1 表示新建
        self.status_text_id = p.addUserDebugText(
            text, [0.02, -0.26, self.table_height + 0.52], [0.1, 0.1, 0.1],
            textSize=1.2,
            replaceItemUniqueId=self.status_text_id if self.status_text_id else -1)

    def capture_frame(self, width, height):
        # 从当前调试相机获取图像
        # getCameraImage(width, height) 返回 (width, height, rgbPixels, depthPixels, segMask)
        # rgbPixels 格式通常是扁平 RGBA，下面转换为 HxWx4，然后取前三通道 RGB
        _, _, rgb, _, _ = p.getCameraImage(width, height)
        return np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

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

        # 禁用关节阻尼，且关闭内置电机以便手动控制/命令速度
        # p.changeDynamics(bodyUniqueId, linkIndex, linearDamping=..., angularDamping=...)
        #  - linearDamping / angularDamping: 阻尼系数（通常对速度积分有影响），设 0 表示无额外阻尼
        # p.setJointMotorControl2(bodyUniqueId, jointIndex, controlMode, targetVelocity=..., force=...)
        #  - controlMode 为 p.VELOCITY_CONTROL 表示使用速度控制模式
        #  - 在这里只把 force=0（最大驱动力为 0），等于禁用内置伺服/电机输出，让关节不被内部控制器驱动
        #  - 若希望使用外部命令驱动关节，可 later 使用 setJointMotorControlArray 或 resetJointState/控制器
        for ji in self.active_joints:
            p.changeDynamics(self.body_id, ji, linearDamping=0, angularDamping=0)
            p.setJointMotorControl2(self.body_id, ji, p.VELOCITY_CONTROL, force=0)

        # q_nominal: 用作默认/参考关节角度向量（长度 = 活动关节自由度）
        # 初始化为零，后续重置或用于 IK/rest pose 参考
        self.q_nominal = np.zeros(self.dof)

    def move_base(self, v_base, dt):
        """
        移动机器人基座（XY 平面）：
        输入:
          - v_base: 可迭代，期望基座速度 (vx, vy, ...)（单位：m/s），函数只使用前两个分量
          - dt: 时间步长（秒）
        输出: None（在仿真中更新 base_pos 与在 pybullet 中重置基座位置）
        行为：将速度裁剪到配置的 base_vel_limit，积分位移，保持原 z 高度不变，更新场景中的可视小车位置
        """
        v = np.clip(np.asarray(v_base, dtype=float).flatten()[:2],
                     -self.config.base_vel_limit, self.config.base_vel_limit)
        self.base_pos += v * dt
        # 获取当前 base 的 z 高度以在更新时保留
        bz = p.getBasePositionAndOrientation(self.body_id)[0][2]
        # resetBasePositionAndOrientation(objectUniqueId, posList, ornList)
        p.resetBasePositionAndOrientation(
            self.body_id, [self.base_pos[0], self.base_pos[1], bz], [0, 0, 0, 1])
        # 更新场景中小车视觉部件的位置
        self.scene.update_cart_pose(self.base_pos)

    def _set_link_colors(self):
        """
        设置各 link 的可视颜色（只修改 visual，不改变碰撞形状）
        输入: 无
        输出: 无（直接调用 p.changeVisualShape）
        说明: colors 字典 key 为 linkIndex（-1 表示 base），value 为 RGBA 列表
        """
        colors = {-1: [0.15]*3+[1], 0: [0.92]*3+[1], 1: [0.12,0.46,0.70,1],
                  2: [0.92]*3+[1], 3: [0.12,0.46,0.70,1], 4: [0.92]*3+[1],
                  5: [0.15]*3+[1]}
        for li, rgba in colors.items():
            # changeVisualShape(bodyUniqueId, linkIndex, rgbaColor=..., specularColor=...)
            p.changeVisualShape(self.body_id, li, rgbaColor=rgba, specularColor=[0.6]*3)

    def get_joint_state(self):
                """
                获取活动关节的当前状态
                输入: 无（使用 self.active_joints）
                输出: (q, dq)
                    - q: np.array，关节位置（弧度），形状 (dof,)
                    - dq: np.array，关节速度（rad/s），形状 (dof,)
                使用 p.getJointStates 返回的元组，每个元素格式为 (position, velocity, reactionForces, torque)
                """
                st = p.getJointStates(self.body_id, self.active_joints)
                return np.array([s[0] for s in st]), np.array([s[1] for s in st])

    def get_ee_pose(self):
                """
                获取末端执行器（EE）的位姿
                输入: 无
                输出: (pos, quat)
                    - pos: np.array 长度 3，世界坐标系下的末端位置 [x,y,z]
                    - quat: np.array 长度 4，末端朝向四元数 [x,y,z,w]
                使用 p.getLinkState 返回的索引 4（worldLinkFramePosition）和 5（worldLinkFrameOrientation）
                """
                s = p.getLinkState(self.body_id, self.ee_link_index, computeForwardKinematics=True)
                return np.array(s[4], dtype=float), np.array(s[5], dtype=float)

    def get_link_origin(self, link_idx):
                """
                获取指定连杆的世界坐标原点位置
                输入:
                    - link_idx: 连杆索引（int）
                输出: pos: np.array 长度 3，世界坐标系下该连杆原点位置
                """
                s = p.getLinkState(self.body_id, link_idx, computeForwardKinematics=True)
                return np.array(s[4], dtype=float)

    def calculate_ik(self, target_pos, target_quat):
        """
        计算逆运动学求解关节角度
        输入:
          - target_pos: 目标位姿位置 [x,y,z]
          - target_quat: 目标位姿四元数 [x,y,z,w]
        输出: np.array 长度 dof，活动关节的关节角度解（弧度）
        说明: 这里为求解设置了 lower/upper limits, joint ranges, rest poses 以及迭代/精度参数
        """
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
        """
        通过 IK 计算并直接重置关节到目标位姿（即时设置，不通过控制器）
        输入: target_pos, target_quat
        输出: None（将关节状态设置为 IK 解并更新 q_nominal）
        注意: resetJointState 会直接修改关节状态，适合初始化或瞬时跳变
        """
        qt = self.calculate_ik(target_pos, target_quat)
        for i, ji in enumerate(self.active_joints):
            p.resetJointState(self.body_id, ji, qt[i])
        self.q_nominal = qt.copy()

    def get_link_jacobian(self, link_idx, q, dq):
                """
                计算指定连杆在给定关节位置/速度下的雅可比矩阵
                输入:
                    - link_idx: 连杆索引
                    - q: 关节位置数组，形状 (dof,)
                    - dq: 关节速度数组，形状 (dof,)
                输出: (jt, jr)
                    - jt: 线速度雅可比，numpy 数组形状 (3, dof)
                    - jr: 角速度雅可比，numpy 数组形状 (3, dof)
                说明: p.calculateJacobian 的第三个参数为 link 相对于自身参考点的位置偏移（这里用 [0,0,0]）
                """
                z = np.zeros_like(q)
                jt, jr = p.calculateJacobian(self.body_id, link_idx, [0, 0, 0],
                                                                            q.tolist(), dq.tolist(), z.tolist())
                return np.array(jt, dtype=float), np.array(jr, dtype=float)

    def get_ee_jacobian(self, q, dq):
        """
        获取末端的完整 6xN 雅可比（先是线速度部分，再是角速度部分）
        输入: q, dq
        输出: numpy 数组，形状 (6, dof)
        """
        jt, jr = self.get_link_jacobian(self.ee_link_index, q, dq)
        return np.vstack([jt, jr])

    def get_augmented_ee_jacobian(self, q, dq):
        """
        返回扩展后的雅可比，包含基座自由度 (x,y) + 机械臂关节
        输入: q, dq
        输出: numpy 数组，形状 (6, 2 + dof)
        说明: base_block 为基座对末端速度的影响矩阵（这里假设基座仅有平移自由度 x,y）
        """
        jt, jr = self.get_link_jacobian(self.ee_link_index, q, dq)
        base_block = np.vstack([
            np.array([[1, 0], [0, 1], [0, 0]], dtype=float),
            np.zeros((3, 2), dtype=float),
        ])
        return np.hstack([base_block, np.vstack([jt, jr])])

    def get_link_cbf_row_aug(self, link_idx, normal, q, dq):
                """
                为 CBF 相关计算返回一行约束表达（扩展基座部分）
                输入:
                    - link_idx: 连杆索引
                    - normal: 法向量（长度 >=3），通常世界坐标系下碰撞法向量
                    - q, dq: 关节位置与速度
                输出: 1D numpy 数组，拼接 [normal_x, normal_y, normal @ jt]
                    - 前两项为基座在 XY 上的法向量分量
                    - 后面为法向量与雅可比的乘积（长度 dof），整体长度为 2 + dof
                备注: 这里 normal @ jt 是 (3,) @ (3 x dof) -> (do f, )
                """
                jt, _ = self.get_link_jacobian(link_idx, q, dq)
                return np.concatenate([normal[:2], normal @ jt])

    def get_closest_point_to_obstacle(self, link_idx, obstacle_body_id, max_dist=1.0):
                """
                查找指定连杆到障碍物的最近点（使用 PyBullet 的 getClosestPoints）
                输入:
                    - link_idx: 在 bodyA 上的 link 索引
                    - obstacle_body_id: 障碍物的 body id
                    - max_dist: 最大搜索距离（米）
                输出: None 或 (pos, dist, n)
                    - pos: 最近点在世界坐标系下的位置（np.array 长度 3）
                    - dist: 最近距离（标量）
                    - n: 单位法向量（从障碍指向本体？依据 PyBullet 返回的 normal）
                说明: contacts 是一个 contact dict 列表，常用字段：
                    - c[5]：接触点位置（world position on B）
                    - c[7]：接触法向量
                    - c[8]：距离（距离为正表示分离）
                """
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
                """
                在给定世界空间点处计算 CBF 行（扩展基座），用于在特定点上评估雅可比
                输入:
                    - link_idx: 连杆索引
                    - world_point: 世界坐标系下的点（长度 3）
                    - normal: 法向量
                    - q, dq: 关节位置与速度
                输出: 1D numpy 数组，格式同 get_link_cbf_row_aug
                说明: 先把 world_point 转为连杆局部坐标，再计算该点处的雅可比
                """
                ls = p.getLinkState(self.body_id, link_idx, computeForwardKinematics=True)
                inv_p, inv_o = p.invertTransform(ls[4], ls[5])
                local_pt, _ = p.multiplyTransforms(inv_p, inv_o, world_point.tolist(), [0, 0, 0, 1])
                z = np.zeros_like(q)
                jt, _ = p.calculateJacobian(self.body_id, link_idx, list(local_pt),
                                                                         q.tolist(), dq.tolist(), z.tolist())
                jt = np.array(jt, dtype=float)
                return np.concatenate([normal[:2], normal @ jt])

    def command_joint_velocities(self, dq_cmd):
                """
                通过 PyBullet 的速度控制接口向活动关节下发速度命令
                输入:
                    - dq_cmd: 期望关节速度数组（长度 dof）
                输出: None（直接在仿真中设置目标速度）
                行为: 将 dq_cmd 裁剪到 dq_limit，然后调用 setJointMotorControlArray
                setJointMotorControlArray(bodyUniqueId, jointIndices, controlMode, targetVelocities=..., forces=...)
                """
                dq_clip = np.clip(dq_cmd, -self.config.dq_limit, self.config.dq_limit)
                p.setJointMotorControlArray(
                        self.body_id, self.active_joints, p.VELOCITY_CONTROL,
                        targetVelocities=dq_clip.tolist(),
                        forces=[self.config.ee_force_limit] * self.dof)