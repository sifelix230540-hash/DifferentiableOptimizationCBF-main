import math

import numpy as np
import pybullet as p

from CBF_experiment.active.pybullet.welding_320_common import (
    ExperimentConfig,
    SimulationScene,
    _prepare_package_urdf,
)
from CBF_experiment.active.pybullet.welding_320_geometry import SurfaceDistanceEngine

WORKPIECE_PACKAGE_NAME = "中组立0725(1).stp.SLDASM"
WORKPIECE_PACKAGE_ALIAS = "workpiece_scene"


class JakaRobot:
    """封装 9 轴机器人加载、运动学和速度指令。"""

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
        self.link_name_by_index = {}
        for joint_index in range(self.num_joints):
            joint_info = p.getJointInfo(self.body_id, joint_index)
            joint_type = joint_info[2]
            self.link_name_by_index[joint_index] = joint_info[12].decode()
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
        self.welding_gun_base_link_index = -1
        self.robobase_link_index = -1
        self.welding_gun_links = []
        for joint_index in range(self.num_joints):
            link_name = p.getJointInfo(self.body_id, joint_index)[12].decode()
            if link_name == "weld_point":
                self.ee_link_index = joint_index
            if link_name == "welding_gun_base":
                self.welding_gun_base_link_index = joint_index
            if link_name == "robobase":
                self.robobase_link_index = joint_index
            if link_name in ("welding_gun_base", "weld_point"):
                self.welding_gun_links.append(joint_index)

        self.cbf_link_indices = self.prismatic_joints[2:] + list(self.revolute_joints) + self.welding_gun_links
        self.rear_six_link_indices = [int(link_index) for link_index in self.revolute_joints]

        for joint_index in self.active_joints:
            p.changeDynamics(self.body_id, joint_index, linearDamping=0, angularDamping=0)
            p.setJointMotorControl2(self.body_id, joint_index, p.VELOCITY_CONTROL, force=0)

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
        self._surface_engine = SurfaceDistanceEngine(config)
        robot_link_roles = {
            int(link_index): "robot_rear_six" if int(link_index) in self.rear_six_link_indices else "robot"
            for link_index in self.cbf_link_indices
        }
        self._surface_engine.register_body(
            self.body_id,
            link_indices=self.cbf_link_indices,
            role="robot",
            link_role_map=robot_link_roles,
        )
        self._surface_obstacle_links: dict[int, list[int] | None] = {}

    def get_link_name(self, link_index: int) -> str:
        return self.link_name_by_index.get(int(link_index), f"link_{int(link_index)}")

    def register_surface_obstacle(self, body_id: int, link_indices: list[int] | None = None):
        normalized = None if link_indices is None else [int(li) for li in link_indices]
        self._surface_obstacle_links[int(body_id)] = normalized
        self._surface_engine.register_body(int(body_id), link_indices=normalized, role="obstacle")

    def get_surface_visualization_clouds(
        self,
        body_id: int,
        link_indices: list[int] | None = None,
        max_points_per_link: int | None = None,
    ) -> list[dict]:
        if getattr(self, "_surface_engine", None) is None:
            return []
        query_center_world = None
        if int(body_id) != int(self.body_id) and bool(getattr(self.config, "obstacle_local_dense_enabled", False)):
            query_center_world, _ = self.get_robobase_pose()
        return self._surface_engine.get_visualization_clouds(
            int(body_id),
            link_indices=None if link_indices is None else [int(li) for li in link_indices],
            max_points_per_link=max_points_per_link,
            query_center_world=query_center_world,
        )

    @staticmethod
    def get_body_link_name(body_id: int, link_index: int) -> str:
        if int(link_index) < 0:
            return "base_link"
        return p.getJointInfo(body_id, int(link_index))[12].decode()

    def get_joint_state(self):
        states = p.getJointStates(self.body_id, self.active_joints)
        return np.array([state[0] for state in states]), np.array([state[1] for state in states])

    def set_joint_state(self, q, dq=None):
        q = np.array(q, dtype=float)
        dq_vec = np.zeros_like(q) if dq is None else np.array(dq, dtype=float)
        for i, joint_index in enumerate(self.active_joints):
            p.resetJointState(self.body_id, joint_index, float(q[i]), float(dq_vec[i]))

    def get_ee_pose(self):
        state = p.getLinkState(self.body_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(state[4], dtype=float), np.array(state[5], dtype=float)

    def get_link_pose(self, link_index):
        state = p.getLinkState(self.body_id, link_index, computeForwardKinematics=True)
        return np.array(state[4], dtype=float), np.array(state[5], dtype=float)

    def get_link_origin(self, link_index):
        pos, _ = self.get_link_pose(link_index)
        return pos

    def transform_link_points_to_world(self, link_index, local_points, local_normals=None):
        world_pos, world_quat = self.get_link_pose(link_index)
        rot = np.array(p.getMatrixFromQuaternion(world_quat.tolist()), dtype=float).reshape(3, 3)
        pts_local = np.asarray(local_points, dtype=float).reshape(-1, 3)
        pts_world = (rot @ pts_local.T).T + world_pos.reshape(1, 3)
        if local_normals is None:
            return pts_world, None
        normals_local = np.asarray(local_normals, dtype=float).reshape(-1, 3)
        normals_world = (rot @ normals_local.T).T
        return pts_world, normals_world

    def get_surface_local_samples(self, link_indices: list[int] | None = None) -> dict[int, dict]:
        selected = None if link_indices is None else {int(li) for li in link_indices}
        body_clouds = getattr(getattr(self, "_surface_engine", None), "_body_clouds", {}).get(int(self.body_id), {})
        samples: dict[int, dict] = {}
        for link_index, cloud in body_clouds.items():
            if selected is not None and int(link_index) not in selected:
                continue
            samples[int(link_index)] = {
                "link_index": int(link_index),
                "link_name": str(cloud.link_name),
                "local_points": np.asarray(cloud.local_points, dtype=float).copy(),
                "local_normals": np.asarray(cloud.local_normals, dtype=float).copy(),
                "role": str(cloud.role),
            }
        return samples

    def get_robobase_pose(self):
        if int(self.robobase_link_index) < 0:
            pos, quat = p.getBasePositionAndOrientation(self.body_id)
            return np.array(pos, dtype=float), np.array(quat, dtype=float)
        state = p.getLinkState(self.body_id, self.robobase_link_index, computeForwardKinematics=True)
        return np.array(state[4], dtype=float), np.array(state[5], dtype=float)

    def _build_rest_poses(self, rest_poses=None):
        rest_full = list(self._ik_rest)
        if rest_poses is None:
            return rest_full
        rest_active = np.array(rest_poses, dtype=float)
        for i, joint_index in enumerate(self.active_joints):
            if i < len(rest_active):
                rest_full[joint_index] = float(rest_active[i])
        return rest_full

    def calculate_ik(self, target_pos, target_quat, rest_poses=None):
        ik = p.calculateInverseKinematics(
            self.body_id,
            self.ee_link_index,
            target_pos,
            target_quat,
            lowerLimits=self._ik_lower,
            upperLimits=self._ik_upper,
            jointRanges=self._ik_ranges,
            restPoses=self._build_rest_poses(rest_poses),
            maxNumIterations=500,
            residualThreshold=1e-6,
        )
        return np.array(ik[: self.dof], dtype=float)

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

    def get_closest_points_to_obstacle(self, link_index, obs_body_id, max_dist=1.0):
        if getattr(self, "_surface_engine", None) is not None:
            query_center_world = None
            if int(obs_body_id) != int(self.body_id) and bool(getattr(self.config, "obstacle_local_dense_enabled", False)):
                query_center_world, _ = self.get_robobase_pose()
            result = self._surface_engine.query_link_to_body(
                robot_body_id=self.body_id,
                robot_link_index=int(link_index),
                obstacle_body_id=int(obs_body_id),
                obstacle_link_indices=self._surface_obstacle_links.get(int(obs_body_id)),
                max_dist=max_dist,
                query_center_world=query_center_world,
            )
            if result is not None:
                return result
        contacts = p.getClosestPoints(self.body_id, obs_body_id, max_dist, linkIndexA=link_index)
        if not contacts:
            return None
        best = min(contacts, key=lambda contact: contact[8])
        obs_link_index = int(best[4])
        point_on_link = np.array(best[5], dtype=float)
        point_on_obstacle = np.array(best[6], dtype=float)
        signed_dist = float(best[8])
        normal_on_obstacle = np.array(best[7], dtype=float)
        nl = np.linalg.norm(normal_on_obstacle)
        if nl <= 1e-9:
            fallback = point_on_link - point_on_obstacle
            fl = np.linalg.norm(fallback)
            normal_on_obstacle = fallback / fl if fl > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            normal_on_obstacle = normal_on_obstacle / nl
        return {
            "obs_link_index": obs_link_index,
            "obs_link_name": self.get_body_link_name(obs_body_id, obs_link_index),
            "point_on_link": point_on_link,
            "point_on_obstacle": point_on_obstacle,
            "signed_dist": signed_dist,
            "normal_on_obstacle": normal_on_obstacle,
        }

    def get_closest_point_to_obstacle(self, link_index, obs_body_id, max_dist=1.0):
        closest = self.get_closest_points_to_obstacle(link_index, obs_body_id, max_dist=max_dist)
        if closest is None:
            return None
        return (
            closest["point_on_link"],
            closest["signed_dist"],
            closest["normal_on_obstacle"],
        )

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
        return np.array([state[0] for state in states])


class WorkpieceModel:
    """加载工件 URDF，并提供焊点坐标系查询。"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        urdf_path, search_root = _prepare_package_urdf(
            config.workpiece_urdf_path,
            package_name=WORKPIECE_PACKAGE_NAME,
            package_alias=WORKPIECE_PACKAGE_ALIAS,
            remove_collision=config.ignore_all_collisions,
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

        if config.ignore_all_collisions:
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


class URDFObstacle:
    """将工件 body 包装成控制器可用的 CBF 障碍物。"""

    def __init__(self, body_id: int, cbf_link_indices: list[int] | None = None):
        self._bid = body_id
        self._cbf_links = cbf_link_indices

    @property
    def body_id(self):
        return self._bid

    @property
    def cbf_link_indices(self):
        return self._cbf_links

    def disable_collision_with(self, robot_body_id, num_joints):
        num_obs_links = p.getNumJoints(self.body_id)
        for obs_link in range(-1, num_obs_links):
            p.setCollisionFilterPair(robot_body_id, self.body_id, -1, obs_link, enableCollision=0)
            for robot_link in range(num_joints):
                p.setCollisionFilterPair(robot_body_id, self.body_id, robot_link, obs_link, enableCollision=0)

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
