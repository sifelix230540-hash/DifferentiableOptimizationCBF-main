"""仿真场景模块：初始化 PyBullet，加载 9 轴机器人与工件。"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "Super_config.json"


# ── 配置加载 ────────────────────────────────────────


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(rel: str) -> str:
    return str(PROJECT_ROOT / rel)


# ── URDF 预处理 ─────────────────────────────────────


def _prepare_urdf(
    urdf_path: str,
    package_name: str | None = None,
    package_alias: str | None = None,
    remove_collision: bool = False,
) -> tuple[str, str]:
    """处理 URDF 中的 package:// 引用，非 ASCII 路径自动复制到临时目录。

    Returns
    -------
    (resolved_urdf, search_root)
        resolved_urdf: 可被 PyBullet 加载的 URDF 绝对路径
        search_root:   需传给 p.setAdditionalSearchPath 的目录
    """
    urdf_path = os.path.abspath(urdf_path)
    source_root = os.path.dirname(os.path.dirname(urdf_path))
    source_pkg_name = os.path.basename(source_root)
    package_name = package_name or source_pkg_name
    package_alias = package_alias or package_name

    copy_required = (package_alias != source_pkg_name)
    try:
        urdf_path.encode("ascii")
        source_root.encode("ascii")
        package_alias.encode("ascii")
    except UnicodeEncodeError:
        copy_required = True

    if not copy_required and not remove_collision:
        return urdf_path, os.path.dirname(source_root)

    tmp_root = os.path.join(tempfile.gettempdir(), "pybullet_urdf")
    tmp_pkg = os.path.join(tmp_root, package_alias)
    if os.path.exists(tmp_pkg):
        shutil.rmtree(tmp_pkg, ignore_errors=True)
        for _ in range(30):
            if not os.path.exists(tmp_pkg):
                break
            time.sleep(0.1)
    shutil.copytree(source_root, tmp_pkg, dirs_exist_ok=True)

    rel_urdf_dir = os.path.relpath(os.path.dirname(urdf_path), source_root)
    new_urdf_dir = os.path.join(tmp_pkg, rel_urdf_dir)
    os.makedirs(new_urdf_dir, exist_ok=True)
    new_urdf = os.path.join(new_urdf_dir, "model.urdf")

    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_text = f.read()

    urdf_text = urdf_text.replace(
        f"package://{package_name}/", f"package://{package_alias}/"
    )
    if source_pkg_name != package_name:
        urdf_text = urdf_text.replace(
            f"package://{source_pkg_name}/", f"package://{package_alias}/"
        )

    if remove_collision:
        root = ET.fromstring(urdf_text)
        for link_el in root.findall("link"):
            for coll_el in list(link_el.findall("collision")):
                link_el.remove(coll_el)
        urdf_text = ET.tostring(root, encoding="unicode")

    with open(new_urdf, "w", encoding="utf-8", newline="\n") as f:
        f.write(urdf_text)

    print(f"[info] URDF 已复制到临时目录: {tmp_pkg}")
    return new_urdf, tmp_root


# ── 仿真场景 ────────────────────────────────────────


class SimulationScene:
    """PyBullet 仿真场景：物理引擎、地面、相机。"""

    def __init__(self, cfg: dict):
        sim = cfg.get("simulation", {})
        self.dt = float(sim.get("dt", 1.0 / 240.0))
        gravity = sim.get("gravity", [0.0, 0.0, -9.81])

        self.client_id = p.connect(p.GUI, options="--width=1920 --height=1080")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*gravity)
        p.setTimeStep(self.dt)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(rgbBackground=[1, 1, 1])

        p.resetDebugVisualizerCamera(
            cameraDistance=float(sim.get("camera_distance", 1.4)),
            cameraYaw=float(sim.get("camera_yaw", -215.0)),
            cameraPitch=float(sim.get("camera_pitch", -26.0)),
            cameraTargetPosition=sim.get("camera_target", [0.3, 0.2, 0.6]),
        )

        plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.95, 0.95, 0.95, 1.0])
        self._draw_axes()

    @staticmethod
    def _draw_axes(length: float = 0.12):
        o = [0, 0, 0.001]
        p.addUserDebugLine(o, [length, 0, 0.001], [1, 0, 0], lineWidth=2)
        p.addUserDebugLine(o, [0, length, 0.001], [0, 0.8, 0], lineWidth=2)
        p.addUserDebugLine(o, [0, 0, 0.001 + length], [0, 0, 1], lineWidth=2)

    def enable_rendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    @staticmethod
    def create_marker(radius: float, color, pos) -> int:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos)

    @staticmethod
    def draw_frame(pos, quat, length: float = 0.10, width: float = 3.0,
                   replace_ids: list[int] | None = None) -> list[int]:
        """画坐标系三轴，返回 3 个 debug line id（可用于下次 replace 实现跟随）。"""
        rot = np.array(p.getMatrixFromQuaternion(
            np.asarray(quat, dtype=float).tolist()
        ), dtype=float).reshape(3, 3)
        origin = np.asarray(pos, dtype=float)
        colors = [[1, 0, 0], [0, 0.8, 0], [0, 0, 1]]
        ids: list[int] = []
        for axis in range(3):
            end = origin + length * rot[:, axis]
            rid = replace_ids[axis] if replace_ids and axis < len(replace_ids) else -1
            ids.append(p.addUserDebugLine(
                origin.tolist(), end.tolist(), colors[axis],
                lineWidth=width, replaceItemUniqueId=rid,
            ))
        return ids

    @staticmethod
    def draw_polyline(pts, color, width: float = 1.5):
        for i in range(len(pts) - 1):
            a = pts[i].tolist() if hasattr(pts[i], "tolist") else list(pts[i])
            b = pts[i + 1].tolist() if hasattr(pts[i + 1], "tolist") else list(pts[i + 1])
            p.addUserDebugLine(a, b, color, lineWidth=width)


# ── 机器人 ──────────────────────────────────────────


class Robot:
    """9 轴机器人：3 轴龙门 + 6 轴关节臂 + 焊枪。"""

    def __init__(self, cfg, scene=None):
        if hasattr(cfg, "_cfg"):
            cfg = cfg._cfg
        robot_cfg = cfg.get("robot", {})
        sim_cfg = cfg.get("simulation", {})
        self.dt = float(sim_cfg.get("dt", 1.0 / 240.0))
        self.dq_limit = float(robot_cfg.get("dq_limit", 1.0))
        self.base_vel_limit = float(robot_cfg.get("base_vel_limit", 0.4))

        urdf_abs = _resolve(robot_cfg.get("urdf", "assets/robots/9_axis/urdf/9_axis.urdf"))
        urdf_file, search_root = _prepare_urdf(urdf_abs)
        p.setAdditionalSearchPath(search_root)

        self.body_id = p.loadURDF(
            urdf_file,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            flags=(p.URDF_USE_MATERIAL_COLORS_FROM_MTL
                   | p.URDF_USE_SELF_COLLISION
                   | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT),
        )

        self.num_joints = p.getNumJoints(self.body_id)
        self.active_joints: list[int] = []
        self.prismatic_joints: list[int] = []
        self.revolute_joints: list[int] = []
        self.link_name_by_index: dict[int, str] = {}

        for i in range(self.num_joints):
            info = p.getJointInfo(self.body_id, i)
            jtype = info[2]
            self.link_name_by_index[i] = info[12].decode()
            if jtype == p.JOINT_PRISMATIC:
                self.active_joints.append(i)
                self.prismatic_joints.append(i)
            elif jtype == p.JOINT_REVOLUTE:
                self.active_joints.append(i)
                self.revolute_joints.append(i)

        self.n_pris = len(self.prismatic_joints)
        self.n_revo = len(self.revolute_joints)
        self.dof = len(self.active_joints)

        self.ee_link_index = self.revolute_joints[-1] if self.revolute_joints else -1
        self.welding_gun_base_link_index = -1
        self.robobase_link_index = -1
        self.welding_gun_links: list[int] = []

        for i in range(self.num_joints):
            name = self.link_name_by_index[i]
            if name == "weld_point":
                self.ee_link_index = i
            elif name == "welding_gun_base":
                self.welding_gun_base_link_index = i
            elif name == "robobase":
                self.robobase_link_index = i
            if name in ("welding_gun_base", "weld_point"):
                self.welding_gun_links.append(i)

        self.rear_six_link_indices: list[int] = [int(j) for j in self.revolute_joints]

        self._ik_lower: list[float] = []
        self._ik_upper: list[float] = []
        self._ik_ranges: list[float] = []
        self._ik_rest: list[float] = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.body_id, i)
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

        for ji in self.active_joints:
            p.changeDynamics(self.body_id, ji, linearDamping=0, angularDamping=0)
            p.setJointMotorControl2(self.body_id, ji, p.VELOCITY_CONTROL, force=0)

        gantry_q = robot_cfg.get("gantry_initial_q", [0.0, 0.0, 0.0])
        for axis_id, ji in enumerate(self.prismatic_joints):
            if axis_id < len(gantry_q):
                p.resetJointState(self.body_id, ji, float(gantry_q[axis_id]))
                self._ik_rest[ji] = float(gantry_q[axis_id])

    # ── 状态 ────────────────────────────────────

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        states = p.getJointStates(self.body_id, self.active_joints)
        return (
            np.array([s[0] for s in states]),
            np.array([s[1] for s in states]),
        )

    def set_joint_state(self, q, dq=None):
        q = np.asarray(q, dtype=float)
        dq = np.zeros_like(q) if dq is None else np.asarray(dq, dtype=float)
        for i, ji in enumerate(self.active_joints):
            p.resetJointState(self.body_id, ji, float(q[i]), float(dq[i]))

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(self.body_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(state[4], dtype=float), np.array(state[5], dtype=float)

    def get_link_pose(self, link_index: int) -> tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(self.body_id, link_index, computeForwardKinematics=True)
        return np.array(state[4], dtype=float), np.array(state[5], dtype=float)

    def get_robobase_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.robobase_link_index < 0:
            pos, quat = p.getBasePositionAndOrientation(self.body_id)
            return np.array(pos, dtype=float), np.array(quat, dtype=float)
        state = p.getLinkState(self.body_id, self.robobase_link_index, computeForwardKinematics=True)
        return np.array(state[4], dtype=float), np.array(state[5], dtype=float)

    # ── 运动学 ──────────────────────────────────

    def calculate_ik(self, target_pos, target_quat, rest_poses=None) -> np.ndarray:
        rest = list(self._ik_rest)
        if rest_poses is not None:
            for i, ji in enumerate(self.active_joints):
                if i < len(rest_poses):
                    rest[ji] = float(rest_poses[i])
        ik = p.calculateInverseKinematics(
            self.body_id,
            self.ee_link_index,
            target_pos,
            target_quat,
            lowerLimits=self._ik_lower,
            upperLimits=self._ik_upper,
            jointRanges=self._ik_ranges,
            restPoses=rest,
            maxNumIterations=500,
            residualThreshold=1e-6,
        )
        return np.array(ik[: self.dof], dtype=float)

    def get_ee_jacobian(self, q, dq) -> np.ndarray:
        zeros = np.zeros_like(q)
        jt, jr = p.calculateJacobian(
            self.body_id,
            self.ee_link_index,
            [0, 0, 0],
            q.tolist(),
            dq.tolist(),
            zeros.tolist(),
        )
        return np.vstack([np.array(jt), np.array(jr)])

    def get_link_jacobian(self, link_index: int, q, dq):
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

    def get_link_linear_jacobian_at_world_point(self, link_index: int, world_point, q, dq) -> np.ndarray:
        link_state = p.getLinkState(self.body_id, int(link_index), computeForwardKinematics=True)
        inv_pos, inv_quat = p.invertTransform(link_state[4], link_state[5])
        local_point, _ = p.multiplyTransforms(
            inv_pos,
            inv_quat,
            np.asarray(world_point, dtype=float).reshape(3).tolist(),
            [0, 0, 0, 1],
        )
        zeros = np.zeros_like(q)
        jt, _jr = p.calculateJacobian(
            self.body_id,
            int(link_index),
            list(local_point),
            q.tolist(),
            dq.tolist(),
            zeros.tolist(),
        )
        return np.asarray(jt, dtype=float)

    # ── 速度指令 ────────────────────────────────

    def command_velocities(self, u_cmd):
        lb = np.concatenate([
            np.full(self.n_pris, -self.base_vel_limit),
            np.full(self.n_revo, -self.dq_limit),
        ])
        u = np.clip(u_cmd, lb, -lb)
        q, _ = self.get_joint_state()
        q_new = q + u * self.dt
        for i, ji in enumerate(self.active_joints):
            p.resetJointState(self.body_id, ji, float(q_new[i]), float(u[i]))

    # ── 兼容旧接口 stub ────────────────────────────

    @property
    def cbf_link_indices(self) -> list[int]:
        excluded = {self.ee_link_index}
        candidates = self.prismatic_joints[2:] + list(self.revolute_joints) + self.welding_gun_links
        return [i for i in candidates if i not in excluded]

    @property
    def total_dof(self) -> int:
        return self.dof

    def register_surface_obstacle(self, body_id, link_indices=None):
        self._surface_obstacles = getattr(self, "_surface_obstacles", {})
        self._surface_obstacles[int(body_id)] = link_indices

    def get_surface_visualization_clouds(self, body_id, **kw) -> list[dict]:
        max_pts = kw.get("max_points_per_link", 500)
        link_indices = kw.get("link_indices", None)
        bid = int(body_id)
        vsd = p.getVisualShapeData(bid)
        if link_indices is not None:
            allowed = set(int(i) for i in link_indices)
        else:
            allowed = None
        clouds: list[dict] = []
        for shape in vsd:
            li = int(shape[1])
            if allowed is not None and li not in allowed:
                continue
            geom_type = int(shape[2])
            mesh_file = shape[4]
            if isinstance(mesh_file, bytes):
                mesh_file = mesh_file.decode("utf-8", errors="replace")
            mesh_file = mesh_file.strip()
            if geom_type != p.GEOM_MESH or not mesh_file:
                continue
            mesh_path = Path(mesh_file)
            if not mesh_path.is_absolute():
                continue
            if not mesh_path.exists():
                continue
            try:
                import trimesh
                mesh = trimesh.load(str(mesh_path), force="mesh")
                pts_local = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
            except Exception:
                continue
            if pts_local.shape[0] == 0:
                continue
            mesh_scale = np.array(shape[3], dtype=float).reshape(3)
            pts_local = pts_local * mesh_scale.reshape(1, 3)
            if li < 0:
                pos, quat = p.getBasePositionAndOrientation(bid)
            else:
                state = p.getLinkState(bid, li, computeForwardKinematics=True)
                pos, quat = state[4], state[5]
            rot = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
            pts_world = pts_local @ rot.T + np.array(pos, dtype=float).reshape(1, 3)
            if max_pts is not None and pts_world.shape[0] > int(max_pts):
                idx = np.random.choice(pts_world.shape[0], int(max_pts), replace=False)
                pts_world = pts_world[idx]
            clouds.append({"link_index": li, "points": pts_world})
        return clouds

    def get_surface_local_samples(self, link_indices=None) -> dict:
        return {}

    def get_closest_points_to_obstacle(self, link_index, obs_body_id, max_dist=1.0):
        contacts = p.getClosestPoints(self.body_id, obs_body_id, max_dist, linkIndexA=link_index)
        if not contacts:
            return None
        best = min(contacts, key=lambda c: c[8])
        normal = np.array(best[7], dtype=float)
        nl = np.linalg.norm(normal)
        if nl > 1e-9:
            normal = normal / nl
        else:
            normal = np.array([1.0, 0.0, 0.0])
        return {
            "obs_link_index": int(best[4]),
            "point_on_link": np.array(best[5], dtype=float),
            "point_on_obstacle": np.array(best[6], dtype=float),
            "signed_dist": float(best[8]),
            "normal_on_obstacle": normal,
        }


# ── 工件 ────────────────────────────────────────────


class Workpiece:
    """工件模型：加载 URDF，提供焊缝坐标系查询。"""

    def __init__(self, cfg):
        if hasattr(cfg, "_cfg"):
            cfg = cfg._cfg
        wp = cfg.get("workpiece", {})
        urdf_abs = _resolve(wp.get("urdf"))
        pkg_name = wp.get("package_name")
        pkg_alias = wp.get("package_alias")
        ignore_coll = bool(wp.get("ignore_all_collisions", False))

        urdf_file, search_root = _prepare_urdf(
            urdf_abs,
            package_name=pkg_name,
            package_alias=pkg_alias,
            remove_collision=ignore_coll,
        )
        p.setAdditionalSearchPath(search_root)

        ori_deg = wp.get("orientation_deg", [0, 0, 0])
        quat = p.getQuaternionFromEuler([math.radians(v) for v in ori_deg])
        self.body_id = p.loadURDF(
            urdf_file,
            basePosition=wp.get("position", [0, 0, 0]),
            baseOrientation=quat,
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )

        self.link_name_to_index: dict[str, int] = {}
        for i in range(p.getNumJoints(self.body_id)):
            name = p.getJointInfo(self.body_id, i)[12].decode()
            self.link_name_to_index[name] = i

        if ignore_coll:
            for li in range(-1, p.getNumJoints(self.body_id)):
                p.setCollisionFilterGroupMask(self.body_id, li, 0, 0)

    def get_frame_pose(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        if link_name == "base_link":
            pos, quat = p.getBasePositionAndOrientation(self.body_id)
            return np.array(pos, dtype=float), np.array(quat, dtype=float)
        if link_name not in self.link_name_to_index:
            raise KeyError(f"工件中未找到 link: {link_name}")
        idx = self.link_name_to_index[link_name]
        state = p.getLinkState(self.body_id, idx, computeForwardKinematics=True)
        return np.array(state[4], dtype=float), np.array(state[5], dtype=float)


# ── 旧接口兼容 ──────────────────────────────────────


class ExperimentConfig:
    """兼容旧接口：从 Super_config.json 读取，提供属性访问。"""

    def __init__(self):
        self._cfg = load_config()
        r = self._cfg.get("robot", {})
        w = self._cfg.get("workpiece", {})
        s = self._cfg.get("simulation", {})
        self.urdf_path = _resolve(r.get("urdf", ""))
        self.workpiece_urdf_path = _resolve(w.get("urdf", ""))
        self.workpiece_position = tuple(w.get("position", [0, 0, 0]))
        self.workpiece_orientation_deg = tuple(w.get("orientation_deg", [0, 0, 0]))
        self.ignore_all_collisions = bool(w.get("ignore_all_collisions", False))
        self.start_link_name = w.get("start_link", "l2")
        self.goal_link_name = w.get("goal_link", "l3")
        self.gantry_initial_q = tuple(r.get("gantry_initial_q", [0, 0, 0]))
        self.dt = float(s.get("dt", 1.0 / 240.0))
        self.dq_limit = float(r.get("dq_limit", 1.0))
        self.base_vel_limit = float(r.get("base_vel_limit", 0.4))
        self.camera_distance = float(s.get("camera_distance", 1.4))
        self.camera_yaw = float(s.get("camera_yaw", -215.0))
        self.camera_pitch = float(s.get("camera_pitch", -26.0))
        self.camera_target = tuple(s.get("camera_target", [0.3, 0.2, 0.6]))


JakaRobot = Robot
WorkpieceModel = Workpiece
_prepare_package_urdf = _prepare_urdf


def load_joint_trajectory(path: str | Path) -> dict:
    traj_path = Path(path)
    with open(traj_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _replay_progress(current: int, total: int, width: int = 40,
                     prefix: str = "", extra: str = "") -> str:
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = f"{100.0 * frac:5.1f}%"
    return f"\r{prefix} |{bar}| {pct} [{current}/{total}] {extra}"


_SEG_COLORS = [
    [0.2, 0.4, 1.0],   # blue
    [1.0, 0.4, 0.0],   # orange
    [0.2, 0.8, 0.2],   # green
    [0.8, 0.2, 0.8],   # purple
    [1.0, 0.0, 0.0],   # red
    [0.0, 0.8, 0.8],   # cyan
    [0.6, 0.6, 0.0],   # olive
    [1.0, 0.6, 0.8],   # pink
    [0.4, 0.4, 0.4],   # gray
]


def replay(
    cfg_path: str | Path | None = None,
    trajectory_json: str | Path | None = None,
    *,
    gui: bool = True,
    realtime: bool = True,
    speed: float = 1.0,
    stay_open: bool = True,
):
    """Replay joint_trajectory.json in PyBullet with per-segment progress.

    Parameters
    ----------
    speed : float
        Playback speed multiplier (2.0 = double speed).
    stay_open : bool
        If True, keep the GUI open after replay for inspection.
    """
    import sys as _sys

    cfg = load_config(cfg_path)
    traj_cfg = cfg.get("trajectory_planning", {})
    traj_path = Path(trajectory_json) if trajectory_json else Path(
        _resolve(traj_cfg.get("output_json", "artifacts/sdf_exp/joint_trajectory.json"))
    )
    payload = load_joint_trajectory(traj_path)
    dt = float(payload.get("dt", cfg.get("simulation", {}).get("dt", 1.0 / 240.0)))

    all_steps = payload.get("steps", [])
    seg_info = payload.get("segments", [])
    n_total = len(all_steps)

    print(f"[replay] {len(seg_info)} segments, {n_total} steps, dt={dt:.4f}s")
    print(f"[replay] playback speed = {speed:.1f}x  (sim {n_total * dt:.1f}s)")
    print("=" * 72)

    scene = None
    created_connection = False
    if gui:
        scene = SimulationScene(cfg)
        scene.enable_rendering()
    elif not p.isConnected():
        p.connect(p.DIRECT)
        created_connection = True

    try:
        robot = Robot(cfg)
        workpiece = Workpiece(cfg)

        ee_frame_ids = None
        ref_frame_ids = None
        if scene is not None:
            ee_pos, ee_quat = robot.get_ee_pose()
            ee_frame_ids = scene.draw_frame(ee_pos, ee_quat, length=0.12)
            wp_cfg = cfg.get("workpiece", {})
            for link_name in [wp_cfg.get("start_link", "l2"),
                              wp_cfg.get("goal_link", "l3")]:
                try:
                    pos, quat = workpiece.get_frame_pose(link_name)
                    scene.create_marker(0.02, [1, 0, 0, 1], pos.tolist())
                    scene.draw_frame(pos, quat, length=0.08)
                except KeyError:
                    pass

        seg_step_groups: list[tuple[str, list[dict]]] = []
        cur_name = None
        cur_list: list[dict] = []
        for st in all_steps:
            sn = st.get("segment_name", "")
            if sn != cur_name:
                if cur_list:
                    seg_step_groups.append((cur_name or "", cur_list))
                cur_name = sn
                cur_list = [st]
            else:
                cur_list.append(st)
        if cur_list:
            seg_step_groups.append((cur_name or "", cur_list))

        if scene is not None:
            print("[replay] drawing reference path ...")
            for seg_idx, (seg_name, seg_steps) in enumerate(seg_step_groups):
                color = _SEG_COLORS[seg_idx % len(_SEG_COLORS)]
                ref_color = [min(c + 0.4, 1.0) for c in color]
                prev_rp = None
                for st in seg_steps:
                    if "ref_pos" in st:
                        rp = np.asarray(st["ref_pos"], dtype=float)
                        if prev_rp is not None:
                            p.addUserDebugLine(
                                prev_rp.tolist(), rp.tolist(),
                                ref_color, lineWidth=1.5,
                            )
                        prev_rp = rp

        prev_ee = None
        global_idx = 0
        sleep_dt = dt / max(speed, 0.01)

        for seg_idx, (seg_name, seg_steps) in enumerate(seg_step_groups):
            n_seg = len(seg_steps)
            motion = seg_steps[0].get("motion_type", "")
            color = _SEG_COLORS[seg_idx % len(_SEG_COLORS)]
            label = f"[{seg_idx+1}/{len(seg_step_groups)}] {seg_name}"
            print(f"\n{label}  ({motion}, {n_seg} steps)")

            update_freq = max(n_seg // 20, 1)

            prev_rb = None
            is_gantry_seg = motion in ("前三轴",)

            for i, step in enumerate(seg_steps):
                q = np.asarray(step.get("q", []), dtype=float)
                dq = np.asarray(step.get("dq", np.zeros_like(q)), dtype=float)
                robot.set_joint_state(q, dq=dq)

                if scene is not None:
                    ee_pos, ee_quat = robot.get_ee_pose()
                    ee_frame_ids = scene.draw_frame(
                        ee_pos, ee_quat, length=0.12,
                        replace_ids=ee_frame_ids,
                    )
                    if "ref_pos" in step:
                        rp = np.asarray(step["ref_pos"], dtype=float)
                        rq = np.asarray(step.get("ref_quat", ee_quat), dtype=float)
                        ref_frame_ids = scene.draw_frame(
                            rp, rq, length=0.06,
                            replace_ids=ref_frame_ids,
                        )
                    if is_gantry_seg:
                        rb_pos, _ = robot.get_robobase_pose()
                        if prev_rb is not None:
                            p.addUserDebugLine(
                                prev_rb.tolist(), rb_pos.tolist(),
                                [0.0, 0.85, 0.0], lineWidth=3.0,
                            )
                        prev_rb = rb_pos.copy()
                    if prev_ee is not None:
                        p.addUserDebugLine(
                            prev_ee.tolist(), ee_pos.tolist(),
                            color, lineWidth=2.5,
                        )
                    prev_ee = ee_pos.copy()

                if (i + 1) % update_freq == 0 or i == n_seg - 1:
                    h_val = step.get("min_h", 0.0)
                    extra = f"h={h_val:+.4f}" if isinstance(h_val, (int, float)) else ""
                    _sys.stdout.write(_replay_progress(
                        i + 1, n_seg, prefix=label, extra=extra,
                    ))
                    _sys.stdout.flush()

                if realtime:
                    time.sleep(sleep_dt)
                global_idx += 1

            print(f"\n  OK seg {seg_idx+1} done")

        print("\n" + "=" * 72)
        print(f"[replay] done: {global_idx} steps played")

        if gui and stay_open and scene is not None:
            print("[replay] GUI open -- close the window to exit")
            while p.isConnected():
                ee_pos, ee_quat = robot.get_ee_pose()
                ee_frame_ids = scene.draw_frame(
                    ee_pos, ee_quat, length=0.12,
                    replace_ids=ee_frame_ids,
                )
                time.sleep(1.0 / 30.0)

        return payload
    finally:
        if created_connection and p.isConnected():
            p.disconnect()


# ── 主入口 ──────────────────────────────────────────


def main():
    cfg = load_config()
    scene = SimulationScene(cfg)
    robot = Robot(cfg)
    workpiece = Workpiece(cfg)
    scene.enable_rendering()

    ee_pos, ee_quat = robot.get_ee_pose()
    print(f"[Robot] DOF={robot.dof}  EE pos={np.round(ee_pos, 3)}  quat={np.round(ee_quat, 3)}")
    ee_frame_ids = scene.draw_frame(ee_pos, ee_quat, length=0.12)

    wp_cfg = cfg.get("workpiece", {})
    for link_name in [wp_cfg.get("start_link", "l2"), wp_cfg.get("goal_link", "l3")]:
        pos, quat = workpiece.get_frame_pose(link_name)
        print(f"[Workpiece] {link_name}: pos={np.round(pos, 3)}  quat={np.round(quat, 3)}")
        scene.create_marker(0.02, [1, 0, 0, 1], pos.tolist())
        scene.draw_frame(pos, quat, length=0.08)

    print("\n场景已加载，可拖动关节滑条观察末端坐标系跟随。关闭窗口退出。")
    while p.isConnected():
        ee_pos, ee_quat = robot.get_ee_pose()
        ee_frame_ids = scene.draw_frame(ee_pos, ee_quat, length=0.12,
                                        replace_ids=ee_frame_ids)
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    import sys as _sys
    if 1:
        speed = 0.2
        replay(speed=speed)
    else:
        main()
