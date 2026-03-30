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
    show_collision_meshes: bool = False
    show_cbf_contacts: bool = True
    cbf_contact_normal_length: float = 0.04
    cbf_contact_cross_size: float = 0.006
    cbf_contact_line_width: float = 2.0
    surface_prefer_gpu: bool = True
    surface_target_density: float = 300.0
    surface_min_samples: int = 96
    surface_max_samples: int = 768
    robot_surface_target_density: float = 300.0
    robot_surface_min_samples: int = 96
    robot_surface_max_samples: int = 768
    robot_rear_six_surface_target_density: float = 1800.0
    robot_rear_six_surface_min_samples: int = 384
    robot_rear_six_surface_max_samples: int = 4096
    obstacle_surface_target_density: float = 1200.0
    obstacle_surface_min_samples: int = 256
    obstacle_surface_max_samples: int = 4096
    obstacle_local_dense_enabled: bool = True
    obstacle_local_dense_radius: float = 2.0
    obstacle_local_dense_target_density: float = 4800.0
    obstacle_local_dense_min_samples: int = 1024
    obstacle_local_dense_max_samples: int = 12000
    obstacle_local_dense_query_enabled: bool = True
    obstacle_local_dense_visual_enabled: bool = True
    obstacle_local_dense_update_interval: int = 5
    surface_gpu_chunk_size: int = 2048
    show_surface_samples: bool = True
    surface_visual_max_points_per_link: int = 48
    robot_surface_visual_max_points_per_link: int = 48
    robot_rear_six_visual_max_points_per_link: int = 240
    obstacle_surface_visual_max_points_per_link: int = 240
    obstacle_local_dense_visual_max_points_per_link: int = 1200
    surface_visual_point_size: int = 4
    surface_visual_update_interval: int = 6
    ee_trace_lifetime: float = 2.0
    use_rrt_nominal_planner: bool = True
    show_nominal_planner_toggle: bool = True
    nominal_planner_toggle_wait_s: float = 3.0

    # 轨迹
    approach_duration: float = 6.0
    weld_duration: float = 7.0
    return_duration: float = 6.0
    hold_duration: float = 3.0
    dq_limit: float = 1.0
    base_vel_limit: float = 0.4
    position_gain: float = 1.0
    orientation_gain: float = 3.0
    use_mesh_cbf: bool = True
    safety_margin: float = 0.005
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
    second_order_nominal_enabled: bool = False
    second_order_nominal_candidate_count: int = 8
    second_order_nominal_preview_tau: float = 0.04
    second_order_nominal_fd_eps: float = 0.01
    second_order_nominal_goal_weight: float = 1.0
    second_order_nominal_clearance_weight: float = 2.0
    second_order_nominal_curvature_weight: float = 0.08
    second_order_nominal_reference_weight: float = 0.35
    second_order_nominal_activation_distance: float = 0.08
    second_order_nominal_active_links: int = 2
    progress_end_tolerance: float = 0.02

    # MPC
    N_mpc: int = 20
    mpc_dt: float = 0.04
    gamma_dcbf: float = 0.5
    mpc_tracking_weight: float = 0.5
    mpc_orientation_tracking_weight: float = 0.2
    mpc_terminal_orientation_window: float = 1.0
    mpc_control_weight: float = 0.2
    mpc_smooth_weight: float = 0.2
    mpc_replan_steps: int = 6
    mpc_progress_step_min: float = 0.01
    mpc_second_order_risk_enabled: bool = True
    mpc_second_order_risk_horizon: int = 4
    mpc_second_order_safe_margin: float = 0.01
    mpc_second_order_preview_tau: float = 0.04
    mpc_second_order_fd_eps: float = 0.01
    mpc_second_order_risk_weight: float = 0.2
    mpc_second_order_shrink_weight: float = 0.1

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
        self._collision_visual_specs: list[dict] = []
        self._cbf_contact_debug_ids: list[int] = []
        self._surface_point_debug_ids: list[int] = []

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

    def add_toggle(self, label: str, default: bool) -> int:
        return p.addUserDebugParameter(label, 0.0, 1.0, 1.0 if default else 0.0)

    def read_toggle(self, item_id: int) -> bool:
        return bool(p.readUserDebugParameter(item_id) >= 0.5)

    def choose_toggle(self, label: str, default: bool, wait_s: float = 0.0) -> bool:
        toggle_id = self.add_toggle(label, default)
        selected = bool(default)
        deadline = time.time() + max(float(wait_s), 0.0)
        first_read = True
        while first_read or time.time() < deadline:
            selected = self.read_toggle(toggle_id)
            first_read = False
            if time.time() < deadline:
                self.update_status(
                    f"{label}: {'ON' if selected else 'OFF'}  |  可在左侧滑条切换，稍后自动开始规划"
                )
                time.sleep(min(0.05, max(deadline - time.time(), 0.0)))
        return selected

    @staticmethod
    def _get_body_link_pose(body_id: int, link_index: int):
        if link_index < 0:
            return p.getBasePositionAndOrientation(body_id)
        state = p.getLinkState(body_id, link_index, computeForwardKinematics=True)
        # Collision-shape local pose is expressed in the link COM/inertial frame.
        return state[0], state[1]

    @staticmethod
    def _decode_shape_filename(filename) -> str:
        if isinstance(filename, (bytes, bytearray)):
            return filename.decode("utf-8")
        return str(filename)

    def _create_visual_shape_from_collision(self, shape_data, rgba) -> int:
        geom_type = int(shape_data[2])
        dims = tuple(float(v) for v in shape_data[3])
        filename = self._decode_shape_filename(shape_data[4])
        if geom_type == p.GEOM_MESH:
            return p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=filename,
                meshScale=dims[:3],
                rgbaColor=rgba,
            )
        if geom_type == p.GEOM_BOX:
            return p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.5 * dims[0], 0.5 * dims[1], 0.5 * dims[2]],
                rgbaColor=rgba,
            )
        if geom_type == p.GEOM_SPHERE:
            return p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=dims[0], rgbaColor=rgba)
        if geom_type == p.GEOM_CYLINDER:
            return p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=dims[1], length=dims[0], rgbaColor=rgba)
        if geom_type == p.GEOM_CAPSULE:
            return p.createVisualShape(shapeType=p.GEOM_CAPSULE, radius=dims[1], length=dims[0], rgbaColor=rgba)
        return -1

    def add_collision_mesh_visuals(self, body_id: int, rgba, link_indices: list[int] | None = None) -> int:
        selected = None if link_indices is None else set(int(li) for li in link_indices)
        created = 0
        for link_index in range(-1, p.getNumJoints(body_id)):
            if selected is not None and link_index not in selected:
                continue
            shape_datas = p.getCollisionShapeData(body_id, link_index)
            if not shape_datas:
                continue
            world_pos, world_orn = self._get_body_link_pose(body_id, link_index)
            for shape_data in shape_datas:
                visual_shape_id = self._create_visual_shape_from_collision(shape_data, rgba)
                if visual_shape_id < 0:
                    continue
                local_pos = tuple(float(v) for v in shape_data[5])
                local_orn = tuple(float(v) for v in shape_data[6])
                visual_pos, visual_orn = p.multiplyTransforms(world_pos, world_orn, local_pos, local_orn)
                visual_body_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=visual_pos,
                    baseOrientation=visual_orn,
                )
                p.setCollisionFilterGroupMask(visual_body_id, -1, 0, 0)
                self._collision_visual_specs.append({
                    "body_id": int(body_id),
                    "link_index": int(link_index),
                    "local_pos": local_pos,
                    "local_orn": local_orn,
                    "visual_body_id": int(visual_body_id),
                })
                created += 1
        return created

    def update_collision_mesh_visuals(self):
        for spec in self._collision_visual_specs:
            world_pos, world_orn = self._get_body_link_pose(spec["body_id"], spec["link_index"])
            visual_pos, visual_orn = p.multiplyTransforms(
                world_pos,
                world_orn,
                spec["local_pos"],
                spec["local_orn"],
            )
            p.resetBasePositionAndOrientation(spec["visual_body_id"], visual_pos, visual_orn)

    def clear_cbf_contact_visuals(self):
        for item_id in self._cbf_contact_debug_ids:
            p.removeUserDebugItem(item_id)
        self._cbf_contact_debug_ids.clear()

    def clear_surface_cloud_visuals(self):
        for item_id in self._surface_point_debug_ids:
            p.removeUserDebugItem(item_id)
        self._surface_point_debug_ids.clear()

    def _add_debug_cross(self, center, color, half_extent: float, width: float):
        center = np.asarray(center, dtype=float)
        segments = (
            (center + np.array([half_extent, 0.0, 0.0]), center - np.array([half_extent, 0.0, 0.0])),
            (center + np.array([0.0, half_extent, 0.0]), center - np.array([0.0, half_extent, 0.0])),
            (center + np.array([0.0, 0.0, half_extent]), center - np.array([0.0, 0.0, half_extent])),
        )
        for start, end in segments:
            self._cbf_contact_debug_ids.append(
                p.addUserDebugLine(start.tolist(), end.tolist(), color, lineWidth=width)
            )

    def update_cbf_contact_visuals(self, contact_specs: list[dict]):
        self.clear_cbf_contact_visuals()
        for spec in contact_specs:
            point_on_link = np.asarray(spec["point_on_link"], dtype=float)
            point_on_obstacle = np.asarray(spec["point_on_obstacle"], dtype=float)
            normal_on_link = np.asarray(spec["normal_on_link"], dtype=float)
            normal_on_obstacle = np.asarray(spec["normal_on_obstacle"], dtype=float)
            normal_length = float(spec["normal_length"])
            line_width = float(spec.get("line_width", self.config.cbf_contact_line_width))
            cross_size = float(spec.get("cross_size", self.config.cbf_contact_cross_size))
            pair_color = spec.get("pair_color", [0.95, 0.75, 0.10])
            label_color = spec.get("label_color", pair_color)

            self._cbf_contact_debug_ids.append(
                p.addUserDebugLine(point_on_link.tolist(), point_on_obstacle.tolist(), pair_color, lineWidth=line_width)
            )
            self._add_debug_cross(point_on_link, [0.95, 0.10, 0.10], cross_size, line_width)
            self._add_debug_cross(point_on_obstacle, [0.10, 0.35, 1.00], cross_size, line_width)
            self._cbf_contact_debug_ids.append(
                p.addUserDebugLine(
                    point_on_link.tolist(),
                    (point_on_link + normal_length * normal_on_link).tolist(),
                    [1.00, 0.20, 0.20],
                    lineWidth=line_width,
                )
            )
            self._cbf_contact_debug_ids.append(
                p.addUserDebugLine(
                    point_on_obstacle.tolist(),
                    (point_on_obstacle + normal_length * normal_on_obstacle).tolist(),
                    [0.10, 0.50, 1.00],
                    lineWidth=line_width,
                )
            )
            midpoint = 0.5 * (point_on_link + point_on_obstacle)
            self._cbf_contact_debug_ids.append(
                p.addUserDebugText(
                    spec["label"],
                    midpoint.tolist(),
                    textColorRGB=label_color,
                    textSize=1.0,
                )
            )

    def update_surface_cloud_visuals(self, cloud_specs: list[dict]):
        self.clear_surface_cloud_visuals()
        for spec in cloud_specs:
            points = np.asarray(spec["points"], dtype=float).reshape(-1, 3)
            if points.shape[0] == 0:
                continue
            colors = np.asarray(spec["colors"], dtype=float).reshape(-1, 3)
            point_size = int(spec.get("point_size", self.config.surface_visual_point_size))
            self._surface_point_debug_ids.append(
                p.addUserDebugPoints(
                    points.tolist(),
                    colors.tolist(),
                    pointSize=point_size,
                )
            )


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


def _coerce_vec3(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _coerce_points_array(value) -> np.ndarray | None:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


def build_cbf_contact_visualization_specs(cbf_contacts: list[dict], normal_length: float = 0.04) -> list[dict]:
    specs = []
    for idx, contact in enumerate(cbf_contacts or []):
        point_on_link = _coerce_vec3(contact.get("point_on_link"))
        point_on_obstacle = _coerce_vec3(contact.get("point_on_obstacle"))
        if point_on_link is None or point_on_obstacle is None:
            continue

        normal_on_link = _coerce_vec3(contact.get("normal_on_link"))
        normal_on_obstacle = _coerce_vec3(contact.get("normal_on_obstacle"))
        if normal_on_link is None or np.linalg.norm(normal_on_link) < 1e-9:
            normal_on_link = None
        if normal_on_obstacle is None or np.linalg.norm(normal_on_obstacle) < 1e-9:
            normal_on_obstacle = _coerce_vec3(contact.get("normal"))
        if normal_on_obstacle is None or np.linalg.norm(normal_on_obstacle) < 1e-9:
            normal_on_obstacle = point_on_link - point_on_obstacle
        normal_on_obstacle = _normalize(normal_on_obstacle)
        if np.linalg.norm(normal_on_obstacle) < 1e-9:
            normal_on_obstacle = np.array([1.0, 0.0, 0.0], dtype=float)
        if normal_on_link is None:
            normal_on_link = -normal_on_obstacle
        else:
            normal_on_link = _normalize(normal_on_link)
            if np.linalg.norm(normal_on_link) < 1e-9:
                normal_on_link = -normal_on_obstacle

        h_val = float(contact.get("h_val", float("nan")))
        link_name = str(contact.get("link_name", f"cbf_{idx}"))
        obs_link_name = contact.get("obs_link_name")
        label_prefix = (
            f"{link_name} -> {obs_link_name}"
            if obs_link_name not in (None, "", "?")
            else link_name
        )
        pair_color = [0.95, 0.20, 0.20] if np.isfinite(h_val) and h_val < 0.0 else [0.95, 0.75, 0.10]
        specs.append({
            "point_on_link": point_on_link,
            "point_on_obstacle": point_on_obstacle,
            "normal_on_link": normal_on_link,
            "normal_on_obstacle": normal_on_obstacle,
            "normal_length": float(normal_length),
            "label": (
                f"{label_prefix} | h={h_val * 1000:.1f}mm"
                if np.isfinite(h_val)
                else label_prefix
            ),
            "pair_color": pair_color,
            "label_color": pair_color,
        })
    return specs


def build_surface_cloud_visualization_specs(clouds: list[dict], point_size: int = 4) -> list[dict]:
    specs = []
    for idx, cloud in enumerate(clouds or []):
        points = _coerce_points_array(cloud.get("points"))
        if points is None or points.shape[0] == 0:
            continue
        color = _coerce_vec3(cloud.get("color"))
        if color is None:
            color = np.array([0.2, 0.2, 0.2], dtype=float)
        colors = np.repeat(color.reshape(1, 3), points.shape[0], axis=0)
        specs.append({
            "link_name": str(cloud.get("link_name", f"cloud_{idx}")),
            "points": points,
            "colors": colors,
            "point_size": int(point_size),
        })
    return specs


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
