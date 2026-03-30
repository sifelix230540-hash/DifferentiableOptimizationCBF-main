from collections import deque
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from CBF_experiment.active.pybullet.welding_320_common import ExperimentConfig, quaternion_error_rotvec
from CBF_experiment.active.pybullet.welding_320_robot import JakaRobot
from CBF_experiment.active.pybullet.welding_320_trajectory import JointWaypointTrajectory, PiecewiseLineSlerpTrajectory

try:
    import pybullet_planning as pp
except ImportError:
    pp = None

DEBUG_LOG_PATH = Path(__file__).resolve().parents[3] / "debug-31a24e.log"
DEBUG_SESSION_ID = "31a24e"


def _append_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class CartesianRRTNominalPlanner:
    """在末端笛卡尔空间生成三段名义 RRT 路径。"""

    def __init__(self, robot: JakaRobot, config: ExperimentConfig, workpiece_body_id: int):
        if pp is None:
            raise RuntimeError("未安装 pybullet_planning，请先安装后再启用 RRT 名义规划器。")
        if robot.welding_gun_base_link_index < 0:
            raise RuntimeError("未找到 welding_gun_base link，无法做笛卡尔 RRT 碰撞检查。")
        self.robot = robot
        self.config = config
        self.workpiece_body_id = workpiece_body_id
        self.last_plan_statuses: list[str] = []

    @staticmethod
    def _rounded_key(pos: np.ndarray) -> tuple[float, float, float]:
        arr = np.array(pos, dtype=float)
        return tuple(np.round(arr, 4).tolist())

    def _linear_fallback(self, start_pos: np.ndarray, goal_pos: np.ndarray, n: int = 24) -> list[np.ndarray]:
        return [
            (1.0 - alpha) * start_pos + alpha * goal_pos
            for alpha in np.linspace(0.0, 1.0, n, endpoint=True)
        ]

    def _make_pose_helpers(self, start_pos, start_quat, goal_pos, goal_quat):
        start_pos = np.array(start_pos, dtype=float)
        goal_pos = np.array(goal_pos, dtype=float)
        start_quat = np.array(start_quat, dtype=float)
        goal_quat = np.array(goal_quat, dtype=float)
        chord = goal_pos - start_pos
        chord_norm_sq = float(np.dot(chord, chord))
        slerp = Slerp([0.0, 1.0], Rotation.from_quat(np.vstack([start_quat, goal_quat])))

        def alpha_from_pos(pos):
            pos = np.array(pos, dtype=float)
            if chord_norm_sq < 1e-12:
                return 0.0
            alpha = float(np.dot(pos - start_pos, chord) / chord_norm_sq)
            return float(np.clip(alpha, 0.0, 1.0))

        def quat_from_pos(pos):
            return slerp([alpha_from_pos(pos)]).as_quat()[0]

        return alpha_from_pos, quat_from_pos

    def _choose_seed_q(self, pos, alpha_from_pos, q_seed_start, q_seed_goal, cache_entries):
        alpha = alpha_from_pos(pos)
        seed_q = (1.0 - alpha) * q_seed_start + alpha * q_seed_goal
        if cache_entries:
            nearest_pos, nearest_q = min(
                cache_entries,
                key=lambda item: np.linalg.norm(np.array(pos, dtype=float) - item[0]),
            )
            if np.linalg.norm(np.array(pos, dtype=float) - nearest_pos) < 0.20:
                seed_q = nearest_q
        return np.array(seed_q, dtype=float)

    def _solve_pose_ik(self, target_pos, target_quat, seed_q):
        q_sol = self.robot.calculate_ik(target_pos, target_quat, rest_poses=seed_q)
        self.robot.set_joint_state(q_sol)
        ee_pos, _ = self.robot.get_ee_pose()
        if np.linalg.norm(np.array(target_pos, dtype=float) - ee_pos) > self.config.rrt_ik_tolerance:
            return None
        return q_sol

    def _base_collides_with_workpiece(self, q):
        if self.config.ignore_all_collisions:
            return False
        self.robot.set_joint_state(q)
        closest = self.robot.get_closest_point_to_obstacle(
            self.robot.welding_gun_base_link_index,
            self.workpiece_body_id,
            max_dist=max(1.0, self.config.rrt_cartesian_margin * 3.0),
        )
        if closest is None:
            return False
        return float(closest[1]) < self.config.safety_margin

    def _plan_single_segment(
        self,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        q_seed_start: np.ndarray,
        q_seed_goal: np.ndarray,
        duration: float,
    ) -> JointWaypointTrajectory:
        start_pos = np.array(start_pos, dtype=float)
        start_quat = np.array(start_quat, dtype=float)
        goal_pos = np.array(goal_pos, dtype=float)
        goal_quat = np.array(goal_quat, dtype=float)
        q_seed_start = np.array(q_seed_start, dtype=float)
        q_seed_goal = np.array(q_seed_goal, dtype=float)

        alpha_from_pos, quat_from_pos = self._make_pose_helpers(start_pos, start_quat, goal_pos, goal_quat)
        cache: dict[tuple[float, float, float], np.ndarray] = {
            self._rounded_key(start_pos): q_seed_start.copy(),
            self._rounded_key(goal_pos): q_seed_goal.copy(),
        }
        cache_entries = [
            (start_pos.copy(), q_seed_start.copy()),
            (goal_pos.copy(), q_seed_goal.copy()),
        ]

        def collision_fn(conf, diagnosis=False, **_kwargs):
            pos = np.array(conf, dtype=float)
            key = self._rounded_key(pos)
            q_candidate = cache.get(key)
            if q_candidate is None:
                seed_q = self._choose_seed_q(pos, alpha_from_pos, q_seed_start, q_seed_goal, cache_entries)
                q_candidate = self._solve_pose_ik(pos, quat_from_pos(pos), seed_q)
                if q_candidate is None:
                    return True
                cache[key] = q_candidate.copy()
                cache_entries.append((pos.copy(), q_candidate.copy()))
            return self._base_collides_with_workpiece(q_candidate)

        margin = self.config.rrt_cartesian_margin
        lower = np.minimum(start_pos, goal_pos) - margin
        upper = np.maximum(start_pos, goal_pos) + margin
        resolution = max(self.config.rrt_cartesian_resolution, 1e-3)

        def distance_fn(q1, q2):
            return float(np.linalg.norm(np.array(q2, dtype=float) - np.array(q1, dtype=float)))

        def sample_fn():
            return np.random.uniform(lower, upper).tolist()

        def extend_fn(q1, q2):
            q1 = np.array(q1, dtype=float)
            q2 = np.array(q2, dtype=float)
            dist = np.linalg.norm(q2 - q1)
            n_steps = max(int(np.ceil(dist / resolution)), 1)
            return [
                (q1 + (idx / n_steps) * (q2 - q1)).tolist()
                for idx in range(1, n_steps + 1)
            ]

        path_pos = None
        try:
            path_pos = pp.birrt(
                start_pos.tolist(),
                goal_pos.tolist(),
                distance_fn,
                sample_fn,
                extend_fn,
                collision_fn,
                max_iterations=self.config.rrt_max_iterations,
                max_time=self.config.rrt_max_time,
                restarts=self.config.rrt_restarts,
                smooth=self.config.rrt_smooth,
            )
        except TypeError:
            path_pos = pp.birrt(
                start_pos.tolist(),
                goal_pos.tolist(),
                distance_fn,
                sample_fn,
                extend_fn,
                collision_fn,
                max_iterations=self.config.rrt_max_iterations,
                max_time=self.config.rrt_max_time,
            )

        planner_status = "rrt"
        if path_pos is None or len(path_pos) == 0:
            print("[warn] RRT 未找到可行路径，使用线性插值名义路径回退。")
            path_pos = [pt.tolist() for pt in self._linear_fallback(start_pos, goal_pos)]
            planner_status = "linear_fallback"

        waypoints_pos = [np.array(pos, dtype=float) for pos in path_pos]
        waypoints_quat = [np.array(quat_from_pos(pos), dtype=float) for pos in waypoints_pos]
        if waypoints_quat:
            waypoints_quat[0] = start_quat.copy()
            waypoints_quat[-1] = goal_quat.copy()
        return JointWaypointTrajectory(
            waypoints_pos,
            waypoints_quat,
            duration,
            self.config.dt,
            planner_status=planner_status,
        )

    def build_three_phase_trajectory(
        self,
        q_init: np.ndarray,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        initial_pose: tuple[np.ndarray, np.ndarray],
        start_ref: tuple[np.ndarray, np.ndarray],
        goal_ref: tuple[np.ndarray, np.ndarray],
    ) -> PiecewiseLineSlerpTrajectory:
        q_backup, dq_backup = self.robot.get_joint_state()
        q_init = np.array(q_init, dtype=float)
        q_start = np.array(q_start, dtype=float)
        q_goal = np.array(q_goal, dtype=float)
        initial_pos, initial_quat = initial_pose
        start_pos, start_quat = start_ref
        goal_pos, goal_quat = goal_ref

        try:
            print("[rrt] 规划第 1 段: 初始 -> 焊接起点 ...")
            seg_1 = self._plan_single_segment(
                initial_pos, initial_quat, start_pos, start_quat, q_init, q_start, self.config.approach_duration
            )
            print(f"[rrt]   -> {len(seg_1.waypoints_pos)} 个路径点 ({seg_1.planner_status})")

            print("[rrt] 规划第 2 段: 焊接起点 -> 焊接终点 ...")
            seg_2 = self._plan_single_segment(
                start_pos, start_quat, goal_pos, goal_quat, q_start, q_goal, self.config.weld_duration
            )
            print(f"[rrt]   -> {len(seg_2.waypoints_pos)} 个路径点 ({seg_2.planner_status})")

            print("[rrt] 规划第 3 段: 焊接终点 -> 初始 ...")
            seg_3 = self._plan_single_segment(
                goal_pos, goal_quat, initial_pos, initial_quat, q_goal, q_init, self.config.return_duration
            )
            print(f"[rrt]   -> {len(seg_3.waypoints_pos)} 个路径点 ({seg_3.planner_status})")
            self.last_plan_statuses = [seg_1.planner_status, seg_2.planner_status, seg_3.planner_status]
        finally:
            self.robot.set_joint_state(q_backup, dq_backup)
        return PiecewiseLineSlerpTrajectory([seg_1, seg_2, seg_3])


class DynamicNominalReferenceMixer:
    """在停滞时沿已执行避障轨迹外推，并与名义参考做动态混合。"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._ee_history: deque[np.ndarray] = deque(maxlen=max(int(config.dynamic_nominal_history_size), 2))
        self._progress_history: deque[float] = deque(maxlen=max(int(config.dynamic_nominal_history_size), 2))
        self._stall_active = False
        self._locked_escape_dir: np.ndarray | None = None

    @staticmethod
    def _safe_normalize(vec: np.ndarray) -> np.ndarray | None:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return None
        return np.array(vec, dtype=float) / norm

    @staticmethod
    def _project_to_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
        return np.array(vec, dtype=float) - float(np.dot(vec, normal)) * normal

    def _build_escape_dir(
        self,
        exec_delta: np.ndarray,
        ee_pos: np.ndarray,
        nominal_positions: list[np.ndarray],
        normal: np.ndarray,
    ) -> np.ndarray:
        exec_tangent = self._safe_normalize(self._project_to_plane(exec_delta, normal))
        nominal_tangent = None
        if nominal_positions:
            nominal_delta = nominal_positions[0] - ee_pos
            nominal_tangent = self._safe_normalize(self._project_to_plane(nominal_delta, normal))

        tangent = exec_tangent
        if tangent is None:
            tangent = nominal_tangent
        elif nominal_tangent is not None and float(np.dot(tangent, nominal_tangent)) < 0.0:
            tangent = nominal_tangent
        if tangent is None:
            tangent = normal

        escape_dir = self._safe_normalize(tangent + self.config.dynamic_nominal_normal_gain * normal)
        if escape_dir is None:
            escape_dir = normal
        return escape_dir

    def mix_positions(
        self,
        ee_pos: np.ndarray,
        current_progress: float,
        nominal_positions: list[np.ndarray],
        signed_dist: float | None,
        obstacle_normal: np.ndarray | None,
    ) -> tuple[list[np.ndarray], dict]:
        ee_pos = np.array(ee_pos, dtype=float)
        nominal_positions = [np.array(pos, dtype=float) for pos in nominal_positions]
        self._ee_history.append(ee_pos.copy())
        self._progress_history.append(float(current_progress))

        progress_gain = (
            float(self._progress_history[-1] - self._progress_history[0]) if len(self._progress_history) >= 2 else 0.0
        )
        exec_delta = self._ee_history[-1] - self._ee_history[0] if len(self._ee_history) >= 2 else np.zeros(3)
        exec_motion = float(np.linalg.norm(exec_delta))
        tracking_error = float(np.linalg.norm(nominal_positions[0] - ee_pos)) if nominal_positions else 0.0
        info = {
            "dynamic_nominal_weight": 0.0,
            "dynamic_nominal_signed_dist": float("inf") if signed_dist is None else float(signed_dist),
            "dynamic_reference_offset_norm": 0.0,
            "dynamic_nominal_progress_gain": progress_gain,
            "dynamic_nominal_exec_motion": exec_motion,
            "dynamic_nominal_tracking_error": tracking_error,
            "dynamic_nominal_stall_active": False,
            "dynamic_escape_dir": None,
            "dynamic_obstacle_normal": None,
        }
        if (not self.config.use_dynamic_nominal_reference) or signed_dist is None or obstacle_normal is None:
            return nominal_positions, info

        normal = np.array(obstacle_normal, dtype=float)
        normal = self._safe_normalize(normal)
        if normal is None:
            return nominal_positions, info

        activation_distance = max(0.10, self.config.safety_margin * 5.0)
        signed_dist = float(signed_dist)
        if signed_dist >= activation_distance:
            return nominal_positions, info

        if len(self._ee_history) < self._ee_history.maxlen or len(self._progress_history) < self._progress_history.maxlen:
            return nominal_positions, info

        is_stall_candidate = (
            progress_gain < self.config.dynamic_nominal_progress_epsilon
            and exec_motion > self.config.dynamic_nominal_exec_motion_trigger
            and tracking_error > self.config.dynamic_nominal_tracking_error_trigger
        )
        if self._stall_active:
            should_release = (
                signed_dist >= activation_distance
                or progress_gain > self.config.dynamic_nominal_release_progress
            )
            if should_release:
                self._stall_active = False
                self._locked_escape_dir = None

        if not self._stall_active and not is_stall_candidate:
            return nominal_positions, info

        if not self._stall_active:
            self._stall_active = True
            self._locked_escape_dir = self._build_escape_dir(exec_delta, ee_pos, nominal_positions, normal)

        escape_dir = normal if self._locked_escape_dir is None else self._locked_escape_dir

        clearance_ratio = float(
            np.clip(
                (activation_distance - signed_dist) / max(activation_distance - self.config.safety_margin, 1e-6),
                0.0,
                1.0,
            )
        )
        stall_ratio = float(
            np.clip(
                1.0 - progress_gain / max(self.config.dynamic_nominal_progress_epsilon, 1e-6),
                0.0,
                1.0,
            )
        )
        weight = min(clearance_ratio * stall_ratio, self.config.dynamic_nominal_max_weight)
        normal_push = max(activation_distance - signed_dist, 0.0) * normal

        escape_positions: list[np.ndarray] = []
        accumulated = 0.0
        anchor = ee_pos
        for nominal_pos in nominal_positions:
            step_len = max(float(np.linalg.norm(nominal_pos - anchor)), self.config.mpc_progress_step_min)
            accumulated += step_len
            escape_positions.append(
                nominal_pos + accumulated * self.config.dynamic_nominal_escape_distance * escape_dir + normal_push
            )
            anchor = nominal_pos

        mixed_positions = [nominal_positions[0].copy()]
        mixed_positions.extend(
            (1.0 - weight) * nominal_pos + weight * escape_pos
            for nominal_pos, escape_pos in zip(nominal_positions[1:], escape_positions[1:])
        )
        offset = mixed_positions[0] - nominal_positions[0] if mixed_positions else np.zeros(3)
        info.update({
            "dynamic_nominal_weight": weight,
            "dynamic_nominal_stall_active": self._stall_active,
            "dynamic_reference_offset_norm": float(np.linalg.norm(offset)),
            "dynamic_escape_dir": np.asarray(escape_dir, dtype=float).tolist(),
            "dynamic_obstacle_normal": np.asarray(normal, dtype=float).tolist(),
        })
        return mixed_positions, info


class MPCDCBFController:
    """按路径进度采样参考并求解 MPC-DCBF 控制量。"""

    def __init__(self, robot, config: ExperimentConfig, trajectory):
        self.robot = robot
        self.config = config
        self.n = robot.total_dof
        self.N = config.N_mpc
        self.trajectory = trajectory
        self.reference_mixer = DynamicNominalReferenceMixer(config)
        self._last_cbf_meta: list[dict] = []
        self._prev_sol: np.ndarray | None = None
        self._cached_u: np.ndarray | None = None
        self._cached_info: dict | None = None
        self._step_count = 0

        single_bounds = (
            [(-config.base_vel_limit, config.base_vel_limit)] * robot.n_pris
            + [(-config.dq_limit, config.dq_limit)] * robot.n_revo
        )
        self._bounds = single_bounds * self.N
        self._lb = np.array([bound[0] for bound in single_bounds])
        self._ub = np.array([bound[1] for bound in single_bounds])

    @staticmethod
    def _safe_unit(vec: np.ndarray) -> np.ndarray | None:
        arr = np.asarray(vec, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-9:
            return None
        return arr / norm

    @staticmethod
    def _project_to_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=float)
        nrm = np.asarray(normal, dtype=float)
        return arr - float(np.dot(arr, nrm)) * nrm

    @classmethod
    def _orthogonal_unit(cls, normal: np.ndarray) -> np.ndarray:
        normal = cls._safe_unit(normal)
        if normal is None:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        trial_axes = (
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        )
        for axis in trial_axes:
            tangent = cls._safe_unit(np.cross(normal, axis))
            if tangent is not None:
                return tangent
        return np.array([1.0, 0.0, 0.0], dtype=float)

    def _build_cbf_data(self, q, dq, obstacles):
        grad_rows, h_vals = [], []
        cbf_meta = []
        for obs in obstacles:
            use_mesh = self.config.use_mesh_cbf and obs.body_id >= 0
            obs_links = getattr(obs, "cbf_link_indices", None)
            check_links = obs_links if obs_links is not None else self.robot.cbf_link_indices
            for link_index in check_links:
                if use_mesh:
                    closest = self.robot.get_closest_points_to_obstacle(link_index, obs.body_id)
                    if closest is None:
                        continue
                    support_point = closest["point_on_link"]
                    signed_dist = closest["signed_dist"]
                    normal_on_link = np.asarray(
                        closest.get("normal_on_link", -np.asarray(closest["normal_on_obstacle"], dtype=float)),
                        dtype=float,
                    )
                    normal_on_obstacle = np.asarray(closest["normal_on_obstacle"], dtype=float)
                    normal = normal_on_obstacle
                    h_val = signed_dist - self.config.safety_margin
                    h_vals.append(h_val)
                    grad_rows.append(self.robot.get_link_cbf_row_at_point(link_index, support_point, normal, q, dq))
                    cbf_meta.append({
                        "link_index": int(link_index),
                        "link_name": self.robot.get_link_name(link_index),
                        "is_ee_link": bool(link_index == self.robot.ee_link_index),
                        "is_welding_gun_link": bool(link_index in self.robot.welding_gun_links),
                        "obs_body_id": int(obs.body_id),
                        "obs_link_index": int(closest["obs_link_index"]),
                        "obs_link_name": str(closest["obs_link_name"]),
                        "use_mesh": True,
                        "signed_dist": float(signed_dist),
                        "h_val": float(h_val),
                        "normal": normal_on_obstacle.tolist(),
                        "normal_on_link": normal_on_link.tolist(),
                        "normal_on_obstacle": normal_on_obstacle.tolist(),
                        "point_on_link": np.asarray(closest["point_on_link"], dtype=float).tolist(),
                        "point_on_obstacle": np.asarray(closest["point_on_obstacle"], dtype=float).tolist(),
                    })
                else:
                    link_pos = self.robot.get_link_origin(link_index)
                    signed_dist, normal = obs.compute_distance(link_pos)
                    h_val = signed_dist - self.config.safety_margin
                    h_vals.append(h_val)
                    grad_rows.append(self.robot.get_link_cbf_row(link_index, normal, q, dq))
                    cbf_meta.append({
                        "link_index": int(link_index),
                        "link_name": self.robot.get_link_name(link_index),
                        "is_ee_link": bool(link_index == self.robot.ee_link_index),
                        "is_welding_gun_link": bool(link_index in self.robot.welding_gun_links),
                        "obs_body_id": int(getattr(obs, "body_id", -1)),
                        "use_mesh": False,
                        "signed_dist": float(signed_dist),
                        "h_val": float(h_val),
                        "normal": np.asarray(normal, dtype=float).tolist(),
                        "normal_on_link": (-np.asarray(normal, dtype=float)).tolist(),
                        "normal_on_obstacle": np.asarray(normal, dtype=float).tolist(),
                        "point_on_link": np.asarray(link_pos, dtype=float).tolist(),
                        "point_on_obstacle": np.asarray(link_pos - signed_dist * np.asarray(normal, dtype=float), dtype=float).tolist(),
                    })
        self._last_cbf_meta = cbf_meta
        return grad_rows, h_vals

    def _build_second_order_risk_terms(
        self,
        q,
        ee_pos: np.ndarray,
        ref_pos: np.ndarray,
        ref_positions: list[np.ndarray],
        active_contexts: list[dict],
        j_pos: np.ndarray,
    ) -> list[dict]:
        if not getattr(self.config, "mpc_second_order_risk_enabled", False):
            return []
        if not active_contexts or not ref_positions or not hasattr(self.robot, "set_joint_state"):
            return []
        risk_horizon = min(max(int(getattr(self.config, "mpc_second_order_risk_horizon", 0)), 0), len(ref_positions), self.N)
        if risk_horizon <= 0:
            return []
        preview_tau = max(float(getattr(self.config, "mpc_second_order_preview_tau", self.config.mpc_dt)), 1e-6)
        fd_eps = max(float(getattr(self.config, "mpc_second_order_fd_eps", 0.25 * self.config.mpc_dt)), 1e-6)
        safe_margin = float(getattr(self.config, "mpc_second_order_safe_margin", self.config.safety_margin))
        anchor = np.asarray(ref_pos, dtype=float)
        risk_terms = []
        for step_idx in range(risk_horizon):
            target = np.asarray(ref_positions[step_idx], dtype=float)
            cart_dir = self._safe_unit(target - anchor)
            anchor = target
            if cart_dir is None:
                continue
            desired_speed = max(float(np.linalg.norm(target - np.asarray(ee_pos, dtype=float))) / max((step_idx + 1) * preview_tau, 1e-6), self.config.mpc_progress_step_min)
            joint_dir = np.linalg.lstsq(j_pos, desired_speed * cart_dir, rcond=None)[0]
            joint_dir = np.clip(joint_dir, self._lb, self._ub)
            if float(np.linalg.norm(joint_dir)) <= 1e-9:
                continue
            predicted_min_h = self._evaluate_directional_barrier_value(
                q=np.asarray(q, dtype=float),
                joint_direction=joint_dir,
                active_contexts=active_contexts,
                step_scale=(step_idx + 1) * preview_tau,
            )
            base_min_h = float(min(float(ctx.get("h_val", float("inf"))) for ctx in active_contexts))
            predicted_curvature = self._estimate_directional_second_order_curvature(
                q=np.asarray(q, dtype=float),
                joint_direction=joint_dir,
                active_contexts=active_contexts,
                base_min_h=base_min_h,
                fd_eps=fd_eps,
            )
            shortfall = max(safe_margin - float(predicted_min_h), 0.0)
            negative_curvature = max(-float(predicted_curvature), 0.0)
            if shortfall <= 0.0 and negative_curvature <= 0.0:
                continue
            risk_terms.append({
                "horizon_step": int(step_idx),
                "joint_direction": np.asarray(joint_dir, dtype=float),
                "risk_weight": float(getattr(self.config, "mpc_second_order_risk_weight", 0.0)),
                "shrink_weight": float(getattr(self.config, "mpc_second_order_shrink_weight", 0.0)),
                "shortfall": float(shortfall),
                "negative_curvature": float(negative_curvature),
            })
        return risk_terms

    def _build_qp(
        self,
        ee_pos,
        j_pos,
        j_rot,
        ref_positions,
        ref_rotvecs,
        grad_rows,
        h_vals,
        orientation_weight,
        second_order_risk_terms: list[dict] | None = None,
    ):
        n, N = self.n, self.N
        mdt = self.config.mpc_dt
        cfg = self.config
        dim = n * N

        jtj_pos = mdt ** 2 * (j_pos.T @ j_pos)
        jtj_rot = mdt ** 2 * (j_rot.T @ j_rot)
        idx = np.arange(N)
        weight_mat = (N - np.maximum(idx[:, None], idx[None, :])).astype(float)
        h_mat = 2.0 * cfg.mpc_tracking_weight * np.kron(weight_mat, jtj_pos)
        h_mat += 2.0 * orientation_weight * np.kron(weight_mat, jtj_rot)

        c_vecs = np.array([ee_pos - ref_positions[k] for k in range(N)])
        c_suffix = np.cumsum(c_vecs[::-1], axis=0)[::-1]
        f_vec = 2.0 * cfg.mpc_tracking_weight * mdt * (c_suffix @ j_pos).ravel()
        rot_vecs = -np.array(ref_rotvecs, dtype=float)
        rot_suffix = np.cumsum(rot_vecs[::-1], axis=0)[::-1]
        f_vec += 2.0 * orientation_weight * mdt * (rot_suffix @ j_rot).ravel()

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

        for term in second_order_risk_terms or []:
            step_idx = int(np.clip(term.get("horizon_step", 0), 0, N - 1))
            joint_direction = np.asarray(term.get("joint_direction", np.zeros(n)), dtype=float).reshape(-1)
            norm = float(np.linalg.norm(joint_direction))
            if norm <= 1e-9:
                continue
            direction = joint_direction / norm
            weight = (
                float(term.get("risk_weight", 0.0)) * max(float(term.get("shortfall", 0.0)), 0.0)
                + float(term.get("shrink_weight", 0.0)) * max(float(term.get("negative_curvature", 0.0)), 0.0)
            )
            if weight <= 0.0:
                continue
            sl = slice(step_idx * n, (step_idx + 1) * n)
            h_mat[sl, sl] += 2.0 * weight * np.outer(direction, direction)
            f_vec[sl] += 2.0 * weight * direction

        return h_mat, f_vec, a_cbf, b_cbf

    def _get_dynamic_nominal_hint(self, obstacles):
        hint_enabled = (
            bool(getattr(self.config, "use_dynamic_nominal_reference", False))
            or bool(getattr(self.config, "second_order_nominal_enabled", False))
            or bool(getattr(self.config, "mpc_second_order_risk_enabled", False))
        )
        if self.config.ignore_all_collisions or not hint_enabled:
            return None, None

        best_signed_dist = None
        best_normal = None
        max_dist = max(1.0, self.config.safety_margin * 10.0)
        for obs in obstacles:
            obs_body_id = getattr(obs, "body_id", -1)
            if obs_body_id < 0:
                continue
            obs_links = getattr(obs, "cbf_link_indices", None)
            check_links = obs_links if obs_links is not None else self.robot.cbf_link_indices
            for link_index in check_links:
                closest = self.robot.get_closest_point_to_obstacle(
                    link_index,
                    obs_body_id,
                    max_dist=max_dist,
                )
                if closest is None:
                    continue
                _, signed_dist, normal = closest
                if best_signed_dist is None or signed_dist < best_signed_dist:
                    best_signed_dist = float(signed_dist)
                    best_normal = np.array(normal, dtype=float)
        return best_signed_dist, best_normal

    def _collect_second_order_barrier_context(self, obstacles, h_vals) -> list[dict]:
        if not getattr(self.config, "second_order_nominal_enabled", False):
            return []
        max_links = max(int(getattr(self.config, "second_order_nominal_active_links", 0)), 0)
        if max_links <= 0 or not self._last_cbf_meta or not h_vals:
            return []
        obstacle_by_body = {int(getattr(obs, "body_id", -1)): obs for obs in obstacles}
        ranked_indices = np.argsort(np.asarray(h_vals, dtype=float))
        contexts: list[dict] = []
        for idx in ranked_indices:
            if len(contexts) >= max_links:
                break
            meta = dict(self._last_cbf_meta[int(idx)])
            obs = obstacle_by_body.get(int(meta.get("obs_body_id", -1)))
            if obs is None:
                continue
            meta["obstacle"] = obs
            meta["h_val"] = float(h_vals[int(idx)])
            contexts.append(meta)
        return contexts

    def _evaluate_barrier_context_min_h(self, q_eval, active_contexts: list[dict]) -> float:
        if not active_contexts:
            return float("inf")
        if not hasattr(self.robot, "set_joint_state"):
            return float(min(float(ctx.get("h_val", float("inf"))) for ctx in active_contexts))
        q_eval = np.asarray(q_eval, dtype=float)
        dq_eval = np.zeros_like(q_eval)
        q_backup = None
        dq_backup = None
        if hasattr(self.robot, "get_joint_state"):
            try:
                q_backup, dq_backup = self.robot.get_joint_state()
            except Exception:
                q_backup, dq_backup = None, None
        try:
            self.robot.set_joint_state(q_eval, dq_eval)
        except TypeError:
            self.robot.set_joint_state(q_eval)
        try:
            min_h = float("inf")
            max_dist = max(1.0, getattr(self.config, "second_order_nominal_activation_distance", 0.08) * 4.0)
            for ctx in active_contexts:
                link_index = int(ctx["link_index"])
                obs = ctx["obstacle"]
                obs_body_id = int(getattr(obs, "body_id", -1))
                if obs_body_id >= 0:
                    closest = self.robot.get_closest_points_to_obstacle(link_index, obs_body_id, max_dist=max_dist)
                    if closest is None:
                        continue
                    signed_dist = float(closest["signed_dist"])
                else:
                    link_pos = self.robot.get_link_origin(link_index)
                    signed_dist, _ = obs.compute_distance(link_pos)
                    signed_dist = float(signed_dist)
                min_h = min(min_h, signed_dist - self.config.safety_margin)
            return min_h
        finally:
            if q_backup is not None:
                try:
                    self.robot.set_joint_state(q_backup, dq_backup)
                except TypeError:
                    self.robot.set_joint_state(q_backup)

    def _evaluate_directional_barrier_value(self, q, joint_direction: np.ndarray, active_contexts: list[dict], step_scale: float) -> float:
        q = np.asarray(q, dtype=float)
        u_dir = np.asarray(joint_direction, dtype=float)
        return self._evaluate_barrier_context_min_h(q + float(step_scale) * u_dir, active_contexts)

    def _estimate_directional_second_order_curvature(
        self,
        q,
        joint_direction: np.ndarray,
        active_contexts: list[dict],
        base_min_h: float,
        fd_eps: float,
    ) -> float:
        eps = max(float(fd_eps), 1e-6)
        forward_h = self._evaluate_directional_barrier_value(q, joint_direction, active_contexts, eps)
        backward_h = self._evaluate_directional_barrier_value(q, joint_direction, active_contexts, -eps)
        return float((forward_h - 2.0 * float(base_min_h) + backward_h) / (eps ** 2))

    def _build_nominal_direction_candidates(
        self,
        ee_pos: np.ndarray,
        ref_pos: np.ndarray,
        ref_positions: list[np.ndarray],
        obstacle_normal: np.ndarray | None,
    ) -> list[np.ndarray]:
        target_point = ref_positions[0] if ref_positions else ref_pos
        goal_dir = self._safe_unit(np.asarray(target_point, dtype=float) - np.asarray(ee_pos, dtype=float))
        ref_dir = self._safe_unit(np.asarray(target_point, dtype=float) - np.asarray(ref_pos, dtype=float))
        normal = self._safe_unit(obstacle_normal) if obstacle_normal is not None else None
        candidates: list[np.ndarray] = []

        def append_candidate(vec):
            unit = self._safe_unit(vec)
            if unit is None:
                return
            if any(abs(float(np.dot(unit, existing))) > 0.995 for existing in candidates):
                return
            candidates.append(unit)

        append_candidate(ref_dir if ref_dir is not None else goal_dir)
        append_candidate(goal_dir)
        if normal is not None:
            tangent = self._safe_unit(self._project_to_plane(goal_dir if goal_dir is not None else normal, normal))
            if tangent is None:
                tangent = self._orthogonal_unit(normal)
            lateral = self._safe_unit(np.cross(normal, tangent))
            append_candidate(tangent)
            append_candidate(normal)
            append_candidate(tangent + 0.4 * normal)
            append_candidate(tangent - 0.4 * normal)
            if lateral is not None:
                append_candidate(lateral)
                append_candidate(-lateral)
                append_candidate(lateral + 0.25 * normal)
        return candidates[: max(int(getattr(self.config, "second_order_nominal_candidate_count", 0)), 1)]

    def _select_second_order_nominal_direction(self, candidate_metrics: list[dict]) -> dict | None:
        if not candidate_metrics:
            return None
        cfg = self.config
        best = None
        for metric in candidate_metrics:
            clearance = float(metric.get("predicted_min_h", -float("inf")))
            curvature = float(metric.get("predicted_curvature", 0.0))
            clearance_shortfall = max(-clearance, 0.0)
            score = (
                float(getattr(cfg, "second_order_nominal_goal_weight", 1.0)) * float(metric.get("goal_alignment", 0.0))
                + float(getattr(cfg, "second_order_nominal_clearance_weight", 1.0)) * clearance
                - float(getattr(cfg, "second_order_nominal_clearance_weight", 1.0)) * 32.0 * clearance_shortfall
                - float(getattr(cfg, "second_order_nominal_curvature_weight", 0.0)) * max(-curvature, 0.0)
                + float(getattr(cfg, "second_order_nominal_reference_weight", 0.0)) * float(metric.get("reference_alignment", 0.0))
            )
            candidate = dict(metric)
            candidate["score"] = float(score)
            if best is None or float(candidate["score"]) > float(best["score"]):
                best = candidate
        return best

    def _apply_second_order_nominal_preview(
        self,
        q,
        ee_pos: np.ndarray,
        ref_pos: np.ndarray,
        ref_positions: list[np.ndarray],
        active_contexts: list[dict],
        obstacle_normal: np.ndarray | None,
        dynamic_info: dict,
        j_pos: np.ndarray,
    ) -> tuple[list[np.ndarray], dict]:
        preview_info = {
            "second_order_nominal_active": False,
            "second_order_nominal_weight": 0.0,
            "second_order_nominal_best_score": 0.0,
            "second_order_nominal_predicted_min_h": float("inf"),
            "second_order_nominal_predicted_curvature": 0.0,
            "second_order_nominal_selected_direction": None,
            "second_order_nominal_skip_reason": None,
        }
        if not getattr(self.config, "second_order_nominal_enabled", False):
            return [np.asarray(pos, dtype=float).copy() for pos in ref_positions], preview_info
        if not ref_positions or not active_contexts or obstacle_normal is None:
            return [np.asarray(pos, dtype=float).copy() for pos in ref_positions], preview_info
        if not hasattr(self.robot, "set_joint_state"):
            return [np.asarray(pos, dtype=float).copy() for pos in ref_positions], preview_info

        signed_dist = float(dynamic_info.get("dynamic_nominal_signed_dist", float("inf")))
        activation_distance = max(
            float(getattr(self.config, "second_order_nominal_activation_distance", 0.08)),
            self.config.safety_margin * 5.0,
        )
        stall_active = bool(dynamic_info.get("dynamic_nominal_stall_active", False))
        # region agent log
        _append_debug_log(
            f"solve_{self._step_count}",
            "SO2",
            "welding_320_control.py:875",
            "Second-order preview gate",
            {
                "step_count": int(self._step_count),
                "signed_dist": float(signed_dist),
                "activation_distance": float(activation_distance),
                "stall_active": bool(stall_active),
                "active_context_count": int(len(active_contexts)),
                "obstacle_normal_norm": float(np.linalg.norm(obstacle_normal)),
                "second_order_enabled": bool(getattr(self.config, "second_order_nominal_enabled", False)),
            },
        )
        # endregion
        if signed_dist >= activation_distance and not stall_active:
            return [np.asarray(pos, dtype=float).copy() for pos in ref_positions], preview_info

        base_min_h = float(min(float(ctx.get("h_val", float("inf"))) for ctx in active_contexts))
        if not np.isfinite(base_min_h):
            return [np.asarray(pos, dtype=float).copy() for pos in ref_positions], preview_info

        candidates = self._build_nominal_direction_candidates(
            ee_pos=np.asarray(ee_pos, dtype=float),
            ref_pos=np.asarray(ref_pos, dtype=float),
            ref_positions=[np.asarray(pos, dtype=float) for pos in ref_positions],
            obstacle_normal=np.asarray(obstacle_normal, dtype=float),
        )
        if not candidates:
            return [np.asarray(pos, dtype=float).copy() for pos in ref_positions], preview_info

        goal_dir = self._safe_unit(np.asarray(ref_positions[0], dtype=float) - np.asarray(ee_pos, dtype=float))
        ref_dir = self._safe_unit(np.asarray(ref_positions[0], dtype=float) - np.asarray(ref_pos, dtype=float))
        preview_tau = max(float(getattr(self.config, "second_order_nominal_preview_tau", self.config.mpc_dt)), 1e-6)
        fd_eps = max(float(getattr(self.config, "second_order_nominal_fd_eps", 0.25 * self.config.mpc_dt)), 1e-6)
        base_step = max(float(np.linalg.norm(np.asarray(ref_positions[0], dtype=float) - np.asarray(ref_pos, dtype=float))), self.config.mpc_progress_step_min)
        desired_speed = base_step / preview_tau

        candidate_metrics = []
        for cart_dir in candidates:
            joint_dir = np.linalg.lstsq(j_pos, desired_speed * cart_dir, rcond=None)[0]
            joint_dir = np.clip(joint_dir, self._lb, self._ub)
            if float(np.linalg.norm(joint_dir)) <= 1e-9:
                continue
            predicted_min_h = self._evaluate_directional_barrier_value(q, joint_dir, active_contexts, preview_tau)
            predicted_curvature = self._estimate_directional_second_order_curvature(
                q=q,
                joint_direction=joint_dir,
                active_contexts=active_contexts,
                base_min_h=base_min_h,
                fd_eps=fd_eps,
            )
            candidate_metrics.append({
                "direction": np.asarray(cart_dir, dtype=float),
                "joint_direction": np.asarray(joint_dir, dtype=float),
                "goal_alignment": float(np.dot(cart_dir, goal_dir)) if goal_dir is not None else 0.0,
                "reference_alignment": float(np.dot(cart_dir, ref_dir)) if ref_dir is not None else 0.0,
                "predicted_min_h": float(predicted_min_h),
                "predicted_curvature": float(predicted_curvature),
            })
        scored_candidates = []
        for metric in candidate_metrics:
            scored = self._select_second_order_nominal_direction([metric])
            if scored is not None:
                scored_candidates.append(scored)
        selected = self._select_second_order_nominal_direction(candidate_metrics)
        ranked_candidates = sorted(scored_candidates, key=lambda item: float(item.get("score", -float("inf"))), reverse=True)
        prev_selected_direction = getattr(self, "_debug_prev_second_order_nominal_direction", None)
        selected_vs_prev_cos = None
        if (
            selected is not None
            and prev_selected_direction is not None
            and np.linalg.norm(prev_selected_direction) > 1e-9
        ):
            selected_vs_prev_cos = float(
                np.dot(np.asarray(selected["direction"], dtype=float), prev_selected_direction)
                / max(
                    np.linalg.norm(np.asarray(selected["direction"], dtype=float)) * np.linalg.norm(prev_selected_direction),
                    1e-9,
                )
            )
        # region agent log
        _append_debug_log(
            f"solve_{self._step_count}",
            "SO1",
            "welding_320_control.py:921",
            "Second-order candidate ranking",
            {
                "step_count": int(self._step_count),
                "base_min_h": float(base_min_h),
                "candidate_count": int(len(candidate_metrics)),
                "selected_direction": None
                if selected is None
                else np.asarray(selected["direction"], dtype=float).tolist(),
                "selected_score": None if selected is None else float(selected.get("score", 0.0)),
                "selected_vs_prev_cos": selected_vs_prev_cos,
                "preview_score_threshold": 0.0,
                "top_candidates": [
                    {
                        "direction": np.asarray(item["direction"], dtype=float).tolist(),
                        "score": float(item.get("score", 0.0)),
                        "predicted_min_h": float(item.get("predicted_min_h", float("inf"))),
                        "predicted_curvature": float(item.get("predicted_curvature", 0.0)),
                        "goal_alignment": float(item.get("goal_alignment", 0.0)),
                        "reference_alignment": float(item.get("reference_alignment", 0.0)),
                    }
                    for item in ranked_candidates[:3]
                ],
            },
        )
        # endregion
        if selected is None:
            return [np.asarray(pos, dtype=float).copy() for pos in ref_positions], preview_info
        if float(selected.get("score", 0.0)) <= 0.0:
            preview_info.update({
                "second_order_nominal_best_score": float(selected.get("score", 0.0)),
                "second_order_nominal_predicted_min_h": float(selected["predicted_min_h"]),
                "second_order_nominal_predicted_curvature": float(selected["predicted_curvature"]),
                "second_order_nominal_selected_direction": np.asarray(selected["direction"], dtype=float).tolist(),
                "second_order_nominal_skip_reason": "non_positive_score",
            })
            # region agent log
            _append_debug_log(
                f"solve_{self._step_count}",
                "SO3",
                "welding_320_control.py:931",
                "Second-order preview suppressed",
                {
                    "step_count": int(self._step_count),
                    "reason": "non_positive_score",
                    "selected_score": float(selected.get("score", 0.0)),
                    "selected_direction": np.asarray(selected["direction"], dtype=float).tolist(),
                    "predicted_min_h": float(selected["predicted_min_h"]),
                    "predicted_curvature": float(selected["predicted_curvature"]),
                },
            )
            # endregion
            return [np.asarray(pos, dtype=float).copy() for pos in ref_positions], preview_info

        clearance_ratio = float(
            np.clip(
                (activation_distance - signed_dist) / max(activation_distance - self.config.safety_margin, 1e-6),
                0.0,
                1.0,
            )
        )
        preview_weight = clearance_ratio * (1.0 if stall_active else 0.4)
        shifted_refs = [np.asarray(pos, dtype=float).copy() for pos in ref_positions]
        for idx, nominal_pos in enumerate(shifted_refs):
            shifted_refs[idx] = nominal_pos + preview_weight * (idx + 1) * base_step * np.asarray(selected["direction"], dtype=float)
        self._debug_prev_second_order_nominal_direction = np.asarray(selected["direction"], dtype=float).copy()
        first_shift_norm = (
            float(np.linalg.norm(shifted_refs[0] - np.asarray(ref_positions[0], dtype=float)))
            if shifted_refs
            else 0.0
        )
        last_shift_norm = (
            float(np.linalg.norm(shifted_refs[-1] - np.asarray(ref_positions[-1], dtype=float)))
            if shifted_refs
            else 0.0
        )
        # region agent log
        _append_debug_log(
            f"solve_{self._step_count}",
            "SO3",
            "welding_320_control.py:935",
            "Second-order reference shift",
            {
                "step_count": int(self._step_count),
                "base_step": float(base_step),
                "preview_tau": float(preview_tau),
                "fd_eps": float(fd_eps),
                "clearance_ratio": float(clearance_ratio),
                "preview_weight": float(preview_weight),
                "first_shift_norm": float(first_shift_norm),
                "last_shift_norm": float(last_shift_norm),
                "selected_direction": np.asarray(selected["direction"], dtype=float).tolist(),
            },
        )
        # endregion
        preview_info.update({
            "second_order_nominal_active": preview_weight > 0.0,
            "second_order_nominal_weight": float(preview_weight),
            "second_order_nominal_best_score": float(selected.get("score", 0.0)),
            "second_order_nominal_predicted_min_h": float(selected["predicted_min_h"]),
            "second_order_nominal_predicted_curvature": float(selected["predicted_curvature"]),
            "second_order_nominal_selected_direction": np.asarray(selected["direction"], dtype=float).tolist(),
        })
        return shifted_refs, preview_info

    def _compute_active_orientation_weight(self, current_progress: float) -> tuple[float, float]:
        cfg = self.config
        segment_index = None
        if self.trajectory is not None and hasattr(self.trajectory, "current_segment_index"):
            try:
                segment_index = int(self.trajectory.current_segment_index(float(current_progress)))
            except Exception:
                segment_index = None
        if segment_index == 1:
            return float(cfg.mpc_orientation_tracking_weight), 1.0

        remaining_progress = max(self.trajectory.progress_end - float(current_progress), 0.0)
        orientation_phase_ratio = float(
            np.clip(
                1.0 - remaining_progress / max(cfg.mpc_terminal_orientation_window, 1e-6),
                0.0,
                1.0,
            )
        )
        return float(cfg.mpc_orientation_tracking_weight * orientation_phase_ratio), orientation_phase_ratio

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
        obstacles,
        current_progress=0.0,
    ):
        self._step_count += 1
        run_id = f"solve_{self._step_count}"
        second_order_cache_guard = False
        if self._cached_info is not None and bool(getattr(self.config, "second_order_nominal_enabled", False)):
            activation_distance = max(
                float(getattr(self.config, "second_order_nominal_activation_distance", 0.08)),
                self.config.safety_margin * 5.0,
            )
            cached_signed_dist = float(self._cached_info.get("dynamic_nominal_signed_dist", float("inf")))
            second_order_cache_guard = (
                bool(self._cached_info.get("second_order_nominal_active", False))
                or cached_signed_dist < activation_distance
            )
        can_use_cache = (
            self._cached_u is not None
            and self._step_count % self.config.mpc_replan_steps != 0
            and self._cached_info is not None
            and float(self._cached_info.get("min_h", 1.0)) >= 0.0
            and not bool(self._cached_info.get("dynamic_nominal_stall_active", False))
            and not second_order_cache_guard
        )
        if can_use_cache:
            # region agent log
            _append_debug_log(
                run_id,
                "H4",
                "welding_320_control.py:516",
                "Returning cached control",
                {
                    "step_count": int(self._step_count),
                    "replan_steps": int(self.config.mpc_replan_steps),
                    "cached_u_norm": float(np.linalg.norm(self._cached_u)),
                    "cached_status": str(self._cached_info.get("status", "unknown")) if self._cached_info else "missing",
                    "cached_stall_active": bool(self._cached_info.get("dynamic_nominal_stall_active", False))
                    if self._cached_info
                    else False,
                },
            )
            # endregion
            return self._cached_u, self._cached_info
        if self._cached_u is not None and self._cached_info is not None and second_order_cache_guard:
            # region agent log
            _append_debug_log(
                run_id,
                "H4",
                "welding_320_control.py:520",
                "Bypassing cached control for second-order zone",
                {
                    "step_count": int(self._step_count),
                    "replan_steps": int(self.config.mpc_replan_steps),
                    "cached_u_norm": float(np.linalg.norm(self._cached_u)),
                    "cached_second_order_active": bool(self._cached_info.get("second_order_nominal_active", False)),
                    "cached_signed_dist": float(self._cached_info.get("dynamic_nominal_signed_dist", float("inf"))),
                    "cached_min_h": float(self._cached_info.get("min_h", float("inf"))),
                },
            )
            # endregion

        n, N = self.n, self.N
        cfg = self.config
        j_full = self.robot.get_ee_jacobian(q, dq)
        j_pos = j_full[:3]
        j_rot = j_full[3:]
        singular_values = np.linalg.svd(j_full, compute_uv=False)

        ref_positions = []
        ref_rotvecs = []
        ds_ref = max(np.linalg.norm(ref_lin_vel) * cfg.mpc_dt, cfg.mpc_progress_step_min)
        for k in range(1, N + 1):
            pk, qk, _, _ = self.trajectory.sample_by_progress(
                min(current_progress + k * ds_ref, self.trajectory.progress_end)
            )
            ref_positions.append(pk)
            ref_rotvecs.append(quaternion_error_rotvec(ee_quat, qk))

        original_ref_pos = np.array(ref_pos, dtype=float)
        original_ref_positions = [np.array(pk, dtype=float) for pk in ref_positions]

        dynamic_signed_dist, dynamic_normal = self._get_dynamic_nominal_hint(obstacles)
        mixed_refs, dynamic_info = self.reference_mixer.mix_positions(
            ee_pos=np.array(ee_pos, dtype=float),
            current_progress=float(current_progress),
            nominal_positions=[original_ref_pos.copy()] + [pk.copy() for pk in original_ref_positions],
            signed_dist=dynamic_signed_dist,
            obstacle_normal=dynamic_normal,
        )
        ref_pos = mixed_refs[0]
        ref_positions = mixed_refs[1:]

        remaining_progress = max(self.trajectory.progress_end - float(current_progress), 0.0)
        active_orientation_weight, orientation_phase_ratio = self._compute_active_orientation_weight(current_progress)

        # region agent log
        _append_debug_log(
            run_id,
            "H2",
            "welding_320_control.py:555",
            "Reference and Jacobian snapshot",
            {
                "step_count": int(self._step_count),
                "current_progress": float(current_progress),
                "ref_delta_norm": float(np.linalg.norm(ref_pos - ee_pos)),
                "ref_lin_speed": float(np.linalg.norm(ref_lin_vel)),
                "jacobian_pos_norm": float(np.linalg.norm(j_pos)),
                "jacobian_rot_norm": float(np.linalg.norm(j_rot)),
                "jacobian_sigma_max": float(singular_values[0]) if singular_values.size else 0.0,
                "jacobian_sigma_min": float(singular_values[-1]) if singular_values.size else 0.0,
                "dynamic_weight": float(dynamic_info.get("dynamic_nominal_weight", 0.0)),
                "dynamic_stall_active": bool(dynamic_info.get("dynamic_nominal_stall_active", False)),
                "dynamic_offset_norm": float(dynamic_info.get("dynamic_reference_offset_norm", 0.0)),
                "active_orientation_weight": float(active_orientation_weight),
                "orientation_phase_ratio": float(orientation_phase_ratio),
            },
        )
        # endregion

        original_step_cos = 0.0
        mixed_step_cos = 0.0
        if original_ref_positions:
            original_ref_step = original_ref_positions[0] - original_ref_pos
            original_ref_err = original_ref_pos - ee_pos
            original_step_cos = float(
                np.dot(original_ref_step, original_ref_err)
                / max(np.linalg.norm(original_ref_step) * np.linalg.norm(original_ref_err), 1e-9)
            )
        if ref_positions:
            mixed_ref_step = ref_positions[0] - ref_pos
            mixed_ref_err = ref_pos - ee_pos
            mixed_step_cos = float(
                np.dot(mixed_ref_step, mixed_ref_err)
                / max(np.linalg.norm(mixed_ref_step) * np.linalg.norm(mixed_ref_err), 1e-9)
            )

        # region agent log
        _append_debug_log(
            run_id,
            "H10",
            "welding_320_control.py:592",
            "Reference geometry comparison",
            {
                "step_count": int(self._step_count),
                "original_ref_err_norm": float(np.linalg.norm(original_ref_pos - ee_pos)),
                "mixed_ref_err_norm": float(np.linalg.norm(ref_pos - ee_pos)),
                "original_step_cos": original_step_cos,
                "mixed_step_cos": mixed_step_cos,
                "dynamic_weight": float(dynamic_info.get("dynamic_nominal_weight", 0.0)),
                "remaining_progress": float(remaining_progress),
                "orientation_phase_ratio": float(orientation_phase_ratio),
            },
        )
        # endregion

        grad_rows, h_vals = self._build_cbf_data(q, dq, obstacles)
        min_h = float(np.min(h_vals)) if h_vals else 1.0
        worst_cbf = None
        if h_vals:
            worst_idx = int(np.argmin(h_vals))
            worst_cbf = dict(self._last_cbf_meta[worst_idx])
            worst_cbf["cbf_index"] = worst_idx
        second_order_context = self._collect_second_order_barrier_context(obstacles, h_vals)
        ref_positions, second_order_info = self._apply_second_order_nominal_preview(
            q=np.asarray(q, dtype=float),
            ee_pos=np.asarray(ee_pos, dtype=float),
            ref_pos=np.asarray(ref_pos, dtype=float),
            ref_positions=[np.asarray(pos, dtype=float) for pos in ref_positions],
            active_contexts=second_order_context,
            obstacle_normal=dynamic_normal,
            dynamic_info=dynamic_info,
            j_pos=j_pos,
        )
        second_order_risk_terms = self._build_second_order_risk_terms(
            q=np.asarray(q, dtype=float),
            ee_pos=np.asarray(ee_pos, dtype=float),
            ref_pos=np.asarray(ref_pos, dtype=float),
            ref_positions=[np.asarray(pos, dtype=float) for pos in ref_positions],
            active_contexts=second_order_context,
            j_pos=j_pos,
        )
        h_mat, f_vec, a_cbf, b_cbf = self._build_qp(
            ee_pos,
            j_pos,
            j_rot,
            ref_positions,
            ref_rotvecs,
            grad_rows,
            h_vals,
            active_orientation_weight,
            second_order_risk_terms=second_order_risk_terms,
        )

        constraints = []
        if len(grad_rows) > 0:
            constraints.append({
                "type": "ineq",
                "fun": lambda x: a_cbf @ x + b_cbf,
                "jac": lambda x: a_cbf,
            })

        pos_err = ref_pos - ee_pos
        rot_err = quaternion_error_rotvec(ee_quat, ref_quat)
        xdot_nom_pos = ref_lin_vel + cfg.position_gain * pos_err
        xdot_nom_rot = ref_ang_vel + cfg.orientation_gain * rot_err
        xdot_nom = np.concatenate([
            xdot_nom_pos,
            xdot_nom_rot,
        ])
        u_nom = np.clip(np.linalg.lstsq(j_pos, xdot_nom_pos, rcond=None)[0], self._lb, self._ub)
        ee_nom_vel = j_pos @ u_nom
        ee_nom_rot = j_rot @ u_nom

        # region agent log
        _append_debug_log(
            run_id,
            "H8",
            "welding_320_control.py:666",
            "Nominal inverse-kinematics breakdown",
            {
                "step_count": int(self._step_count),
                "pos_err_norm": float(np.linalg.norm(pos_err)),
                "rot_err_norm": float(np.linalg.norm(rot_err)),
                "ref_lin_speed": float(np.linalg.norm(ref_lin_vel)),
                "ref_ang_speed": float(np.linalg.norm(ref_ang_vel)),
                "xdot_nom_lin_speed": float(np.linalg.norm(xdot_nom[:3])),
                "xdot_nom_rot_speed": float(np.linalg.norm(xdot_nom[3:])),
                "ee_nom_lin_speed": float(np.linalg.norm(ee_nom_vel)),
                "ee_nom_rot_speed": float(np.linalg.norm(ee_nom_rot)),
                "nominal_solver_mode": "position_only_lstsq",
                "xdot_nom_vs_pos_err_cos": float(
                    np.dot(xdot_nom[:3], pos_err)
                    / max(np.linalg.norm(xdot_nom[:3]) * np.linalg.norm(pos_err), 1e-9)
                ),
                "ee_nom_vs_pos_err_cos": float(
                    np.dot(ee_nom_vel, pos_err)
                    / max(np.linalg.norm(ee_nom_vel) * np.linalg.norm(pos_err), 1e-9)
                ),
                "ref_vel_vs_pos_err_cos": float(
                    np.dot(ref_lin_vel, pos_err)
                    / max(np.linalg.norm(ref_lin_vel) * np.linalg.norm(pos_err), 1e-9)
                ),
                "nominal_lin_residual": float(np.linalg.norm(ee_nom_vel - xdot_nom[:3])),
                "nominal_rot_residual": float(np.linalg.norm(ee_nom_rot - xdot_nom[3:])),
            },
        )
        # endregion

        if self._prev_sol is not None:
            x0 = np.empty(n * N)
            x0[: (N - 1) * n] = self._prev_sol[n:]
            x0[(N - 1) * n :] = self._prev_sol[(N - 1) * n :]
        else:
            x0 = np.tile(u_nom, N)

        zero_constraints = b_cbf if len(grad_rows) > 0 else np.array([], dtype=float)
        zero_feasible = bool(np.all(zero_constraints >= -1e-9)) if zero_constraints.size else True
        x0_constraint_margin = float(np.min(a_cbf @ x0 + b_cbf)) if len(grad_rows) > 0 else float("inf")

        # region agent log
        _append_debug_log(
            run_id,
            "H3",
            "welding_320_control.py:604",
            "QP pre-solve feasibility snapshot",
            {
                "step_count": int(self._step_count),
                "min_h": float(min_h),
                "cbf_count": int(len(grad_rows)),
                "zero_feasible": bool(zero_feasible),
                "zero_constraint_margin": float(np.min(zero_constraints)) if zero_constraints.size else float("inf"),
                "x0_norm": float(np.linalg.norm(x0)),
                "x0_constraint_margin": x0_constraint_margin,
                "f_vec_norm": float(np.linalg.norm(f_vec)),
                "h_mat_norm": float(np.linalg.norm(h_mat)),
                "worst_cbf": worst_cbf,
            },
        )
        # endregion

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

        tightest_constraint = None
        if len(grad_rows) > 0:
            solved_margin = a_cbf @ res.x + b_cbf if hasattr(res, "x") else a_cbf @ x0 + b_cbf
            tight_row = int(np.argmin(solved_margin))
            tightest_constraint = {
                "row_index": tight_row,
                "margin": float(solved_margin[tight_row]),
                "cbf_index": int(tight_row // N),
                "horizon_step": int(tight_row % N),
            }
            if tightest_constraint["cbf_index"] < len(self._last_cbf_meta):
                tightest_constraint.update(self._last_cbf_meta[tightest_constraint["cbf_index"]])

        # region agent log
        _append_debug_log(
            run_id,
            "H1",
            "welding_320_control.py:640",
            "QP solve result",
            {
                "step_count": int(self._step_count),
                "solver_success": bool(res.success),
                "solver_status": int(getattr(res, "status", -1)),
                "status_label": status,
                "objective_value": float(res.fun) if hasattr(res, "fun") else float("nan"),
                "u_cmd_norm": float(np.linalg.norm(u_cmd)),
                "u_cmd_max_abs": float(np.max(np.abs(u_cmd))) if np.size(u_cmd) else 0.0,
                "x0_norm": float(np.linalg.norm(x0)),
                "min_h": float(min_h),
                "zero_feasible": bool(zero_feasible),
                "tightest_constraint": tightest_constraint,
            },
        )
        # endregion

        ee_cmd_vel = j_pos @ u_cmd
        pos_err_norm = float(np.linalg.norm(pos_err))
        task_dir = pos_err / pos_err_norm if pos_err_norm > 1e-9 else np.zeros(3, dtype=float)
        escape_dir = np.array(dynamic_info.get("dynamic_escape_dir") or np.zeros(3), dtype=float)
        obstacle_normal = np.array(dynamic_info.get("dynamic_obstacle_normal") or np.zeros(3), dtype=float)

        # region agent log
        _append_debug_log(
            run_id,
            "H12",
            "welding_320_control.py:847",
            "Dynamic escape-vs-task snapshot",
            {
                "step_count": int(self._step_count),
                "stall_active": bool(dynamic_info.get("dynamic_nominal_stall_active", False)),
                "dynamic_weight": float(dynamic_info.get("dynamic_nominal_weight", 0.0)),
                "signed_dist": float(dynamic_info.get("dynamic_nominal_signed_dist", float("inf"))),
                "escape_dir": escape_dir.tolist(),
                "obstacle_normal": obstacle_normal.tolist(),
                "escape_vs_task_cos": float(
                    np.dot(escape_dir, task_dir)
                    / max(np.linalg.norm(escape_dir) * np.linalg.norm(task_dir), 1e-9)
                ),
                "normal_vs_task_cos": float(
                    np.dot(obstacle_normal, task_dir)
                    / max(np.linalg.norm(obstacle_normal) * np.linalg.norm(task_dir), 1e-9)
                ),
            },
        )
        # endregion

        cbf_motion_summary = []
        if len(grad_rows) > 0:
            for cbf_idx, grad in enumerate(grad_rows):
                current_margin = float(cfg.mpc_dt * np.dot(grad, u_cmd) + cfg.gamma_dcbf * h_vals[cbf_idx])
                meta = dict(self._last_cbf_meta[cbf_idx]) if cbf_idx < len(self._last_cbf_meta) else {}
                meta.update({
                    "cbf_index": int(cbf_idx),
                    "grad_dot_u": float(np.dot(grad, u_cmd)),
                    "current_margin": current_margin,
                })
                cbf_motion_summary.append(meta)
            cbf_motion_summary.sort(key=lambda item: item["current_margin"])

        # region agent log
        _append_debug_log(
            run_id,
            "H5",
            "welding_320_control.py:664",
            "Command-vs-CBF motion summary",
            {
                "step_count": int(self._step_count),
                "ee_cmd_speed": float(np.linalg.norm(ee_cmd_vel)),
                "task_progress_rate": float(np.dot(ee_cmd_vel, task_dir)),
                "task_alignment_cos": float(
                    np.dot(ee_cmd_vel, task_dir) / max(np.linalg.norm(ee_cmd_vel), 1e-9)
                ),
                "negative_h_count": int(sum(1 for h in h_vals if h < 0.0)),
                "active_cbf_count": int(sum(1 for item in cbf_motion_summary if item["current_margin"] <= 1e-6)),
                "tightest_current_cbfs": cbf_motion_summary[:3],
            },
        )
        # endregion

        nominal_cbf_summary = []
        if len(grad_rows) > 0:
            for cbf_idx, grad in enumerate(grad_rows):
                nominal_margin = float(cfg.mpc_dt * np.dot(grad, u_nom) + cfg.gamma_dcbf * h_vals[cbf_idx])
                nominal_cbf_summary.append({
                    "cbf_index": int(cbf_idx),
                    "grad_dot_u_nom": float(np.dot(grad, u_nom)),
                    "nominal_margin": nominal_margin,
                })
            nominal_cbf_summary.sort(key=lambda item: item["nominal_margin"])

        # region agent log
        _append_debug_log(
            run_id,
            "H6",
            "welding_320_control.py:690",
            "Nominal-vs-constrained command comparison",
            {
                "step_count": int(self._step_count),
                "u_nom_norm": float(np.linalg.norm(u_nom)),
                "u_cmd_norm": float(np.linalg.norm(u_cmd)),
                "u_diff_norm": float(np.linalg.norm(u_cmd - u_nom)),
                "ee_nom_speed": float(np.linalg.norm(ee_nom_vel)),
                "ee_cmd_speed": float(np.linalg.norm(ee_cmd_vel)),
                "nominal_task_progress_rate": float(np.dot(ee_nom_vel, task_dir)),
                "command_task_progress_rate": float(np.dot(ee_cmd_vel, task_dir)),
                "nominal_task_alignment_cos": float(
                    np.dot(ee_nom_vel, task_dir) / max(np.linalg.norm(ee_nom_vel), 1e-9)
                ),
                "command_task_alignment_cos": float(
                    np.dot(ee_cmd_vel, task_dir) / max(np.linalg.norm(ee_cmd_vel), 1e-9)
                ),
                "tightest_nominal_cbfs": nominal_cbf_summary[:3],
            },
        )
        # endregion

        obj_u = None
        try:
            obj_x = np.linalg.solve(h_mat + 1e-9 * np.eye(h_mat.shape[0]), -f_vec)
            obj_u = np.clip(obj_x[:n], self._lb, self._ub)
        except np.linalg.LinAlgError:
            obj_u = None

        if obj_u is not None:
            ee_obj_vel = j_pos @ obj_u
            obj_cbf_summary = []
            if len(grad_rows) > 0:
                for cbf_idx, grad in enumerate(grad_rows):
                    obj_margin = float(cfg.mpc_dt * np.dot(grad, obj_u) + cfg.gamma_dcbf * h_vals[cbf_idx])
                    obj_cbf_summary.append({
                        "cbf_index": int(cbf_idx),
                        "grad_dot_u_obj": float(np.dot(grad, obj_u)),
                        "obj_margin": obj_margin,
                    })
                obj_cbf_summary.sort(key=lambda item: item["obj_margin"])

            mdt = cfg.mpc_dt
            jtj_pos = mdt ** 2 * (j_pos.T @ j_pos)
            idx = np.arange(N)
            weight_mat = (N - np.maximum(idx[:, None], idx[None, :])).astype(float)
            h_pos_only = 2.0 * cfg.mpc_tracking_weight * np.kron(weight_mat, jtj_pos)
            h_pos_only += 2.0 * cfg.mpc_control_weight * np.eye(h_pos_only.shape[0])
            if N > 1:
                diff = np.zeros((N - 1, N))
                for k in range(N - 1):
                    diff[k, k] = -1.0
                    diff[k, k + 1] = 1.0
                h_pos_only += 2.0 * cfg.mpc_smooth_weight * np.kron(diff.T @ diff, np.eye(n))
            c_vecs = np.array([ee_pos - ref_positions[k] for k in range(N)])
            c_suffix = np.cumsum(c_vecs[::-1], axis=0)[::-1]
            f_pos_only = 2.0 * cfg.mpc_tracking_weight * mdt * (c_suffix @ j_pos).ravel()
            obj_pos_x = np.linalg.solve(h_pos_only + 1e-9 * np.eye(h_pos_only.shape[0]), -f_pos_only)
            obj_pos_u = np.clip(obj_pos_x[:n], self._lb, self._ub)
            ee_obj_pos_vel = j_pos @ obj_pos_u

            # region agent log
            _append_debug_log(
                run_id,
                "H7",
                "welding_320_control.py:720",
                "Objective-only command comparison",
                {
                    "step_count": int(self._step_count),
                    "u_obj_norm": float(np.linalg.norm(obj_u)),
                    "u_cmd_norm": float(np.linalg.norm(u_cmd)),
                    "obj_vs_cmd_diff_norm": float(np.linalg.norm(obj_u - u_cmd)),
                    "obj_task_progress_rate": float(np.dot(ee_obj_vel, task_dir)),
                    "obj_task_alignment_cos": float(
                        np.dot(ee_obj_vel, task_dir) / max(np.linalg.norm(ee_obj_vel), 1e-9)
                    ),
                    "tightest_obj_cbfs": obj_cbf_summary[:3],
                },
            )
            # endregion

            # region agent log
            _append_debug_log(
                run_id,
                "H11",
                "welding_320_control.py:748",
                "Position-vs-full objective comparison",
                {
                    "step_count": int(self._step_count),
                    "u_obj_pos_norm": float(np.linalg.norm(obj_pos_u)),
                    "u_obj_full_norm": float(np.linalg.norm(obj_u)),
                    "obj_pos_task_progress_rate": float(np.dot(ee_obj_pos_vel, task_dir)),
                    "obj_full_task_progress_rate": float(np.dot(ee_obj_vel, task_dir)),
                    "obj_pos_alignment_cos": float(
                        np.dot(ee_obj_pos_vel, task_dir) / max(np.linalg.norm(ee_obj_pos_vel), 1e-9)
                    ),
                    "obj_full_alignment_cos": float(
                        np.dot(ee_obj_vel, task_dir) / max(np.linalg.norm(ee_obj_vel), 1e-9)
                    ),
                },
            )
            # endregion

        info = {
            "min_h": min_h,
            "max_slack": 0.0,
            "status": status,
            "tracking_error": float(np.linalg.norm(pos_err)),
            "orientation_error": float(np.linalg.norm(rot_err)),
            "progress_step": float(ds_ref),
            "cbf_contacts": [dict(meta) for meta in self._last_cbf_meta],
            **dynamic_info,
            **second_order_info,
            "mpc_second_order_risk_term_count": int(len(second_order_risk_terms)),
            "mpc_second_order_risk_cost": float(
                sum(
                    float(term.get("risk_weight", 0.0)) * float(term.get("shortfall", 0.0))
                    + float(term.get("shrink_weight", 0.0)) * float(term.get("negative_curvature", 0.0))
                    for term in second_order_risk_terms
                )
            ),
        }
        self._cached_u = u_cmd
        self._cached_info = info
        return u_cmd, info


def create_controller(robot, config: ExperimentConfig, trajectory):
    return MPCDCBFController(robot, config, trajectory)
