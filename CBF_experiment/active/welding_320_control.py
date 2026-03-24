from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from CBF_experiment.active.welding_320_common import ExperimentConfig, quaternion_error_rotvec
from CBF_experiment.active.welding_320_ik import IKCandidate
from CBF_experiment.active.welding_320_robot import JakaRobot
from CBF_experiment.active.welding_320_trajectory import JointWaypointTrajectory, PiecewiseLineSlerpTrajectory

try:
    import pybullet_planning as pp
except ImportError:
    pp = None


@dataclass
class PlannerDiagnostics:
    ik_calls: int = 0
    ik_failures: int = 0
    collision_checks: int = 0
    collision_failures: int = 0
    sampled_states: int = 0
    ik_blocked_nodes: int = 0
    fk_errors: list[float] = field(default_factory=list)

    def record_ik_attempt(self, position_error: float, success: bool):
        self.ik_calls += 1
        self.fk_errors.append(float(position_error))
        if not success:
            self.ik_failures += 1

    def record_collision_check(self, is_collision: bool):
        self.collision_checks += 1
        if is_collision:
            self.collision_failures += 1

    @property
    def max_fk_error(self) -> float:
        return float(max(self.fk_errors)) if self.fk_errors else 0.0

    @property
    def mean_fk_error(self) -> float:
        return float(np.mean(self.fk_errors)) if self.fk_errors else 0.0

    def summary(self) -> dict:
        return {
            "ik_calls": self.ik_calls,
            "ik_failures": self.ik_failures,
            "collision_checks": self.collision_checks,
            "collision_failures": self.collision_failures,
            "sampled_states": self.sampled_states,
            "ik_blocked_nodes": self.ik_blocked_nodes,
            "max_fk_error": self.max_fk_error,
            "mean_fk_error": self.mean_fk_error,
        }


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
        self._rng = np.random.default_rng(config.planner_seed)
        self.last_plan_statuses: list[str] = []
        self.last_plan_diagnostics: list[dict] = []

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

    def _sample_cartesian_point(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        return np.array(self._rng.uniform(lower, upper), dtype=float)

    def _solve_pose_ik(self, target_pos, target_quat, seed_q, diagnostics: PlannerDiagnostics | None = None):
        q_sol = self.robot.calculate_ik(target_pos, target_quat, rest_poses=seed_q)
        self.robot.set_joint_state(q_sol)
        ee_pos, _ = self.robot.get_ee_pose()
        position_error = float(np.linalg.norm(np.array(target_pos, dtype=float) - ee_pos))
        success = position_error <= self.config.rrt_ik_tolerance
        if diagnostics is not None:
            diagnostics.record_ik_attempt(position_error, success=success)
        if not success:
            return None
        return q_sol

    def _base_collides_with_workpiece(self, q, diagnostics: PlannerDiagnostics | None = None):
        if self.config.ignore_all_collisions:
            if diagnostics is not None:
                diagnostics.record_collision_check(False)
            return False
        self.robot.set_joint_state(q)
        closest = self.robot.get_closest_point_to_obstacle(
            self.robot.welding_gun_base_link_index,
            self.workpiece_body_id,
            max_dist=max(1.0, self.config.rrt_cartesian_margin * 3.0),
        )
        if closest is None:
            if diagnostics is not None:
                diagnostics.record_collision_check(False)
            return False
        is_collision = float(closest[1]) < self.config.safety_margin
        if diagnostics is not None:
            diagnostics.record_collision_check(is_collision)
        return is_collision

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
        diagnostics = PlannerDiagnostics()
        cache: dict[tuple[float, float, float], np.ndarray] = {
            self._rounded_key(start_pos): q_seed_start.copy(),
            self._rounded_key(goal_pos): q_seed_goal.copy(),
        }
        cache_entries = [
            (start_pos.copy(), q_seed_start.copy()),
            (goal_pos.copy(), q_seed_goal.copy()),
        ]

        def collision_fn(conf, diagnosis=False, **_kwargs):
            diagnostics.sampled_states += 1
            pos = np.array(conf, dtype=float)
            key = self._rounded_key(pos)
            q_candidate = cache.get(key)
            if q_candidate is None:
                seed_q = self._choose_seed_q(pos, alpha_from_pos, q_seed_start, q_seed_goal, cache_entries)
                q_candidate = self._solve_pose_ik(pos, quat_from_pos(pos), seed_q, diagnostics=diagnostics)
                if q_candidate is None:
                    diagnostics.ik_blocked_nodes += 1
                    return True
                cache[key] = q_candidate.copy()
                cache_entries.append((pos.copy(), q_candidate.copy()))
            return self._base_collides_with_workpiece(q_candidate, diagnostics=diagnostics)

        margin = self.config.rrt_cartesian_margin
        lower = np.minimum(start_pos, goal_pos) - margin
        upper = np.maximum(start_pos, goal_pos) + margin
        resolution = max(self.config.rrt_cartesian_resolution, 1e-3)

        def distance_fn(q1, q2):
            return float(np.linalg.norm(np.array(q2, dtype=float) - np.array(q1, dtype=float)))

        def sample_fn():
            return self._sample_cartesian_point(lower, upper).tolist()

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
            planning_diagnostics=diagnostics.summary(),
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
            self.last_plan_diagnostics = [
                dict(seg_1.planning_diagnostics),
                dict(seg_2.planning_diagnostics),
                dict(seg_3.planning_diagnostics),
            ]
        finally:
            self.robot.set_joint_state(q_backup, dq_backup)
        return PiecewiseLineSlerpTrajectory([seg_1, seg_2, seg_3])


class JointSpaceNominalPlanner:
    """在关节空间规划靠近/撤离段，避免笛卡尔节点内嵌 IK 假失败。"""

    def __init__(self, robot: JakaRobot, config: ExperimentConfig, workpiece_body_id: int):
        if pp is None:
            raise RuntimeError("未安装 pybullet_planning，请先安装后再启用关节空间规划器。")
        self.robot = robot
        self.config = config
        self.workpiece_body_id = workpiece_body_id
        self._rng = np.random.default_rng(config.planner_seed + 1)

    def _sample_joint_state(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        return np.array(self._rng.uniform(lower, upper), dtype=float)

    def _select_goal_candidate(self, current_q, goal_candidates, is_free_fn=None):
        current_q = np.array(current_q, dtype=float)
        ordered = sorted(
            [np.array(q_goal, dtype=float) for q_goal in goal_candidates],
            key=lambda q_goal: float(np.linalg.norm(q_goal - current_q)),
        )
        if is_free_fn is None:
            return ordered[0]
        for q_goal in ordered:
            if is_free_fn(q_goal):
                return q_goal
        return ordered[0]

    def _joint_path_to_trajectory(self, q_path, duration: float, planner_status: str, diagnostics: dict | None = None):
        q_backup, dq_backup = self.robot.get_joint_state()
        waypoints_pos = []
        waypoints_quat = []
        try:
            for q_waypoint in q_path:
                self.robot.set_joint_state(q_waypoint)
                pos, quat = self.robot.get_ee_pose()
                waypoints_pos.append(pos.copy())
                waypoints_quat.append(quat.copy())
        finally:
            self.robot.set_joint_state(q_backup, dq_backup)
        return JointWaypointTrajectory(
            waypoints_pos,
            waypoints_quat,
            duration,
            self.config.dt,
            planner_status=planner_status,
            planning_diagnostics={} if diagnostics is None else diagnostics,
            joint_waypoints=q_path,
        )

    def plan_to_goal_set(
        self,
        start_q: np.ndarray,
        goal_candidates: list[IKCandidate] | list[np.ndarray],
        duration: float,
        label: str,
    ) -> JointWaypointTrajectory:
        if not goal_candidates:
            raise RuntimeError(f"{label} 段没有可用目标候选。")

        lower, upper = self.robot.get_active_joint_limits()
        diagnostics = PlannerDiagnostics()

        def is_free(q):
            diagnostics.sampled_states += 1
            is_collision = not self.robot.is_state_collision_free(
                q,
                obstacle_body_ids=[self.workpiece_body_id],
                safety_margin=self.config.safety_margin,
            )
            diagnostics.record_collision_check(is_collision)
            return not is_collision

        processed_goals = [cand.q if hasattr(cand, "q") else np.array(cand, dtype=float) for cand in goal_candidates]
        goal_q = self._select_goal_candidate(start_q, processed_goals, is_free_fn=is_free)

        resolution = max(self.config.rrt_cartesian_resolution, 1e-3)

        def distance_fn(q1, q2):
            return float(np.linalg.norm(np.array(q2, dtype=float) - np.array(q1, dtype=float)))

        def sample_fn():
            return self._sample_joint_state(lower, upper).tolist()

        def extend_fn(q1, q2):
            q1 = np.array(q1, dtype=float)
            q2 = np.array(q2, dtype=float)
            dist = np.linalg.norm(q2 - q1)
            n_steps = max(int(np.ceil(dist / resolution)), 1)
            return [
                (q1 + (idx / n_steps) * (q2 - q1)).tolist()
                for idx in range(1, n_steps + 1)
            ]

        path_q = None
        try:
            path_q = pp.birrt(
                np.array(start_q, dtype=float).tolist(),
                goal_q.tolist(),
                distance_fn,
                sample_fn,
                extend_fn,
                lambda q, **_kwargs: not is_free(np.array(q, dtype=float)),
                max_iterations=self.config.rrt_max_iterations,
                max_time=self.config.rrt_max_time,
                restarts=self.config.rrt_restarts,
                smooth=self.config.rrt_smooth,
            )
        except TypeError:
            path_q = pp.birrt(
                np.array(start_q, dtype=float).tolist(),
                goal_q.tolist(),
                distance_fn,
                sample_fn,
                extend_fn,
                lambda q, **_kwargs: not is_free(np.array(q, dtype=float)),
                max_iterations=self.config.rrt_max_iterations,
                max_time=self.config.rrt_max_time,
            )

        planner_status = f"{label}_joint_rrt"
        if path_q is None or len(path_q) == 0:
            planner_status = f"{label}_joint_fallback"
            path_q = [np.array(start_q, dtype=float), np.array(goal_q, dtype=float)]

        diagnostics_summary = diagnostics.summary()
        diagnostics_summary["goal_count"] = len(processed_goals)
        diagnostics_summary["selected_goal_norm"] = float(np.linalg.norm(np.array(goal_q, dtype=float) - np.array(start_q, dtype=float)))
        q_path = [np.array(q_waypoint, dtype=float) for q_waypoint in path_q]
        return self._joint_path_to_trajectory(q_path, duration, planner_status, diagnostics=diagnostics_summary)


class MPCDCBFController:
    """按路径进度采样参考并求解 MPC-DCBF 控制量。"""

    def __init__(self, robot, config: ExperimentConfig, trajectory):
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
        self._lb = np.array([bound[0] for bound in single_bounds])
        self._ub = np.array([bound[1] for bound in single_bounds])

    def _build_cbf_data(self, q, dq, obstacles):
        grad_rows, h_vals = [], []
        for obs in obstacles:
            use_mesh = self.config.use_mesh_cbf and obs.body_id >= 0
            obs_links = getattr(obs, "cbf_link_indices", None)
            check_links = obs_links if obs_links is not None else self.robot.cbf_link_indices
            for link_index in check_links:
                if use_mesh:
                    cp = self.robot.get_closest_point_to_obstacle(link_index, obs.body_id)
                    if cp is None:
                        continue
                    support_point, signed_dist, normal = cp
                    h_vals.append(signed_dist - self.config.safety_margin)
                    grad_rows.append(self.robot.get_link_cbf_row_at_point(link_index, support_point, normal, q, dq))
                else:
                    link_pos = self.robot.get_link_origin(link_index)
                    signed_dist, normal = obs.compute_distance(link_pos)
                    h_vals.append(signed_dist - self.config.safety_margin)
                    grad_rows.append(self.robot.get_link_cbf_row(link_index, normal, q, dq))
        return grad_rows, h_vals

    def _build_qp(self, ee_pos, j_pos, j_rot, ref_positions, ref_rotvecs, grad_rows, h_vals):
        n, N = self.n, self.N
        mdt = self.config.mpc_dt
        cfg = self.config
        dim = n * N

        jtj_pos = mdt ** 2 * (j_pos.T @ j_pos)
        jtj_rot = mdt ** 2 * (j_rot.T @ j_rot)
        idx = np.arange(N)
        weight_mat = (N - np.maximum(idx[:, None], idx[None, :])).astype(float)
        h_mat = 2.0 * cfg.mpc_tracking_weight * np.kron(weight_mat, jtj_pos)
        h_mat += 2.0 * cfg.mpc_orientation_tracking_weight * np.kron(weight_mat, jtj_rot)

        c_vecs = np.array([ee_pos - ref_positions[k] for k in range(N)])
        c_suffix = np.cumsum(c_vecs[::-1], axis=0)[::-1]
        f_vec = 2.0 * cfg.mpc_tracking_weight * mdt * (c_suffix @ j_pos).ravel()
        rot_vecs = -np.array(ref_rotvecs, dtype=float)
        rot_suffix = np.cumsum(rot_vecs[::-1], axis=0)[::-1]
        f_vec += 2.0 * cfg.mpc_orientation_tracking_weight * mdt * (rot_suffix @ j_rot).ravel()

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
        if self._cached_u is not None and self._step_count % self.config.mpc_replan_steps != 0:
            return self._cached_u, self._cached_info

        n, N = self.n, self.N
        cfg = self.config
        j_full = self.robot.get_ee_jacobian(q, dq)
        j_pos = j_full[:3]
        j_rot = j_full[3:]

        ref_positions = []
        ref_rotvecs = []
        ds_ref = max(np.linalg.norm(ref_lin_vel) * cfg.mpc_dt, cfg.mpc_progress_step_min)
        for k in range(1, N + 1):
            pk, qk, _, _ = self.trajectory.sample_by_progress(
                min(current_progress + k * ds_ref, self.trajectory.progress_end)
            )
            ref_positions.append(pk)
            ref_rotvecs.append(quaternion_error_rotvec(ee_quat, qk))

        grad_rows, h_vals = self._build_cbf_data(q, dq, obstacles)
        min_h = float(np.min(h_vals)) if h_vals else 1.0
        h_mat, f_vec, a_cbf, b_cbf = self._build_qp(
            ee_pos, j_pos, j_rot, ref_positions, ref_rotvecs, grad_rows, h_vals
        )

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
            rot_err = quaternion_error_rotvec(ee_quat, ref_quat)
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
        rot_err = quaternion_error_rotvec(ee_quat, ref_quat)
        info = {
            "min_h": min_h,
            "max_slack": 0.0,
            "status": status,
            "tracking_error": float(np.linalg.norm(pos_err)),
            "orientation_error": float(np.linalg.norm(rot_err)),
            "progress_step": float(ds_ref),
        }
        self._cached_u = u_cmd
        self._cached_info = info
        return u_cmd, info


def create_controller(robot, config: ExperimentConfig, trajectory):
    return MPCDCBFController(robot, config, trajectory)
