"""Drake 九轴机械臂 IRIS + CBF-MPC 两层控制器

两层架构：
  1. 名义控制器 — IRIS C-space 安全凸区域 -> GCS 轨迹优化 -> 平滑 q_ref(t)
  2. 安全滤波器 — CBF-MPC：Drake 碰撞距离查询 -> DCBF 约束 -> QP

模块化类：
  TaskConfig           — 任务级参数（目标、IRIS、MPC、CBF）
  TargetPoseComputer   — 从场景 l2 frame 计算世界系目标位姿 + IK 求 q_goal
  IrisRegionBuilder    — 多种子 IrisNp 计算 C-space 安全凸区域
  GcsPathPlanner       — IRIS 区域上 GCS 轨迹优化
  CBFMPCSafetyFilter   — Drake 碰撞查询 + DCBF 约束 + QP
  TwoLayerController   — 组合名义 + 安全
  Experiment           — 主仿真循环 + MeshCat 可视化
"""

from __future__ import annotations

import importlib
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── 路径与基础模块复用 ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_viz = importlib.import_module("CBF_experiment.active.3_25_drake_visualization")
DrakeConfig = _viz.DrakeConfig
DrakeEnvironment = _viz.DrakeEnvironment
MeshConverter = _viz.MeshConverter
RobotLoader = _viz.RobotLoader
SceneLoader = _viz.SceneLoader
URDFPackageResolver = _viz.URDFPackageResolver
SCENE_PKG_DIR = _viz.SCENE_PKG_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TaskConfig
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class TaskConfig:
    """IRIS + CBF-MPC 任务参数。"""

    # ── 目标 ──
    target_z_in_l2_local: list[float] = field(
        default_factory=lambda: [0.0, 1.0, -1.0]
    )

    # ── IRIS ──
    iris_num_seeds: int = 7
    iris_iteration_limit: int = 50
    iris_configuration_space_margin: float = 0.01
    iris_num_collision_infeasible_samples: int = 5

    # ── GCS ──
    gcs_bezier_order: int = 3
    gcs_path_length_weight: float = 1.0
    gcs_time_weight: float = 1.0
    gcs_max_rounded_paths: int = 10

    # ── MPC ──
    N_mpc: int = 5
    mpc_dt: float = 0.05
    mpc_tracking_weight: float = 10.0
    mpc_control_weight: float = 0.01
    mpc_smooth_weight: float = 0.1

    # ── CBF ──
    gamma_dcbf: float = 8
    safety_margin: float = 0.05
    cbf_max_distance: float = 0.01

    # ── 任务空间模式 ──
    use_task_space: bool = True
    task_space_Kp: float = 3.0
    task_space_vel: float = 0.5
    pre_approach_dist: float = 2.0
    approach_switch_dist: float = 0.2
    approach_cart_vel: float = 0.15
    pointcloud_samples: int = 200
    pointcloud_inflate: float = 0.05

    # ── 仿真 ──
    sim_dt: float = 0.01
    vel_limit: float = 1.0
    max_steps: int = 10000
    sim_realtime_factor: float = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TargetPoseComputer
# ═══════════════════════════════════════════════════════════════════════════════
class TargetPoseComputer:
    """计算世界系目标位姿，并用 IK 求解 q_goal。"""

    def __init__(
        self,
        env: DrakeEnvironment,
        robot: RobotLoader,
        scene: SceneLoader,
        task: TaskConfig,
    ) -> None:
        self.env = env
        self.robot = robot
        self.scene = scene
        self.task = task
        self.target_pose = None
        self.q_goal: Optional[np.ndarray] = None

    def compute_target_pose(self):
        from pydrake.all import RigidTransform, RotationMatrix

        plant = self.env.plant
        ctx = self.env.plant_context

        l2_body = plant.GetBodyByName("l2", self.scene.model_instance)
        X_WL2 = plant.CalcRelativeTransform(
            ctx, plant.world_frame(), l2_body.body_frame()
        )
        target_pos_world = X_WL2.translation()
        R_WL2 = X_WL2.rotation()

        z_local = np.array(self.task.target_z_in_l2_local, dtype=float)
        z_local /= np.linalg.norm(z_local)
        z_world = R_WL2.matrix() @ z_local

        R_target = self._rotation_from_z_axis(z_world)
        self.target_pose = RigidTransform(RotationMatrix(R_target), target_pos_world)

        print(f"[TargetPose] 目标位置（世界系）：{target_pos_world}")
        print(f"[TargetPose] 目标 z 轴（世界系）：{z_world}")
        return self.target_pose

    def solve_ik(self) -> np.ndarray:
        """多起点 IK 求解，渐进式放宽碰撞约束。"""
        from pydrake.all import InverseKinematics, RotationMatrix, Solve

        if self.target_pose is None:
            self.compute_target_pose()

        plant = self.env.plant
        nq_robot = plant.num_positions(self.robot.model_instance)
        q_lb = plant.GetPositionLowerLimits()[:nq_robot]
        q_ub = plant.GetPositionUpperLimits()[:nq_robot]
        q_lb = np.where(np.isinf(q_lb), -6.28, q_lb)
        q_ub = np.where(np.isinf(q_ub), 6.28, q_ub)

        stages = [
            (0.005, 5.0,  0.02),
            (0.005, 5.0,  0.01),
            (0.005, 5.0,  0.005),
            (0.05,  30.0, 0.02),
            (0.05,  30.0, 0.01),
            (0.05,  30.0, 0.005),
            (0.05,  30.0, 0.0),
        ]

        rng = np.random.RandomState(0)
        n_random = 8

        for stage_i, (pos_tol, theta_deg, col_dist) in enumerate(stages):
            guesses = [plant.GetPositions(self.env.plant_context).copy()]
            for _ in range(n_random):
                q_rand = rng.uniform(q_lb, q_ub)
                q_full = plant.GetPositions(self.env.plant_context).copy()
                q_full[:nq_robot] = q_rand
                guesses.append(q_full)

            for gi, q0 in enumerate(guesses):
                ik = InverseKinematics(plant, self.env.plant_context)
                ee_frame = plant.GetBodyByName(
                    "weld_point", self.robot.model_instance
                ).body_frame()

                ik.AddPositionConstraint(
                    frameB=ee_frame, p_BQ=np.zeros(3),
                    frameA=plant.world_frame(),
                    p_AQ_lower=self.target_pose.translation() - pos_tol,
                    p_AQ_upper=self.target_pose.translation() + pos_tol,
                )
                ik.AddOrientationConstraint(
                    frameAbar=plant.world_frame(),
                    R_AbarA=self.target_pose.rotation(),
                    frameBbar=ee_frame,
                    R_BbarB=RotationMatrix(),
                    theta_bound=np.deg2rad(theta_deg),
                )
                if col_dist > 0:
                    ik.AddMinimumDistanceLowerBoundConstraint(
                        bound=col_dist, influence_distance_offset=0.05,
                    )

                prog = ik.prog()
                prog.SetInitialGuess(ik.q(), q0)
                result = Solve(prog)

                if result.is_success():
                    q_sol = result.GetSolution(ik.q())
                    self.q_goal = q_sol[:nq_robot]
                    tag = f"无碰撞 d={col_dist}" if col_dist > 0 else "可能有碰撞"
                    print(f"[TargetPose] IK 成功（{tag}，阶段{stage_i} 猜测{gi}），"
                          f"q_goal = {np.array2string(self.q_goal, precision=3)}")
                    return self.q_goal

            if stage_i < len(stages) - 1:
                next_tol, next_theta, next_col = stages[stage_i + 1]
                print(f"[TargetPose] 阶段{stage_i} (tol={pos_tol}, θ={theta_deg}°, d={col_dist}) "
                      f"全部失败，进入阶段{stage_i+1}...")

        raise RuntimeError("IK 求解全部失败（所有阶段和多起点均未收敛）")

    @staticmethod
    def _rotation_from_z_axis(z: np.ndarray) -> np.ndarray:
        z = z / np.linalg.norm(z)
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(z, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        x = np.cross(ref, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. IrisRegionBuilder
# ═══════════════════════════════════════════════════════════════════════════════
class IrisRegionBuilder:
    """使用 IrisNp 在多个种子位形上计算 C-space 无碰撞凸区域。"""

    def __init__(self, env: DrakeEnvironment, robot: RobotLoader, task: TaskConfig):
        self.env = env
        self.robot = robot
        self.task = task
        self.regions: list = []

    def build(self, q_start: np.ndarray, q_goal: np.ndarray) -> list:
        from pydrake.geometry.optimization import IrisNp, IrisOptions

        plant = self.env.plant
        plant_context = self.env.plant_context

        options = IrisOptions()
        options.iteration_limit = self.task.iris_iteration_limit
        options.configuration_space_margin = self.task.iris_configuration_space_margin
        options.num_collision_infeasible_samples = self.task.iris_num_collision_infeasible_samples

        seeds = self._generate_seeds(q_start, q_goal)

        self.regions = []
        n_skipped = 0
        for i, seed in enumerate(seeds):
            try:
                plant.SetPositions(plant_context, seed)

                sg_context = self.env.scene_graph.GetMyContextFromRoot(self.env.context)
                query_object = self.env.scene_graph.get_query_output_port().Eval(sg_context)
                if query_object.HasCollisions():
                    n_skipped += 1
                    if n_skipped <= 3:
                        print(f"[IRIS] 种子 {i} 碰撞，跳过", flush=True)
                    continue

                print(f"[IRIS] 种子 {i} 无碰撞，计算 IrisNp...", flush=True)
                region = IrisNp(plant, plant_context, options)
                vol = region.MaximumVolumeInscribedEllipsoid().CalcVolume()
                if vol < 1e-10:
                    print(f"[IRIS] 种子 {i}: 区域体积过小 ({vol:.2e})，丢弃", flush=True)
                    continue
                self.regions.append(region)
                print(f"[IRIS] 区域 {len(self.regions)-1}: 成功 (dim={region.ambient_dimension()}, vol={vol:.2e})", flush=True)
            except Exception as e:
                print(f"[IRIS] 种子 {i} 失败: {e}", flush=True)
        if n_skipped > 3:
            print(f"[IRIS] ...共跳过 {n_skipped} 个碰撞种子", flush=True)

        plant.SetPositions(plant_context, q_start)
        print(f"[IRIS] 共生成 {len(self.regions)} 个 C-space 安全凸区域", flush=True)

        n_s = sum(1 for r in self.regions if r.PointInSet(q_start))
        n_g = sum(1 for r in self.regions if r.PointInSet(q_goal))
        print(f"[IRIS] q_start 在 {n_s}/{len(self.regions)} 个区域中"
              f"  q_goal 在 {n_g}/{len(self.regions)} 个区域中", flush=True)

        return self.regions

    def _generate_seeds(self, q_start: np.ndarray, q_goal: np.ndarray) -> list[np.ndarray]:
        plant = self.env.plant
        nq = q_start.shape[0]
        q_lb = plant.GetPositionLowerLimits()[:nq]
        q_ub = plant.GetPositionUpperLimits()[:nq]
        q_lb = np.where(np.isinf(q_lb), -6.28, q_lb)
        q_ub = np.where(np.isinf(q_ub), 6.28, q_ub)

        seeds = [q_start.copy(), q_goal.copy()]
        rng = np.random.RandomState(42)

        n_interp = max(0, self.task.iris_num_seeds - 2)
        for i in range(n_interp):
            alpha = (i + 1) / (n_interp + 1)
            q_mid = (1 - alpha) * q_start + alpha * q_goal
            q_mid += rng.randn(nq) * 0.1
            q_mid = np.clip(q_mid, q_lb, q_ub)
            seeds.append(q_mid)

        for _ in range(3):
            q_near = q_start + rng.randn(nq) * 0.3
            seeds.append(np.clip(q_near, q_lb, q_ub))

        for _ in range(3):
            q_near = q_goal + rng.randn(nq) * 0.5
            seeds.append(np.clip(q_near, q_lb, q_ub))

        for _ in range(4):
            seeds.append(rng.uniform(q_lb, q_ub))

        return seeds


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GcsPathPlanner — 名义控制器
# ═══════════════════════════════════════════════════════════════════════════════
class GcsPathPlanner:
    """在 IRIS 凸区域上用 GCS 求解最短平滑路径。"""

    def __init__(self, env: DrakeEnvironment, task: TaskConfig):
        self.env = env
        self.task = task
        self.trajectory = None

    def plan(self, regions: list, q_start: np.ndarray, q_goal: np.ndarray):
        import sys
        self._q_start = q_start.copy()
        self._q_goal = q_goal.copy()
        self._fallback = False

        if not regions:
            print("[GCS] 没有 IRIS 区域，使用线性插值回退轨迹", flush=True)
            self._setup_fallback()
            return None, None

        from pydrake.geometry.optimization import (
            GraphOfConvexSetsOptions, HPolyhedron, Point,
        )
        from pydrake.planning import GcsTrajectoryOptimization

        nq = q_start.shape[0]
        gcs = GcsTrajectoryOptimization(num_positions=nq)

        eps = 0.05
        source_box = HPolyhedron.MakeBox(q_start - eps, q_start + eps)
        target_box = HPolyhedron.MakeBox(q_goal - eps, q_goal + eps)
        source_sg = gcs.AddRegions([source_box], order=1, name="source")
        target_sg = gcs.AddRegions([target_box], order=1, name="target")

        main_sg = gcs.AddRegions(
            regions, order=self.task.gcs_bezier_order, name="iris",
        )

        gcs.AddEdges(source_sg, main_sg)
        gcs.AddEdges(main_sg, main_sg)
        gcs.AddEdges(main_sg, target_sg)

        gcs.AddPathLengthCost(weight=self.task.gcs_path_length_weight)
        gcs.AddTimeCost(weight=self.task.gcs_time_weight)

        v_max = np.full(nq, self.task.vel_limit)
        gcs.AddVelocityBounds(-v_max, v_max)

        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = self.task.gcs_max_rounded_paths

        print(f"[GCS] 求解中... ({len(regions)} 个 IRIS 区域)", flush=True)
        try:
            traj, result = gcs.SolvePath(source_sg, target_sg, options)
            if result.is_success():
                dur = traj.end_time() - traj.start_time()
                print(f"[GCS] 路径求解成功，时长 = {dur:.2f}s", flush=True)
                self.trajectory = traj
                return traj, result
            else:
                print(f"[GCS] 求解失败 ({result.get_solution_result()})，使用线性插值回退", flush=True)
        except Exception as e:
            print(f"[GCS] 异常: {e}，使用线性插值回退", flush=True)

        self._setup_fallback()
        return None, None

    def _setup_fallback(self) -> None:
        """GCS 求解失败时的线性插值回退轨迹。"""
        from pydrake.trajectories import PiecewisePolynomial

        dist = np.linalg.norm(self._q_goal - self._q_start)
        duration = max(dist / self.task.vel_limit, 1.0)
        self.trajectory = PiecewisePolynomial.FirstOrderHold(
            [0.0, duration],
            np.column_stack([self._q_start, self._q_goal]),
        )
        self._fallback = True
        print(f"[GCS] 回退轨迹: 线性插值，时长 = {duration:.2f}s")

    def sample(self, t: float) -> np.ndarray:
        if self.trajectory is None:
            return self._q_start.copy()
        tc = np.clip(t, self.trajectory.start_time(), self.trajectory.end_time())
        return self.trajectory.value(tc).flatten()

    @property
    def duration(self) -> float:
        if self.trajectory is None:
            return 0.0
        return self.trajectory.end_time() - self.trajectory.start_time()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CBFMPCSafetyFilter
# ═══════════════════════════════════════════════════════════════════════════════
class CBFMPCSafetyFilter:
    """
    多步 MPC + 离散 CBF 约束的安全滤波器。

    每个控制步：
      1. 查询 Drake 碰撞距离 -> h(q)
      2. 计算雅可比 -> grad_h
      3. 构造 QP -> 求解 u_safe
    """

    def __init__(self, env: DrakeEnvironment, robot: RobotLoader,
                 scene: SceneLoader, task: TaskConfig):
        self.env = env
        self.robot = robot
        self.scene = scene
        self.task = task
        self._robot_body_indices = None
        self._scene_body_indices = None

    def _ensure_body_indices(self) -> None:
        if self._robot_body_indices is not None:
            return
        plant = self.env.plant
        self._robot_body_indices = set(
            plant.GetBodyIndices(self.robot.model_instance)
        )
        self._scene_body_indices = set(
            plant.GetBodyIndices(self.scene.model_instance)
        )

    def solve(self, q_current: np.ndarray, q_ref: np.ndarray,
              dq_ref: np.ndarray) -> np.ndarray:
        from pydrake.solvers import MathematicalProgram, Solve as SolveProg

        nq = q_current.shape[0]
        dt = self.task.mpc_dt

        prog = MathematicalProgram()
        u = prog.NewContinuousVariables(nq, "u")

        # 跟踪代价：‖u - dq_ref‖² (使速度跟踪名义速度)
        w_track = self.task.mpc_tracking_weight
        prog.AddQuadraticErrorCost(w_track * np.eye(nq), dq_ref, u)

        # 位置跟踪代价：‖q + u*dt - q_ref‖² => ‖u - (q_ref - q)/dt‖² * dt²
        u_pos_ref = (q_ref - q_current) / dt
        prog.AddQuadraticErrorCost(w_track * 0.5 * np.eye(nq), u_pos_ref, u)

        # 控制正则
        prog.AddQuadraticErrorCost(
            self.task.mpc_control_weight * np.eye(nq), np.zeros(nq), u
        )

        # 速度界限
        prog.AddBoundingBoxConstraint(
            -self.task.vel_limit * np.ones(nq),
            self.task.vel_limit * np.ones(nq),
            u,
        )

        # CBF 约束
        cbf_data = self._compute_cbf_data(q_current)
        gamma = self.task.gamma_dcbf

        for h_val, grad_h in cbf_data:
            if h_val > self.task.cbf_max_distance:
                continue
            # DCBF: grad_h @ u * dt >= -gamma * h_val
            lb = np.array([-gamma * h_val / dt])
            prog.AddLinearConstraint(
                grad_h.reshape(1, nq), lb, np.array([np.inf]), u
            )

        result = SolveProg(prog)
        if result.is_success():
            return result.GetSolution(u)

        print("[CBF-MPC] QP 失败，回退到纯跟踪")
        return np.clip(u_pos_ref, -self.task.vel_limit, self.task.vel_limit)

    def filter_velocity(self, q_current: np.ndarray,
                        u_nominal: np.ndarray,
                        safety_margin: Optional[float] = None,
                        vel_limit: Optional[float] = None,
                        ) -> np.ndarray:
        """最小化 ‖u − u_nominal‖² 并满足 CBF + 速度约束。"""
        from pydrake.solvers import MathematicalProgram, Solve as SolveProg

        nq = q_current.shape[0]
        v_lim = vel_limit if vel_limit is not None else self.task.vel_limit
        prog = MathematicalProgram()
        u = prog.NewContinuousVariables(nq, "u")

        prog.AddQuadraticErrorCost(np.eye(nq), u_nominal, u)
        prog.AddBoundingBoxConstraint(-v_lim * np.ones(nq),
                                      v_lim * np.ones(nq), u)

        dt = self.task.mpc_dt
        gamma = self.task.gamma_dcbf
        margin = safety_margin if safety_margin is not None else self.task.safety_margin
        cbf_data = self._compute_cbf_data(q_current, margin_override=margin)

        for h_val, grad_h in cbf_data:
            if h_val > self.task.cbf_max_distance:
                continue
            lb = np.array([-gamma * h_val / dt])
            prog.AddLinearConstraint(
                grad_h.reshape(1, nq), lb, np.array([np.inf]), u,
            )

        result = SolveProg(prog)
        if result.is_success():
            return result.GetSolution(u)

        return np.clip(u_nominal, -v_lim, v_lim)

    def _compute_cbf_data(self, q: np.ndarray,
                          margin_override: Optional[float] = None,
                          ) -> list[tuple[float, np.ndarray]]:
        """计算所有机器人-场景碰撞对的 (h, grad_h)。"""
        from pydrake.multibody.tree import JacobianWrtVariable

        self._ensure_body_indices()
        plant = self.env.plant
        ctx = self.env.plant_context
        nq = q.shape[0]
        margin = margin_override if margin_override is not None else self.task.safety_margin

        plant.SetPositions(ctx, self.robot.model_instance, q)

        sg_context = self.env.scene_graph.GetMyContextFromRoot(self.env.context)
        query_object = self.env.scene_graph.get_query_output_port().Eval(sg_context)
        pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(
            self.task.cbf_max_distance
        )

        inspector = self.env.scene_graph.model_inspector()
        cbf_data = []

        for pair in pairs:
            frame_id_A = inspector.GetFrameId(pair.id_A)
            frame_id_B = inspector.GetFrameId(pair.id_B)
            body_A = plant.GetBodyFromFrameId(frame_id_A)
            body_B = plant.GetBodyFromFrameId(frame_id_B)

            is_robot_A = body_A.index() in self._robot_body_indices
            is_robot_B = body_B.index() in self._robot_body_indices
            is_scene_A = body_A.index() in self._scene_body_indices
            is_scene_B = body_B.index() in self._scene_body_indices

            if not ((is_robot_A and is_scene_B) or (is_robot_B and is_scene_A)):
                continue

            h = pair.distance - margin

            if is_robot_A:
                robot_body = body_A
                p_B = pair.p_ACa
                normal = pair.nhat_BA_W
            else:
                robot_body = body_B
                p_B = pair.p_BCb
                normal = -pair.nhat_BA_W

            # 雅可比 (3 x nq_full)
            J_W = plant.CalcJacobianTranslationalVelocity(
                ctx,
                JacobianWrtVariable.kQDot,
                robot_body.body_frame(),
                p_B,
                plant.world_frame(),
                plant.world_frame(),
            )

            # 只取机器人的 q 列
            robot_q_start = plant.GetJointByName(
                "pris01", self.robot.model_instance
            ).position_start()
            J_robot = J_W[:, robot_q_start:robot_q_start + nq]

            grad_h = normal @ J_robot
            cbf_data.append((h, grad_h))

        return cbf_data


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TwoLayerController
# ═══════════════════════════════════════════════════════════════════════════════
class TwoLayerController:
    """两层控制器：GCS 名义轨迹 + CBF-MPC 安全滤波。"""

    def __init__(self, gcs_planner: GcsPathPlanner, cbf_filter: CBFMPCSafetyFilter,
                 env: DrakeEnvironment):
        self.gcs = gcs_planner
        self.cbf = cbf_filter
        self.env = env

    def compute(self, q_current: np.ndarray, t: float) -> np.ndarray:
        dt_num = 0.01
        q_ref = self.gcs.sample(t)
        q_ref_next = self.gcs.sample(t + dt_num)
        dq_ref = (q_ref_next - q_ref) / dt_num

        return self.cbf.solve(q_current, q_ref, dq_ref)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TaskSpaceIrisBuilder — 三维任务空间 IRIS
# ═══════════════════════════════════════════════════════════════════════════════
class TaskSpaceIrisBuilder:
    """在 3D 笛卡尔空间中用 Iris 构建无碰撞凸区域。

    将场景 mesh 凸包作为障碍物，在末端可达工作空间中生长自由区域。
    """

    def __init__(self, env: DrakeEnvironment, scene: SceneLoader) -> None:
        self.env = env
        self.scene = scene

    def _extract_obstacles(self, sample_count: int = 300,
                           inflate_radius: float = 0.03,
                           ) -> tuple[list, np.ndarray, float]:
        """将 mesh 表面均匀采样为点云，每个点膨胀为微型立方体 VPolytope。

        使用 VPolytope（LP 基础）而非 Hyperellipsoid（SOCP），
        避免 Drake Iris 求解器数值不稳定。每个立方体边长 = 2*inflate_radius，
        仅覆盖采样点周围极小区域，不引入凸包多余体积。

        Returns:
            (obstacles, centers, inflate_radius)
            obstacles: list[VPolytope]  用于 IRIS
            centers:   ndarray (N, 3)   所有采样点坐标
            inflate_radius: float       立方体半边长
        """
        import trimesh
        from pydrake.geometry.optimization import VPolytope

        plant = self.env.plant
        ctx = self.env.plant_context
        mesh_dir = SCENE_PKG_DIR / "meshes"

        r = inflate_radius
        box_offsets = np.array([
            [ r,  r,  r], [ r,  r, -r], [ r, -r,  r], [ r, -r, -r],
            [-r,  r,  r], [-r,  r, -r], [-r, -r,  r], [-r, -r, -r],
        ])

        all_centers: list[np.ndarray] = []
        obstacles: list = []

        for body_idx in plant.GetBodyIndices(self.scene.model_instance):
            body = plant.get_body(body_idx)
            X_WB = plant.CalcRelativeTransform(
                ctx, plant.world_frame(), body.body_frame()
            )
            obj_path = mesh_dir / f"{body.name()}.obj"
            if not obj_path.exists():
                continue

            mesh = trimesh.load(str(obj_path), force="mesh")
            R = X_WB.rotation().matrix()
            t_vec = X_WB.translation()
            n_verts = mesh.vertices.shape[0]

            if n_verts < 10:
                pts_local = mesh.vertices
            else:
                count = min(sample_count, n_verts)
                pts_local, _ = trimesh.sample.sample_surface(mesh, count)

            pts_world = (R @ pts_local.T).T + t_vec

            for pt in pts_world:
                cube_verts = pt + box_offsets
                obstacles.append(VPolytope(cube_verts.T))
            all_centers.append(pts_world)

            print(f"[3D-IRIS] '{body.name()}': {n_verts} 顶点 → "
                  f"{pts_world.shape[0]} 个微型 VPolytope (边长={2*r}m)",
                  flush=True)

        centers = np.vstack(all_centers) if all_centers else np.zeros((0, 3))
        print(f"[3D-IRIS] 共 {len(obstacles)} 个微型立方体障碍物 "
              f"(点云采样，无凸包多余体积)", flush=True)
        return obstacles, centers, inflate_radius

    def build(self, start_pos: np.ndarray, goal_pos: np.ndarray,
              n_seeds: int = 15) -> list:
        obstacles, _, _ = self._extract_obstacles()
        return self.build_with_obstacles(obstacles, start_pos, goal_pos, n_seeds)

    def build_with_obstacles(self, obstacles: list,
                             start_pos: np.ndarray, goal_pos: np.ndarray,
                             n_seeds: int = 15) -> list:
        from pydrake.geometry.optimization import Iris, IrisOptions, HPolyhedron

        if not obstacles:
            print("[3D-IRIS] 无障碍物，跳过", flush=True)
            return []

        lb = np.minimum(start_pos, goal_pos) - 3.0
        ub = np.maximum(start_pos, goal_pos) + 3.0
        domain = HPolyhedron.MakeBox(lb, ub)

        seeds = self._generate_seeds(start_pos, goal_pos, n_seeds)
        regions: list = []
        options = IrisOptions()

        for i, seed in enumerate(seeds):
            if not domain.PointInSet(seed):
                continue
            if any(obs.PointInSet(seed) for obs in obstacles):
                print(f"[3D-IRIS] 种子 {i} 在障碍物内，跳过", flush=True)
                continue
            try:
                region = Iris(obstacles, seed, domain, options)
                vol = region.MaximumVolumeInscribedEllipsoid().CalcVolume()
                if vol < 1e-6:
                    continue
                regions.append(region)
                print(f"[3D-IRIS] 区域 {len(regions)-1}: vol={vol:.2e}", flush=True)
            except Exception as e:
                print(f"[3D-IRIS] 种子 {i} 失败: {e}", flush=True)

        n_s = sum(1 for r in regions if r.PointInSet(start_pos))
        n_g = sum(1 for r in regions if r.PointInSet(goal_pos))
        print(f"[3D-IRIS] 共 {len(regions)} 个区域 | "
              f"start∈{n_s} | goal∈{n_g}", flush=True)
        return regions

    @staticmethod
    def _generate_seeds(start: np.ndarray, goal: np.ndarray,
                        n: int) -> list[np.ndarray]:
        seeds = [start.copy(), goal.copy()]
        rng = np.random.RandomState(42)
        for i in range(min(n - 2, 6)):
            alpha = (i + 1) / (min(n - 2, 6) + 1)
            s = (1 - alpha) * start + alpha * goal + rng.randn(3) * 0.2
            seeds.append(s)
        for _ in range(2):
            seeds.append(start + rng.randn(3) * 0.8)
        for _ in range(2):
            seeds.append(goal + rng.randn(3) * 0.8)
        return seeds


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TaskSpaceGcsPlanner — 三维 GCS 轨迹规划
# ═══════════════════════════════════════════════════════════════════════════════
class TaskSpaceGcsPlanner:
    """在 3D IRIS 区域上用 GCS 规划平滑末端轨迹。"""

    def __init__(self, task: TaskConfig) -> None:
        self.task = task
        self.trajectory = None
        self._start: Optional[np.ndarray] = None
        self._goal: Optional[np.ndarray] = None

    def plan(self, regions: list, start_pos: np.ndarray,
             goal_pos: np.ndarray):
        self._start = start_pos.copy()
        self._goal = goal_pos.copy()

        if not regions:
            print("[3D-GCS] 无 IRIS 区域，使用直线插值", flush=True)
            self._setup_fallback()
            return self.trajectory

        from pydrake.geometry.optimization import (
            GraphOfConvexSetsOptions, HPolyhedron,
        )
        from pydrake.planning import GcsTrajectoryOptimization

        gcs = GcsTrajectoryOptimization(num_positions=3)

        eps = 0.1
        source_sg = gcs.AddRegions(
            [HPolyhedron.MakeBox(start_pos - eps, start_pos + eps)],
            order=1, name="source",
        )
        target_sg = gcs.AddRegions(
            [HPolyhedron.MakeBox(goal_pos - eps, goal_pos + eps)],
            order=1, name="target",
        )
        main_sg = gcs.AddRegions(regions, order=2, name="iris3d")

        gcs.AddEdges(source_sg, main_sg)
        gcs.AddEdges(main_sg, main_sg)
        gcs.AddEdges(main_sg, target_sg)

        gcs.AddPathLengthCost(weight=1.0)
        gcs.AddTimeCost(weight=1.0)
        v_max = np.full(3, self.task.task_space_vel)
        gcs.AddVelocityBounds(-v_max, v_max)

        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = 5

        print(f"[3D-GCS] 求解中... ({len(regions)} 个区域)", flush=True)
        try:
            traj, result = gcs.SolvePath(source_sg, target_sg, options)
            if result.is_success():
                dur = traj.end_time() - traj.start_time()
                print(f"[3D-GCS] 成功，时长 = {dur:.2f}s", flush=True)
                self.trajectory = traj
                return self.trajectory
            print(f"[3D-GCS] 失败 ({result.get_solution_result()})，直线回退",
                  flush=True)
        except Exception as e:
            print(f"[3D-GCS] 异常: {e}，直线回退", flush=True)

        self._setup_fallback()
        return self.trajectory

    def _setup_fallback(self) -> None:
        from pydrake.trajectories import PiecewisePolynomial
        dist = np.linalg.norm(self._goal - self._start)
        duration = max(dist / self.task.task_space_vel, 1.0)
        self.trajectory = PiecewisePolynomial.FirstOrderHold(
            [0.0, duration],
            np.column_stack([self._start, self._goal]),
        )
        print(f"[3D-GCS] 回退直线轨迹，时长 = {duration:.2f}s", flush=True)

    def sample(self, t: float) -> np.ndarray:
        if self.trajectory is None:
            return self._start.copy() if self._start is not None else np.zeros(3)
        tc = np.clip(t, self.trajectory.start_time(), self.trajectory.end_time())
        return self.trajectory.value(tc).flatten()

    def sample_velocity(self, t: float) -> np.ndarray:
        if self.trajectory is None:
            return np.zeros(3)
        tc = np.clip(t, self.trajectory.start_time(), self.trajectory.end_time())
        return self.trajectory.EvalDerivative(tc).flatten()

    @property
    def duration(self) -> float:
        if self.trajectory is None:
            return 0.0
        return self.trajectory.end_time() - self.trajectory.start_time()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. JacobianTrackingController — 雅可比末端跟踪
# ═══════════════════════════════════════════════════════════════════════════════
class JacobianTrackingController:
    """用阻尼最小二乘 J† 将 3D 末端速度指令转为关节速度。"""

    def __init__(self, env: DrakeEnvironment, robot: RobotLoader,
                 task: TaskConfig) -> None:
        self.env = env
        self.robot = robot
        self.task = task
        self._col0: Optional[int] = None

    def _col_start(self) -> int:
        if self._col0 is None:
            self._col0 = self.env.plant.GetJointByName(
                "pris01", self.robot.model_instance
            ).position_start()
        return self._col0

    def _jacobian(self, q: np.ndarray):
        """返回 (ee_pos, J_robot)。"""
        from pydrake.multibody.tree import JacobianWrtVariable
        plant = self.env.plant
        ctx = self.env.plant_context
        nq = q.shape[0]
        plant.SetPositions(ctx, self.robot.model_instance, q)
        ee = plant.GetBodyByName("weld_point", self.robot.model_instance)
        ee_pos = plant.CalcRelativeTransform(
            ctx, plant.world_frame(), ee.body_frame()
        ).translation()
        J_full = plant.CalcJacobianTranslationalVelocity(
            ctx, JacobianWrtVariable.kQDot,
            ee.body_frame(), np.zeros(3),
            plant.world_frame(), plant.world_frame(),
        )
        c0 = self._col_start()
        return ee_pos, J_full[:, c0:c0 + nq]

    def compute(self, q: np.ndarray, desired_pos: np.ndarray,
                desired_vel: Optional[np.ndarray] = None) -> np.ndarray:
        nq = q.shape[0]
        ee_pos, J = self._jacobian(q)
        pos_err = desired_pos - ee_pos

        v_desired = self.task.task_space_Kp * pos_err
        if desired_vel is not None:
            v_desired += desired_vel

        damping = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))
        u = J_pinv @ v_desired
        return np.clip(u, -self.task.vel_limit, self.task.vel_limit)

    def compute_approach(self, q: np.ndarray, target_pos: np.ndarray,
                         max_cart_vel: float = 0.15) -> np.ndarray:
        """限速笛卡尔直线接近，关节速度上限降为 30%。"""
        nq = q.shape[0]
        ee_pos, J = self._jacobian(q)
        err = target_pos - ee_pos
        err_norm = np.linalg.norm(err)

        if err_norm > 1e-4:
            speed = min(max_cart_vel, err_norm)
            v_cart = (err / err_norm) * speed
        else:
            v_cart = np.zeros(3)

        damping = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))
        u = J_pinv @ v_cart
        jlim = self.task.vel_limit * 0.3
        return np.clip(u, -jlim, jlim)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Experiment — 主仿真循环
# ═══════════════════════════════════════════════════════════════════════════════
class Experiment:
    """组装全部模块，运行仿真主循环并实时可视化。"""

    def __init__(
        self,
        drake_config: Optional[DrakeConfig] = None,
        task_config: Optional[TaskConfig] = None,
    ) -> None:
        self.drake_cfg = drake_config or DrakeConfig()
        self.task_cfg = task_config or TaskConfig()
        self._use_ts = self.task_cfg.use_task_space

        MeshConverter.default_for_project().convert_all(force=False)
        self.resolver = URDFPackageResolver.default_for_project()
        self.env = DrakeEnvironment(self.drake_cfg)

        self.robot = RobotLoader(self.env, self.resolver)
        self.scene = SceneLoader(
            self.env, self.resolver,
            translation=self.drake_cfg.scene_translation,
        )

        self._add_frame_visualization()
        self._setup_collision_filter()
        self.env.finalize()

        self.robot.set_initial_positions()
        self.robot.print_joint_info()

        q_start = self.env.plant.GetPositions(
            self.env.plant_context, self.robot.model_instance
        ).copy()
        print(f"\n── 初始位形 q_start = {np.array2string(q_start, precision=3)} ──",
              flush=True)

        # ── 目标位姿（两种模式都需要）──
        self.target = TargetPoseComputer(
            self.env, self.robot, self.scene, self.task_cfg
        )
        self.target.compute_target_pose()
        self._target_pos_world = self.target.target_pose.translation().copy()

        # CBF 安全滤波器（两种模式共用）
        self.cbf_filter = CBFMPCSafetyFilter(
            self.env, self.robot, self.scene, self.task_cfg
        )

        if self._use_ts:
            self._init_task_space(q_start)
        else:
            self._init_config_space(q_start)

        self._add_target_frame()
        self.env.diagram.ForcedPublish(self.env.context)

    # ── 任务空间初始化 ──────────────────────────────────────────────
    def _init_task_space(self, q_start: np.ndarray) -> None:
        print("\n══ 任务空间模式 (3D IRIS + GCS + J† + CBF-MPC) ══", flush=True)
        plant = self.env.plant
        ctx = self.env.plant_context

        plant.SetPositions(ctx, self.robot.model_instance, q_start)
        ee = plant.GetBodyByName("weld_point", self.robot.model_instance)
        X_WE = plant.CalcRelativeTransform(ctx, plant.world_frame(), ee.body_frame())
        ee_start = X_WE.translation().copy()
        ee_goal = self._target_pos_world.copy()
        self._ee_goal_final = ee_goal.copy()

        # 点云表面采样 → 小球体障碍物（后续 IRIS 和预到位检查共用）
        print("\n── 3D IRIS 障碍物提取（点云采样） ──", flush=True)
        iris3d = TaskSpaceIrisBuilder(self.env, self.scene)
        obstacles, obs_centers, obs_radius = iris3d._extract_obstacles(
            sample_count=self.task_cfg.pointcloud_samples,
            inflate_radius=self.task_cfg.pointcloud_inflate,
        )
        self._obs_centers = obs_centers
        self._obs_radius = obs_radius

        z_world = self.target.target_pose.rotation().matrix()[:, 2]
        d = self.task_cfg.pre_approach_dist
        max_d = 8.0
        min_clearance = 0.3

        # 矢量化间隙检查：candidate 到所有球心距离 - 球半径 >= min_clearance
        while d <= max_d:
            candidate = ee_goal - z_world * d
            if len(obs_centers) > 0:
                surface_dists = (np.linalg.norm(obs_centers - candidate, axis=1)
                                 - obs_radius)
                too_close = surface_dists.min() < min_clearance
            else:
                too_close = False
            if not too_close:
                break
            print(f"[TaskSpace] d={d:.2f}m → 预到位点距障碍物过近，增大偏移",
                  flush=True)
            d += 0.3

        self._pre_approach = ee_goal - z_world * d

        print(f"[TaskSpace] EE 起点:   {np.array2string(ee_start, precision=3)}",
              flush=True)
        print(f"[TaskSpace] 预到位点:  {np.array2string(self._pre_approach, precision=3)} "
              f"(沿 -z 偏移 {d:.2f}m)", flush=True)
        print(f"[TaskSpace] 最终焊点:  {np.array2string(ee_goal, precision=3)}",
              flush=True)

        # 3D IRIS 构建 + 重试：如果 goal 不在任何区域内则增大偏移重试
        max_retries = 5
        for attempt in range(max_retries):
            print(f"\n── 3D IRIS 区域构建 (尝试 {attempt+1}) ──", flush=True)
            regions = iris3d.build_with_obstacles(
                obstacles, ee_start, self._pre_approach
            )
            n_g = sum(1 for r in regions if r.PointInSet(self._pre_approach))
            if n_g > 0:
                break

            d += 0.5
            if d > max_d:
                print("[TaskSpace] 预到位距离已达上限，使用当前结果", flush=True)
                break
            self._pre_approach = ee_goal - z_world * d
            print(f"[TaskSpace] goal∈0 → 增大偏移至 d={d:.2f}m, "
                  f"新预到位: {np.array2string(self._pre_approach, precision=3)}",
                  flush=True)

        # 3D GCS（规划到预到位点）
        print("\n── 3D GCS 轨迹规划 ──", flush=True)
        self.ts_planner = TaskSpaceGcsPlanner(self.task_cfg)
        self.ts_planner.plan(regions, ee_start, self._pre_approach)
        self._approach_phase = False

        # Jacobian 跟踪器
        self.jac_ctrl = JacobianTrackingController(
            self.env, self.robot, self.task_cfg
        )

        # IK 求解最终目标关节角（approach 阶段用关节空间插值）
        print("\n── 最终焊点 IK 求解 ──", flush=True)
        try:
            self._q_goal_approach = self.target.solve_ik()
            print(f"[Approach] IK 成功，q_goal = "
                  f"{np.array2string(self._q_goal_approach, precision=3)}")
        except RuntimeError as e:
            print(f"[Approach] IK 失败: {e}，approach 阶段将使用雅可比回退")
            self._q_goal_approach = None

        # 恢复机器人到当前位形（IK 内部可能修改了 plant 位形）
        plant.SetPositions(ctx, self.robot.model_instance, q_start)

        # ── 可视化 ──
        self._visualize_waypoints(ee_start, self._pre_approach, ee_goal)
        self._visualize_point_obstacles(obs_centers, obs_radius)
        self._visualize_gcs_trajectory()

    # ── C-space 初始化（保留原逻辑）──────────────────────────────
    def _init_config_space(self, q_start: np.ndarray) -> None:
        print("\n══ 配置空间模式 (C-space IRIS + GCS + CBF-MPC) ══", flush=True)
        print("── 计算 IK ──", flush=True)
        self.target.solve_ik()
        self.env.plant.SetPositions(
            self.env.plant_context, self.robot.model_instance, q_start
        )

        print("\n── C-space IRIS 区域构建 ──", flush=True)
        iris_builder = IrisRegionBuilder(self.env, self.robot, self.task_cfg)
        regions = iris_builder.build(q_start, self.target.q_goal)

        print("\n── C-space GCS 轨迹规划 ──", flush=True)
        self.gcs_planner = GcsPathPlanner(self.env, self.task_cfg)
        self.gcs_planner.plan(regions, q_start, self.target.q_goal)

        self.controller = TwoLayerController(
            self.gcs_planner, self.cbf_filter, self.env
        )

    def run(self) -> None:
        url = self.env.meshcat.web_url()
        print(f"\n[Experiment] MeshCat -> {url}", flush=True)
        print("[Experiment] 仿真开始，按 Ctrl+C 停止\n", flush=True)

        try:
            webbrowser.open(url)
        except Exception:
            pass

        plant = self.env.plant
        ctx = self.env.plant_context
        dt = self.task_cfg.sim_dt
        nq_robot = plant.num_positions(self.robot.model_instance)
        q_lb = plant.GetPositionLowerLimits()[:nq_robot]
        q_ub = plant.GetPositionUpperLimits()[:nq_robot]

        t = 0.0
        for step in range(self.task_cfg.max_steps):
            q = plant.GetPositions(ctx, self.robot.model_instance)

            # 检查 EE 位置到达
            ee = plant.GetBodyByName("weld_point", self.robot.model_instance)
            X_WE = plant.CalcRelativeTransform(ctx, plant.world_frame(), ee.body_frame())
            pos_err = np.linalg.norm(X_WE.translation() - self._target_pos_world)
            if pos_err < 0.03:
                print(f"\n[Experiment] 已到达目标！step={step}, t={t:.2f}s, "
                      f"pos_err={pos_err:.4f}", flush=True)
                break

            # ── 计算控制指令 ──
            if self._use_ts:
                u = self._step_task_space(q, t, step)
            else:
                u = self._step_config_space(q, t, step, dt)

            # 运动学积分
            q_new = np.clip(q + u * dt, q_lb, q_ub)
            plant.SetPositions(ctx, self.robot.model_instance, q_new)
            self.env.diagram.ForcedPublish(self.env.context)
            t += dt

            if step % 200 == 0:
                self._print_status(step, t, q_new)

            time.sleep(dt * self.task_cfg.sim_realtime_factor)

        print("[Experiment] 仿真结束", flush=True)
        self._hold_meshcat()

    # ── 任务空间控制步 ──
    def _step_task_space(self, q: np.ndarray, t: float,
                         step: int) -> np.ndarray:
        plant = self.env.plant
        ctx = self.env.plant_context
        ee = plant.GetBodyByName("weld_point", self.robot.model_instance)
        ee_pos = plant.CalcRelativeTransform(
            ctx, plant.world_frame(), ee.body_frame()
        ).translation()

        if not self._approach_phase:
            pre_err = np.linalg.norm(ee_pos - self._pre_approach)
            if pre_err < self.task_cfg.approach_switch_dist:
                self._approach_phase = True
                if step > 0:
                    print(f"\n[TaskSpace] 到达预到位点附近 (err={pre_err:.3f})，"
                          f"进入最终接近阶段（关节空间 IK 引导）", flush=True)

        if self._approach_phase:
            return self._step_approach(q, step)

        desired_pos = self.ts_planner.sample(t)
        desired_vel = self.ts_planner.sample_velocity(t)
        try:
            u_nom = self.jac_ctrl.compute(q, desired_pos, desired_vel)
            u_safe = self.cbf_filter.filter_velocity(q, u_nom)
            return u_safe
        except Exception as e:
            if step % 500 == 0:
                print(f"[Experiment] CBF 异常: {e}，跳过安全滤波", flush=True)
            return self.jac_ctrl.compute(q, desired_pos, desired_vel)

    def _step_approach(self, q: np.ndarray, step: int) -> np.ndarray:
        """approach 阶段：IK 关节空间 P 控制 → 雅可比微调。

        焊接任务要求末端工具接触工件，CBF 与此目标矛盾。
        平滑性由 P 控制器 + 速度裁剪保证（无瞬移）。
        两个子阶段：
          1) IK 引导：关节空间 P 控制趋向 IK 目标
          2) 微调：到达 IK 位形后用雅可比消除残余笛卡尔误差
        """
        vlim = self.task_cfg.vel_limit

        if self._q_goal_approach is not None:
            q_err_norm = np.linalg.norm(self._q_goal_approach - q)
            if q_err_norm > 0.05:
                Kp = self.task_cfg.task_space_Kp
                q_err = self._q_goal_approach - q
                u = np.clip(Kp * q_err, -vlim, vlim)
                phase_tag = "IK引导"
            else:
                u = self.jac_ctrl.compute_approach(
                    q, self._ee_goal_final,
                    max_cart_vel=0.3,
                )
                phase_tag = "雅可比微调"
        else:
            u = self.jac_ctrl.compute_approach(
                q, self._ee_goal_final,
                max_cart_vel=self.task_cfg.approach_cart_vel,
            )
            phase_tag = "雅可比回退"

        if step % 200 == 0:
            q_err_val = (np.linalg.norm(self._q_goal_approach - q)
                         if self._q_goal_approach is not None else -1)
            print(f"    [approach:{phase_tag}] q_err={q_err_val:.4f}  "
                  f"|u|={np.linalg.norm(u):.4f}  "
                  f"pris03={q[2]:.4f}", flush=True)

        return u

    # ── C-space 控制步（保留原逻辑）──
    def _step_config_space(self, q: np.ndarray, t: float,
                           step: int, dt: float) -> np.ndarray:
        try:
            return self.controller.compute(q, t)
        except Exception as e:
            if step % 200 == 0:
                print(f"[Experiment] C-space 控制异常: {e}", flush=True)
            try:
                q_ref = self.gcs_planner.sample(t)
            except Exception:
                q_ref = self.target.q_goal if self.target.q_goal is not None else q
            return np.clip(
                (q_ref - q) / dt,
                -self.task_cfg.vel_limit, self.task_cfg.vel_limit,
            )

    def _print_status(self, step: int, t: float, q: np.ndarray) -> None:
        plant = self.env.plant
        ctx = self.env.plant_context
        nq = q.shape[0]
        q_lb = plant.GetPositionLowerLimits()[:nq]
        q_ub = plant.GetPositionUpperLimits()[:nq]

        ee = plant.GetBodyByName("weld_point", self.robot.model_instance)
        X_WE = plant.CalcRelativeTransform(ctx, plant.world_frame(), ee.body_frame())
        pos = X_WE.translation()
        target_pos = self.target.target_pose.translation()
        pos_err = np.linalg.norm(pos - target_pos)
        print(f"  step={step:5d}  t={t:.2f}s  "
              f"ee={np.array2string(pos, precision=3)}  pos_err={pos_err:.4f}")

        margin = 0.01
        at_lo = [i for i in range(nq) if q[i] - q_lb[i] < margin]
        at_hi = [i for i in range(nq) if q_ub[i] - q[i] < margin]
        if at_lo or at_hi:
            print(f"    [!] 关节触限: 下限{at_lo} 上限{at_hi}  "
                  f"q[:3]={np.array2string(q[:3], precision=3)}", flush=True)

    def _hold_meshcat(self) -> None:
        print("[Experiment] MeshCat 保持运行，按 Ctrl+C 退出")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[Experiment] 已退出")

    def _setup_collision_filter(self) -> None:
        """排除自碰撞 + 龙门架(前3平动关节)与场景的碰撞。"""
        from pydrake.geometry import CollisionFilterDeclaration

        plant = self.env.plant
        sg = self.env.scene_graph
        cfm = sg.collision_filter_manager()

        robot_bodies = [
            plant.get_body(idx)
            for idx in plant.GetBodyIndices(self.robot.model_instance)
        ]
        robot_geoms = plant.CollectRegisteredGeometries(robot_bodies)
        cfm.Apply(CollisionFilterDeclaration().ExcludeWithin(robot_geoms))

        scene_bodies = [
            plant.get_body(idx)
            for idx in plant.GetBodyIndices(self.scene.model_instance)
        ]
        scene_geoms = plant.CollectRegisteredGeometries(scene_bodies)
        cfm.Apply(CollisionFilterDeclaration().ExcludeWithin(scene_geoms))

        # 排除前 3 个平动关节（龙门架）的连杆与场景的碰撞检测
        gantry_joints = ["pris01", "pris02", "pris03"]
        gantry_body_indices = set()
        for jname in gantry_joints:
            joint = plant.GetJointByName(jname, self.robot.model_instance)
            gantry_body_indices.add(joint.child_body().index())
            parent = joint.parent_body()
            if parent.index() != plant.world_body().index():
                gantry_body_indices.add(parent.index())
        try:
            robobase = plant.GetBodyByName("robobase", self.robot.model_instance)
            gantry_body_indices.add(robobase.index())
        except Exception:
            pass

        gantry_bodies = [plant.get_body(idx) for idx in gantry_body_indices]
        if gantry_bodies:
            gantry_geoms = plant.CollectRegisteredGeometries(gantry_bodies)
            cfm.Apply(
                CollisionFilterDeclaration().ExcludeBetween(
                    gantry_geoms, scene_geoms
                )
            )
            names = [plant.get_body(i).name() for i in gantry_body_indices]
            print(f"[Experiment] 龙门架连杆 {names} 已排除与场景的碰撞检测")

        # 排除焊枪末端连杆与工件的碰撞检测（焊接任务必须接触工件）
        tool_link_names = ["weld_point", "welding_gun_base", "link09", "link08"]
        tool_body_indices = set()
        for name in tool_link_names:
            try:
                body = plant.GetBodyByName(name, self.robot.model_instance)
                tool_body_indices.add(body.index())
            except Exception:
                pass
        if tool_body_indices:
            tool_bodies = [plant.get_body(idx) for idx in tool_body_indices]
            tool_geoms = plant.CollectRegisteredGeometries(tool_bodies)
            cfm.Apply(
                CollisionFilterDeclaration().ExcludeBetween(
                    tool_geoms, scene_geoms
                )
            )
            names = [plant.get_body(i).name() for i in tool_body_indices]
            print(f"[Experiment] 工具连杆 {names} 已排除与场景的碰撞检测"
                  f"（焊接需接触工件）")

        print("[Experiment] 已添加碰撞过滤（自碰撞 + 龙门架-场景 + 工具-场景）")

    def _add_frame_visualization(self) -> None:
        from pydrake.all import AddFrameTriadIllustration

        cfg = self.drake_cfg
        plant = self.env.plant
        sg = self.env.scene_graph

        frames = [
            ("world", None, None, 1.2),
            ("robot_base", "base_link", self.robot.model_instance, 1.0),
            ("weld_point", "weld_point", self.robot.model_instance, 0.8),
            ("scene_base", "base_link", self.scene.model_instance, 1.0),
            ("scene_l2", "l2", self.scene.model_instance, 0.8),
        ]
        for name, body_name, model, scale in frames:
            kw = dict(
                scene_graph=sg, plant=plant, name=f"{name}_frame",
                length=cfg.frame_axis_length * scale,
                radius=cfg.frame_axis_radius * scale,
                opacity=cfg.frame_axis_opacity,
            )
            if body_name is None:
                kw["frame"] = plant.world_frame()
            else:
                kw["body"] = plant.GetBodyByName(body_name, model)
            AddFrameTriadIllustration(**kw)

    def _add_target_frame(self) -> None:
        """在 MeshCat 中用圆柱绘制目标坐标系。"""
        if self.target.target_pose is None:
            return
        from pydrake.all import Rgba, Cylinder
        from pydrake.math import RigidTransform, RotationMatrix

        meshcat = self.env.meshcat
        R = self.target.target_pose.rotation().matrix()
        p = self.target.target_pose.translation()
        length, radius = 0.4, 0.008

        colors = [Rgba(1, 0, 0, 0.8), Rgba(0, 1, 0, 0.8), Rgba(0, 0, 1, 0.8)]
        for i, (color, label) in enumerate(zip(colors, ["x", "y", "z"])):
            axis = R[:, i]
            mid = p + axis * length / 2
            R_cyl = self._align_z_to(axis)
            meshcat.SetObject(f"target_frame/{label}", Cylinder(radius, length), color)
            meshcat.SetTransform(
                f"target_frame/{label}",
                RigidTransform(RotationMatrix(R_cyl), mid),
            )
        print("[Experiment] 已绘制目标坐标系")

    # ── 任务空间可视化 ──────────────────────────────────────────────
    def _visualize_waypoints(self, ee_start: np.ndarray,
                             pre_approach: np.ndarray,
                             ee_goal: np.ndarray) -> None:
        from pydrake.all import Rgba, Sphere
        from pydrake.math import RigidTransform

        meshcat = self.env.meshcat
        points = [
            ("waypoints/ee_start",     ee_start,     0.08, Rgba(0, 0.8, 0, 0.9)),
            ("waypoints/pre_approach", pre_approach,  0.08, Rgba(1, 0.6, 0, 0.9)),
            ("waypoints/weld_goal",    ee_goal,       0.08, Rgba(1, 0, 0, 0.9)),
        ]
        for name, pos, r, color in points:
            meshcat.SetObject(name, Sphere(r), color)
            meshcat.SetTransform(name, RigidTransform(pos))

        line_pts = np.column_stack([ee_start, pre_approach, ee_goal])
        meshcat.SetLine("waypoints/approach_line", line_pts,
                        line_width=2.0, rgba=Rgba(1, 1, 1, 0.4))
        print("[Vis] 起点(绿)、预到位(橙)、焊点(红) 已标记", flush=True)

    def _visualize_point_obstacles(self, centers: np.ndarray,
                                   radius: float) -> None:
        """在 MeshCat 中绘制点云障碍物采样点（每隔 stride 个画一个小球）。"""
        from pydrake.all import Rgba, Sphere
        from pydrake.math import RigidTransform

        if centers.shape[0] == 0:
            return

        meshcat = self.env.meshcat
        stride = max(1, centers.shape[0] // 120)
        n_drawn = 0
        vis_r = radius * 2.5

        for i in range(0, centers.shape[0], stride):
            meshcat.SetObject(f"obs_pts/{i:04d}", Sphere(vis_r),
                              Rgba(1.0, 0.3, 0.0, 0.4))
            meshcat.SetTransform(f"obs_pts/{i:04d}",
                                 RigidTransform(centers[i]))
            n_drawn += 1

        print(f"[Vis] 点云障碍物: 绘制 {n_drawn}/{centers.shape[0]} 个 "
              f"(r={radius}m, 无多余凸包)", flush=True)

    def _visualize_gcs_trajectory(self) -> None:
        from pydrake.all import Rgba
        from pydrake.math import RigidTransform, RotationMatrix

        meshcat = self.env.meshcat
        planner = self.ts_planner
        if planner.trajectory is None:
            return

        n_samples = 80
        t0 = planner.trajectory.start_time()
        t1 = planner.trajectory.end_time()
        ts = np.linspace(t0, t1, n_samples)
        pts = np.array([planner.sample(t) for t in ts]).T

        meshcat.SetLine("gcs_path/line", pts,
                        line_width=4.0, rgba=Rgba(0, 1, 1, 0.9))

        from pydrake.all import Sphere
        for idx in [0, n_samples // 4, n_samples // 2,
                    3 * n_samples // 4, n_samples - 1]:
            p = pts[:, idx]
            meshcat.SetObject(f"gcs_path/pt_{idx}", Sphere(0.03),
                              Rgba(0, 1, 1, 0.8))
            meshcat.SetTransform(f"gcs_path/pt_{idx}", RigidTransform(p))

        dur = t1 - t0
        length = sum(np.linalg.norm(pts[:, j+1] - pts[:, j])
                     for j in range(pts.shape[1] - 1))
        print(f"[Vis] GCS 轨迹: {n_samples} 采样点, "
              f"长度={length:.2f}m, 时长={dur:.2f}s", flush=True)

    @staticmethod
    def _align_z_to(axis: np.ndarray) -> np.ndarray:
        """计算将 z 轴旋转到 axis 方向的旋转矩阵。"""
        z = np.array([0, 0, 1.0])
        v = np.cross(z, axis)
        c = np.dot(z, axis)
        if np.linalg.norm(v) < 1e-8:
            return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx / (1 + c)


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    drake_config = DrakeConfig()
    task_config = TaskConfig()
    exp = Experiment(drake_config, task_config)
    exp.run()


if __name__ == "__main__":
    main()
