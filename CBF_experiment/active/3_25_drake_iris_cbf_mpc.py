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
    gamma_dcbf: float = 0.3
    safety_margin: float = 0.05
    cbf_max_distance: float = 1.0

    # ── 仿真 ──
    sim_dt: float = 0.01
    vel_limit: float = 1.0
    max_steps: int = 5000


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

    def _compute_cbf_data(self, q: np.ndarray) -> list[tuple[float, np.ndarray]]:
        """计算所有机器人-场景碰撞对的 (h, grad_h)。"""
        from pydrake.multibody.tree import JacobianWrtVariable

        self._ensure_body_indices()
        plant = self.env.plant
        ctx = self.env.plant_context
        nq = q.shape[0]

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

            h = pair.distance - self.task.safety_margin

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
# 7. Experiment — 主仿真循环
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

        # ① STL -> OBJ
        MeshConverter.default_for_project().convert_all(force=False)

        # ② 解析器
        self.resolver = URDFPackageResolver.default_for_project()

        # ③ Drake 环境
        self.env = DrakeEnvironment(self.drake_cfg)

        # ④ 加载模型
        self.robot = RobotLoader(self.env, self.resolver)
        self.scene = SceneLoader(
            self.env, self.resolver,
            translation=self.drake_cfg.scene_translation,
        )

        # ⑤ 坐标系可视化
        self._add_frame_visualization()

        # ⑤-bis 碰撞过滤：排除机器人和场景各自的自碰撞
        self._setup_collision_filter()

        # ⑥ Finalize
        self.env.finalize()

        # ⑦ 初始位形
        self.robot.set_initial_positions()
        self.robot.print_joint_info()

        # ⑧ 保存初始位形，然后计算目标位姿 + IK
        q_start = self.env.plant.GetPositions(
            self.env.plant_context, self.robot.model_instance
        ).copy()
        print(f"\n── 初始位形 q_start = {np.array2string(q_start, precision=3)} ──")

        print("── 计算目标位姿 ──")
        self.target = TargetPoseComputer(
            self.env, self.robot, self.scene, self.task_cfg
        )
        self.target.compute_target_pose()
        self.target.solve_ik()

        # IK 会修改 plant context，恢复到初始位形
        self.env.plant.SetPositions(
            self.env.plant_context, self.robot.model_instance, q_start
        )

        # ⑨ 在 MeshCat 中绘制目标坐标系
        self._add_target_frame()

        # ⑩ IRIS
        print("\n── IRIS 区域构建 ──")
        iris_builder = IrisRegionBuilder(self.env, self.robot, self.task_cfg)
        regions = iris_builder.build(q_start, self.target.q_goal)

        # ⑪ GCS
        print("\n── GCS 轨迹规划 ──")
        self.gcs_planner = GcsPathPlanner(self.env, self.task_cfg)
        self.gcs_planner.plan(regions, q_start, self.target.q_goal)

        # ⑫ CBF-MPC
        self.cbf_filter = CBFMPCSafetyFilter(
            self.env, self.robot, self.scene, self.task_cfg
        )

        # ⑬ 两层控制器
        self.controller = TwoLayerController(
            self.gcs_planner, self.cbf_filter, self.env
        )

        # 发布初始状态
        self.env.diagram.ForcedPublish(self.env.context)

    def run(self) -> None:
        url = self.env.meshcat.web_url()
        print(f"\n[Experiment] MeshCat -> {url}")
        print("[Experiment] 仿真开始，按 Ctrl+C 停止\n")

        try:
            webbrowser.open(url)
        except Exception:
            pass

        plant = self.env.plant
        ctx = self.env.plant_context
        dt = self.task_cfg.sim_dt
        max_steps = self.task_cfg.max_steps

        t = 0.0
        for step in range(max_steps):
            q = plant.GetPositions(ctx, self.robot.model_instance)

            # 检查到达
            if self.target.q_goal is not None:
                err = np.linalg.norm(q - self.target.q_goal)
                if err < 0.02:
                    print(f"\n[Experiment] 已到达目标！step={step}, t={t:.2f}s, err={err:.4f}")
                    break

            # 控制
            try:
                u = self.controller.compute(q, t)
            except Exception as e:
                if step % 200 == 0:
                    print(f"[Experiment] 控制器异常: {e}，纯跟踪回退")
                try:
                    q_ref = self.gcs_planner.sample(t)
                except Exception:
                    q_ref = self.target.q_goal if self.target.q_goal is not None else q
                u = np.clip(
                    (q_ref - q) / dt,
                    -self.task_cfg.vel_limit,
                    self.task_cfg.vel_limit,
                )

            # 运动学积分
            q_new = q + u * dt
            q_lb = plant.GetPositionLowerLimits()
            q_ub = plant.GetPositionUpperLimits()
            nq = q.shape[0]
            q_new = np.clip(q_new, q_lb[:nq], q_ub[:nq])
            plant.SetPositions(ctx, self.robot.model_instance, q_new)

            self.env.diagram.ForcedPublish(self.env.context)
            t += dt

            if step % 200 == 0:
                self._print_status(step, t, q_new)

            time.sleep(dt * 0.1)  # 10x 加速播放

        print("[Experiment] 仿真结束")
        self._hold_meshcat()

    def _print_status(self, step: int, t: float, q: np.ndarray) -> None:
        plant = self.env.plant
        ctx = self.env.plant_context
        ee = plant.GetBodyByName("weld_point", self.robot.model_instance)
        X_WE = plant.CalcRelativeTransform(ctx, plant.world_frame(), ee.body_frame())
        pos = X_WE.translation()
        target_pos = self.target.target_pose.translation()
        pos_err = np.linalg.norm(pos - target_pos)
        print(f"  step={step:5d}  t={t:.2f}s  "
              f"ee={np.array2string(pos, precision=3)}  pos_err={pos_err:.4f}")

    def _hold_meshcat(self) -> None:
        print("[Experiment] MeshCat 保持运行，按 Ctrl+C 退出")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[Experiment] 已退出")

    def _setup_collision_filter(self) -> None:
        """排除机器人内部自碰撞和场景内部自碰撞，仅保留机器人-场景碰撞对。"""
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

        print("[Experiment] 已添加碰撞过滤（排除机器人和场景各自的自碰撞）")

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
