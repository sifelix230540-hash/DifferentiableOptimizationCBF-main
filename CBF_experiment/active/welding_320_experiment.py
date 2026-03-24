import math
from pathlib import Path
import sys
import time

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBF_experiment.active.welding_320_common import (
    ExperimentConfig,
    SimulationScene,
    build_weld_reference_quat,
)
from CBF_experiment.active.welding_320_control import (
    CartesianRRTNominalPlanner,
    JointSpaceNominalPlanner,
    create_controller,
)
from CBF_experiment.active.welding_320_corridor import PyBulletModelSnapshot
from CBF_experiment.active.welding_320_ik import MultiSeedIKSolver
from CBF_experiment.active.welding_320_robot import JakaRobot, URDFObstacle, WorkpieceModel
from CBF_experiment.active.welding_320_trajectory import PathProgressTrajectory, PiecewiseLineSlerpTrajectory


def build_pre_approach_pose(target_pos, target_quat, weld_direction, retract_distance, lift_distance):
    target_pos = np.array(target_pos, dtype=float)
    target_quat = np.array(target_quat, dtype=float)
    weld_direction = np.array(weld_direction, dtype=float)
    norm = np.linalg.norm(weld_direction)
    unit_dir = np.array([0.0, 0.0, -1.0]) if norm < 1e-9 else weld_direction / norm
    pre_pos = target_pos - unit_dir * float(retract_distance)
    pre_pos = pre_pos + np.array([0.0, 0.0, float(lift_distance)])
    return pre_pos, target_quat.copy()


class AvoidanceExperiment:
    """串起场景、机器人、工件、规划与控制的主实验流程。"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.scene = SimulationScene(config)
        self.robot = JakaRobot(config, self.scene)
        self.workpiece = WorkpieceModel(config)

        ee_pos_init, ee_quat_init = self.robot.get_ee_pose()
        self.initial_pos = ee_pos_init.copy()
        self.initial_quat = ee_quat_init.copy()
        self.obstacles = []
        if not config.ignore_all_collisions:
            wp_obs = URDFObstacle(
                self.workpiece.body_id,
                cbf_link_indices=[li for li in self.robot.cbf_link_indices if li != self.robot.ee_link_index],
            )
            self.obstacles.append(wp_obs)

        start_pos, start_frame_quat = self.workpiece.get_frame_pose(config.start_link_name)
        goal_pos, goal_frame_quat = self.workpiece.get_frame_pose(config.goal_link_name)
        start_quat = build_weld_reference_quat(
            start_frame_quat,
            config.weld_local_direction,
            prev_quat=ee_quat_init,
        )
        goal_quat = build_weld_reference_quat(
            goal_frame_quat,
            config.weld_local_direction,
            prev_quat=start_quat,
        )

        self.start_ref = (start_pos, np.array(start_quat, dtype=float))
        self.goal_ref = (goal_pos, np.array(goal_quat, dtype=float))

        q_init, _ = self.robot.get_joint_state()
        obstacle_body_ids = [] if config.ignore_all_collisions else [self.workpiece.body_id]
        self.ik_solver = MultiSeedIKSolver(self.robot, config)
        start_candidates = self.ik_solver.solve(start_pos, start_quat, reference_q=q_init, obstacle_body_ids=obstacle_body_ids)
        q_start = (
            start_candidates[0].q.copy()
            if start_candidates
            else self.robot.calculate_ik(start_pos, start_quat, rest_poses=q_init)
        )
        goal_candidates = self.ik_solver.solve(goal_pos, goal_quat, reference_q=q_start, obstacle_body_ids=obstacle_body_ids)
        q_goal = (
            goal_candidates[0].q.copy()
            if goal_candidates
            else self.robot.calculate_ik(goal_pos, goal_quat, rest_poses=q_start)
        )

        weld_dir_start = p.getMatrixFromQuaternion(start_frame_quat)
        weld_dir_goal = p.getMatrixFromQuaternion(goal_frame_quat)
        weld_dir_start = np.array(weld_dir_start, dtype=float).reshape(3, 3) @ np.array(config.weld_local_direction, dtype=float)
        weld_dir_goal = np.array(weld_dir_goal, dtype=float).reshape(3, 3) @ np.array(config.weld_local_direction, dtype=float)

        pre_start_pos, pre_start_quat = build_pre_approach_pose(
            start_pos,
            start_quat,
            weld_dir_start,
            config.pre_approach_distance,
            config.pre_approach_lift,
        )
        pre_goal_pos, pre_goal_quat = build_pre_approach_pose(
            goal_pos,
            goal_quat,
            weld_dir_goal,
            config.pre_approach_distance,
            config.pre_approach_lift,
        )
        pre_start_candidates = self.ik_solver.solve(
            pre_start_pos,
            pre_start_quat,
            reference_q=q_init,
            obstacle_body_ids=obstacle_body_ids,
            extra_seed_qs=[q_start],
        )
        q_pre_start = (
            pre_start_candidates[0].q.copy()
            if pre_start_candidates
            else self.robot.calculate_ik(pre_start_pos, pre_start_quat, rest_poses=q_init)
        )
        pre_goal_candidates = self.ik_solver.solve(
            pre_goal_pos,
            pre_goal_quat,
            reference_q=q_goal,
            obstacle_body_ids=obstacle_body_ids,
            extra_seed_qs=[q_goal, q_init],
        )
        q_pre_goal = (
            pre_goal_candidates[0].q.copy()
            if pre_goal_candidates
            else self.robot.calculate_ik(pre_goal_pos, pre_goal_quat, rest_poses=q_goal)
        )

        self.nominal_planner = JointSpaceNominalPlanner(
            self.robot,
            config,
            workpiece_body_id=self.workpiece.body_id,
        )
        self.cartesian_planner = CartesianRRTNominalPlanner(
            self.robot,
            config,
            workpiece_body_id=self.workpiece.body_id,
        )
        seg_1 = self.nominal_planner.plan_to_goal_set(
            q_init,
            pre_start_candidates if pre_start_candidates else [q_pre_start],
            config.approach_duration * 0.55,
            "approach_pre",
        )
        seg_2 = self.nominal_planner.plan_to_goal_set(
            q_pre_start,
            start_candidates if start_candidates else [q_start],
            config.approach_duration * 0.45,
            "approach_align",
        )
        seg_3 = self.cartesian_planner._plan_single_segment(
            start_pos,
            start_quat,
            goal_pos,
            goal_quat,
            q_start,
            q_goal,
            config.weld_duration,
        )
        seg_4 = self.nominal_planner.plan_to_goal_set(
            q_goal,
            pre_goal_candidates if pre_goal_candidates else [q_pre_goal],
            config.return_duration * 0.45,
            "return_release",
        )
        seg_5 = self.nominal_planner.plan_to_goal_set(
            q_pre_goal,
            [q_init],
            config.return_duration * 0.55,
            "return_home",
        )
        base_trajectory = PiecewiseLineSlerpTrajectory([seg_1, seg_2, seg_3, seg_4, seg_5])
        self.trajectory = PathProgressTrajectory(base_trajectory)
        self.plan_segments = list(self.trajectory.segments)
        self.model_snapshot = PyBulletModelSnapshot(
            robot_urdf=config.urdf_path,
            workpiece_urdf=config.workpiece_urdf_path,
            active_joint_names=tuple(
                p.getJointInfo(self.robot.body_id, joint_index)[1].decode()
                for joint_index in self.robot.active_joints
            ),
            active_joint_limits=tuple(
                (float(lo), float(hi))
                for lo, hi in zip(*self.robot.get_active_joint_limits())
            ),
        )
        self.ik_diagnostics = {
            "start_candidate_count": len(start_candidates),
            "goal_candidate_count": len(goal_candidates),
            "pre_start_candidate_count": len(pre_start_candidates),
            "pre_goal_candidate_count": len(pre_goal_candidates),
            "start_position_error": float(self.robot.evaluate_pose_candidate(q_start, start_pos, start_quat)["position_error"]),
            "goal_position_error": float(self.robot.evaluate_pose_candidate(q_goal, goal_pos, goal_quat)["position_error"]),
        }

        if config.ignore_all_collisions:
            print("[info] 名义轨迹使用关节空间 approach/return + 焊缝段参考轨迹（当前忽略全部碰撞）。")
            print("[info] CBF/名义规划均已忽略碰撞，仅保留轨迹跟踪与种子筛选诊断。")
        else:
            print("[info] 名义轨迹使用关节空间 approach/return + 焊缝段参考轨迹（统一用关节态碰撞筛选种子）。")
        print("[info] IK 候选统计: " + ", ".join(f"{k}={v}" for k, v in self.ik_diagnostics.items()))
        print("[info] 模型签名: " + self.model_snapshot.signature()[:12])
        print("[info] 各段规划状态: " + ", ".join(
            f"seg{i + 1}={getattr(seg, 'planner_status', 'unknown')}"
            for i, seg in enumerate(self.plan_segments)
        ))
        for idx, seg in enumerate(self.plan_segments):
            diag = getattr(seg, "planning_diagnostics", {})
            if diag:
                print(
                    f"[info] seg{idx + 1} 诊断: "
                    f"ik_calls={diag.get('ik_calls', 0)} "
                    f"ik_failures={diag.get('ik_failures', 0)} "
                    f"collision_checks={diag.get('collision_checks', 0)} "
                    f"collision_failures={diag.get('collision_failures', 0)} "
                    f"max_fk={diag.get('max_fk_error', 0.0):.4f}"
                )

        seg_colors = ([0.85, 0.35, 0.15], [0.95, 0.15, 0.15], [0.20, 0.45, 0.90])
        for seg_idx, seg in enumerate(self.trajectory.segments):
            color = seg_colors[min(seg_idx, len(seg_colors) - 1)]
            status = getattr(seg, "planner_status", "")
            is_rrt = "rrt" in status
            draw_color = color if is_rrt else [0.35, 0.35, 0.35]
            draw_width = 2.8 if is_rrt else 1.4
            marker_rgba = (*color, 0.90) if is_rrt else (0.15, 0.15, 0.15, 0.75)
            self.scene.draw_polyline(seg.waypoints_pos, color=draw_color, width=draw_width)
            stride = max(1, len(seg.waypoints_pos) // 20)
            for i in range(0, len(seg.waypoints_pos), stride):
                self.scene.create_marker(0.005 if is_rrt else 0.004, marker_rgba, seg.waypoints_pos[i].tolist())
            p.addUserDebugText(
                status,
                seg.waypoints_pos[0].tolist(),
                textColorRGB=draw_color,
                textSize=1.1,
            )
        self.scene.draw_direction(start_pos, weld_dir_start, [0.90, 0.20, 0.20], length=0.12)
        self.scene.draw_direction(goal_pos, weld_dir_goal, [0.20, 0.20, 0.90], length=0.12)
        self.scene.create_marker(0.010, (0.9, 0.6, 0.2, 0.85), pre_start_pos.tolist())
        self.scene.create_marker(0.010, (0.2, 0.6, 0.9, 0.85), pre_goal_pos.tolist())

        focus = 0.5 * (start_pos + goal_pos)
        p.resetDebugVisualizerCamera(
            cameraDistance=config.camera_distance,
            cameraYaw=config.camera_yaw,
            cameraPitch=config.camera_pitch,
            cameraTargetPosition=focus.tolist(),
        )

        self.ee_marker = self.scene.create_marker(0.012, (0.1, 0.9, 0.2, 0.9), ee_pos_init.tolist())
        self.ref_marker = self.scene.create_marker(0.012, (0.95, 0.2, 0.2, 0.9), self.initial_pos.tolist())
        self.start_marker = self.scene.create_marker(0.014, (0.95, 0.15, 0.15, 0.85), start_pos.tolist())
        self.goal_marker = self.scene.create_marker(0.014, (0.15, 0.25, 0.95, 0.85), goal_pos.tolist())

        self.controller = create_controller(self.robot, config, self.trajectory)
        self.prev_ee = ee_pos_init.copy()
        self.sim_step = 0
        self.scene.enable_rendering()
        print("===== 焊接实验 (控制器: mpc_dcbf, 障碍: workpiece_only) =====")

    def _progress_step_limit(self, progress_value: float) -> float:
        seg_idx = self.trajectory.current_segment_index(progress_value)
        status = getattr(self.trajectory.segments[min(seg_idx, len(self.trajectory.segments) - 1)], "planner_status", "")
        scale = 1.0
        if status.startswith("approach_"):
            scale = self.config.approach_progress_scale
        elif status.startswith("return_"):
            scale = self.config.return_progress_scale
        return max(self.config.progress_step_limit * scale, self.config.mpc_progress_step_min)

    def _update_visuals(self, ee_pos, ref_pos, info, progress_value):
        self.scene.update_marker(self.ee_marker, ee_pos.tolist())
        self.scene.update_marker(self.ref_marker, ref_pos.tolist())
        if np.linalg.norm(ee_pos - self.prev_ee) > 1e-3:
            p.addUserDebugLine(self.prev_ee.tolist(), ee_pos.tolist(), [0.1, 0.8, 0.2], lineWidth=1.5)
            self.prev_ee = ee_pos.copy()
        seg_idx = min(self.trajectory.current_segment_index(progress_value) + 1, len(self.trajectory.segments))
        self.scene.update_status(
            f"seg={seg_idx}  step={self.sim_step}  "
            f"err={info['tracking_error']*1000:.1f}mm  "
            f"rot={math.degrees(info.get('orientation_error', 0.0)):.1f}deg  "
            f"h={info['min_h']*1000:.1f}mm  "
            f"{info['status']}"
        )

    def _solve_step(self, q, dq, ee_pos, ee_quat, ref_pos, ref_quat, ref_lv, ref_av, progress_value):
        for obs in self.obstacles:
            obs.update_from_slider()
        return self.controller.solve(
            q,
            dq,
            ee_pos,
            ee_quat,
            ref_pos,
            ref_quat,
            ref_lv,
            ref_av,
            self.obstacles,
            current_progress=progress_value,
        )

    def run(self):
        progress_exec = 0.0
        hold_steps = int(self.config.hold_duration / self.config.dt)
        hold_counter = 0

        try:
            while p.isConnected():
                q, dq = self.robot.get_joint_state()
                ee_pos, ee_quat = self.robot.get_ee_pose()
                progress_proj = self.trajectory.project_progress(
                    ee_pos,
                    hint_progress=progress_exec,
                    search_radius=max(0.5, self.config.mpc_progress_step_min * self.config.N_mpc * 4.0),
                )
                progress_exec = self.trajectory.advance_progress(
                    progress_exec,
                    progress_proj,
                    max_step=self._progress_step_limit(progress_exec),
                )

                if progress_exec >= self.trajectory.progress_end - self.config.progress_end_tolerance:
                    hold_counter += 1
                    if hold_counter >= hold_steps:
                        break
                else:
                    hold_counter = 0

                ref = self.trajectory.sample_by_progress(progress_exec)
                u_cmd, info = self._solve_step(q, dq, ee_pos, ee_quat, *ref, progress_exec)
                lag_error, contour_error = self.trajectory.compute_path_errors(ee_pos, progress_exec)
                info["progress_exec"] = float(progress_exec)
                info["progress_proj"] = float(progress_proj)
                info["lag_error"] = float(lag_error)
                info["contour_error"] = float(contour_error)

                alpha_q = self.config.q_nominal_tracking
                self.robot.q_nominal = (1 - alpha_q) * self.robot.q_nominal + alpha_q * q
                self.robot.command_velocities(u_cmd)

                p.stepSimulation()
                self._update_visuals(ee_pos, ref[0], info, progress_exec)
                if self.sim_step % self.config.print_every == 0:
                    gp = self.robot.get_gantry_pos()
                    seg_idx = min(self.trajectory.current_segment_index(progress_exec) + 1, len(self.trajectory.segments))
                    print(
                        f"[step {self.sim_step:4d}] "
                        f"seg={seg_idx} "
                        f"s={progress_exec:.3f}/{self.trajectory.progress_end:.3f} "
                        f"gantry=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                        f"err={info['tracking_error']*1000:.1f}mm "
                        f"rot={math.degrees(info.get('orientation_error', 0.0)):.1f}deg "
                        f"lag={info['lag_error']*1000:.1f}mm "
                        f"cont={info['contour_error']*1000:.1f}mm "
                        f"h={info['min_h']*1000:.1f}mm "
                        f"{info['status']}"
                    )
                self.sim_step += 1
                time.sleep(self.config.dt)

            if p.isConnected():
                self.robot.command_velocities(np.zeros(self.robot.dof))
                print("===== 轨迹结束，保持窗口 (Ctrl+C 退出) =====")
            while p.isConnected():
                q, dq = self.robot.get_joint_state()
                ee_pos, ee_quat = self.robot.get_ee_pose()
                ref = self.trajectory.sample_by_progress(self.trajectory.progress_end)
                u_cmd, info = self._solve_step(q, dq, ee_pos, ee_quat, *ref, self.trajectory.progress_end)
                self.robot.command_velocities(u_cmd)
                p.stepSimulation()
                self._update_visuals(ee_pos, ref[0], info, self.trajectory.progress_end)
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            print("\n用户中断。")
        finally:
            if p.isConnected():
                p.disconnect()
            print(f"仿真结束，共 {self.sim_step} 步。")


def main():
    AvoidanceExperiment(ExperimentConfig()).run()
