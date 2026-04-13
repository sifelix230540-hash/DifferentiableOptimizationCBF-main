"""Trajectory planning module: plan_poses.json -> joint_trajectory.json.

带逐段进度条输出，可随时看到每段完成情况。
"""

from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.cbf_qp_module import solve_cbf_qp_step  # noqa: E402
from CBF_experiment.active.pybullet.configuration_metrics import evaluate_configuration_quality, summarize_clearance_entries  # noqa: E402
from CBF_experiment.active.pybullet.geometry_module import GeometryEngine  # noqa: E402
from CBF_experiment.active.pybullet.simulation_module import Robot, Workpiece, load_config, _resolve  # noqa: E402


# ── 工具 ────────────────────────────────────────────


def _load_payload(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_segment_joint_targets(poses: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    q_home = np.asarray(poses["q_home"], dtype=float)
    q_init = np.asarray(poses["q_initconfig"], dtype=float)
    q_mid = np.asarray(poses["q_mid"], dtype=float)
    q_gantry_home = q_mid.copy()
    q_gantry_home[:3] = q_init[:3]
    return {
        "joint_to_initconfig": (q_home, q_init),
        "gantry_to_mid": (q_init, q_mid),
        "converge_to_mid": (q_mid, q_mid),
        "gantry_to_home": (q_mid, q_gantry_home),
    }


def _segment_duration(cfg: dict, seg_name: str, motion_type: str) -> float:
    durations = dict(cfg.get("trajectory_planning", {}).get("segment_durations", {}))
    if seg_name in durations:
        return float(durations[seg_name])
    legacy = cfg.get("trajectory", {})
    if motion_type in ("后六轴", "前三轴", "全轴收敛"):
        return float(legacy.get("approach_duration", 3.0))
    if seg_name == "weld":
        return float(legacy.get("weld_duration", 7.0))
    return float(legacy.get("approach_duration", 6.0))


def _step_count(duration: float, dt: float, min_steps: int) -> int:
    return max(int(round(duration / dt)), min_steps, 2)


def _joint_velocity_limits(cfg: dict, robot) -> np.ndarray:
    robot_cfg = cfg.get("robot", {})
    base_vel_limit = float(robot_cfg.get("base_vel_limit", 0.4))
    dq_limit = float(robot_cfg.get("dq_limit", 1.0))
    return np.concatenate([
        np.full(int(robot.n_pris), base_vel_limit, dtype=float),
        np.full(int(robot.n_revo), dq_limit, dtype=float),
    ])


def _resample_cartesian(positions, quaternions, n_steps: int):
    pts = np.asarray(positions, dtype=float).reshape(-1, 3)
    quats = np.asarray(quaternions, dtype=float).reshape(-1, 4)
    if pts.shape[0] == 1:
        return np.repeat(pts, n_steps, axis=0), np.repeat(quats, n_steps, axis=0)
    chord = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(chord)])
    total = max(float(s[-1]), 1e-9)
    alpha = np.linspace(0.0, total, n_steps)
    pos_out = np.column_stack([np.interp(alpha, s, pts[:, ax]) for ax in range(3)])
    slerp = Slerp(s, Rotation.from_quat(quats))
    return pos_out, slerp(alpha).as_quat()


def _resample_joint_waypoints(waypoints: np.ndarray, n_steps: int) -> np.ndarray:
    """沿 joint_waypoints 做弧长参数化多点插值，返回 (n_steps, dof)。"""
    if waypoints.shape[0] == 1:
        return np.repeat(waypoints, n_steps, axis=0)
    chord = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(chord)])
    total = max(float(s[-1]), 1e-9)
    alpha = np.linspace(0.0, total, n_steps)
    return np.column_stack(
        [np.interp(alpha, s, waypoints[:, j]) for j in range(waypoints.shape[1])]
    )


def _resample_positions(positions, n_steps: int) -> np.ndarray:
    """对 3D 位置序列做弧长参数化重采样。"""
    pts = np.asarray(positions, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 1:
        return np.repeat(pts, n_steps, axis=0)
    chord = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(chord)])
    total = max(float(s[-1]), 1e-9)
    alpha = np.linspace(0.0, total, n_steps)
    return np.column_stack([np.interp(alpha, s, pts[:, ax]) for ax in range(3)])


def _joint_path_min_duration(waypoints: np.ndarray, vel_limits: np.ndarray) -> float:
    """给定关节航点和速度上限，估计物理可达的最短时长。"""
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.shape[0] <= 1:
        return 0.0
    delta = np.abs(np.diff(waypoints, axis=0))
    seg_times = np.max(delta / np.maximum(vel_limits.reshape(1, -1), 1e-9), axis=1)
    return float(np.sum(seg_times))


def _min_clearance_for_q(robot, geometry, q: np.ndarray) -> float:
    robot.set_joint_state(q, dq=np.zeros_like(q))
    distances, _normals, _link_indices = geometry.get_cbf_distances(robot, q)
    d = np.asarray(distances, dtype=float).reshape(-1)
    return float(np.min(d)) if d.size else float("inf")


def _chomp_optimize_joint_refs(q_refs: np.ndarray, *, robot, geometry, cfg: dict) -> tuple[np.ndarray, dict]:
    tp_cfg = cfg.get("trajectory_planning", {})
    q_opt = np.asarray(q_refs, dtype=float).copy()
    if q_opt.shape[0] <= 2 or not bool(tp_cfg.get("use_chomp_optimizer", False)):
        clearance = min(_min_clearance_for_q(robot, geometry, q) for q in q_opt)
        return q_opt, {"clearance_before": clearance, "clearance_after": clearance}

    iters = int(tp_cfg.get("chomp_iters", 20))
    step_size = float(tp_cfg.get("chomp_step_size", 0.10))
    smooth_weight = float(tp_cfg.get("chomp_smooth_weight", 0.2))
    obstacle_weight = float(tp_cfg.get("chomp_obstacle_weight", 1.0))
    clearance_margin = float(tp_cfg.get("chomp_clearance_margin", 0.05))
    fd_eps = float(tp_cfg.get("chomp_fd_eps", 1e-3))

    clearance_before = min(_min_clearance_for_q(robot, geometry, q) for q in q_opt)
    q_start = q_opt[0].copy()
    q_goal = q_opt[-1].copy()

    def _obstacle_penalty(q_local: np.ndarray) -> float:
        clearance = _min_clearance_for_q(robot, geometry, q_local)
        violation = max(clearance_margin - clearance, 0.0)
        return 0.5 * violation * violation

    for _ in range(iters):
        q_prev = q_opt.copy()
        for idx in range(1, q_opt.shape[0] - 1):
            smooth_grad = 2.0 * q_prev[idx] - q_prev[idx - 1] - q_prev[idx + 1]
            obstacle_grad = np.zeros(q_opt.shape[1], dtype=float)
            for dim in range(q_opt.shape[1]):
                q_plus = q_prev[idx].copy()
                q_minus = q_prev[idx].copy()
                q_plus[dim] += fd_eps
                q_minus[dim] -= fd_eps
                obstacle_grad[dim] = (
                    _obstacle_penalty(q_plus) - _obstacle_penalty(q_minus)
                ) / max(2.0 * fd_eps, 1e-9)
            q_opt[idx] = q_prev[idx] - step_size * (
                smooth_weight * smooth_grad + obstacle_weight * obstacle_grad
            )
        q_opt[0] = q_start
        q_opt[-1] = q_goal

    clearance_after = min(_min_clearance_for_q(robot, geometry, q) for q in q_opt)
    return q_opt, {
        "clearance_before": clearance_before,
        "clearance_after": clearance_after,
    }


def _quat_angle_error(q_a: np.ndarray, q_b: np.ndarray) -> float:
    qa = Rotation.from_quat(np.asarray(q_a, dtype=float).reshape(4))
    qb = Rotation.from_quat(np.asarray(q_b, dtype=float).reshape(4))
    return float(np.linalg.norm((qb * qa.inv()).as_rotvec()))


def _compute_configuration_quality(robot, q, dq, info: dict | None, geometry) -> dict:
    clearance_summary = None
    if isinstance(info, dict):
        maybe_summary = info.get("clearance_summary")
        if isinstance(maybe_summary, dict):
            clearance_summary = dict(maybe_summary)
    if clearance_summary is None and geometry is not None:
        clearance_summary = summarize_clearance_entries(getattr(geometry, "last_query_meta", []))
    return evaluate_configuration_quality(
        robot,
        q,
        dq=dq,
        motion_component="linear",
        clearance_summary=clearance_summary,
    )


def _summarize_configuration_quality(history: list[dict]) -> dict:
    if not history:
        return {
            "min_manipulability": 0.0,
            "min_inverse_condition": 0.0,
            "min_self_collision_distance": 0.0,
            "min_environment_distance": 0.0,
            "min_joint_limit_margin": 0.0,
        }
    inv_cond = np.asarray([item.get("inverse_condition", 0.0) for item in history], dtype=float)
    manipulability = np.asarray([item.get("manipulability", 0.0) for item in history], dtype=float)
    self_clearance = np.asarray([item.get("self_collision_distance", np.inf) for item in history], dtype=float)
    env_clearance = np.asarray([item.get("environment_distance", np.inf) for item in history], dtype=float)
    joint_margin = np.asarray([item.get("joint_limit_margin", 0.0) for item in history], dtype=float)
    return {
        "min_manipulability": float(np.min(manipulability)),
        "min_inverse_condition": float(np.min(inv_cond)),
        "min_self_collision_distance": float(np.min(self_clearance)),
        "min_environment_distance": float(np.min(env_clearance)),
        "min_joint_limit_margin": float(np.min(joint_margin)),
    }


# ── 进度条 ──────────────────────────────────────────


def _progress_bar(current: int, total: int, width: int = 40,
                  prefix: str = "", extra: str = "") -> str:
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = f"{100.0 * frac:5.1f}%"
    return f"\r{prefix} |{bar}| {pct} [{current}/{total}] {extra}"


# ── 主流程 ──────────────────────────────────────────


def run(
    cfg_path: str | Path | None = None,
    poses_json: str | Path | None = None,
    output_json: str | Path | None = None,
):
    cfg = load_config(cfg_path)
    tp_cfg = cfg.get("trajectory_planning", {})
    poses_path = Path(poses_json or _resolve(
        tp_cfg.get("input_json", "artifacts/sdf_exp/plan_poses.json")
    ))
    output_path = Path(output_json or _resolve(
        tp_cfg.get("output_json", "artifacts/sdf_exp/joint_trajectory.json")
    ))
    poses = _load_payload(poses_path)
    dt = float(tp_cfg.get("dt",
        cfg.get("control", {}).get("mpc_dt",
            cfg.get("simulation", {}).get("dt", 1.0 / 240.0))))
    alpha_ema = float(np.clip(tp_cfg.get("velocity_ema_alpha", 1.0), 0.0, 1.0))
    legacy_traj_cfg = cfg.get("trajectory", {})
    pos_tol = float(tp_cfg.get(
        "segment_pos_tolerance",
        legacy_traj_cfg.get("progress_end_tolerance", 0.02),
    ))
    ori_tol = float(tp_cfg.get("segment_ori_tolerance", 0.08))
    joint_tol = float(tp_cfg.get("segment_joint_tolerance", 0.02))
    robobase_tol = float(tp_cfg.get("segment_robobase_tolerance", pos_tol))
    max_extra_time = float(tp_cfg.get("segment_max_extension", 8.0))
    hold_steps = max(int(tp_cfg.get("segment_converged_hold_steps", 3)), 1)
    track_quality = bool(cfg.get("configuration_quality", {}).get("track_during_trajectory", True))

    created_connection = False
    if not p.isConnected():
        p.connect(p.DIRECT)
        created_connection = True
    try:
        robot = Robot(cfg)
        _workpiece = Workpiece(cfg)
        geometry = GeometryEngine(cfg, robot)

        q0 = np.asarray(
            poses.get("q_home", np.zeros(robot.dof, dtype=float)), dtype=float
        ).reshape(robot.dof)
        robot.set_joint_state(q0)
        vel_limits = _joint_velocity_limits(cfg, robot)

        joint_targets = {}
        if all(k in poses for k in ("q_home", "q_initconfig", "q_mid")):
            joint_targets = _build_segment_joint_targets(poses)

        all_segments = poses.get("segments", [])
        total_segments = len(all_segments)

        total_steps_est = 0
        seg_step_counts = []
        seg_durations = []
        for seg in all_segments:
            dur = _segment_duration(cfg, seg["name"], seg.get("motion_type", "笛卡尔"))
            if seg.get("joint_waypoints") and len(seg["joint_waypoints"]) > 1:
                dur = max(dur, _joint_path_min_duration(
                    np.asarray(seg["joint_waypoints"], dtype=float), vel_limits,
                ))
            elif seg["name"] in joint_targets:
                q_start, q_goal = joint_targets[seg["name"]]
                dur = max(dur, _joint_path_min_duration(
                    np.vstack([q_start, q_goal]), vel_limits,
                ))
            n = _step_count(dur, dt, int(seg.get("n_points", 2)))
            seg_durations.append(dur)
            seg_step_counts.append(n)
            total_steps_est += n

        print(f"[traj] {total_segments} segments, ~{total_steps_est} steps, dt={dt:.4f}s")
        print(f"[traj] est. duration ~ {total_steps_est * dt:.1f}s")
        print("=" * 72)

        steps: list[dict] = []
        segment_reports: list[dict] = []
        t_now = 0.0
        global_step = 0

        for seg_idx, seg in enumerate(all_segments):
            seg_name = str(seg["name"])
            motion_type = str(seg.get("motion_type", "笛卡尔"))
            duration = seg_durations[seg_idx]
            n_steps = seg_step_counts[seg_idx]
            max_steps = n_steps + max(int(round(max_extra_time / max(dt, 1e-9))), 0)

            seg_label = f"[{seg_idx+1}/{total_segments}] {seg_name}"
            print(f"\n{seg_label}  ({motion_type}, {duration:.1f}s min, {n_steps} steps + converge)")

            has_joint_wp = (
                "joint_waypoints" in seg
                and seg["joint_waypoints"]
                and len(seg["joint_waypoints"]) > 1
            )
            is_joint_seg = (not has_joint_wp) and (seg_name in joint_targets)
            u_prev = None

            if has_joint_wp:
                jwps = np.asarray(seg["joint_waypoints"], dtype=float)
                q_wp_refs = _resample_joint_waypoints(jwps, n_steps)
                rb_ref_positions = _resample_positions(
                    seg["positions"], n_steps,
                )
                print(f"  → multi-waypoint joint interp ({jwps.shape[0]} wp → {n_steps} steps)")
            elif is_joint_seg:
                q_start, q_goal = joint_targets[seg_name]
                alphas = np.linspace(0.0, 1.0, n_steps)
            else:
                ref_positions, ref_quaternions = _resample_cartesian(
                    seg["positions"], seg["quaternions"], n_steps,
                )

            seg_t0 = _time.perf_counter()
            min_h_seg = np.inf
            cbf_active_count = 0
            converged_hold_count = 0
            terminal_error = {}
            terminated_by_convergence = False
            quality_history: list[dict] = []

            for i in range(max_steps):
                q, dq = robot.get_joint_state()
                ref_idx = min(i, n_steps - 1)

                if has_joint_wp:
                    q_ref = q_wp_refs[ref_idx]
                    u_raw, info = solve_cbf_qp_step(
                        robot, q, dq,
                        dt=dt,
                        q_ref=q_ref, geometry_engine=geometry, cfg=cfg,
                    )
                    ref_pos_out = rb_ref_positions[ref_idx]
                    ref_quat_out = np.array([0, 0, 0, 1], dtype=float)
                elif is_joint_seg:
                    alpha = alphas[ref_idx]
                    q_ref = (1.0 - alpha) * q_start + alpha * q_goal
                    u_raw, info = solve_cbf_qp_step(
                        robot, q, dq,
                        dt=dt,
                        q_ref=q_ref, geometry_engine=geometry, cfg=cfg,
                    )
                    ref_pos_out = None
                    ref_quat_out = None
                else:
                    rp = np.asarray(ref_positions[ref_idx], dtype=float)
                    rq = np.asarray(ref_quaternions[ref_idx], dtype=float)
                    u_raw, info = solve_cbf_qp_step(
                        robot, q, dq,
                        dt=dt,
                        pos_ref=rp, quat_ref=rq,
                        geometry_engine=geometry, cfg=cfg,
                    )
                    ref_pos_out = rp
                    ref_quat_out = rq

                if u_prev is None or alpha_ema >= 1.0:
                    u_cmd = np.asarray(u_raw, dtype=float).reshape(-1)
                else:
                    u_cmd = alpha_ema * np.asarray(u_raw, dtype=float).reshape(-1) + (1.0 - alpha_ema) * u_prev
                u_prev = u_cmd.copy()

                q_next = q + u_cmd * dt
                robot.set_joint_state(q_next, dq=u_cmd)
                ee_pos, ee_quat = robot.get_ee_pose()
                rb_pos, _ = robot.get_robobase_pose()

                if ref_pos_out is None:
                    ref_pos_out = ee_pos
                    ref_quat_out = ee_quat

                min_h_seg = min(min_h_seg, float(info["min_h"]))
                if info["cbf_active"]:
                    cbf_active_count += 1

                step_dict = {
                    "t": round(t_now, 6),
                    "segment_name": seg_name,
                    "motion_type": motion_type,
                    "q": q_next.tolist(),
                    "dq": np.asarray(u_cmd, dtype=float).tolist(),
                    "ee_pos": np.asarray(ee_pos, dtype=float).tolist(),
                    "ee_quat": np.asarray(ee_quat, dtype=float).tolist(),
                    "ref_pos": np.asarray(ref_pos_out, dtype=float).tolist(),
                    "ref_quat": np.asarray(ref_quat_out, dtype=float).tolist(),
                    "status": str(info["status"]),
                    "min_h": round(float(info["min_h"]), 6),
                    "cbf_active": bool(info["cbf_active"]),
                }
                if track_quality:
                    quality_metrics = _compute_configuration_quality(robot, q_next, u_cmd, info, geometry)
                    quality_history.append(quality_metrics)
                    step_dict["configuration_quality"] = quality_metrics
                if has_joint_wp:
                    step_dict["robobase_pos"] = np.asarray(rb_pos, dtype=float).tolist()
                steps.append(step_dict)
                t_now += dt
                global_step += 1

                if has_joint_wp:
                    rb_goal = np.asarray(seg["positions"][-1], dtype=float).reshape(3)
                    rb_err = float(np.linalg.norm(rb_goal - rb_pos))
                    q_err = float(np.max(np.abs(q_wp_refs[-1] - q_next)))
                    terminal_error = {
                        "robobase_pos": round(rb_err, 6),
                        "joint_inf": round(q_err, 6),
                    }
                    converged_now = (rb_err <= robobase_tol) and (q_err <= joint_tol)
                elif is_joint_seg:
                    q_goal_err = float(np.max(np.abs(q_goal - q_next)))
                    terminal_error = {
                        "joint_inf": round(q_goal_err, 6),
                    }
                    converged_now = q_goal_err <= joint_tol
                else:
                    pos_goal = np.asarray(ref_positions[-1], dtype=float).reshape(3)
                    quat_goal = np.asarray(ref_quaternions[-1], dtype=float).reshape(4)
                    pos_err = float(np.linalg.norm(pos_goal - ee_pos))
                    ori_err = _quat_angle_error(ee_quat, quat_goal)
                    terminal_error = {
                        "pos": round(pos_err, 6),
                        "ori": round(ori_err, 6),
                    }
                    converged_now = (pos_err <= pos_tol) and (ori_err <= ori_tol)

                if converged_now:
                    converged_hold_count += 1
                else:
                    converged_hold_count = 0

                if (i + 1) >= n_steps and converged_hold_count >= hold_steps:
                    terminated_by_convergence = True
                    break

                if (i + 1) % max(max_steps // 20, 1) == 0 or i == max_steps - 1:
                    elapsed = _time.perf_counter() - seg_t0
                    rate = (i + 1) / max(elapsed, 1e-6)
                    eta = (max_steps - i - 1) / max(rate, 1e-6)
                    extra = (
                        f"h_min={min_h_seg:+.4f} "
                        f"cbf={cbf_active_count} "
                        f"goal={terminal_error} "
                        f"{rate:.0f}it/s "
                        f"ETA {eta:.0f}s"
                    )
                    sys.stdout.write(_progress_bar(
                        i + 1, max_steps, prefix=seg_label, extra=extra,
                    ))
                    sys.stdout.flush()

            seg_elapsed = _time.perf_counter() - seg_t0
            executed_steps = i + 1
            status_text = "converged" if terminated_by_convergence else "max-extension-reached"
            print(
                f"\n  {status_text} {seg_elapsed:.1f}s  "
                f"h_min={min_h_seg:+.4f}  cbf_active={cbf_active_count}/{executed_steps}  "
                f"goal={terminal_error}"
            )

            segment_reports.append({
                "name": seg_name,
                "motion_type": motion_type,
                "duration": round(executed_steps * dt, 4),
                "min_duration": round(duration, 4),
                "n_steps": executed_steps,
                "planned_steps": n_steps,
                "wall_time": round(seg_elapsed, 2),
                "min_h": round(min_h_seg, 6) if np.isfinite(min_h_seg) else 0.0,
                "cbf_active_ratio": round(cbf_active_count / max(executed_steps, 1), 4),
                "converged": bool(terminated_by_convergence),
                "terminal_error": terminal_error,
            })
            if track_quality:
                segment_reports[-1]["configuration_quality"] = _summarize_configuration_quality(quality_history)

        print("\n" + "=" * 72)
        print(f"[traj] done: {len(steps)} steps, sim time {t_now:.2f}s")

        payload = {
            "dt": round(dt, 8),
            "n_segments": len(segment_reports),
            "n_steps": len(steps),
            "segments": segment_reports,
            "steps": steps,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"[traj] saved -> {output_path}")
        return payload
    finally:
        if created_connection and p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    run()
