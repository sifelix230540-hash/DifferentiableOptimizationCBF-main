"""位姿规划模块：读 plan_path.json → 9 段位姿轨迹 → 写 plan_poses.json。

完整路径（9 段）
--------------
  起始位置 ──(1 后六轴)──▸ initconfig
           ──(2 前三轴)──▸ 中间位置
           ──(3 笛卡尔)──▸ retreat 起点
           ──(4 笛卡尔)──▸ 焊接起点
           ──(5 焊接)────▸ 焊接终点
           ──(6 笛卡尔)──▸ retreat 终点
           ──(7 笛卡尔)──▸ 中间位置
           ──(8 后六轴)──▸ initconfig
           ──(9 前三轴)──▸ 起始位置

输入: plan_path.json + Super_config.json
输出: plan_poses.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.simulation_module import (  # noqa: E402
    Robot, Workpiece, load_config, _resolve,
)


# ── 工具函数 ────────────────────────────────────────


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def _as_vec3_list(raw) -> list[np.ndarray]:
    if raw is None:
        return []
    return [np.asarray(pt, dtype=float).reshape(3) for pt in raw]


def _concat_paths(*paths: list[np.ndarray]) -> list[np.ndarray]:
    merged: list[np.ndarray] = []
    for path in paths:
        for pt in path:
            pt = np.asarray(pt, dtype=float).reshape(3)
            if merged and np.allclose(merged[-1], pt, atol=1e-6):
                continue
            merged.append(pt)
    return merged


def _slerp_quats(q_start: np.ndarray, q_end: np.ndarray, count: int) -> list[np.ndarray]:
    q0 = np.asarray(q_start, dtype=float).reshape(4)
    q1 = np.asarray(q_end, dtype=float).reshape(4)
    if count <= 1:
        return [q0.copy()]
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    slerp = Slerp([0.0, 1.0], Rotation.from_quat(np.vstack([q0, q1])))
    ts = np.linspace(0.0, 1.0, count, endpoint=True)
    return list(slerp(ts).as_quat())


# ── 焊枪四元数 ──────────────────────────────────────


def build_gun_quat(frame_quat: np.ndarray, z_local: np.ndarray) -> np.ndarray:
    """根据焊点帧四元数和局部 z 方向构建焊枪世界四元数。

    gun_z = frame_rot · normalize(z_local)
    gun_x = project( frame_rot·(-1,0,0) , onto plane ⊥ gun_z )
    gun_y = gun_z × gun_x
    """
    frame_rot = Rotation.from_quat(frame_quat)
    gun_z = _normalize(frame_rot.apply(_normalize(np.asarray(z_local, dtype=float))))

    ref_x = frame_rot.apply(np.array([-1.0, 0.0, 0.0]))
    proj = ref_x - float(np.dot(ref_x, gun_z)) * gun_z
    gun_x = _normalize(proj)
    if np.linalg.norm(gun_x) < 1e-6:
        gun_x = _normalize(np.cross(gun_z, np.array([0.0, 1.0, 0.0])))

    gun_y = _normalize(np.cross(gun_z, gun_x))
    gun_x = _normalize(np.cross(gun_y, gun_z))

    return Rotation.from_matrix(np.column_stack([gun_x, gun_y, gun_z])).as_quat()


# ── 关节空间 FK 插值 ────────────────────────────────


def _interpolate_joint_segment(
    robot: Robot,
    q_start: np.ndarray,
    q_end: np.ndarray,
    steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """在关节空间线性插值，用 FK 计算每步 EE 位姿。"""
    positions: list[np.ndarray] = []
    quats: list[np.ndarray] = []
    for i in range(steps):
        alpha = i / max(steps - 1, 1)
        q = (1 - alpha) * q_start + alpha * q_end
        robot.set_joint_state(q)
        pos, quat = robot.get_ee_pose()
        positions.append(pos.copy())
        quats.append(quat.copy())
    return positions, quats


# ── 主流程 ──────────────────────────────────────────


def run(
    cfg_path: str | Path | None = None,
    plan_path_json: str | Path | None = None,
    output_json: str | Path | None = None,
):
    cfg = load_config(cfg_path)
    pp_cfg = cfg.get("pose_planning", {})

    plan_json = Path(plan_path_json or _resolve(
        pp_cfg.get("plan_path_json", "artifacts/sdf_exp/plan_path.json")
    ))
    out_json = Path(output_json or _resolve(
        pp_cfg.get("output_json", "artifacts/sdf_exp/plan_poses.json")
    ))
    n_joint_steps = int(pp_cfg.get("joint_interpolation_steps", 20))

    with open(plan_json, "r", encoding="utf-8") as f:
        plan = json.load(f)

    wp_cfg = cfg.get("workpiece", {})
    z_start_local = np.asarray(wp_cfg.get("weld_start_z_local", [-1, 1, -1]), dtype=float)
    z_end_local = np.asarray(wp_cfg.get("weld_end_z_local", [1, 1, -1]), dtype=float)
    start_link = wp_cfg.get("start_link", "l2")
    goal_link = wp_cfg.get("goal_link", "l3")

    gantry_home = np.asarray(
        cfg.get("robot", {}).get("gantry_initial_q", [0, 0, 0]), dtype=float
    )

    # ── PyBullet DIRECT: 提取所有关键帧信息 ─────────
    p.connect(p.DIRECT)
    try:
        robot = Robot(cfg)
        workpiece = Workpiece(cfg)

        n_pris = robot.n_pris
        n_revo = robot.n_revo

        # 起始位置：机器人默认位姿
        q_home, _ = robot.get_joint_state()
        home_ee_pos, home_ee_quat = robot.get_ee_pose()

        # initconfig：只改后六轴到规划目标值，龙门不动
        q_plan = np.asarray(plan["robot_q_at_goal"], dtype=float)
        q_initconfig = q_home.copy()
        q_initconfig[n_pris:] = q_plan[n_pris:]  # 后六轴 → 规划值

        robot.set_joint_state(q_initconfig)
        initconfig_ee_pos, initconfig_ee_quat = robot.get_ee_pose()

        # 中间位置：再把前三轴移到规划值（= q_plan 完整配置）
        q_mid = q_plan.copy()
        robot.set_joint_state(q_mid)
        mid_ee_pos, mid_ee_quat = robot.get_ee_pose()

        # 焊点帧四元数
        _, start_frame_quat = workpiece.get_frame_pose(start_link)
        _, goal_frame_quat = workpiece.get_frame_pose(goal_link)

        # ── 关节空间段用 FK 插值 ────────────────────
        # Seg 1: 起始 → initconfig (后六轴)
        seg1_pos, seg1_quat = _interpolate_joint_segment(
            robot, q_home, q_initconfig, n_joint_steps,
        )

        # Seg 2: initconfig → 中间位置 (前三轴, 沿 RRT robobase 路径)
        #   主角 = robobase；positions 存 robobase 坐标，joint_waypoints 存对应关节构型
        rrt_path_pb = plan.get("path", [])
        seg2_joint_waypoints = None
        if rrt_path_pb and len(rrt_path_pb) > 1 and plan.get("path_is_robobase", False):
            robot.set_joint_state(q_initconfig)
            base_at_init, _ = robot.get_robobase_pose()
            gantry_offset = np.asarray(base_at_init, dtype=float) - q_initconfig[:n_pris]
            seg2_pos, seg2_quat = [], []
            seg2_joint_waypoints = []
            for wp in rrt_path_pb:
                q_wp = q_initconfig.copy()
                q_wp[:n_pris] = np.asarray(wp, dtype=float).reshape(3) - gantry_offset
                robot.set_joint_state(q_wp)
                rb_pos, _ = robot.get_robobase_pose()
                seg2_pos.append(np.asarray(rb_pos, dtype=float).copy())
                seg2_quat.append(np.array([0, 0, 0, 1], dtype=float))
                seg2_joint_waypoints.append(q_wp.tolist())
            print(f"[pose] Seg 2 uses RRT robobase path: {len(seg2_pos)} waypoints (robobase coords)")
        else:
            seg2_pos, seg2_quat = _interpolate_joint_segment(
                robot, q_initconfig, q_mid, n_joint_steps,
            )
            print(f"[pose] Seg 2 uses linear joint interpolation: {n_joint_steps} steps")

        # 返程
        # Seg 8: 焊接完成后全九轴收敛回 q_mid
        #   焊接过程中关节会因 IK 偏离 q_mid，此段确保回到
        #   {前三轴: mid, 后六轴: init_config} 的精确构型
        robot.set_joint_state(q_mid)
        mid_ee_pos_check, _ = robot.get_ee_pose()
        seg8_pos = [mid_ee_pos_check.copy()] * max(n_joint_steps // 2, 2)
        seg8_quat = [mid_ee_quat.copy()] * len(seg8_pos)
        print(f"[pose] Seg 8: converge to q_mid (all 9 axes), {len(seg8_pos)} steps")

        # Seg 9: 前三轴原路返回 (后六轴保持 init_config 不变, RRT 反向)
        #   主角 = robobase；positions 存 robobase 坐标
        seg9_joint_waypoints = None
        if rrt_path_pb and len(rrt_path_pb) > 1 and plan.get("path_is_robobase", False):
            robot.set_joint_state(q_mid)
            base_at_mid, _ = robot.get_robobase_pose()
            gantry_offset_ret = np.asarray(base_at_mid, dtype=float) - q_mid[:n_pris]
            seg9_pos, seg9_quat = [], []
            seg9_joint_waypoints = []
            for wp in reversed(rrt_path_pb):
                q_wp = q_mid.copy()
                q_wp[:n_pris] = np.asarray(wp, dtype=float).reshape(3) - gantry_offset_ret
                robot.set_joint_state(q_wp)
                rb_pos, _ = robot.get_robobase_pose()
                seg9_pos.append(np.asarray(rb_pos, dtype=float).copy())
                seg9_quat.append(np.array([0, 0, 0, 1], dtype=float))
                seg9_joint_waypoints.append(q_wp.tolist())
            print(f"[pose] Seg 9 uses RRT robobase path (reversed): {len(seg9_pos)} waypoints (robobase coords)")
        else:
            seg9_pos, seg9_quat = _interpolate_joint_segment(
                robot, q_mid, q_initconfig, n_joint_steps,
            )
            print(f"[pose] Seg 9 uses linear joint interpolation: {n_joint_steps} steps")
    finally:
        p.disconnect()

    # ── 焊枪四元数 ──────────────────────────────────
    weld_start_quat = build_gun_quat(start_frame_quat, z_start_local)
    weld_end_quat = build_gun_quat(goal_frame_quat, z_end_local)

    if float(np.dot(mid_ee_quat, weld_start_quat)) < 0.0:
        weld_start_quat = -weld_start_quat
    if float(np.dot(weld_start_quat, weld_end_quat)) < 0.0:
        weld_end_quat = -weld_end_quat

    print(f"[pose] weld_start_quat = {np.round(weld_start_quat, 4).tolist()}")
    print(f"[pose] weld_end_quat   = {np.round(weld_end_quat, 4).tolist()}")

    # ── 笛卡尔段位置（从 plan_path.json）──────────
    weld_start = np.asarray(plan["weld_start_point"], dtype=float)
    weld_goal = np.asarray(plan["weld_goal_point"], dtype=float)

    # Seg 3: 中间位置 → retreat 起点 (ee_bezier_path)
    seg3_cartesian = _as_vec3_list(plan.get("ee_bezier_path"))

    # Seg 4: retreat 起点 → 焊接起点 (approach_line_path)
    seg4_cartesian = _as_vec3_list(
        plan.get("approach_line_path") or plan.get("approach_bezier_path")
    )
    if seg4_cartesian and not np.allclose(seg4_cartesian[-1], weld_start, atol=1e-4):
        seg4_cartesian = _concat_paths(seg4_cartesian, [weld_start])

    # Seg 5: 焊接
    seg5_cartesian = [weld_start.copy(), weld_goal.copy()]

    # Seg 6: 焊接终点 → retreat 终点 (直线)
    #   approach_end_line_path: retreat_end → weld_goal (直线, 需翻转)
    seg6_cartesian = _as_vec3_list(
        plan.get("approach_end_line_path") or plan.get("approach_end_bezier_path")
    )
    seg6_cartesian.reverse()
    if seg6_cartesian and not np.allclose(seg6_cartesian[0], weld_goal, atol=1e-4):
        seg6_cartesian = _concat_paths([weld_goal], seg6_cartesian)

    # Seg 7: retreat 终点 → 中间位置
    #   ee_bezier_path_return / ee_path_return 方向: mid → retreat_end (反向)
    #   需要翻转为 retreat_end → mid
    seg7_cartesian = _as_vec3_list(
        plan.get("ee_bezier_path_return") or plan.get("ee_path_return")
    )
    seg7_cartesian.reverse()

    # ── 笛卡尔段四元数 SLERP ──────────────────────
    def _slerp_for(positions, q_a, q_b):
        return _slerp_quats(q_a, q_b, len(positions))

    seg3_quat = _slerp_for(seg3_cartesian, mid_ee_quat, weld_start_quat)
    seg4_quat = _slerp_for(seg4_cartesian, weld_start_quat, weld_start_quat)
    seg5_quat = _slerp_for(seg5_cartesian, weld_start_quat, weld_end_quat)
    seg6_quat = _slerp_for(seg6_cartesian, weld_end_quat, weld_end_quat)
    seg7_quat = _slerp_for(seg7_cartesian, weld_end_quat, mid_ee_quat)

    # ── 组装 ────────────────────────────────────────
    #   (name, positions, quaternions, motion_type, joint_waypoints_or_None)
    segments = [
        ("joint_to_initconfig",  seg1_pos,       seg1_quat,  "后六轴",    None),
        ("gantry_to_mid",        seg2_pos,       seg2_quat,  "前三轴",    seg2_joint_waypoints),
        ("approach_to_retreat",  seg3_cartesian, seg3_quat,  "笛卡尔",   None),
        ("retreat_to_weld_start", seg4_cartesian, seg4_quat, "笛卡尔",   None),
        ("weld",                 seg5_cartesian, seg5_quat,  "焊接",     None),
        ("weld_end_retreat",     seg6_cartesian, seg6_quat,  "笛卡尔",   None),
        ("return_to_mid",       seg7_cartesian, seg7_quat,  "笛卡尔",    None),
        ("converge_to_mid",     seg8_pos,       seg8_quat,  "全轴收敛",  None),
        ("gantry_to_home",      seg9_pos,       seg9_quat,  "前三轴",    seg9_joint_waypoints),
    ]

    output_segments = []
    for name, positions, quats, motion_type, jwp in segments:
        n = len(positions)
        entry = {
            "name": name,
            "motion_type": motion_type,
            "n_points": n,
            "positions": [np.asarray(pt, dtype=float).tolist() for pt in positions],
            "quaternions": [np.asarray(q, dtype=float).tolist() for q in quats],
        }
        if jwp is not None:
            entry["joint_waypoints"] = jwp
        output_segments.append(entry)
        tag = f"  (joint_wp={len(jwp)})" if jwp else ""
        print(f"[pose] {name:25s}  {motion_type}  n={n}{tag}")

    # ── 写出 ────────────────────────────────────────
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_segments": len(output_segments),
        "weld_start_quat": weld_start_quat.tolist(),
        "weld_end_quat": weld_end_quat.tolist(),
        "home_ee_pos": home_ee_pos.tolist(),
        "home_ee_quat": home_ee_quat.tolist(),
        "mid_ee_pos": mid_ee_pos.tolist(),
        "mid_ee_quat": mid_ee_quat.tolist(),
        "q_home": q_home.tolist(),
        "q_initconfig": q_initconfig.tolist(),
        "q_mid": q_mid.tolist(),
        "weld_start_z_local": z_start_local.tolist(),
        "weld_end_z_local": z_end_local.tolist(),
        "segments": output_segments,
    }
    with open(str(out_json), "w", encoding="utf-8") as _f:
        _f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[pose] 完成 → {out_json}")
    return payload


if __name__ == "__main__":
    run()
