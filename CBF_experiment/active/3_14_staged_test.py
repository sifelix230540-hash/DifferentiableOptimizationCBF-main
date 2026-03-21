"""3_14 分阶段测试脚本。

阶段 1: 加载 9 轴 URDF 模型，可视化机器人初始姿态
阶段 2: 加载障碍物，可视化障碍物与机器人的相对位置
阶段 3: 启动 CBF 控制器，执行避障轨迹跟踪

每个阶段完成后暂停，在 PyBullet GUI 中确认无误后按 Enter 继续。
"""

import math
import time
import numpy as np
import pybullet as p

# 复用 3_14 中的所有类
from importlib import import_module as _im

_mod = _im("3_14_jaka_3d_cbf_qp_experiment")

ExperimentConfig = _mod.ExperimentConfig
SimulationScene = _mod.SimulationScene
JakaRobot = _mod.JakaRobot
create_obstacle = _mod.create_obstacle
LineSlerpTrajectory = _mod.LineSlerpTrajectory
create_controller = _mod.create_controller


def wait_for_user(msg: str):
    """在终端等待用户确认，同时保持 PyBullet 窗口响应。"""
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"  >>> 在 PyBullet 窗口中检查，确认无误后在终端按 Enter 继续 <<<")
    print(f"{'='*60}\n")
    try:
        input()
    except EOFError:
        pass


# ── 阶段 1: 加载机器人 ─────────────────────────────────────────────
def stage1_load_robot(cfg: ExperimentConfig):
    print("\n[阶段 1] 初始化仿真场景 & 加载 9 轴 URDF ...")
    scene = SimulationScene(cfg)
    robot = JakaRobot(cfg, scene)

    # 打印关节信息
    print(f"  总关节数: {robot.num_joints}")
    print(f"  可动关节: {robot.dof}  (移动副 {robot.n_pris} + 转动副 {robot.n_revo})")
    print(f"  末端 link index: {robot.ee_link_index}")
    print("  关节详情:")
    for ji in robot.active_joints:
        info = p.getJointInfo(robot.body_id, ji)
        jname = info[1].decode("utf-8")
        jtype = {p.JOINT_PRISMATIC: "移动", p.JOINT_REVOLUTE: "转动"}.get(info[2], "?")
        lo, hi = float(info[8]), float(info[9])
        print(f"    [{ji:2d}] {jname:20s}  {jtype}  限位=[{lo:.3f}, {hi:.3f}]")

    q, _ = robot.get_joint_state()
    ee_pos, ee_quat = robot.get_ee_pose()
    print(f"  初始关节角: {np.round(q, 4)}")
    print(f"  末端位置:   {np.round(ee_pos, 4)}")

    # 标记末端（绿色）
    scene.create_marker(0.015, (0.1, 0.9, 0.2, 0.9), ee_pos.tolist())

    # 标记各 link 原点（蓝色小球）
    for li in robot.cbf_link_indices:
        lp = robot.get_link_origin(li)
        scene.create_marker(0.008, (0.2, 0.4, 1.0, 0.7), lp.tolist())

    wait_for_user("阶段 1 完成 — 机器人已加载 (零位姿态)，绿色球=末端，蓝色球=各连杆原点")
    return scene, robot


# ── 阶段 2: 加载障碍物 ─────────────────────────────────────────────
def _create_sphere_near_ee(cfg, scene, ee_pos):
    """在末端附近创建球障碍物，滑块范围 ±0.5m，不受原始硬编码限制。"""
    r = cfg.sphere_radius
    init_pos = np.array([
        ee_pos[0],
        ee_pos[1] - 0.5,
        ee_pos[2] - 0.2,
    ])
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=cfg.sphere_rgba)
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
    bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                            baseVisualShapeIndex=vis, basePosition=init_pos.tolist())
    # 滑块范围：末端 ±0.5m
    sliders = {
        "x": p.addUserDebugParameter("obs_x", ee_pos[0] - 0.5, ee_pos[0] + 0.5, float(init_pos[0])),
        "y": p.addUserDebugParameter("obs_y", ee_pos[1] - 0.5, ee_pos[1] + 0.5, float(init_pos[1])),
        "z": p.addUserDebugParameter("obs_z", ee_pos[2] - 0.5, ee_pos[2] + 0.5, float(init_pos[2])),
    }

    # 包装成与 SphereObstacle 兼容的对象
    from types import SimpleNamespace
    obs = SimpleNamespace(
        _bid=bid, _r=r, sliders=sliders,
        body_id=bid,
    )
    obs.update_from_slider = lambda: _slider_update(obs)
    obs.get_position = lambda: np.array(p.getBasePositionAndOrientation(bid)[0], dtype=float)
    obs.compute_distance = lambda pt: _sphere_dist(obs, pt)
    obs.disable_collision_with = lambda rbid, nj: _disable_col(obs, rbid, nj)
    return obs


def _slider_update(obs):
    pos = np.array([p.readUserDebugParameter(obs.sliders[k]) for k in "xyz"])
    p.resetBasePositionAndOrientation(obs._bid, pos.tolist(), [0, 0, 0, 1])
    return pos


def _sphere_dist(obs, pt):
    d = pt - obs.get_position()
    dist = np.linalg.norm(d)
    if dist < 1e-9:
        return -obs._r, np.array([1, 0, 0.])
    return dist - obs._r, d / dist


def _disable_col(obs, robot_bid, nj):
    p.setCollisionFilterPair(robot_bid, obs._bid, -1, -1, enableCollision=0)
    for li in range(nj):
        p.setCollisionFilterPair(robot_bid, obs._bid, li, -1, enableCollision=0)


def stage2_load_obstacle(cfg, scene, robot):
    print("\n[阶段 2] 加载障碍物 ...")

    ee_pos, _ = robot.get_ee_pose()
    print(f"  当前末端位置: {np.round(ee_pos, 4)}")

    obs = _create_sphere_near_ee(cfg, scene, ee_pos)
    obs.disable_collision_with(robot.body_id, robot.num_joints)
    obs_pos = obs.update_from_slider()
    print(f"  障碍物类型: sphere (末端附近)")
    print(f"  障碍物位置: {np.round(obs_pos, 4)}")
    print(f"  末端↔障碍物距离: {np.linalg.norm(ee_pos - obs_pos)*1000:.1f} mm")

    # 检查各 link 到障碍物的距离
    q, dq = robot.get_joint_state()
    print("  各连杆到障碍物距离:")
    for li in robot.cbf_link_indices:
        cp = robot.get_closest_point_to_obstacle(li, obs.body_id)
        if cp is not None:
            _, dist, _ = cp
            tag = " ⚠ 过近!" if dist < cfg.safety_margin else ""
            print(f"    link {li:2d}: {dist*1000:7.1f} mm{tag}")
        else:
            print(f"    link {li:2d}: 无接触点")

    wait_for_user("阶段 2 完成 — 障碍物在末端 ±0.5m 范围内，可拖动滑块调整")
    return [obs]


# ── 阶段 3: IK → 轨迹 → CBF 控制 ──────────────────────────────────
def stage3_run_cbf(cfg, scene, robot, obstacles):
    print("\n[阶段 3] 求解 IK 起始位姿 → 构建轨迹 → 启动 CBF 控制器 ...")

    # ---- 3a: 根据障碍物位置计算起点/终点 ----
    obs_center = obstacles[0].update_from_slider()
    start_pos = np.array([
        obs_center[0] - cfg.line_half_span,
        obs_center[1] + cfg.line_bias_y,
        obs_center[2] + cfg.line_bias_z,
    ])
    goal_pos = np.array([
        obs_center[0] + cfg.line_half_span,
        obs_center[1] + cfg.line_bias_y,
        obs_center[2] + cfg.line_bias_z,
    ])
    sq = p.getQuaternionFromEuler([math.radians(v) for v in cfg.start_euler_deg])
    gq = p.getQuaternionFromEuler([math.radians(v) for v in cfg.goal_euler_deg])

    print(f"  目标起始位置: {np.round(start_pos, 4)}")
    print(f"  目标终止位置: {np.round(goal_pos, 4)}")

    # ---- 3b: IK 求解 + 验证 ----
    print("  求解 IK ...")
    robot.reset_to_pose(start_pos, sq)
    ee_pos, ee_quat = robot.get_ee_pose()
    ik_err = np.linalg.norm(ee_pos - start_pos)
    print(f"  IK 后末端位置: {np.round(ee_pos, 4)}")
    print(f"  IK 位置误差:   {ik_err*1000:.2f} mm")
    if ik_err > 0.01:
        print("  ⚠ IK 误差较大 (>10mm)，可能无解或精度不足！")
        print("    建议检查: 龙门吊初始位置 / 目标点是否在工作空间内")

    # 标记起点(绿)和终点(红)
    scene.create_marker(0.012, (0.1, 0.9, 0.2, 0.9), ee_pos.tolist())
    scene.create_marker(0.012, (0.95, 0.2, 0.2, 0.9), goal_pos.tolist())

    wait_for_user("阶段 3a — IK 已求解，绿球=实际末端，红球=目标终点。确认位姿合理后继续")

    # ---- 3c: 构建轨迹 ----
    traj = LineSlerpTrajectory(
        start_pos, np.array(sq, dtype=float),
        goal_pos, np.array(gq, dtype=float),
        cfg.trajectory_duration, cfg.dt,
    )

    # 画参考轨迹
    scene.draw_polyline(
        traj.reference_points(cfg.reference_samples),
        color=[0.85, 0.1, 0.1], width=1.2,
    )

    controller = create_controller(robot, cfg, traj)
    print(f"  控制器: {cfg.controller_type}")
    print(f"  轨迹时长: {cfg.trajectory_duration}s + 保持 {cfg.hold_duration}s")

    ee_pos, _ = robot.get_ee_pose()
    ee_marker = scene.create_marker(0.012, (0.1, 0.9, 0.2, 0.9), ee_pos.tolist())
    ref_marker = scene.create_marker(0.012, (0.95, 0.2, 0.2, 0.9), start_pos.tolist())
    prev_ee = ee_pos.copy()

    wait_for_user("阶段 3 准备就绪 — 红线=参考轨迹，按 Enter 开始运行控制器")

    total_steps = int((cfg.trajectory_duration + cfg.hold_duration) / cfg.dt)
    sim_step = 0

    try:
        while p.isConnected() and sim_step < total_steps:
            t = sim_step * cfg.dt
            q, dq = robot.get_joint_state()
            ee_pos, ee_quat = robot.get_ee_pose()
            ref_pos, ref_quat, ref_lv, ref_av = traj.sample(
                min(t, cfg.trajectory_duration))

            for obs in obstacles:
                obs.update_from_slider()

            u_cmd, info = controller.solve(
                q, dq, ee_pos, ee_quat,
                ref_pos, ref_quat, ref_lv, ref_av,
                obstacles, current_time=t,
            )

            aq = cfg.q_nominal_tracking
            robot.q_nominal = (1 - aq) * robot.q_nominal + aq * q
            robot.command_velocities(u_cmd)
            p.stepSimulation()

            # 可视化
            scene.update_marker(ee_marker, ee_pos.tolist())
            scene.update_marker(ref_marker, ref_pos.tolist())
            if np.linalg.norm(ee_pos - prev_ee) > 1e-3:
                p.addUserDebugLine(
                    prev_ee.tolist(), ee_pos.tolist(),
                    [0.1, 0.8, 0.2], lineWidth=1.5)
                prev_ee = ee_pos.copy()

            if sim_step % cfg.print_every == 0:
                gp = robot.get_gantry_pos()
                print(
                    f"  [step {sim_step:4d}] "
                    f"gantry=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                    f"err={info['tracking_error']*1000:.1f}mm "
                    f"h={info['min_h']*1000:.1f}mm "
                    f"{info['status']}"
                )

            sim_step += 1
            time.sleep(cfg.dt)

        # 轨迹结束，保持
        print("\n  轨迹执行完毕，保持末端位置 (Ctrl+C 退出) ...")
        robot.command_velocities(np.zeros(robot.dof))
        while p.isConnected():
            q, dq = robot.get_joint_state()
            ee_pos, ee_quat = robot.get_ee_pose()
            ref = traj.sample(cfg.trajectory_duration)
            for obs in obstacles:
                obs.update_from_slider()
            u_cmd, _ = controller.solve(
                q, dq, ee_pos, ee_quat, *ref, obstacles,
                current_time=cfg.trajectory_duration)
            robot.command_velocities(u_cmd)
            p.stepSimulation()
            time.sleep(1 / 60)

    except KeyboardInterrupt:
        print("\n  用户中断。")
    finally:
        if p.isConnected():
            p.disconnect()
        print(f"  仿真结束，共 {sim_step} 步。")


# ── 主入口 ─────────────────────────────────────────────────────────
def main():
    cfg = ExperimentConfig()

    # 阶段 1
    scene, robot = stage1_load_robot(cfg)

    # 阶段 2
    obstacles = stage2_load_obstacle(cfg, scene, robot)

    # 阶段 3
    stage3_run_cbf(cfg, scene, robot, obstacles)


if __name__ == "__main__":
    main()
