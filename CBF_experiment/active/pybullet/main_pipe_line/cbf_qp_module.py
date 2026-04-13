from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from CBF_experiment.active.pybullet.configuration_metrics import summarize_clearance_entries


def _bounds_from_robot(robot, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    # 从配置里读取机器人速度上界:
    # 前 `n_pris` 个自由度对应底座/龙门等移动轴,
    # 后 `n_revo` 个自由度对应转动关节。
    robot_cfg = cfg.get("robot", {})
    base_vel_limit = float(robot_cfg.get("base_vel_limit", 0.4))
    dq_limit = float(robot_cfg.get("dq_limit", 1.0))

    # 组装控制输入上下界:
    # u = [prismatic_joint_velocities, revolute_joint_velocities]。
    lb = np.concatenate([
        np.full(int(robot.n_pris), -base_vel_limit, dtype=float),
        np.full(int(robot.n_revo), -dq_limit, dtype=float),
    ])

    # 上界默认与下界对称。
    ub = -lb
    return lb, ub


def _build_cbf_rows(robot, q, geometry_engine, cfg: dict):
    # 这里把几何模块输出的“采样点到障碍物的距离/法向”
    # 转成 QP 里的线性 CBF 不等式:
    #     row @ u + rhs >= 0
    # 其中 row 由点 Jacobian 在障碍法向上的投影得到。
    control_cfg = cfg.get("trajectory_planning", {})
    safety_margin = float(cfg.get("control", {}).get("safety_margin", 0.005))
    self_collision_margin = float(cfg.get("control", {}).get("self_collision_margin", safety_margin))
    cbf_alpha = float(control_cfg.get("cbf_alpha", 1.0))

    # 几何模块负责:
    # 1. 选取哪些连杆/采样点参与 CBF
    # 2. 返回每个点的 signed distance 和 outward normal
    distances, normals, link_indices = geometry_engine.get_cbf_distances(robot, q)

    if len(link_indices) == 0:
        # 没有任何需要约束的点时, 返回空约束矩阵。
        return np.zeros((0, robot.dof), dtype=float), np.zeros(0, dtype=float)

    rows = []
    rhs = []

    # 约定 geometry_engine 除了返回距离和法向, 还要把本次查询用到的世界坐标采样点
    # 写到 `last_query_points` 里, 这样控制层才能在“那个点”上算 Jacobian。
    sample_points = getattr(geometry_engine, "last_query_points", None)
    query_meta = getattr(geometry_engine, "last_query_meta", None)
    if sample_points is None:
        raise RuntimeError("GeometryEngine.get_cbf_distances() must populate last_query_points")

    for idx, link_index in enumerate(link_indices):
        meta = None
        if isinstance(query_meta, list) and idx < len(query_meta):
            meta = query_meta[idx]
        # 当前这条 CBF 对应的采样点世界坐标。
        world_point = np.asarray(sample_points[idx], dtype=float).reshape(3)

        # 障碍物表面法向, 后面会投影到该点的线速度 Jacobian 上。
        normal = np.asarray(normals[idx], dtype=float).reshape(3)

        if meta is not None and str(meta.get("kind", "")) == "self_collision":
            point_a = np.asarray(meta.get("point_on_link", world_point), dtype=float).reshape(3)
            point_b = np.asarray(meta.get("point_on_other_link"), dtype=float).reshape(3)
            link_b = int(meta.get("other_link_index"))
            if hasattr(robot, "get_link_linear_jacobian_at_world_point"):
                jt_a = robot.get_link_linear_jacobian_at_world_point(
                    int(link_index),
                    point_a,
                    q,
                    np.zeros_like(q),
                )
                jt_b = robot.get_link_linear_jacobian_at_world_point(
                    link_b,
                    point_b,
                    q,
                    np.zeros_like(q),
                )
            else:
                jt_a, _jr = robot.get_link_jacobian(int(link_index), q, np.zeros_like(q))
                jt_b, _jr = robot.get_link_jacobian(link_b, q, np.zeros_like(q))
            row = normal.reshape(1, 3) @ (np.asarray(jt_a, dtype=float) - np.asarray(jt_b, dtype=float))
            rows.append(row.reshape(-1))
            rhs.append(cbf_alpha * float(np.asarray(distances[idx], dtype=float) - self_collision_margin))
        else:
            # 在“当前采样点世界坐标”上计算线速度 Jacobian。
            if hasattr(robot, "get_link_linear_jacobian_at_world_point"):
                jt = robot.get_link_linear_jacobian_at_world_point(
                    int(link_index),
                    world_point,
                    q,
                    np.zeros_like(q),
                )
            else:
                jt, _jr = robot.get_link_jacobian(int(link_index), q, np.zeros_like(q))

            # 法向速度 = normal^T * J_trans * u, 所以 CBF 行向量就是 normal^T * J_trans。
            row = normal.reshape(1, 3) @ np.asarray(jt, dtype=float)
            rows.append(row.reshape(-1))

            # 对应的一阶 CBF 条件写成:
            #     row @ u + alpha * (h - margin) >= 0
            # 这里 h 用 signed distance 近似。
            rhs.append(cbf_alpha * float(np.asarray(distances[idx], dtype=float) - safety_margin))
    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def solve_cbf_qp_step(
    robot,
    q: np.ndarray,
    dq: np.ndarray,
    *,
    dt: float | None = None,
    pos_ref: np.ndarray | None = None,
    quat_ref: np.ndarray | None = None,
    q_ref: np.ndarray | None = None,
    geometry_engine=None,
    cfg: dict | None = None,
):
    # 单步 CBF-QP 求解器:
    # 输入当前状态 q, dq 和参考目标, 输出当前时刻速度命令 u_cmd。
    #
    # 整体流程:
    # 1. 先构造一个“名义控制” u_nominal
    # 2. 再把 CBF 安全约束加入 QP
    # 3. 用 SLSQP 求解最接近 u_nominal 的安全控制量
    cfg = {} if cfg is None else cfg

    # 强制把输入整理成一维 numpy 向量, 避免后面矩阵计算维度不一致。
    q = np.asarray(q, dtype=float).reshape(-1)
    dq = np.asarray(dq, dtype=float).reshape(-1)

    # 获取控制输入边界。
    lb, ub = _bounds_from_robot(robot, cfg)

    # 读取跟踪相关超参数。
    tracking_cfg = cfg.get("trajectory_planning", {})
    pos_gain = float(tracking_cfg.get("tracking_gain_pos", 1.0))
    ori_gain = float(tracking_cfg.get("tracking_gain_ori", 3.0))
    nullspace_weight = float(tracking_cfg.get("nullspace_weight", 1e-3))
    joint_nominal_scale = float(tracking_cfg.get("joint_nominal_scale", 0.35))
    cartesian_nominal_scale = float(tracking_cfg.get("cartesian_nominal_scale", 0.5))

    # 控制时间步长优先级:
    # 显式传入 dt > trajectory_planning.dt > control.mpc_dt > simulation.dt。
    control_dt = float(
        dt
        if dt is not None
        else tracking_cfg.get(
            "dt",
            cfg.get("control", {}).get(
                "mpc_dt",
                cfg.get("simulation", {}).get("dt", 1.0 / 240.0),
            ),
        )
    )

    # 两种参考模式:
    # 1. 给定关节参考 q_ref
    # 2. 给定末端位姿参考 (pos_ref, quat_ref)
    mode = "joint_ref" if q_ref is not None else "cartesian_ref"

    if q_ref is not None:
        # 关节参考模式:
        # 直接把“当前位置到目标关节”的一阶差分速度作为名义控制。
        q_ref = np.asarray(q_ref, dtype=float).reshape(-1)
        u_nominal = joint_nominal_scale * (q_ref - q) / max(control_dt, 1e-6)

        def objective(u):
            # QP 目标: 找一个尽量接近关节名义速度的控制量。
            u = np.asarray(u, dtype=float).reshape(-1)
            return float(np.sum((u - u_nominal) ** 2))

        # 先把名义解裁剪到输入上下界, 作为优化初值。
        x0 = np.clip(u_nominal, lb, ub)
    else:
        # 笛卡尔参考模式:
        # 先算末端位置误差和姿态误差, 再反推一个期望关节速度。
        ee_pos, ee_quat = robot.get_ee_pose()
        pos_err = np.asarray(pos_ref, dtype=float).reshape(3) - np.asarray(ee_pos, dtype=float).reshape(3)
        rot_err = (
            Rotation.from_quat(np.asarray(quat_ref, dtype=float).reshape(4))
            * Rotation.from_quat(np.asarray(ee_quat, dtype=float).reshape(4)).inv()
        ).as_rotvec()

        # 位置和姿态分别乘增益, 得到期望末端 twist。
        xdot_ref = cartesian_nominal_scale * np.concatenate([pos_gain * pos_err, ori_gain * rot_err])

        # 末端几何 Jacobian: [Jv; Jw]。
        j_ee = np.asarray(robot.get_ee_jacobian(q, dq), dtype=float)

        # 最小二乘得到“最接近目标末端速度”的名义关节速度。
        u_nominal = np.linalg.lstsq(j_ee, xdot_ref, rcond=None)[0]

        def objective(u):
            # 目标函数 = 末端速度跟踪误差 + 一个很小的控制正则项。
            # 第二项用于抑制过大关节速度, 同时改善数值稳定性。
            u = np.asarray(u, dtype=float).reshape(-1)
            return float(np.sum((j_ee @ u - xdot_ref) ** 2) + nullspace_weight * np.sum(u ** 2))

        x0 = np.clip(u_nominal, lb, ub)

    # 开始构造 CBF 不等式约束。
    constraints = []
    min_h = np.inf
    clearance_summary = summarize_clearance_entries([])
    if geometry_engine is not None:
        # 从几何模块拿到:
        # rows: 每一条 CBF 的线性项
        # rhs : 每一条 CBF 的常数项
        rows, rhs = _build_cbf_rows(robot, q, geometry_engine, cfg)
        clearance_summary = summarize_clearance_entries(getattr(geometry_engine, "last_query_meta", []))
        if rhs.size:
            # 记录当前最危险的 barrier 值, 便于日志和可视化。
            min_h = float(np.min(rhs))
        for i in range(rows.shape[0]):
            row = rows[i].copy()
            b = float(rhs[i])

            def ineq(u, row=row, b=b):
                # SciPy SLSQP 的不等式约束形式是:
                #     fun(u) >= 0
                # 因此这里直接实现 row @ u + b >= 0。
                return float(row @ np.asarray(u, dtype=float).reshape(-1) + b)

            constraints.append({"type": "ineq", "fun": ineq})

    # 求解带边界和 CBF 约束的非线性优化。
    # 由于 objective 是二次型、约束是线性的, 本质上它是一个 QP,
    # 这里只是借助 SLSQP 这个通用求解器来算。
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=list(zip(lb.tolist(), ub.tolist())),
        constraints=constraints,
        options={"maxiter": 80, "ftol": 1e-6},
    )
    if result.success:
        # 求解成功时, 直接采用最优解。
        u_cmd = np.asarray(result.x, dtype=float).reshape(-1)
        status = "cbf_optimal"
    else:
        # 如果带 CBF 的优化失败, 退回到裁剪后的名义控制。
        # 这一步能保证控制器至少还能继续工作, 但安全性不再严格保证。
        u_cmd = np.clip(u_nominal, lb, ub)
        status = "cbf_fallback"
    return u_cmd, {
        # status: 本次是否真正解出了 CBF-QP
        "status": status,
        # mode: 当前使用的是关节参考还是笛卡尔参考
        "mode": mode,
        # min_h: 当前最危险约束的 barrier 值, 越小越危险
        "min_h": float(min_h if np.isfinite(min_h) else 0.0),
        # cbf_active: 本次是否真的加入了 CBF 约束
        "cbf_active": bool(len(constraints) > 0),
        # clearance_summary: 复用几何查询结果导出的净空摘要，为后续在线二级目标留接口
        "clearance_summary": clearance_summary,
    }
