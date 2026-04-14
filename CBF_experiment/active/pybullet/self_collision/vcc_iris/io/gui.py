"""区域内 Bézier 曲线采样、碰撞评估与 PyBullet GUI 轨迹回放。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pybullet as p

from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import SimulationScene, load_config
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.robot_model import compose_full_q, load_robot_metadata


def sample_curve_in_region(region, *, num_points: int, rng: np.random.Generator) -> np.ndarray:
    center = np.asarray(region.center, dtype=float).reshape(-1)
    C = np.asarray(region.C, dtype=float)
    A = np.asarray(region.A, dtype=float)
    b = np.asarray(region.b, dtype=float)
    dim = center.shape[0]
    for _ in range(64):
        dirs = rng.normal(size=(4, dim))
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        radii = rng.random((4, 1)) ** (1.0 / max(dim, 1))
        controls = center.reshape(1, -1) + (dirs @ C.T) * radii * 0.8
        if not np.all(controls @ A.T <= b.reshape(1, -1) + 1e-9):
            continue
        t = np.linspace(0.0, 1.0, int(num_points), dtype=float).reshape(-1, 1)
        omt = 1.0 - t
        curve = (
            (omt ** 3) * controls[0].reshape(1, -1)
            + 3.0 * (omt ** 2) * t * controls[1].reshape(1, -1)
            + 3.0 * omt * (t ** 2) * controls[2].reshape(1, -1)
            + (t ** 3) * controls[3].reshape(1, -1)
        )
        if np.all(curve @ A.T <= b.reshape(1, -1) + 1e-9):
            return curve
    raise RuntimeError("Failed to sample a collision-free candidate curve inside region polytope.")


def evaluate_curve(oracle, curve: np.ndarray) -> dict:
    min_clearance = float("inf")
    worst_pair = None
    any_collision = False
    step_reports = []
    for step_idx, q in enumerate(np.asarray(curve, dtype=float), start=1):
        metric = oracle.query(q)
        any_collision = any_collision or bool(metric["is_collision"])
        if float(metric["min_clearance"]) < min_clearance:
            min_clearance = float(metric["min_clearance"])
            worst_pair = metric["active_pair"]
        step_reports.append({
            "step": int(step_idx),
            "min_clearance": float(metric["min_clearance"]),
            "is_collision": bool(metric["is_collision"]),
            "active_pair": list(metric["active_pair"]) if metric["active_pair"] else None,
        })
    return {
        "curve": np.asarray(curve, dtype=float).tolist(),
        "steps": step_reports,
        "min_clearance": float(min_clearance),
        "worst_pair": list(worst_pair) if worst_pair else None,
        "any_collision": bool(any_collision),
    }


def playback_curve_gui(robot_cfg, curve_report: dict, *, sleep_dt: float = 0.15, hold_seconds: float = 5.0):
    if p.isConnected():
        p.disconnect()
    scene = SimulationScene(load_config(robot_cfg.CFG_PATH))
    scene.enable_rendering()
    robot, metadata, _ = load_robot_metadata(robot_cfg)
    num_steps = len(curve_report["curve"])
    status_ids = [-1] * 6
    try:
        for step_idx, q6 in enumerate(np.asarray(curve_report["curve"], dtype=float), start=1):
            q_full = compose_full_q(metadata, q6)
            robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
            base_pos, _ = robot.get_robobase_pose()
            anchor = np.asarray(base_pos, dtype=float) + np.array([0.0, -0.36, 0.85], dtype=float)
            step_report = curve_report["steps"][step_idx - 1]
            lines = [
                "region_growth=IRIS-ZO",
                f"step={step_idx}/{num_steps}",
                f"curve_min_clearance={float(curve_report['min_clearance']):+.6f}",
                f"step_clearance={float(step_report['min_clearance']):+.6f}",
                f"collision={bool(step_report['is_collision'])}",
                f"worst_pair={curve_report.get('worst_pair')}",
            ]
            for idx, line in enumerate(lines):
                pos = (anchor + np.array([0.0, 0.0, -0.07 * idx], dtype=float)).tolist()
                status_ids[idx] = p.addUserDebugText(
                    line,
                    pos,
                    textColorRGB=[0.08, 0.08, 0.08],
                    textSize=1.15,
                    replaceItemUniqueId=status_ids[idx],
                )
            time.sleep(float(sleep_dt))
        print(f"[GUI] 播放结束，保持窗口 {hold_seconds}s ...")
        end_time = time.time() + max(float(hold_seconds), 0.0)
        while p.isConnected() and time.time() < end_time:
            time.sleep(1.0 / 30.0)
    finally:
        if p.isConnected():
            p.disconnect()


def replay_from_json(
    experiment_json: str | Path,
    *,
    sleep_dt: float = 0.15,
    hold_seconds: float = 5.0,
    robot_cfg=None,
):
    """从已有的 experiment JSON 文件直接启动 GUI 重播，无需重新规划。"""
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig

    path = Path(experiment_json)
    if not path.exists():
        raise FileNotFoundError(f"找不到实验文件: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    curve_report = data.get("curve_report")
    if not curve_report or not curve_report.get("curve"):
        raise ValueError(f"实验文件中缺少有效的 curve_report: {path}")
    if robot_cfg is None:
        robot_cfg = RobotQueryConfig()
    print(f"[replay] 从 {path.name} 加载曲线 ({len(curve_report['curve'])} 步)")
    print(f"[replay] min_clearance={curve_report['min_clearance']:.6f}  collision={curve_report['any_collision']}")
    playback_curve_gui(robot_cfg, curve_report, sleep_dt=sleep_dt, hold_seconds=hold_seconds)

