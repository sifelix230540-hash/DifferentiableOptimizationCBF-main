"""区域内 Bézier 曲线采样、碰撞评估与 PyBullet GUI 轨迹回放。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

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


def _capture_gui_frame(width: int, height: int) -> np.ndarray:
    cam = p.getDebugVisualizerCamera()
    view_matrix = cam[2]
    proj_matrix = cam[3]
    _, _, rgb, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    return np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


def _append_video_frame(video_frames: list[np.ndarray] | None, width: int, height: int):
    if video_frames is None:
        return
    video_frames.append(_capture_gui_frame(width, height))


def _save_video_frames(out_path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    if imageio is None:
        raise RuntimeError("未安装 imageio，无法输出视频。")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = frames or []
    if not data:
        print("[GUI] 警告：无视频帧，跳过保存视频。")
        return
    uri = str(out_path.resolve())
    fps_i = int(fps)
    if out_path.suffix.lower() == ".mp4":
        try:
            imageio.mimsave(
                uri,
                data,
                fps=fps_i,
                format="FFMPEG",
                codec="libx264",
                macro_block_size=1,
                quality=8,
            )
        except (ImportError, ValueError, OSError, TypeError):
            try:
                import cv2

                h, w = data[0].shape[:2]
                writer = cv2.VideoWriter(
                    uri,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps_i),
                    (w, h),
                )
                if not writer.isOpened():
                    raise RuntimeError("cv2.VideoWriter 无法打开")
                try:
                    for fr in data:
                        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                        writer.write(bgr)
                finally:
                    writer.release()
            except Exception:
                gif_path = out_path.with_suffix(".gif")
                imageio.mimsave(str(gif_path.resolve()), data, fps=fps_i)
                print(
                    "[GUI] 未安装 imageio[ffmpeg] 且 OpenCV 写 MP4 失败，已改为 GIF: "
                    f"{gif_path}"
                )
                return
    else:
        imageio.mimsave(uri, data, fps=fps_i)
    print(f"[GUI] 视频已保存: {out_path}")


def _play_curve_on_scene(
    robot,
    metadata,
    curve_report: dict,
    *,
    sleep_dt: float,
    status_ids: list[int],
    curve_label: str = "curve",
    video_frames: list[np.ndarray] | None = None,
    video_width: int = 1280,
    video_height: int = 720,
):
    num_steps = len(curve_report["curve"])
    for step_idx, q6 in enumerate(np.asarray(curve_report["curve"], dtype=float), start=1):
        q_full = compose_full_q(metadata, q6)
        robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        base_pos, _ = robot.get_robobase_pose()
        anchor = np.asarray(base_pos, dtype=float) + np.array([0.0, -0.36, 0.85], dtype=float)
        step_report = curve_report["steps"][step_idx - 1]
        lines = [
            "region_growth=IRIS-ZO",
            f"id={curve_label}",
            f"step={step_idx}/{num_steps}",
            f"curve_min_clearance={float(curve_report['min_clearance']):+.6f}",
            f"step_clearance={float(step_report['min_clearance']):+.6f}",
            f"collision={bool(step_report['is_collision'])}",
            f"worst_pair={curve_report.get('worst_pair')}",
        ]
        while len(status_ids) < len(lines):
            status_ids.append(-1)
        for idx, line in enumerate(lines):
            pos = (anchor + np.array([0.0, 0.0, -0.07 * idx], dtype=float)).tolist()
            status_ids[idx] = p.addUserDebugText(
                line,
                pos,
                textColorRGB=[0.08, 0.08, 0.08],
                textSize=1.15,
                replaceItemUniqueId=status_ids[idx],
            )
        _append_video_frame(video_frames, video_width, video_height)
        time.sleep(float(sleep_dt))


def playback_curve_gui(
    robot_cfg,
    curve_report: dict,
    *,
    sleep_dt: float = 0.15,
    hold_seconds: float = 5.0,
    video_output_path: str | Path | None = None,
    video_fps: int = 15,
    video_width: int = 1280,
    video_height: int = 720,
):
    if p.isConnected():
        p.disconnect()
    scene = SimulationScene(load_config(robot_cfg.CFG_PATH))
    scene.enable_rendering()
    robot, metadata, _ = load_robot_metadata(robot_cfg)
    status_ids = [-1] * 7
    video_frames = [] if video_output_path else None
    try:
        _play_curve_on_scene(
            robot,
            metadata,
            curve_report,
            sleep_dt=sleep_dt,
            status_ids=status_ids,
            curve_label="curve-1",
            video_frames=video_frames,
            video_width=video_width,
            video_height=video_height,
        )
        print(f"[GUI] 播放结束，保持窗口 {hold_seconds}s ...")
        end_time = time.time() + max(float(hold_seconds), 0.0)
        while p.isConnected() and time.time() < end_time:
            _append_video_frame(video_frames, video_width, video_height)
            time.sleep(1.0 / 30.0)
    finally:
        if p.isConnected():
            p.disconnect()
    if video_output_path:
        _save_video_frames(Path(video_output_path), video_frames or [], fps=int(video_fps))


def playback_multiple_curves_gui(
    robot_cfg,
    curve_reports: list[dict],
    *,
    sleep_dt: float = 0.15,
    hold_seconds: float = 5.0,
    between_curves_seconds: float = 0.8,
    video_output_path: str | Path | None = None,
    video_fps: int = 15,
    video_width: int = 1280,
    video_height: int = 720,
):
    if not curve_reports:
        raise ValueError("curve_reports 为空，无法回放。")
    if p.isConnected():
        p.disconnect()
    scene = SimulationScene(load_config(robot_cfg.CFG_PATH))
    scene.enable_rendering()
    robot, metadata, _ = load_robot_metadata(robot_cfg)
    status_ids = [-1] * 7
    video_frames = [] if video_output_path else None
    try:
        for curve_idx, curve_report in enumerate(curve_reports, start=1):
            _play_curve_on_scene(
                robot,
                metadata,
                curve_report,
                sleep_dt=sleep_dt,
                status_ids=status_ids,
                curve_label=f"curve-{curve_idx}/{len(curve_reports)}",
                video_frames=video_frames,
                video_width=video_width,
                video_height=video_height,
            )
            t_end = time.time() + max(float(between_curves_seconds), 0.0)
            while p.isConnected() and time.time() < t_end:
                _append_video_frame(video_frames, video_width, video_height)
                time.sleep(1.0 / 30.0)
        print(f"[GUI] 多段播放结束，保持窗口 {hold_seconds}s ...")
        end_time = time.time() + max(float(hold_seconds), 0.0)
        while p.isConnected() and time.time() < end_time:
            _append_video_frame(video_frames, video_width, video_height)
            time.sleep(1.0 / 30.0)
    finally:
        if p.isConnected():
            p.disconnect()
    if video_output_path:
        _save_video_frames(Path(video_output_path), video_frames or [], fps=int(video_fps))


def replay_from_json(
    experiment_json: str | Path,
    *,
    sleep_dt: float = 0.15,
    hold_seconds: float = 5.0,
    robot_cfg=None,
    video_output_path: str | Path | None = None,
    video_fps: int = 15,
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
    playback_curve_gui(
        robot_cfg,
        curve_report,
        sleep_dt=sleep_dt,
        hold_seconds=hold_seconds,
        video_output_path=video_output_path,
        video_fps=video_fps,
    )

