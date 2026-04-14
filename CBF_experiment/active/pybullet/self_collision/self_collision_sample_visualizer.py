from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.self_collision.self_collision_cspace_benchmark import BenchmarkParameters  # noqa: E402
from CBF_experiment.active.pybullet.self_collision.self_collision_cspace_hulls import (  # noqa: E402
    build_monitored_link_pairs,
    classify_self_collision_sample,
    extract_revolute_metadata,
    extract_self_collision_monitor_metadata,
)
from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import SimulationScene, Robot, load_config, _resolve  # noqa: E402


@dataclass(frozen=True)
class VisualizationParameters:
    CFG_PATH: str | None = None
    SAMPLE_JSON: str = str(Path(_resolve("artifacts/sdf_exp/self_collision_eval_samples.json")))
    OUTPUT_JSON: str = str(Path(_resolve("artifacts/sdf_exp/self_collision_sample_gallery.json")))
    NUM_COLLISION: int = 3
    NUM_FREE: int = 3
    MIN_INDEX_GAP: int = int(BenchmarkParameters.MIN_INDEX_GAP)
    QUERY_DISTANCE: float = float(BenchmarkParameters.QUERY_DISTANCE)
    PENETRATION_THRESH: float = float(BenchmarkParameters.PENETRATION_THRESH)
    CAMERA_DISTANCE: float = 1.45
    CAMERA_YAW: float = -220.0
    CAMERA_PITCH: float = -28.0
    SLEEP_DT: float = 1.0 / 30.0


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _select_representative_indices(distances: np.ndarray, count: int, *, prefer_small_abs: bool) -> list[int]:
    dist = np.asarray(distances, dtype=float).reshape(-1)
    if dist.size == 0 or count <= 0:
        return []
    order = np.argsort(np.abs(dist) if prefer_small_abs else dist)
    candidates: list[int] = []
    anchor_positions = np.linspace(0, max(order.size - 1, 0), num=min(count, order.size))
    for pos in anchor_positions:
        idx = int(order[int(round(float(pos)))])
        if idx not in candidates:
            candidates.append(idx)
    if len(candidates) < min(count, order.size):
        for idx in order.tolist():
            if int(idx) not in candidates:
                candidates.append(int(idx))
            if len(candidates) >= min(count, order.size):
                break
    return candidates[: min(count, order.size)]


def select_samples(sample_payload: dict, *, num_collision: int, num_free: int) -> list[dict]:
    collision_samples = np.asarray(sample_payload.get("collision_samples", []), dtype=float)
    free_samples = np.asarray(sample_payload.get("free_samples", []), dtype=float)
    if collision_samples.ndim == 1 and collision_samples.size:
        collision_samples = collision_samples.reshape(1, -1)
    if free_samples.ndim == 1 and free_samples.size:
        free_samples = free_samples.reshape(1, -1)
    collision_dist = np.asarray(sample_payload.get("collision_distances", []), dtype=float).reshape(-1)
    free_dist = np.asarray(sample_payload.get("free_distances", []), dtype=float).reshape(-1)

    selected: list[dict] = []
    for idx in _select_representative_indices(collision_dist, num_collision, prefer_small_abs=False):
        selected.append({
            "label": "collision",
            "sample_index": int(idx),
            "revolute_q": collision_samples[int(idx)].tolist(),
            "stored_min_distance": float(collision_dist[int(idx)]),
        })
    for idx in _select_representative_indices(free_dist, num_free, prefer_small_abs=True):
        selected.append({
            "label": "free",
            "sample_index": int(idx),
            "revolute_q": free_samples[int(idx)].tolist(),
            "stored_min_distance": float(free_dist[int(idx)]),
        })
    return selected


def _compose_full_q(q_base: np.ndarray, q_indices: list[int], revolute_q) -> np.ndarray:
    q_full = np.asarray(q_base, dtype=float).copy()
    q_full[np.asarray(q_indices, dtype=int)] = np.asarray(revolute_q, dtype=float).reshape(-1)
    return q_full


def _pick_camera_target(robot) -> list[float]:
    ee_pos, _ee_quat = robot.get_ee_pose()
    base_pos, _base_quat = robot.get_robobase_pose()
    target = 0.65 * np.asarray(ee_pos, dtype=float) + 0.35 * np.asarray(base_pos, dtype=float)
    target[2] = max(float(target[2]), 0.35)
    return target.tolist()


def _pair_to_text(active_pair, link_names_by_id: dict[int, str]) -> str:
    if active_pair is None:
        return "none"
    names = [link_names_by_id.get(int(link_id), f"link_{int(link_id)}") for link_id in active_pair]
    return f"{list(active_pair)} ({names[0]} <-> {names[1]})"


def _is_triggered(keys: dict, key_code: int) -> bool:
    return bool(keys.get(int(key_code), 0) & p.KEY_WAS_TRIGGERED)


def visualize_selected_samples_gui(params=VisualizationParameters) -> dict:
    sample_payload = load_json(params.SAMPLE_JSON)
    selected = select_samples(
        sample_payload,
        num_collision=int(params.NUM_COLLISION),
        num_free=int(params.NUM_FREE),
    )
    if not selected:
        raise ValueError("No samples available for visualization.")

    cfg = load_config(params.CFG_PATH)
    scene = SimulationScene(cfg)
    scene.enable_rendering()
    robot = Robot(cfg)
    q_base, dq_base = robot.get_joint_state()
    revolute_ids, revolute_names, _joint_limits, q_indices = extract_revolute_metadata(robot)
    monitored_link_ids, monitored_link_names = extract_self_collision_monitor_metadata(robot)
    monitored_pairs = build_monitored_link_pairs(monitored_link_ids, min_index_gap=int(params.MIN_INDEX_GAP))
    from CBF_experiment.active.pybullet.self_collision.self_collision_backend_coal import (
        build_coal_link_models,
    )
    link_models = build_coal_link_models(robot, monitored_link_ids)
    link_names_by_id = {
        int(link_id): str(name)
        for link_id, name in zip(monitored_link_ids, monitored_link_names)
    }

    inspected: list[dict] = []
    for item in selected:
        q_full = _compose_full_q(q_base, q_indices, item["revolute_q"])
        robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        metric = classify_self_collision_sample(
            robot,
            monitored_pairs=monitored_pairs,
            link_models=link_models,
            penetration_thresh=float(params.PENETRATION_THRESH),
        )
        inspected.append({
            **item,
            "recomputed_min_distance": float(metric["min_distance"]),
            "recomputed_is_collision": bool(metric["is_collision"]),
            "active_pair": metric.get("active_pair"),
        })

    status_ids = [-1, -1, -1, -1]
    current_idx = 0
    last_rendered_idx = None

    def render_current(index: int) -> None:
        nonlocal status_ids, last_rendered_idx
        item = inspected[index]
        q_full = _compose_full_q(q_base, q_indices, item["revolute_q"])
        robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        camera_target = _pick_camera_target(robot)
        p.resetDebugVisualizerCamera(
            cameraDistance=float(params.CAMERA_DISTANCE),
            cameraYaw=float(params.CAMERA_YAW),
            cameraPitch=float(params.CAMERA_PITCH),
            cameraTargetPosition=camera_target,
        )
        base_pos, _base_quat = robot.get_robobase_pose()
        anchor = np.asarray(base_pos, dtype=float) + np.array([0.00, -0.35, 0.85], dtype=float)
        pair_text = _pair_to_text(item["active_pair"], link_names_by_id)
        lines = [
            f"[{index + 1}/{len(inspected)}] label={item['label']} sample_index={item['sample_index']}",
            f"stored_min_distance={item['stored_min_distance']:+.6f}  recomputed_min_distance={item['recomputed_min_distance']:+.6f}",
            f"is_collision={item['recomputed_is_collision']}  active_pair={pair_text}",
            "Controls: N/Right=next, P/Left=prev, Q=quit",
        ]
        for text_idx, line in enumerate(lines):
            text_pos = (anchor + np.array([0.0, 0.0, -0.08 * text_idx], dtype=float)).tolist()
            status_ids[text_idx] = p.addUserDebugText(
                line,
                text_pos,
                textColorRGB=[0.08, 0.08, 0.08],
                textSize=1.2,
                replaceItemUniqueId=status_ids[text_idx],
            )
        print(
            "[self-cspace-vis] "
            f"show {index + 1}/{len(inspected)} | "
            f"label={item['label']} | "
            f"stored={item['stored_min_distance']:+.6f} | "
            f"recomputed={item['recomputed_min_distance']:+.6f} | "
            f"pair={pair_text}"
        )
        last_rendered_idx = int(index)

    output_payload = {
        "sample_json": str(Path(params.SAMPLE_JSON)),
        "joint_indices": [int(j) for j in revolute_ids],
        "joint_names": [str(name) for name in revolute_names],
        "monitored_link_indices": [int(j) for j in monitored_link_ids],
        "monitored_link_names": [str(name) for name in monitored_link_names],
        "selected_samples": inspected,
    }
    output_json = Path(params.OUTPUT_JSON)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[self-cspace-vis] metadata json -> {output_json}")
    print("[self-cspace-vis] GUI ready. Use N/P/Q or arrow keys.")

    try:
        while p.isConnected():
            if last_rendered_idx != current_idx:
                render_current(current_idx)
            keys = p.getKeyboardEvents()
            if _is_triggered(keys, ord("q")) or _is_triggered(keys, ord("Q")):
                break
            if _is_triggered(keys, ord("n")) or _is_triggered(keys, ord("N")) or _is_triggered(keys, p.B3G_RIGHT_ARROW):
                current_idx = (current_idx + 1) % len(inspected)
            elif _is_triggered(keys, ord("p")) or _is_triggered(keys, ord("P")) or _is_triggered(keys, p.B3G_LEFT_ARROW):
                current_idx = (current_idx - 1) % len(inspected)
            time.sleep(float(params.SLEEP_DT))
    finally:
        robot.set_joint_state(q_base, dq_base)
        if p.isConnected():
            p.disconnect()
    return output_payload


if __name__ == "__main__":
    visualize_selected_samples_gui(VisualizationParameters)
