from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import SimulationScene, Robot, load_config  # noqa: E402
from CBF_experiment.active.pybullet.self_collision.self_collision_coal_validation import (  # noqa: E402
    CoalValidationParameters,
)


@dataclass(frozen=True)
class ComparisonVisualizationParameters:
    CFG_PATH: str | None = None
    VALIDATION_JSON: str = str(Path(CoalValidationParameters.OUTPUT_JSON))
    DISAGREEMENTS_FIRST: bool = True
    ONLY_DISAGREEMENTS: bool = False
    CAMERA_DISTANCE: float = 1.45
    CAMERA_YAW: float = -220.0
    CAMERA_PITCH: float = -28.0
    SLEEP_DT: float = 1.0 / 30.0


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compose_full_q(q_base: np.ndarray, q_indices: list[int], revolute_q) -> np.ndarray:
    q_full = np.asarray(q_base, dtype=float).copy()
    q_full[np.asarray(q_indices, dtype=int)] = np.asarray(revolute_q, dtype=float).reshape(-1)
    return q_full


def _pick_camera_target(robot) -> list[float]:
    ee_pos, _ = robot.get_ee_pose()
    base_pos, _ = robot.get_robobase_pose()
    target = 0.65 * np.asarray(ee_pos, dtype=float) + 0.35 * np.asarray(base_pos, dtype=float)
    target[2] = max(float(target[2]), 0.35)
    return target.tolist()


def _is_triggered(keys: dict, key_code: int) -> bool:
    return bool(keys.get(int(key_code), 0) & p.KEY_WAS_TRIGGERED)


def _pair_to_text(active_pair, joint_names_by_id: dict[int, str]) -> str:
    if not active_pair:
        return "none"
    a, b = [int(x) for x in active_pair]
    return f"[{a}, {b}] ({joint_names_by_id.get(a, a)} <-> {joint_names_by_id.get(b, b)})"


def _format_optional_float(value) -> str:
    if value is None:
        return "none"
    return f"{float(value):+.6f}"


def _top_pair_lines(item: dict, joint_names_by_id: dict[int, str], *, limit: int = 3) -> list[str]:
    pair_reports = list(item.get("coal_pair_reports", []))
    if not pair_reports:
        return ["coal pairs: none"]
    ordered = sorted(
        pair_reports,
        key=lambda pair: (
            0 if bool(pair.get("is_collision")) else 1,
            float(pair.get("contact_penetration_depth")) if pair.get("contact_penetration_depth") is not None else float("inf"),
            float(pair.get("distance", float("inf"))),
        ),
    )
    lines = []
    for pair in ordered[: max(int(limit), 1)]:
        pair_text = _pair_to_text(pair.get("pair"), joint_names_by_id)
        lines.append(
            "coal pair: "
            f"{pair_text} | dist={float(pair.get('distance', float('inf'))):+.6f} "
            f"| collide={bool(pair.get('collide_flag'))} "
            f"| depth={_format_optional_float(pair.get('contact_penetration_depth'))}"
        )
    return lines


def _ordered_samples(report: dict, *, disagreements_first: bool, only_disagreements: bool) -> list[dict]:
    disagreements = list(report.get("disagreements", []))
    samples = list(report.get("samples", []))
    if only_disagreements:
        return disagreements
    if not disagreements_first:
        return samples
    disagree_keys = {(str(item.get("label")), int(item.get("sample_index", -1))) for item in disagreements}
    rest = [
        item for item in samples
        if (str(item.get("label")), int(item.get("sample_index", -1))) not in disagree_keys
    ]
    return disagreements + rest


def _highlight_links(robot, link_indices: list[int], pybullet_pair, coal_pair) -> None:
    neutral = [0.82, 0.82, 0.82, 1.0]
    pybullet_color = [0.95, 0.25, 0.10, 1.0]
    coal_color = [0.10, 0.40, 0.95, 1.0]
    both_color = [0.70, 0.10, 0.85, 1.0]
    py_set = {int(x) for x in (pybullet_pair or [])}
    coal_set = {int(x) for x in (coal_pair or [])}
    for link_index in link_indices:
        idx = int(link_index)
        color = neutral
        if idx in py_set and idx in coal_set:
            color = both_color
        elif idx in py_set:
            color = pybullet_color
        elif idx in coal_set:
            color = coal_color
        p.changeVisualShape(int(robot.body_id), idx, rgbaColor=color)


def visualize_comparison_gui(params=ComparisonVisualizationParameters) -> dict:
    report = load_json(params.VALIDATION_JSON)
    inspected = _ordered_samples(
        report,
        disagreements_first=bool(params.DISAGREEMENTS_FIRST),
        only_disagreements=bool(params.ONLY_DISAGREEMENTS),
    )
    if not inspected:
        raise ValueError("No comparison samples available for visualization.")

    cfg = load_config(params.CFG_PATH)
    scene = SimulationScene(cfg)
    scene.enable_rendering()
    robot = Robot(cfg)
    q_base, dq_base = robot.get_joint_state()
    revolute_ids = [int(j) for j in report.get("joint_indices", [])]
    joint_names = [str(name) for name in report.get("joint_names", [])]
    q_indices = list(range(robot.n_pris, robot.n_pris + len(revolute_ids)))
    joint_names_by_id = {int(j): str(name) for j, name in zip(revolute_ids, joint_names)}

    status_ids = [-1] * 12
    current_idx = 0
    last_rendered_idx = None

    def render_current(index: int) -> None:
        nonlocal last_rendered_idx
        item = inspected[index]
        q_full = _compose_full_q(q_base, q_indices, item["revolute_q"])
        robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        _highlight_links(robot, revolute_ids, item.get("pybullet_active_pair"), item.get("coal_active_pair"))
        camera_target = _pick_camera_target(robot)
        p.resetDebugVisualizerCamera(
            cameraDistance=float(params.CAMERA_DISTANCE),
            cameraYaw=float(params.CAMERA_YAW),
            cameraPitch=float(params.CAMERA_PITCH),
            cameraTargetPosition=camera_target,
        )
        base_pos, _ = robot.get_robobase_pose()
        anchor = np.asarray(base_pos, dtype=float) + np.array([0.00, -0.38, 0.92], dtype=float)
        lines = [
            f"[{index + 1}/{len(inspected)}] label={item['label']} sample_index={item['sample_index']} agree={item['collision_agree']}",
            f"PyBullet: dist={float(item['pybullet_min_distance']):+.6f} collision={bool(item['pybullet_is_collision'])}",
            f"PyBullet pair: {_pair_to_text(item.get('pybullet_active_pair'), joint_names_by_id)}",
            f"coal: dist={float(item['coal_min_distance']):+.6f} collision={bool(item['coal_is_collision'])}",
            f"coal pair: {_pair_to_text(item.get('coal_active_pair'), joint_names_by_id)}",
            f"coal contact depth: {_format_optional_float(item.get('coal_contact_penetration_depth'))}",
            f"coal contact pair: {_pair_to_text(item.get('coal_contact_active_pair'), joint_names_by_id)}",
        ]
        lines.extend(_top_pair_lines(item, joint_names_by_id, limit=3))
        lines.append("Controls: N/Right=next, P/Left=prev, Q=quit | colors: PyBullet=red, coal=blue, both=purple")
        if len(status_ids) < len(lines):
            status_ids.extend([-1] * (len(lines) - len(status_ids)))
        for text_idx, line in enumerate(lines):
            text_pos = (anchor + np.array([0.0, 0.0, -0.07 * text_idx], dtype=float)).tolist()
            status_ids[text_idx] = p.addUserDebugText(
                line,
                text_pos,
                textColorRGB=[0.08, 0.08, 0.08],
                textSize=1.15,
                replaceItemUniqueId=status_ids[text_idx],
            )
        print(
            "[self-cspace-coal-vis] "
            f"show {index + 1}/{len(inspected)} | "
            f"agree={item['collision_agree']} | "
            f"pybullet={float(item['pybullet_min_distance']):+.6f} | "
            f"coal={float(item['coal_min_distance']):+.6f} | "
            f"depth={_format_optional_float(item.get('coal_contact_penetration_depth'))}"
        )
        last_rendered_idx = int(index)

    print(f"[self-cspace-coal-vis] validation json -> {params.VALIDATION_JSON}")
    print("[self-cspace-coal-vis] GUI ready. Use N/P/Q or arrow keys.")

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
    return report


if __name__ == "__main__":
    visualize_comparison_gui(ComparisonVisualizationParameters)
