"""PyBullet 机器人模型加载与关节空间辅助函数。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pybullet as p

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import Robot, load_config  # noqa: E402
from CBF_experiment.active.pybullet.self_collision.self_collision_cspace_hulls import (  # noqa: E402
    build_monitored_link_pairs,
    extract_revolute_metadata,
    extract_self_collision_monitor_metadata,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig  # noqa: E402
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import RobotModelMetadata  # noqa: E402


def load_robot_metadata(robot_cfg: RobotQueryConfig) -> tuple[Robot, RobotModelMetadata, bool]:
    created_connection = False
    if not p.isConnected():
        p.connect(p.DIRECT)
        created_connection = True

    robot = Robot(load_config(robot_cfg.CFG_PATH))
    q_base, dq_base = robot.get_joint_state()
    revolute_ids, revolute_names, joint_limits, q_indices = extract_revolute_metadata(robot)
    monitored_link_ids, monitored_link_names = extract_self_collision_monitor_metadata(
        robot,
        include_welding_gun_base=bool(robot_cfg.INCLUDE_WELDING_GUN_BASE),
        include_third_axis_chain=bool(robot_cfg.INCLUDE_THIRD_AXIS_CHAIN),
    )
    monitored_pairs = build_monitored_link_pairs(monitored_link_ids, min_index_gap=int(robot_cfg.MIN_INDEX_GAP))
    metadata = RobotModelMetadata(
        revolute_ids=tuple(int(x) for x in revolute_ids),
        revolute_names=tuple(str(x) for x in revolute_names),
        joint_limits=tuple((float(lo), float(hi)) for lo, hi in joint_limits),
        q_indices=tuple(int(x) for x in q_indices),
        q_base=np.asarray(q_base, dtype=float),
        dq_base=np.asarray(dq_base, dtype=float),
        monitored_link_ids=tuple(int(x) for x in monitored_link_ids),
        monitored_link_names=tuple(str(x) for x in monitored_link_names),
        monitored_pairs=tuple((int(a), int(b)) for a, b in monitored_pairs),
    )
    return robot, metadata, created_connection


def compose_full_q(metadata: RobotModelMetadata, q6: np.ndarray) -> np.ndarray:
    q_full = np.asarray(metadata.q_base, dtype=float).copy()
    q_full[np.asarray(metadata.q_indices, dtype=int)] = np.asarray(q6, dtype=float).reshape(-1)
    return q_full


def joint_box_halfspaces(metadata: RobotModelMetadata) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    rhs = []
    for axis, (lo, hi) in enumerate(metadata.joint_limits):
        upper = np.zeros(len(metadata.joint_limits), dtype=float)
        upper[axis] = 1.0
        rows.append(upper)
        rhs.append(float(hi))
        lower = np.zeros(len(metadata.joint_limits), dtype=float)
        lower[axis] = -1.0
        rows.append(lower)
        rhs.append(-float(lo))
    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def sample_joint_box(metadata: RobotModelMetadata, rng: np.random.Generator, num_samples: int) -> np.ndarray:
    lower = np.asarray([float(lo) for lo, _ in metadata.joint_limits], dtype=float)
    upper = np.asarray([float(hi) for _, hi in metadata.joint_limits], dtype=float)
    return rng.uniform(lower.reshape(1, -1), upper.reshape(1, -1), size=(int(num_samples), len(metadata.joint_limits)))

