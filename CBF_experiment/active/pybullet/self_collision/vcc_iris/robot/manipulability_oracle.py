"""可操作度预言机：将 Jacobian 条件数 / 可操作度阈值视为 "碰撞"，
复用 VCC+IRIS-ZO 框架生成高可操作度的 C-space 凸区域。

用法与 CoalSelfCollisionOracle 完全对等：
    oracle = ManipulabilityOracle(config)
    oracle.is_self_collision(q)  # True ⇔ 可操作度 < 阈值（"不可行"）
    report = oracle.query(q)     # 含 min_clearance = manipulability - threshold
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.robot_model import (
    compose_full_q,
    load_robot_metadata,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import RobotModelMetadata


def _yoshikawa_manipulability(J: np.ndarray) -> float:
    """Yoshikawa manipulability: sqrt(det(J @ J^T))."""
    JJT = J @ J.T
    det = float(np.linalg.det(JJT))
    return float(np.sqrt(max(det, 0.0)))


def _condition_number(J: np.ndarray) -> float:
    sv = np.linalg.svd(J, compute_uv=False)
    if sv[-1] < 1e-12:
        return float("inf")
    return float(sv[0] / sv[-1])


class ManipulabilityOracle:
    """将"可操作度不足"视为"碰撞"，实现与 CoalSelfCollisionOracle 相同的接口。

    Parameters
    ----------
    config : RobotQueryConfig
        机器人加载配置（与 coal oracle 共用）。
    manipulability_threshold : float
        Yoshikawa 可操作度下限。低于此值视为"碰撞"。
    condition_number_threshold : float or None
        条件数上限（可选）。超过此值也视为"碰撞"。
        为 None 时不检查条件数。
    use_position_only : bool
        若 True，仅使用平移 Jacobian (3×n)；否则使用完整 6×n Jacobian。
    accept_below_threshold : bool
        若 True，则“低于阈值”为可行区域，用于低操作度区域分解。
    """

    def __init__(
        self,
        config: RobotQueryConfig,
        *,
        manipulability_threshold: float = 0.01,
        condition_number_threshold: float | None = None,
        use_position_only: bool = False,
        accept_below_threshold: bool = False,
    ):
        self.config = config
        self.robot, self.metadata, self._created_connection = load_robot_metadata(config)
        self._manip_thresh = float(manipulability_threshold)
        self._cond_thresh = float(condition_number_threshold) if condition_number_threshold is not None else None
        self._use_pos_only = use_position_only
        self._accept_below_threshold = bool(accept_below_threshold)

    @property
    def dim(self) -> int:
        return len(self.metadata.joint_limits)

    def close(self):
        import pybullet as p
        self.robot.set_joint_state(self.metadata.q_base, self.metadata.dq_base)
        if self._created_connection and p.isConnected():
            p.disconnect()

    def _get_jacobian(self, q6: np.ndarray) -> np.ndarray:
        q_full = compose_full_q(self.metadata, q6)
        dq_zero = np.zeros_like(q_full)
        self.robot.set_joint_state(q_full, dq_zero)

        J_full = self.robot.get_ee_jacobian(q_full, dq_zero)

        q_indices = np.array(self.metadata.q_indices, dtype=int)
        J = J_full[:, q_indices]

        if self._use_pos_only:
            J = J[:3, :]
        return J

    def _compute_metrics(self, q6: np.ndarray) -> dict:
        J = self._get_jacobian(q6)
        manip = _yoshikawa_manipulability(J)
        cond = _condition_number(J)
        if self._accept_below_threshold:
            is_bad = manip > self._manip_thresh
            clearance = self._manip_thresh - manip
        else:
            is_bad = manip < self._manip_thresh
            clearance = manip - self._manip_thresh
        if self._cond_thresh is not None and cond > self._cond_thresh:
            is_bad = True
        return {
            "manipulability": float(manip),
            "condition_number": float(cond),
            "is_bad": bool(is_bad),
            "clearance": float(clearance),
        }

    def is_self_collision(self, q6: np.ndarray) -> bool:
        """True ⇔ 可操作度低于阈值（类比"碰撞"）。"""
        return self._compute_metrics(q6)["is_bad"]

    def min_clearance(self, q6: np.ndarray) -> float:
        return self._compute_metrics(q6)["clearance"]

    def query(self, q6: np.ndarray) -> dict:
        m = self._compute_metrics(q6)
        return {
            "is_collision": m["is_bad"],
            "min_clearance": m["clearance"],
            "active_pair": None,
            "manipulability": m["manipulability"],
            "condition_number": m["condition_number"],
        }

    def segment_is_collision_free(self, q_a: np.ndarray, q_b: np.ndarray, *, num_steps: int) -> bool:
        q_a = np.asarray(q_a, dtype=float).reshape(-1)
        q_b = np.asarray(q_b, dtype=float).reshape(-1)
        for alpha in np.linspace(0.0, 1.0, int(num_steps) + 1, dtype=float):
            q = (1.0 - alpha) * q_a + alpha * q_b
            if self.is_self_collision(q):
                return False
        return True

    def first_collision_on_segment_fast(
        self,
        q_free: np.ndarray,
        q_target: np.ndarray,
        *,
        bisection_steps: int,
    ) -> np.ndarray | None:
        q_free = np.asarray(q_free, dtype=float).reshape(-1)
        q_target = np.asarray(q_target, dtype=float).reshape(-1)
        if not self.is_self_collision(q_target):
            return None

        low_q = q_free.copy()
        high_q = q_target.copy()
        for _ in range(int(bisection_steps)):
            mid_q = 0.5 * (low_q + high_q)
            if self.is_self_collision(mid_q):
                high_q = mid_q
            else:
                low_q = mid_q
        return high_q

    def first_collision_on_segment(
        self,
        q_free: np.ndarray,
        q_target: np.ndarray,
        *,
        num_steps: int,
        bisection_steps: int,
    ) -> dict | None:
        q_free = np.asarray(q_free, dtype=float).reshape(-1)
        q_target = np.asarray(q_target, dtype=float).reshape(-1)
        if self.is_self_collision(q_free):
            raise ValueError("Segment start must be in high-manipulability region.")
        if not self.is_self_collision(q_target):
            return None

        alphas = np.linspace(0.0, 1.0, int(num_steps) + 1, dtype=float)
        prev_free = q_free.copy()
        high_alpha = None
        for alpha in alphas[1:]:
            q = (1.0 - alpha) * q_free + alpha * q_target
            if self.is_self_collision(q):
                high_alpha = float(alpha)
                break
            prev_free = q
        if high_alpha is None:
            return None

        low_q = prev_free
        high_q = (1.0 - high_alpha) * q_free + high_alpha * q_target
        for _ in range(int(bisection_steps)):
            mid_q = 0.5 * (low_q + high_q)
            if self.is_self_collision(mid_q):
                high_q = mid_q
            else:
                low_q = mid_q

        delta = high_q - low_q
        norm = float(np.linalg.norm(delta))
        normal = delta / norm if norm > 1e-12 else np.zeros_like(delta)
        metric = self.query(low_q)
        return {
            "q_free": low_q,
            "q_collision": high_q,
            "normal": normal,
            "offset": float(np.dot(normal, low_q)),
            "active_pair": None,
            "clearance": float(metric["min_clearance"]),
        }
