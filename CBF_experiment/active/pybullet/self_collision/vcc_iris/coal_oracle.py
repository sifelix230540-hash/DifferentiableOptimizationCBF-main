from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pybullet as p

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.self_collision.self_collision_backend_coal import (  # noqa: E402
    build_coal_link_models,
    classify_self_collision_sample,
    compute_pairwise_self_collision_distance,
    is_any_pair_collision_fast,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.config import RobotQueryConfig  # noqa: E402
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot_model import compose_full_q, load_robot_metadata  # noqa: E402
from CBF_experiment.active.pybullet.self_collision.vcc_iris.types import RobotModelMetadata  # noqa: E402


class CoalSelfCollisionOracle:
    def __init__(self, config: RobotQueryConfig):
        self.config = config
        self.robot, self.metadata, self._created_connection = load_robot_metadata(config)
        self.link_models = build_coal_link_models(self.robot, self.metadata.monitored_link_ids)

    @property
    def dim(self) -> int:
        return len(self.metadata.joint_limits)

    def close(self):
        self.robot.set_joint_state(self.metadata.q_base, self.metadata.dq_base)
        if self._created_connection and p.isConnected():
            p.disconnect()

    def _query_metric(self, q6: np.ndarray) -> dict:
        q_full = compose_full_q(self.metadata, q6)
        self.robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        metric = classify_self_collision_sample(
            self.robot,
            link_models=self.link_models,
            monitored_pairs=self.metadata.monitored_pairs,
            penetration_thresh=float(self.config.PENETRATION_THRESH),
        )
        return metric

    def _query_pair_metric(self, q6: np.ndarray, pair: tuple[int, int]) -> dict:
        q_full = compose_full_q(self.metadata, q6)
        self.robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        return compute_pairwise_self_collision_distance(
            self.robot,
            link_models=self.link_models,
            monitored_pairs=[(int(pair[0]), int(pair[1]))],
            penetration_thresh=float(self.config.PENETRATION_THRESH),
        )

    def _is_collision_fast(self, q6: np.ndarray) -> bool:
        """Collide-only check, early-exit. ~2x faster than _query_metric."""
        q_full = compose_full_q(self.metadata, q6)
        self.robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        return is_any_pair_collision_fast(
            self.robot,
            link_models=self.link_models,
            monitored_pairs=self.metadata.monitored_pairs,
        )

    def is_self_collision(self, q6: np.ndarray) -> bool:
        return self._is_collision_fast(q6)

    def min_clearance(self, q6: np.ndarray) -> float:
        return float(self._query_metric(q6)["min_distance"])

    def active_pair(self, q6: np.ndarray) -> tuple[int, int] | None:
        pair = self._query_metric(q6).get("active_pair")
        if not pair:
            return None
        return int(pair[0]), int(pair[1])

    def query(self, q6: np.ndarray) -> dict:
        metric = self._query_metric(q6)
        pair = metric.get("active_pair")
        return {
            "is_collision": bool(metric["is_collision"]),
            "min_clearance": float(metric["min_distance"]),
            "active_pair": (int(pair[0]), int(pair[1])) if pair else None,
            "contact_penetration_depth": metric.get("contact_penetration_depth"),
        }

    def pair_query(self, q6: np.ndarray, pair: tuple[int, int]) -> dict:
        metric = self._query_pair_metric(q6, pair)
        active_pair = metric.get("active_pair")
        return {
            "pair": tuple(int(x) for x in pair),
            "is_collision": bool(metric["is_collision"]),
            "min_clearance": float(metric["min_distance"]),
            "active_pair": (int(active_pair[0]), int(active_pair[1])) if active_pair else None,
            "contact_penetration_depth": metric.get("contact_penetration_depth"),
        }

    def pair_distances_at(self, q6: np.ndarray) -> list[dict]:
        q_full = compose_full_q(self.metadata, q6)
        self.robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        metric = classify_self_collision_sample(
            self.robot,
            link_models=self.link_models,
            monitored_pairs=self.metadata.monitored_pairs,
            penetration_thresh=float(self.config.PENETRATION_THRESH),
        )
        reports = []
        for report in metric.get("pair_reports", []):
            pair = report.get("pair") or []
            if len(pair) != 2:
                continue
            reports.append({
                "pair": (int(pair[0]), int(pair[1])),
                "distance": float(report["distance"]),
                "is_collision": bool(report["is_collision"]),
            })
        return reports

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
        """Boolean-only bisection returning the collision-boundary point.

        Walks inward from q_target toward q_free (center) using bisection.
        Returns q_collision (the closest point to center that is still in
        collision), or None if q_target is actually free.
        No full-pair distance query is performed.
        """
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
            raise ValueError("Segment start must be collision-free.")
        if not self.is_self_collision(q_target):
            return None

        alphas = np.linspace(0.0, 1.0, int(num_steps) + 1, dtype=float)
        low_alpha = 0.0
        high_alpha = None
        prev_free = q_free.copy()
        for alpha in alphas[1:]:
            q = (1.0 - alpha) * q_free + alpha * q_target
            if self.is_self_collision(q):
                high_alpha = float(alpha)
                break
            low_alpha = float(alpha)
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
        normal = np.zeros_like(delta)
        if norm > 1e-12:
            normal = delta / norm
        metric = self.query(low_q)
        return {
            "q_free": low_q,
            "q_collision": high_q,
            "normal": normal,
            "offset": float(np.dot(normal, low_q)),
            "active_pair": metric["active_pair"],
            "clearance": float(metric["min_clearance"]),
        }

    def first_pair_collision_on_segment(
        self,
        q_free: np.ndarray,
        q_target: np.ndarray,
        pair: tuple[int, int],
        *,
        num_steps: int,
        bisection_steps: int,
    ) -> dict | None:
        q_free = np.asarray(q_free, dtype=float).reshape(-1)
        q_target = np.asarray(q_target, dtype=float).reshape(-1)
        start_metric = self.pair_query(q_free, pair)
        if bool(start_metric["is_collision"]):
            raise ValueError("Segment start must be free for the queried pair.")
        end_metric = self.pair_query(q_target, pair)
        if not bool(end_metric["is_collision"]):
            return None

        alphas = np.linspace(0.0, 1.0, int(num_steps) + 1, dtype=float)
        prev_free = q_free.copy()
        high_alpha = None
        for alpha in alphas[1:]:
            q = (1.0 - alpha) * q_free + alpha * q_target
            metric = self.pair_query(q, pair)
            if bool(metric["is_collision"]):
                high_alpha = float(alpha)
                break
            prev_free = q
        if high_alpha is None:
            return None

        low_q = prev_free
        high_q = (1.0 - high_alpha) * q_free + high_alpha * q_target
        for _ in range(int(bisection_steps)):
            mid_q = 0.5 * (low_q + high_q)
            mid_metric = self.pair_query(mid_q, pair)
            if bool(mid_metric["is_collision"]):
                high_q = mid_q
            else:
                low_q = mid_q

        delta = high_q - low_q
        norm = float(np.linalg.norm(delta))
        normal = np.zeros_like(delta)
        if norm > 1e-12:
            normal = delta / norm
        metric = self.pair_query(low_q, pair)
        return {
            "pair": tuple(int(x) for x in pair),
            "q_free": low_q,
            "q_collision": high_q,
            "normal": normal,
            "offset": float(np.dot(normal, low_q)),
            "active_pair": metric["active_pair"],
            "clearance": float(metric["min_clearance"]),
        }

