"""对偶分解所需的两个 oracle 包装层。

NegationOracle:
    把 base.is_self_collision(q) 的语义反转 —— "可行" ⇔ base 认为是碰撞。
    用于在 C-obs 内部增长负 region：从 IRIS-ZO 的视角，C-obs 才是 "free space"。

DualOracle:
    在 base oracle 的基础上，把 "对方 region 多面体的内部" 也视为 "障碍"。
    实现 mutual pseudo-obstacle 机制：保证正负 region 不重叠（带 ε margin）。

两者都遵循 base oracle 的接口，可直接喂给 build_visibility_graph / run_iris_zo。
为支持 multiprocessing 并行，二者都提供 _factory_spec 元信息，
visibility.py 的 _build_oracle_factory_spec 会识别它们。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ────────────────────────── 工具：多面体判定 ──────────────────────────


def _point_in_polytope_with_margin(
    q: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    margin: float,
) -> bool:
    """判定 q 是否落在 {x : A x <= b - margin} 中（即把 polytope 收缩 margin 后判定）。"""
    return bool(np.all(A @ q <= b - float(margin)))


def _point_in_any_polytope(
    q: np.ndarray,
    polytopes: Sequence[tuple[np.ndarray, np.ndarray]],
    margin: float,
) -> bool:
    for A, b in polytopes:
        if _point_in_polytope_with_margin(q, A, b, margin):
            return True
    return False


def _polytopes_to_arrays(polytopes: Sequence) -> list[tuple[np.ndarray, np.ndarray]]:
    """把 IrisRegion 或 (A, b) 二元组的列表统一成 (A, b) numpy 列表。"""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for item in polytopes:
        if isinstance(item, tuple) and len(item) == 2:
            A, b = item
        else:
            A = item.A
            b = item.b
        out.append((np.asarray(A, dtype=float), np.asarray(b, dtype=float).reshape(-1)))
    return out


# ────────────────────────── NegationOracle ──────────────────────────


class NegationOracle:
    """把 base oracle 的 is_self_collision 语义整体反转。

    适合与 CoalSelfCollisionOracle 配对，用于负 region 增长（C-obs 当成 "free"）。

    所有 IRIS-ZO 用到的接口都对应反转：
      * is_self_collision(q): base 不碰撞 → True (对 neg 而言是 "障碍")
      * segment_is_collision_free(a, b): 整段都需要 base 碰撞，才视为 "neg-free"
      * first_collision_on_segment_fast(...): 二分找 base-非碰撞的边界点
    """

    # 默认不支持普通采样（sample_free_configurations 直接调用 base.metadata，无问题）
    # min_clearance / query 也镜像反转

    def __init__(self, base_oracle):
        self.base = base_oracle
        self.config = base_oracle.config
        # metadata / robot 透传
        self.metadata = base_oracle.metadata
        # NegationOracle 不创建新 pybullet 连接
        self._created_connection = False

    # ── pybullet 资源生命周期 ──
    def close(self):
        # 反转 oracle 不持有独立资源；不主动关 base，由 caller 决定
        pass

    @property
    def dim(self) -> int:
        return self.base.dim

    # ── 必备接口（被 sampling / visibility / iris-zo 调用）──
    def is_self_collision(self, q: np.ndarray) -> bool:
        return not self.base.is_self_collision(q)

    def min_clearance(self, q: np.ndarray) -> float:
        # 反向 clearance：base 越深进入碰撞，对 neg 而言越 "安全"
        if hasattr(self.base, "min_clearance"):
            return -float(self.base.min_clearance(q))
        return 0.0

    def query(self, q: np.ndarray) -> dict:
        if hasattr(self.base, "query"):
            base_q = self.base.query(q)
            return {
                "is_collision": not bool(base_q.get("is_collision", False)),
                "min_clearance": -float(base_q.get("min_clearance", 0.0)),
                "active_pair": base_q.get("active_pair"),
            }
        return {"is_collision": self.is_self_collision(q), "min_clearance": 0.0, "active_pair": None}

    def segment_is_collision_free(self, q_a, q_b, *, num_steps: int) -> bool:
        q_a = np.asarray(q_a, dtype=float).reshape(-1)
        q_b = np.asarray(q_b, dtype=float).reshape(-1)
        for alpha in np.linspace(0.0, 1.0, int(num_steps) + 1, dtype=float):
            q = (1.0 - alpha) * q_a + alpha * q_b
            # neg-free ⇔ base 认为是碰撞
            if not self.base.is_self_collision(q):
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
        # 对 neg：q_target 是 "neg 视角的碰撞" ⇔ base 不碰撞
        if self.base.is_self_collision(q_target):
            return None
        low_q = q_free.copy()
        high_q = q_target.copy()
        for _ in range(int(bisection_steps)):
            mid_q = 0.5 * (low_q + high_q)
            if not self.base.is_self_collision(mid_q):
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
            raise ValueError("NegationOracle.first_collision_on_segment: q_free 必须 neg-free（即 base 是碰撞）。")
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
        return {
            "q_free": low_q,
            "q_collision": high_q,
            "normal": normal,
            "offset": float(np.dot(normal, low_q)),
            "active_pair": None,
            "clearance": 0.0,
        }


# ────────────────────────── DualOracle ──────────────────────────


class DualOracle:
    """在 base oracle 之上叠加 "对方 region 内部 = 伪障碍" 机制。

    Parameters
    ----------
    base : oracle 接口
        实际语义的 oracle（CoalSelfCollisionOracle / NegationOracle / ManipulabilityOracle 等）。
    opposite_polytopes : list of (A, b) or list of IrisRegion
        对方已增长的 region 列表。被视为伪障碍。
    margin : float
        判定时把 polytope 收缩 margin（避免边界数值抖动 + 制造 ε 间隙）。

    所有点检查都先查 base，再查是否落在任一对方 polytope。
    """

    def __init__(self, base_oracle, opposite_polytopes: Sequence = (), *, margin: float = 1e-3):
        self.base = base_oracle
        self.config = base_oracle.config
        self.metadata = base_oracle.metadata
        self._opposite = _polytopes_to_arrays(opposite_polytopes)
        self._margin = float(margin)
        self._created_connection = False

    def close(self):
        pass

    @property
    def dim(self) -> int:
        return self.base.dim

    def _in_opposite(self, q: np.ndarray) -> bool:
        if not self._opposite:
            return False
        return _point_in_any_polytope(np.asarray(q, dtype=float).reshape(-1), self._opposite, self._margin)

    def is_self_collision(self, q: np.ndarray) -> bool:
        if self._in_opposite(q):
            return True
        return self.base.is_self_collision(q)

    def min_clearance(self, q: np.ndarray) -> float:
        if hasattr(self.base, "min_clearance"):
            return float(self.base.min_clearance(q))
        return 0.0

    def query(self, q: np.ndarray) -> dict:
        if hasattr(self.base, "query"):
            return self.base.query(q)
        return {"is_collision": self.is_self_collision(q), "min_clearance": 0.0, "active_pair": None}

    def segment_is_collision_free(self, q_a, q_b, *, num_steps: int) -> bool:
        q_a = np.asarray(q_a, dtype=float).reshape(-1)
        q_b = np.asarray(q_b, dtype=float).reshape(-1)
        for alpha in np.linspace(0.0, 1.0, int(num_steps) + 1, dtype=float):
            q = (1.0 - alpha) * q_a + alpha * q_b
            if self._in_opposite(q):
                return False
            if self.base.is_self_collision(q):
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
            raise ValueError("DualOracle.first_collision_on_segment: q_free 必须可行。")
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
        return {
            "q_free": low_q,
            "q_collision": high_q,
            "normal": normal,
            "offset": float(np.dot(normal, low_q)),
            "active_pair": None,
            "clearance": 0.0,
        }
