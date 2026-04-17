"""在 free 样本上构造全连接可见性图（支持多进程并行）。

并行策略：
  子进程通过 oracle_factory_spec = (module_path, class_name, kwargs_dict) 来
  独立重建 oracle，因此不再耦合具体 oracle 类。
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
from itertools import combinations

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import VisibilityConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import FreeSample, VisibilityGraph


_worker_oracle = None


def _worker_init(oracle_factory_spec: tuple):
    """在子进程中根据工厂规格重建 oracle（支持 DualOracle/NegationOracle 嵌套）。"""
    global _worker_oracle
    _worker_oracle = _instantiate_from_spec(oracle_factory_spec)


def _worker_check_edges(task):
    global _worker_oracle
    edges, vertices, num_steps = task
    visible = []
    for i, j in edges:
        if _worker_oracle.segment_is_collision_free(vertices[i], vertices[j], num_steps=num_steps):
            visible.append((int(i), int(j)))
    return visible


def _parallel_visibility(
    vertices: np.ndarray,
    candidate_pairs: list[tuple[int, int]],
    cfg: VisibilityConfig,
    oracle_factory_spec: tuple,
    num_workers: int,
) -> list[tuple[int, int]]:
    chunk_size = max(1, len(candidate_pairs) // (num_workers * 4))
    chunks = []
    for start in range(0, len(candidate_pairs), chunk_size):
        chunk = candidate_pairs[start : start + chunk_size]
        chunks.append((chunk, vertices, int(cfg.SEGMENT_INTERPOLATION_STEPS)))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=_worker_init, initargs=(oracle_factory_spec,)) as pool:
        results = pool.map(_worker_check_edges, chunks)
    visible = []
    for result_chunk in results:
        visible.extend(result_chunk)
    return visible


def _build_oracle_factory_spec(oracle) -> tuple | None:
    """从 oracle 实例推导出可序列化的工厂规格。

    返回 (module_path, class_name, kwargs_dict)，或 None 表示不支持并行。
    支持嵌套包装（DualOracle / NegationOracle）—— 通过递归构建 base 的 spec，
    在 worker 端再用同样方式恢复。
    """
    # ── DualOracle / NegationOracle：递归构建 base，再包一层 ──
    try:
        from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.dual_oracle import (
            DualOracle,
            NegationOracle,
        )
    except ImportError:
        DualOracle = None  # type: ignore
        NegationOracle = None  # type: ignore

    if DualOracle is not None and isinstance(oracle, DualOracle):
        base_spec = _build_oracle_factory_spec(oracle.base)
        if base_spec is None:
            return None
        # 把对方多面体序列化为 list of (A_list, b_list)
        opp_serialized = [
            (np.asarray(A, dtype=float).tolist(), np.asarray(b, dtype=float).tolist())
            for A, b in oracle._opposite
        ]
        return (
            "CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.dual_oracle",
            "DualOracle",
            {
                "__base_spec__": base_spec,
                "opposite_polytopes_serialized": opp_serialized,
                "margin": float(oracle._margin),
            },
        )

    if NegationOracle is not None and isinstance(oracle, NegationOracle):
        base_spec = _build_oracle_factory_spec(oracle.base)
        if base_spec is None:
            return None
        return (
            "CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.dual_oracle",
            "NegationOracle",
            {"__base_spec__": base_spec},
        )

    # ── 叶子 oracle：CoalSelfCollisionOracle / ManipulabilityOracle ──
    if not hasattr(oracle, "config") or not hasattr(oracle.config, "__dataclass_fields__"):
        return None

    cls = type(oracle)
    module_path = cls.__module__
    class_name = cls.__name__

    config_dict = {
        field: getattr(oracle.config, field)
        for field in oracle.config.__dataclass_fields__
    }

    from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.coal_oracle import CoalSelfCollisionOracle
    if isinstance(oracle, CoalSelfCollisionOracle):
        from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
        return (module_path, class_name, {"config": RobotQueryConfig(**config_dict)})

    try:
        from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.manipulability_oracle import ManipulabilityOracle
        if isinstance(oracle, ManipulabilityOracle):
            from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
            return (
                module_path,
                class_name,
                {
                    "config": RobotQueryConfig(**config_dict),
                    "manipulability_threshold": oracle._manip_thresh,
                    "condition_number_threshold": oracle._cond_thresh,
                    "use_position_only": oracle._use_pos_only,
                    "accept_below_threshold": oracle._accept_below_threshold,
                },
            )
    except ImportError:
        pass

    return None


def _instantiate_from_spec(spec: tuple):
    """从 spec 在 worker 进程中重建 oracle。支持递归 (DualOracle / NegationOracle)。"""
    module_path, class_name, kwargs = spec
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    if "__base_spec__" in kwargs:
        kwargs = dict(kwargs)
        base_spec = kwargs.pop("__base_spec__")
        base = _instantiate_from_spec(base_spec)
        if class_name == "DualOracle":
            opp_serialized = kwargs.pop("opposite_polytopes_serialized", [])
            opp = [
                (np.asarray(A, dtype=float), np.asarray(b, dtype=float))
                for A, b in opp_serialized
            ]
            return cls(base, opp, margin=kwargs.get("margin", 1e-3))
        elif class_name == "NegationOracle":
            return cls(base)
        else:
            raise RuntimeError(f"未知的包装 oracle: {class_name}")

    return cls(**kwargs)


def build_visibility_graph(
    samples: list[FreeSample],
    oracle,
    cfg: VisibilityConfig,
    *,
    parallel_workers: int = 0,
) -> VisibilityGraph:
    vertices = np.asarray([np.asarray(s.q, dtype=float) for s in samples], dtype=float)
    n = len(samples)
    all_pairs = [(int(i), int(j)) for i, j in combinations(range(n), 2)]
    total_pairs = len(all_pairs)

    if parallel_workers <= 0:
        parallel_workers = max(1, os.cpu_count() or 1)

    factory_spec = _build_oracle_factory_spec(oracle) if parallel_workers > 1 else None
    can_parallel = factory_spec is not None and total_pairs >= 20

    if can_parallel:
        from CBF_experiment.active.pybullet.self_collision.vcc_iris.utils.progress import stage_print
        stage_print(f"visibility: {total_pairs} pairs (全连接), {parallel_workers} workers")
        visible_edges = _parallel_visibility(vertices, all_pairs, cfg, factory_spec, parallel_workers)
    else:
        from CBF_experiment.active.pybullet.self_collision.vcc_iris.utils.progress import ProgressBar
        pb = ProgressBar(total_pairs, prefix="[visibility]")
        visible_edges = []
        for idx, (i, j) in enumerate(all_pairs):
            if oracle.segment_is_collision_free(
                vertices[i],
                vertices[j],
                num_steps=int(cfg.SEGMENT_INTERPOLATION_STEPS),
            ):
                visible_edges.append((int(i), int(j)))
            pb.set(idx + 1, suffix=f"visible={len(visible_edges)}")
        pb.close(suffix=f"visible={len(visible_edges)}")

    adjacency: list[set[int]] = [set() for _ in range(n)]
    edges: list[tuple[int, int]] = []
    for i, j in visible_edges:
        adjacency[i].add(int(j))
        adjacency[j].add(int(i))
        edges.append((int(i), int(j)))

    return VisibilityGraph(
        vertices=vertices,
        adjacency=tuple(frozenset(int(x) for x in nbrs) for nbrs in adjacency),
        edges=tuple((int(a), int(b)) for a, b in edges),
        num_candidate_pairs=total_pairs,
        num_visible_edges=len(edges),
    )
