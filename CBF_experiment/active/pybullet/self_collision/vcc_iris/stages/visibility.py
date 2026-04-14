"""在 free 样本上构造可见性图（支持多进程并行）。"""
from __future__ import annotations

import multiprocessing as mp
import os
from itertools import combinations

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import VisibilityConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import FreeSample, VisibilityGraph


def _choose_candidate_pairs(num_vertices: int, cfg: VisibilityConfig) -> list[tuple[int, int]]:
    all_pairs = list(combinations(range(int(num_vertices)), 2))
    if cfg.MAX_CANDIDATE_PAIRS is None or len(all_pairs) <= int(cfg.MAX_CANDIDATE_PAIRS):
        return [(int(i), int(j)) for i, j in all_pairs]
    rng = np.random.default_rng(int(cfg.RANDOM_SEED))
    selected = rng.choice(len(all_pairs), size=int(cfg.MAX_CANDIDATE_PAIRS), replace=False)
    return [tuple(int(x) for x in all_pairs[int(idx)]) for idx in selected.tolist()]


def _sort_by_distance(pairs: list[tuple[int, int]], vertices: np.ndarray) -> list[tuple[int, int]]:
    if len(pairs) == 0:
        return pairs
    dists = [float(np.linalg.norm(vertices[i] - vertices[j])) for i, j in pairs]
    order = np.argsort(dists)
    return [pairs[int(idx)] for idx in order]


_worker_oracle = None


def _worker_init(config_dict):
    global _worker_oracle
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.coal_oracle import CoalSelfCollisionOracle
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
    _worker_oracle = CoalSelfCollisionOracle(RobotQueryConfig(**config_dict))


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
    config_dict: dict,
    num_workers: int,
) -> list[tuple[int, int]]:
    chunk_size = max(1, len(candidate_pairs) // num_workers)
    chunks = []
    for start in range(0, len(candidate_pairs), chunk_size):
        chunk = candidate_pairs[start : start + chunk_size]
        chunks.append((chunk, vertices, int(cfg.SEGMENT_INTERPOLATION_STEPS)))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=_worker_init, initargs=(config_dict,)) as pool:
        results = pool.map(_worker_check_edges, chunks)
    visible = []
    for result_chunk in results:
        visible.extend(result_chunk)
    return visible


def build_visibility_graph(
    samples: list[FreeSample],
    oracle,
    cfg: VisibilityConfig,
    *,
    parallel_workers: int = 0,
) -> VisibilityGraph:
    vertices = np.asarray([np.asarray(sample.q, dtype=float) for sample in samples], dtype=float)
    adjacency: list[set[int]] = [set() for _ in range(len(samples))]
    edges: list[tuple[int, int]] = []
    candidate_pairs = _choose_candidate_pairs(len(samples), cfg)
    candidate_pairs = _sort_by_distance(candidate_pairs, vertices)

    if parallel_workers <= 0:
        parallel_workers = max(1, os.cpu_count() or 1)

    can_parallel = (
        parallel_workers > 1
        and len(candidate_pairs) >= 20
        and hasattr(oracle, "config")
        and hasattr(oracle.config, "__dataclass_fields__")
    )
    if can_parallel:
        from CBF_experiment.active.pybullet.self_collision.vcc_iris.utils.progress import stage_print
        stage_print(f"visibility: {len(candidate_pairs)} edges, {parallel_workers} workers (parallel)")
        config_dict = {
            field: getattr(oracle.config, field)
            for field in oracle.config.__dataclass_fields__
        }
        visible_edges = _parallel_visibility(vertices, candidate_pairs, cfg, config_dict, parallel_workers)
    else:
        from CBF_experiment.active.pybullet.self_collision.vcc_iris.utils.progress import ProgressBar
        pb = ProgressBar(len(candidate_pairs), prefix="[visibility]")
        visible_edges = []
        for idx, (i, j) in enumerate(candidate_pairs):
            if oracle.segment_is_collision_free(
                vertices[i],
                vertices[j],
                num_steps=int(cfg.SEGMENT_INTERPOLATION_STEPS),
            ):
                visible_edges.append((int(i), int(j)))
            pb.set(idx + 1, suffix=f"visible={len(visible_edges)}")
        pb.close(suffix=f"visible={len(visible_edges)}")

    for i, j in visible_edges:
        adjacency[i].add(int(j))
        adjacency[j].add(int(i))
        edges.append((int(i), int(j)))

    return VisibilityGraph(
        vertices=vertices,
        adjacency=tuple(frozenset(int(x) for x in nbrs) for nbrs in adjacency),
        edges=tuple((int(i), int(j)) for i, j in edges),
        num_candidate_pairs=int(len(candidate_pairs)),
        num_visible_edges=int(len(edges)),
    )
