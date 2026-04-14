"""可见性图上的贪心极大团覆盖，生成 IRIS-ZO 种子团。"""
from __future__ import annotations

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import CliqueCoverConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import Clique, VisibilityGraph


def _grow_maximal_clique(seed: int, remaining: set[int], adjacency: tuple[frozenset[int], ...]) -> tuple[int, ...]:
    clique = [int(seed)]
    candidates = set(int(x) for x in adjacency[int(seed)] if int(x) in remaining)
    while candidates:
        next_vertex = max(
            candidates,
            key=lambda vertex: len(candidates.intersection(adjacency[int(vertex)])),
        )
        if all(int(next_vertex) in adjacency[int(member)] for member in clique):
            clique.append(int(next_vertex))
            candidates &= set(int(x) for x in adjacency[int(next_vertex)] if int(x) in remaining)
        else:
            candidates.remove(int(next_vertex))
    return tuple(sorted(int(x) for x in clique))


def greedy_clique_cover(graph: VisibilityGraph, cfg: CliqueCoverConfig) -> list[Clique]:
    remaining = set(range(len(graph.vertices)))
    cliques: list[Clique] = []
    while remaining and len(cliques) < int(cfg.MAX_CLIQUES):
        seed = max(remaining, key=lambda vertex: len(set(graph.adjacency[int(vertex)]).intersection(remaining)))
        clique_vertices = _grow_maximal_clique(int(seed), remaining, graph.adjacency)
        if len(clique_vertices) < int(cfg.MIN_CLIQUE_SIZE):
            break
        score = float(len(clique_vertices))
        cliques.append(Clique(vertex_indices=clique_vertices, score=score))
        remaining.difference_update(int(x) for x in clique_vertices)
    return cliques

