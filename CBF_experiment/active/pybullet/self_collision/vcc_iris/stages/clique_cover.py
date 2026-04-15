"""可见性图上的截断团覆盖 (Truncated Clique Cover)。

支持两种策略：
  igraph_exact  — 反复求解精确 MAXCLIQUE (igraph C 后端)，论文推荐
  greedy        — 贪心极大团启发式，无需额外依赖
"""
from __future__ import annotations

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import CliqueCoverConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import Clique, VisibilityGraph
from CBF_experiment.active.pybullet.self_collision.vcc_iris.utils.progress import stage_print


# ── igraph 精确 MAXCLIQUE ──────────────────────────────────

def _igraph_truncated_clique_cover(graph: VisibilityGraph, cfg: CliqueCoverConfig) -> list[Clique]:
    """论文 Algorithm 1 中 TRUNCATEDCLIQUECOVER 的精确实现。

    每轮在剩余子图上用 igraph 精确求解 MAXCLIQUE (ILP 级别的 C 实现)，
    然后移除被选中的顶点，直到最大团 < smin 或达到上限。
    """
    try:
        import igraph as ig
    except ImportError:
        stage_print("igraph 未安装, 回退到 greedy 策略")
        return _greedy_truncated_clique_cover(graph, cfg)

    n = len(graph.vertices)
    smin = int(cfg.MIN_CLIQUE_SIZE)
    max_cliques = int(cfg.MAX_CLIQUES_PER_ROUND)

    edge_list = [(int(i), int(j)) for i, j in graph.edges]
    g = ig.Graph(n=n, edges=edge_list, directed=False)
    g.vs["orig_id"] = list(range(n))

    cliques: list[Clique] = []

    while len(cliques) < max_cliques and g.vcount() >= smin:
        mc = g.largest_cliques()
        if not mc:
            break
        best = max(mc, key=len)
        if len(best) < smin:
            break

        orig_indices = tuple(sorted(int(g.vs[v]["orig_id"]) for v in best))
        cliques.append(Clique(vertex_indices=orig_indices, score=float(len(orig_indices))))
        stage_print(f"  clique #{len(cliques)}: size={len(orig_indices)}")

        g.delete_vertices(sorted(best, reverse=True))

    return cliques


# ── 贪心后备 ───────────────────────────────────────────────

def _grow_maximal_clique(seed: int, remaining: set[int], adjacency: tuple[frozenset[int], ...]) -> tuple[int, ...]:
    clique = [int(seed)]
    candidates = set(int(x) for x in adjacency[int(seed)] if int(x) in remaining)
    while candidates:
        next_vertex = max(
            candidates,
            key=lambda v: len(candidates.intersection(adjacency[int(v)])),
        )
        if all(int(next_vertex) in adjacency[int(m)] for m in clique):
            clique.append(int(next_vertex))
            candidates &= set(int(x) for x in adjacency[int(next_vertex)] if int(x) in remaining)
        else:
            candidates.remove(int(next_vertex))
    return tuple(sorted(int(x) for x in clique))


def _greedy_truncated_clique_cover(graph: VisibilityGraph, cfg: CliqueCoverConfig) -> list[Clique]:
    remaining = set(range(len(graph.vertices)))
    cliques: list[Clique] = []
    smin = int(cfg.MIN_CLIQUE_SIZE)
    max_cliques = int(cfg.MAX_CLIQUES_PER_ROUND)

    while remaining and len(cliques) < max_cliques:
        seed = max(remaining, key=lambda v: len(set(graph.adjacency[int(v)]).intersection(remaining)))
        clique_verts = _grow_maximal_clique(int(seed), remaining, graph.adjacency)
        if len(clique_verts) < smin:
            break
        cliques.append(Clique(vertex_indices=clique_verts, score=float(len(clique_verts))))
        remaining.difference_update(int(x) for x in clique_verts)
    return cliques


# ── 入口 ───────────────────────────────────────────────────

def truncated_clique_cover(graph: VisibilityGraph, cfg: CliqueCoverConfig) -> list[Clique]:
    if cfg.STRATEGY == "igraph_exact":
        return _igraph_truncated_clique_cover(graph, cfg)
    return _greedy_truncated_clique_cover(graph, cfg)
