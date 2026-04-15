"""流水线各阶段共用的数据结构定义（dataclass）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RobotModelMetadata:
    revolute_ids: tuple[int, ...]
    revolute_names: tuple[str, ...]
    joint_limits: tuple[tuple[float, float], ...]
    q_indices: tuple[int, ...]
    q_base: np.ndarray
    dq_base: np.ndarray
    monitored_link_ids: tuple[int, ...]
    monitored_link_names: tuple[str, ...]
    monitored_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class FreeSample:
    q: np.ndarray
    clearance: float
    active_pair: tuple[int, int] | None


@dataclass(frozen=True)
class VisibilityGraph:
    vertices: np.ndarray
    adjacency: tuple[frozenset[int], ...]
    edges: tuple[tuple[int, int], ...]
    num_candidate_pairs: int
    num_visible_edges: int


@dataclass(frozen=True)
class Clique:
    vertex_indices: tuple[int, ...]
    score: float
    bad_vertex_count: int = 0


@dataclass(frozen=True)
class CliqueEllipsoid:
    vertex_indices: tuple[int, ...]
    center: np.ndarray
    C: np.ndarray
    clique_size: int


@dataclass(frozen=True)
class IrisRegion:
    region_id: int
    source_clique_indices: tuple[int, ...]
    A: np.ndarray
    b: np.ndarray
    center: np.ndarray
    C: np.ndarray
    log_det: float
    iterations: tuple[dict, ...]


@dataclass(frozen=True)
class CoverageEstimate:
    num_hits: int
    num_samples: int
    ratio: float
    confidence_radius: float


@dataclass(frozen=True)
class RoundStats:
    """每轮 VCC 迭代的统计信息。"""
    round_id: int
    num_samples: int
    num_pairs: int
    num_visible_edges: int
    num_cliques: int
    clique_sizes: tuple[int, ...]
    num_regions_grown: int
    coverage_after: float
    elapsed_seconds: float


@dataclass(frozen=True)
class ExperimentReport:
    regions: tuple[IrisRegion, ...]
    coverage: CoverageEstimate
    round_stats: tuple[RoundStats, ...] = ()
    visibility_stats: dict = field(default_factory=dict)
    clique_stats: dict = field(default_factory=dict)
    sample_stats: dict = field(default_factory=dict)
    curve_report: dict = field(default_factory=dict)
