"""采样估计多面体区域并集对 C-free 的覆盖率及置信半径。"""
from __future__ import annotations

import math

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import CoverageEstimate, FreeSample, IrisRegion


def points_in_polytope(points: np.ndarray, A: np.ndarray, b: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.all(np.asarray(points, dtype=float) @ np.asarray(A, dtype=float).T <= np.asarray(b, dtype=float).reshape(1, -1) + float(tol), axis=1)


def estimate_region_coverage(samples: list[FreeSample], regions: list[IrisRegion] | tuple[IrisRegion, ...]) -> CoverageEstimate:
    points = np.asarray([np.asarray(sample.q, dtype=float) for sample in samples], dtype=float)
    covered = np.zeros(points.shape[0], dtype=bool)
    for region in regions:
        covered |= points_in_polytope(points, region.A, region.b, tol=1e-9)
    num_hits = int(np.sum(covered))
    num_samples = int(points.shape[0])
    ratio = float(num_hits / max(num_samples, 1))
    confidence_radius = float(1.96 * math.sqrt(max(ratio * (1.0 - ratio), 1e-12) / max(num_samples, 1)))
    return CoverageEstimate(
        num_hits=num_hits,
        num_samples=num_samples,
        ratio=ratio,
        confidence_radius=confidence_radius,
    )

