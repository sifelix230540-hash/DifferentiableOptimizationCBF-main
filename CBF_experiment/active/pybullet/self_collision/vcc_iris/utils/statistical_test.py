"""Chernoff 界 / 并界形式的非自适应碰撞统计检验。"""
from __future__ import annotations

import math


def union_bound_delta(*, total_delta: float, outer_iter: int, inner_iter: int) -> float:
    outer_iter = max(int(outer_iter), 1)
    inner_iter = max(int(inner_iter), 1)
    total_delta = min(max(float(total_delta), 1e-12), 1.0 - 1e-12)
    weight = 36.0 / (math.pi**4 * float(outer_iter**2) * float(inner_iter**2))
    return min(max(total_delta * weight, 1e-12), 1.0 - 1e-12)


def required_trials(*, epsilon: float, delta: float, tau: float) -> int:
    """Chernoff-based sample bound (Theorem 1 of Werner et al. 2024).

    M = ceil(2 * log(1/delta) / (epsilon * tau^2))
    """
    epsilon = max(float(epsilon), 1e-9)
    tau = min(max(float(tau), 1e-6), 1.0 - 1e-6)
    delta = min(max(float(delta), 1e-12), 1.0 - 1e-12)
    return max(1, int(math.ceil(2.0 * math.log(1.0 / delta) / (epsilon * tau * tau))))


def unadaptive_collision_test(
    num_collisions: int,
    *,
    num_samples: int,
    epsilon: float,
    total_delta: float,
    tau: float,
    outer_iter: int,
    inner_iter: int,
) -> dict:
    delta_ik = union_bound_delta(total_delta=total_delta, outer_iter=outer_iter, inner_iter=inner_iter)
    min_trials = required_trials(epsilon=epsilon, delta=delta_ik, tau=tau)
    threshold = (1.0 - float(tau)) * float(epsilon)
    empirical_rate = float(num_collisions) / max(int(num_samples), 1)
    accept = bool(int(num_samples) >= min_trials and empirical_rate <= threshold)
    return {
        "accept": accept,
        "delta_ik": float(delta_ik),
        "min_trials": int(min_trials),
        "num_samples": int(num_samples),
        "num_collisions": int(num_collisions),
        "empirical_rate": float(empirical_rate),
        "threshold": float(threshold),
    }
