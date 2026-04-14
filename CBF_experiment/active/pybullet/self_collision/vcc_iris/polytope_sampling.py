from __future__ import annotations

import numpy as np


def is_inside_polytope(A: np.ndarray, b: np.ndarray, x: np.ndarray, *, tol: float = 1e-9) -> bool:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    return bool(np.all(A @ x <= b + float(tol)))


def _line_interval(A: np.ndarray, b: np.ndarray, x: np.ndarray, direction: np.ndarray) -> tuple[float, float] | None:
    t_low = -float("inf")
    t_high = float("inf")
    for row, rhs in zip(np.asarray(A, dtype=float), np.asarray(b, dtype=float).reshape(-1)):
        denom = float(np.dot(row, direction))
        slack = float(rhs) - float(np.dot(row, x))
        if abs(denom) <= 1e-12:
            if slack < 0.0:
                return None
            continue
        bound = slack / denom
        if denom > 0.0:
            t_high = min(t_high, bound)
        else:
            t_low = max(t_low, bound)
        if t_low > t_high:
            return None
    return float(t_low), float(t_high)


def hit_and_run_step(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    direction = rng.normal(size=x.shape[0])
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return x.copy()
    direction = direction / norm
    interval = _line_interval(A, b, x, direction)
    if interval is None:
        raise ValueError("Current point is infeasible for the provided polytope.")
    t_low, t_high = interval
    if not np.isfinite(t_low) or not np.isfinite(t_high):
        raise ValueError("Hit-and-run requires a bounded polytope.")
    if t_high - t_low <= 1e-12:
        return x.copy()
    return x + direction * rng.uniform(t_low, t_high)


def sample_polytope_hit_and_run(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    *,
    num_samples: int,
    rng: np.random.Generator,
    mixing_steps: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    if int(num_samples) <= 0:
        return np.zeros((0, np.asarray(x0, dtype=float).size), dtype=float), np.asarray(x0, dtype=float).copy()
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    current = np.asarray(x0, dtype=float).reshape(-1)
    if not is_inside_polytope(A, b, current):
        raise ValueError("Initial point must lie inside the polytope.")
    samples = []
    steps_per_sample = max(int(mixing_steps), 1)
    for _ in range(int(num_samples)):
        for _ in range(steps_per_sample):
            current = hit_and_run_step(A, b, current, rng)
        samples.append(current.copy())
    return np.asarray(samples, dtype=float), current.copy()
