from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_POINT_BATCH_SIZE = 1024
DEFAULT_EQUATION_BATCH_SIZE = 4096


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_samples(payload: dict, positive_keys: list[str], distance_keys: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    samples = None
    for key in positive_keys:
        if key in payload:
            samples = np.asarray(payload[key], dtype=float)
            break
    if samples is None:
        raise KeyError(f"Missing sample keys: {positive_keys}")
    distances = None
    for key in distance_keys:
        if key in payload:
            distances = np.asarray(payload[key], dtype=float).reshape(-1)
            break
    if distances is not None and distances.shape[0] != samples.shape[0]:
        raise ValueError("Distance array length must match sample count.")
    return samples.reshape(samples.shape[0], -1), distances


def _max_violation_against_equations(points: np.ndarray, equations: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, equations.shape[1] - 1)
    eq = np.asarray(equations, dtype=float)
    lhs = pts @ eq[:, :-1].T + eq[:, -1].reshape(1, -1)
    return np.max(lhs, axis=1)


def _min_violation_against_equations_chunked(
    points: np.ndarray,
    equations: np.ndarray,
    *,
    point_batch_size: int = DEFAULT_POINT_BATCH_SIZE,
    equation_batch_size: int = DEFAULT_EQUATION_BATCH_SIZE,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, equations.shape[1] - 1)
    eq = np.asarray(equations, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=float)
    result = np.full(pts.shape[0], -np.inf, dtype=float)
    point_batch = max(int(point_batch_size), 1)
    equation_batch = max(int(equation_batch_size), 1)
    for p_start in range(0, pts.shape[0], point_batch):
        p_end = min(p_start + point_batch, pts.shape[0])
        point_block = pts[p_start:p_end]
        block_result = np.full(point_block.shape[0], -np.inf, dtype=float)
        for e_start in range(0, eq.shape[0], equation_batch):
            e_end = min(e_start + equation_batch, eq.shape[0])
            eq_block = eq[e_start:e_end]
            lhs = point_block @ eq_block[:, :-1].T + eq_block[:, -1].reshape(1, -1)
            block_result = np.maximum(block_result, np.max(lhs, axis=1))
        result[p_start:p_end] = block_result
    return result


def classify_points_against_hulls(
    points,
    hull_payload: dict,
    *,
    space: str = "joint",
    tol: float = 1e-9,
    point_batch_size: int = DEFAULT_POINT_BATCH_SIZE,
    equation_batch_size: int = DEFAULT_EQUATION_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float).reshape(-1, np.asarray(points, dtype=float).shape[-1])
    clusters = list(hull_payload.get("clusters", []))
    if not clusters:
        return np.zeros(pts.shape[0], dtype=bool), np.full(pts.shape[0], np.inf, dtype=float)
    field_name = "equations_joint" if str(space).strip().lower() == "joint" else "equations_normalized"
    min_violation = np.full(pts.shape[0], np.inf, dtype=float)
    for cluster in clusters:
        equations = np.asarray(cluster[field_name], dtype=float)
        cluster_violation = _min_violation_against_equations_chunked(
            pts,
            equations,
            point_batch_size=point_batch_size,
            equation_batch_size=equation_batch_size,
        )
        min_violation = np.minimum(min_violation, cluster_violation)
    inside = min_violation <= float(tol)
    return inside, min_violation


def _safe_mean(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask.astype(float)))


def _compute_boundary_metrics(
    inside_mask: np.ndarray,
    distances: np.ndarray | None,
    *,
    boundary_band: float,
    metric_name: str,
) -> dict:
    if distances is None:
        return {
            f"{metric_name}_count": 0,
            f"{metric_name}_coverage" if "collision" in metric_name else f"{metric_name}_false_positive_rate": 0.0,
        }
    band_mask = np.abs(np.asarray(distances, dtype=float).reshape(-1)) <= float(boundary_band)
    count = int(np.sum(band_mask))
    if "collision" in metric_name:
        value_key = f"{metric_name}_coverage"
    else:
        value_key = f"{metric_name}_false_positive_rate"
    return {
        f"{metric_name}_count": count,
        value_key: _safe_mean(inside_mask[band_mask]) if count > 0 else 0.0,
    }


def evaluate_hulls(
    hull_payload: dict,
    collision_samples,
    free_samples,
    *,
    collision_distances=None,
    free_distances=None,
    boundary_band: float = 0.02,
    space: str = "joint",
) -> dict:
    collision_pts = np.asarray(collision_samples, dtype=float).reshape(-1, np.asarray(collision_samples, dtype=float).shape[-1])
    free_pts = np.asarray(free_samples, dtype=float).reshape(-1, np.asarray(free_samples, dtype=float).shape[-1])
    collision_inside, collision_violation = classify_points_against_hulls(collision_pts, hull_payload, space=space)
    free_inside, free_violation = classify_points_against_hulls(free_pts, hull_payload, space=space)

    coverage = _safe_mean(collision_inside)
    false_positive_rate = _safe_mean(free_inside)
    report = {
        "space": str(space),
        "boundary_band": float(boundary_band),
        "num_collision_samples": int(collision_pts.shape[0]),
        "num_free_samples": int(free_pts.shape[0]),
        "coverage": coverage,
        "miss_rate": float(1.0 - coverage),
        "false_positive_rate": false_positive_rate,
        "collision_inside_count": int(np.sum(collision_inside)),
        "collision_missed_count": int(np.sum(~collision_inside)),
        "free_inside_count": int(np.sum(free_inside)),
        "free_outside_count": int(np.sum(~free_inside)),
        "collision_violation_mean": float(np.mean(collision_violation)) if collision_violation.size else 0.0,
        "free_violation_mean": float(np.mean(free_violation)) if free_violation.size else 0.0,
    }
    report.update(_compute_boundary_metrics(
        collision_inside,
        None if collision_distances is None else np.asarray(collision_distances, dtype=float),
        boundary_band=float(boundary_band),
        metric_name="boundary_collision",
    ))
    report.update(_compute_boundary_metrics(
        free_inside,
        None if free_distances is None else np.asarray(free_distances, dtype=float),
        boundary_band=float(boundary_band),
        metric_name="boundary_free",
    ))
    return report


def evaluate_hulls_from_files(
    *,
    hull_json: str | Path,
    sample_json: str | Path,
    output_json: str | Path | None = None,
    boundary_band: float = 0.02,
    space: str = "joint",
) -> dict:
    hull_payload = load_json(hull_json)
    sample_payload = load_json(sample_json)
    collision_samples, collision_distances = _extract_samples(
        sample_payload,
        ["collision_samples", "negative_samples"],
        ["collision_distances", "negative_distances"],
    )
    free_samples, free_distances = _extract_samples(
        sample_payload,
        ["free_samples", "safe_samples", "positive_samples"],
        ["free_distances", "safe_distances", "positive_distances"],
    )
    report = evaluate_hulls(
        hull_payload,
        collision_samples,
        free_samples,
        collision_distances=collision_distances,
        free_distances=free_distances,
        boundary_band=float(boundary_band),
        space=space,
    )
    report["hull_json"] = str(Path(hull_json))
    report["sample_json"] = str(Path(sample_json))
    if output_json is not None:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate self-collision C-space hulls against collision and free samples.")
    parser.add_argument("--hull-json", type=str, required=True)
    parser.add_argument("--sample-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--boundary-band", type=float, default=0.02)
    parser.add_argument("--space", type=str, default="joint", choices=["joint", "normalized"])
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = _build_arg_parser().parse_args(argv)
    report = evaluate_hulls_from_files(
        hull_json=args.hull_json,
        sample_json=args.sample_json,
        output_json=args.output_json,
        boundary_band=args.boundary_band,
        space=args.space,
    )
    print(
        "[self-cspace-eval] "
        f"coverage={report['coverage']:.4f} "
        f"miss_rate={report['miss_rate']:.4f} "
        f"false_positive_rate={report['false_positive_rate']:.4f}"
    )
    print(
        "[self-cspace-eval] "
        f"boundary_collision_coverage={report['boundary_collision_coverage']:.4f} "
        f"boundary_free_false_positive_rate={report['boundary_free_false_positive_rate']:.4f}"
    )
    if args.output_json:
        print(f"[self-cspace-eval] saved -> {args.output_json}")
    return report


if __name__ == "__main__":
    main()
