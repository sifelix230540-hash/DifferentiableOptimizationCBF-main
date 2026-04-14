from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.config import SamplingConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot_model import sample_joint_box
from CBF_experiment.active.pybullet.self_collision.vcc_iris.types import FreeSample, IrisRegion, RobotModelMetadata


def _point_in_polytope(point: np.ndarray, A: np.ndarray, b: np.ndarray, *, tol: float = 1e-9) -> bool:
    return bool(np.all(np.asarray(A, dtype=float) @ np.asarray(point, dtype=float) <= np.asarray(b, dtype=float) + float(tol)))


def _covered_by_regions(q: np.ndarray, regions: list[IrisRegion] | tuple[IrisRegion, ...] | None) -> bool:
    if not regions:
        return False
    for region in regions:
        if _point_in_polytope(q, region.A, region.b):
            return True
    return False


def sample_free_configurations(oracle, cfg: SamplingConfig, *, regions=None) -> list[FreeSample]:
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.progress import ProgressBar
    rng = np.random.default_rng(int(cfg.RNG_SEED))
    metadata: RobotModelMetadata = oracle.metadata
    target = int(cfg.NUM_SEED_SAMPLES)
    accepted: list[FreeSample] = []
    total_checked = 0
    pb = ProgressBar(target, prefix="[sampling]")
    while len(accepted) < target:
        batch = sample_joint_box(metadata, rng, num_samples=max(8, int(cfg.BATCH_SIZE)))
        for q in batch:
            total_checked += 1
            if oracle.is_self_collision(q):
                continue
            metric = oracle.query(q)
            if cfg.MODE == "uncovered_resampling" and _covered_by_regions(q, regions):
                continue
            accepted.append(
                FreeSample(
                    q=np.asarray(q, dtype=float),
                    clearance=float(metric["min_clearance"]),
                    active_pair=metric["active_pair"],
                )
            )
            pb.set(len(accepted), suffix=f"checked={total_checked}")
            if len(accepted) >= target:
                break
    pb.close(suffix=f"checked={total_checked}")
    return accepted


def save_free_samples(samples: list[FreeSample], cache_path: str):
    payload = [
        {
            "q": np.asarray(sample.q, dtype=float).tolist(),
            "clearance": float(sample.clearance),
            "active_pair": list(sample.active_pair) if sample.active_pair else None,
        }
        for sample in samples
    ]
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_free_samples(cache_path: str) -> list[FreeSample]:
    path = Path(cache_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for item in payload:
        active_pair = item.get("active_pair")
        samples.append(
            FreeSample(
                q=np.asarray(item["q"], dtype=float),
                clearance=float(item["clearance"]),
                active_pair=(int(active_pair[0]), int(active_pair[1])) if active_pair else None,
            )
        )
    return samples

