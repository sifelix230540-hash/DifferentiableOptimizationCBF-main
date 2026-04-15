"""关节盒内均匀拒绝采样无碰撞构型，支持排除已覆盖区域与 JSON 缓存。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import SamplingConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.robot_model import sample_joint_box
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import FreeSample, IrisRegion, RobotModelMetadata


def _point_in_polytope(point: np.ndarray, A: np.ndarray, b: np.ndarray, *, tol: float = 1e-9) -> bool:
    return bool(np.all(A @ point <= b + tol))


def _covered_by_regions(q: np.ndarray, regions: list[IrisRegion] | tuple[IrisRegion, ...]) -> bool:
    for region in regions:
        if _point_in_polytope(q, region.A, region.b):
            return True
    return False


def sample_free_configurations(
    oracle,
    cfg: SamplingConfig,
    *,
    existing_regions: list[IrisRegion] | tuple[IrisRegion, ...] | None = None,
) -> list[FreeSample]:
    """从未被 existing_regions 覆盖的 C-free 区域采 K 个 free samples。"""
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.utils.progress import ProgressBar
    rng = np.random.default_rng(int(cfg.RNG_SEED))
    metadata: RobotModelMetadata = oracle.metadata
    target = int(cfg.NUM_SAMPLES_PER_ROUND)
    accepted: list[FreeSample] = []
    total_checked = 0
    pb = ProgressBar(target, prefix="[sampling]")
    while len(accepted) < target:
        batch = sample_joint_box(metadata, rng, num_samples=max(8, int(cfg.BATCH_SIZE)))
        for q in batch:
            total_checked += 1
            if oracle.is_self_collision(q):
                continue
            if existing_regions and _covered_by_regions(q, existing_regions):
                continue
            metric = oracle.query(q)
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


def sample_coverage_test_points(oracle, num_samples: int, rng_seed: int = 42) -> list[FreeSample]:
    """独立采样一批 free 点用于覆盖率估计（不排除已覆盖区域）。"""
    from CBF_experiment.active.pybullet.self_collision.vcc_iris.utils.progress import ProgressBar
    rng = np.random.default_rng(rng_seed)
    metadata: RobotModelMetadata = oracle.metadata
    accepted: list[FreeSample] = []
    total_checked = 0
    pb = ProgressBar(num_samples, prefix="[coverage-pts]")
    while len(accepted) < num_samples:
        batch = sample_joint_box(metadata, rng, num_samples=max(8, 256))
        for q in batch:
            total_checked += 1
            if oracle.is_self_collision(q):
                continue
            accepted.append(FreeSample(q=np.asarray(q, dtype=float), clearance=0.0, active_pair=None))
            pb.set(len(accepted), suffix=f"checked={total_checked}")
            if len(accepted) >= num_samples:
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
