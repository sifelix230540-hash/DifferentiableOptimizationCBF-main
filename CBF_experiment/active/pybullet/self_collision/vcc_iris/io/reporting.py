"""将实验结果序列化为 cover / experiment 两份 JSON 文件。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import ExperimentReport


def _region_payload(region) -> dict:
    return {
        "region_id": int(region.region_id),
        "source_clique_indices": [int(x) for x in region.source_clique_indices],
        "A": np.asarray(region.A, dtype=float).tolist(),
        "b": np.asarray(region.b, dtype=float).tolist(),
        "center": np.asarray(region.center, dtype=float).tolist(),
        "C": np.asarray(region.C, dtype=float).tolist(),
        "log_det": float(region.log_det),
        "iterations": list(region.iterations),
    }


def write_experiment_report(report: ExperimentReport, cover_json_path: str, experiment_json_path: str):
    cover_payload = {
        "regions": [_region_payload(region) for region in report.regions],
        "coverage": {
            "num_hits": int(report.coverage.num_hits),
            "num_samples": int(report.coverage.num_samples),
            "ratio": float(report.coverage.ratio),
            "confidence_radius": float(report.coverage.confidence_radius),
        },
    }
    experiment_payload = {
        "sample_stats": dict(report.sample_stats),
        "visibility_stats": dict(report.visibility_stats),
        "clique_stats": dict(report.clique_stats),
        "coverage": cover_payload["coverage"],
        "curve_report": dict(report.curve_report),
        "regions": [_region_payload(region) for region in report.regions],
    }

    cover_path = Path(cover_json_path)
    cover_path.parent.mkdir(parents=True, exist_ok=True)
    cover_path.write_text(json.dumps(cover_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    experiment_path = Path(experiment_json_path)
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(json.dumps(experiment_payload, ensure_ascii=False, indent=2), encoding="utf-8")

