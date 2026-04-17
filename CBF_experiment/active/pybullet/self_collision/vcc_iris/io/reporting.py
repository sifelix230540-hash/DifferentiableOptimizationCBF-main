"""将实验结果序列化为 cover / experiment 两份 JSON 文件。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import (
    DualCoverageStats,
    DualExperimentReport,
    ExperimentReport,
)


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
    round_stats_payload = []
    for rs in (report.round_stats or ()):
        round_stats_payload.append({
            "round_id": int(rs.round_id),
            "num_samples": int(rs.num_samples),
            "num_pairs": int(rs.num_pairs),
            "num_visible_edges": int(rs.num_visible_edges),
            "num_cliques": int(rs.num_cliques),
            "clique_sizes": list(rs.clique_sizes),
            "num_regions_grown": int(rs.num_regions_grown),
            "coverage_after": float(rs.coverage_after),
            "elapsed_seconds": float(rs.elapsed_seconds),
        })
    experiment_payload = {
        "sample_stats": dict(report.sample_stats),
        "visibility_stats": dict(report.visibility_stats),
        "clique_stats": dict(report.clique_stats),
        "round_stats": round_stats_payload,
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


# ────────────────────────── Dual report 序列化 ──────────────────────────


def _dual_coverage_payload(stats: DualCoverageStats) -> dict:
    return {
        "num_uniform_samples": int(stats.num_uniform_samples),
        "num_pos_samples": int(stats.num_pos_samples),
        "num_neg_samples": int(stats.num_neg_samples),
        "num_in_pos_region": int(stats.num_in_pos_region),
        "num_in_neg_region": int(stats.num_in_neg_region),
        "num_in_overlap": int(stats.num_in_overlap),
        "cov_pos": float(stats.cov_pos),
        "cov_neg": float(stats.cov_neg),
        "cov_combined": float(stats.cov_combined),
        "balance": float(stats.balance),
        "cov_uncov_in_Cfree": float(stats.cov_uncov_in_Cfree),
        "cov_uncov_in_Cobs": float(stats.cov_uncov_in_Cobs),
        "cov_pos_boosted": float(stats.cov_pos_boosted),
        "cov_neg_boosted": float(stats.cov_neg_boosted),
        "cov_combined_confidence_radius": float(stats.cov_combined_confidence_radius),
    }


def write_dual_experiment_report(
    report: DualExperimentReport,
    *,
    cover_json_path: str,
    experiment_json_path: str,
):
    pos_payload = [_region_payload(r) for r in report.pos_regions]
    neg_payload = [_region_payload(r) for r in report.neg_regions]
    coverage_payload = _dual_coverage_payload(report.final_coverage)

    cover_payload = {
        "schema_version": "dual-1",
        "regions_pos": pos_payload,
        "regions_neg": neg_payload,
        "coverage": coverage_payload,
    }

    round_stats_payload = []
    for rs in (report.round_stats or ()):
        round_stats_payload.append({
            "macro_round_id": int(rs.macro_round_id),
            "side": str(rs.side),
            "sub_round_id": int(rs.sub_round_id),
            "num_samples": int(rs.num_samples),
            "num_pairs": int(rs.num_pairs),
            "num_visible_edges": int(rs.num_visible_edges),
            "num_cliques": int(rs.num_cliques),
            "clique_sizes": list(rs.clique_sizes),
            "num_regions_grown": int(rs.num_regions_grown),
            "elapsed_seconds": float(rs.elapsed_seconds),
            "coverage_after": _dual_coverage_payload(rs.coverage_after) if rs.coverage_after is not None else None,
        })

    experiment_payload = {
        "schema_version": "dual-1",
        "sample_stats": dict(report.sample_stats),
        "config_summary": dict(report.config_summary),
        "coverage_final": coverage_payload,
        "round_stats": round_stats_payload,
        "regions_pos": pos_payload,
        "regions_neg": neg_payload,
    }

    cover_path = Path(cover_json_path)
    cover_path.parent.mkdir(parents=True, exist_ok=True)
    cover_path.write_text(json.dumps(cover_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    experiment_path = Path(experiment_json_path)
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(json.dumps(experiment_payload, ensure_ascii=False, indent=2), encoding="utf-8")

