"""从已有 VCC+IRIS-ZO cover 中采样多段随机安全路径，并回放录制视频。

示例:
    python generate_random_vcc_paths_and_replay.py
    python generate_random_vcc_paths_and_replay.py \
        --cover /root/autodl-tmp/DifferentiableOptimizationCBF-main/artifacts/sdf_exp/vcc_iris_cover.json \
        --num-curves 6 --video /root/autodl-tmp/DifferentiableOptimizationCBF-main/artifacts/videos/vcc_random_paths.mp4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.types import IrisRegion
from CBF_experiment.active.pybullet.self_collision.vcc_iris.io.gui import (
    evaluate_curve,
    playback_multiple_curves_gui,
    sample_curve_in_region,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.coal_oracle import CoalSelfCollisionOracle


DEFAULT_COVER_JSON = str(REPO_ROOT / "artifacts" / "sdf_exp" / "vcc_iris_cover.json")
DEFAULT_EXPERIMENT_JSON = str(REPO_ROOT / "artifacts" / "sdf_exp" / "vcc_iris_experiment.json")
DEFAULT_OUTPUT_JSON = str(REPO_ROOT / "artifacts" / "sdf_exp" / "vcc_random_paths_report.json")
DEFAULT_VIDEO = str(REPO_ROOT / "artifacts" / "videos" / "vcc_random_paths.mp4")


def _load_regions(cover_json: str | Path) -> tuple[list[IrisRegion], dict]:
    path = Path(cover_json)
    if not path.exists():
        raise FileNotFoundError(f"找不到 cover 文件: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    regions = []
    for item in data.get("regions", []):
        regions.append(
            IrisRegion(
                region_id=int(item["region_id"]),
                source_clique_indices=tuple(int(x) for x in item.get("source_clique_indices", [])),
                A=np.asarray(item["A"], dtype=float),
                b=np.asarray(item["b"], dtype=float),
                center=np.asarray(item["center"], dtype=float),
                C=np.asarray(item["C"], dtype=float),
                log_det=float(item["log_det"]),
                iterations=tuple(item.get("iterations", [])),
            )
        )
    return regions, data.get("coverage", {})


def _pick_candidate_regions(regions: list[IrisRegion], *, max_pool: int, rng: random.Random) -> list[IrisRegion]:
    ranked = sorted(regions, key=lambda r: float(r.log_det), reverse=True)
    pool = ranked[: max(1, min(max_pool, len(ranked)))]
    rng.shuffle(pool)
    return pool


def _build_region_only_curve_report(region: IrisRegion, curve: np.ndarray) -> dict:
    A = np.asarray(region.A, dtype=float)
    b = np.asarray(region.b, dtype=float).reshape(-1)
    step_reports = []
    min_margin = float("inf")
    for step_idx, q in enumerate(np.asarray(curve, dtype=float), start=1):
        margin = float(np.min(b - A @ q))
        min_margin = min(min_margin, margin)
        step_reports.append({
            "step": int(step_idx),
            "min_clearance": float(margin),
            "is_collision": False,
            "active_pair": None,
        })
    return {
        "curve": np.asarray(curve, dtype=float).tolist(),
        "steps": step_reports,
        "min_clearance": float(min_margin),
        "worst_pair": None,
        "any_collision": False,
        "clearance_source": "polytope_margin",
    }


def _sample_safe_curves(
    regions: list[IrisRegion],
    *,
    oracle=None,
    num_curves: int,
    num_points: int,
    seed: int,
    max_attempts_per_region: int = 12,
) -> list[dict]:
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    candidate_regions = _pick_candidate_regions(regions, max_pool=max(24, num_curves * 4), rng=py_rng)
    curve_reports: list[dict] = []
    used_region_ids: set[int] = set()

    for region in candidate_regions:
        if len(curve_reports) >= num_curves:
            break
        for _ in range(max_attempts_per_region):
            curve = sample_curve_in_region(region, num_points=num_points, rng=np_rng)
            if oracle is not None:
                try:
                    report = evaluate_curve(oracle, curve)
                except Exception as exc:
                    print(f"[warn] oracle 复核失败，回退到 polytope margin: {exc}")
                    oracle = None
                    report = _build_region_only_curve_report(region, curve)
                if bool(report["any_collision"]):
                    continue
            else:
                report = _build_region_only_curve_report(region, curve)
            report["region_id"] = int(region.region_id)
            report["region_log_det"] = float(region.log_det)
            report["num_points"] = int(num_points)
            curve_reports.append(report)
            used_region_ids.add(int(region.region_id))
            break

    if len(curve_reports) < num_curves:
        ranked = sorted(regions, key=lambda r: float(r.log_det), reverse=True)
        for region in ranked:
            if len(curve_reports) >= num_curves:
                break
            for _ in range(max_attempts_per_region):
                curve = sample_curve_in_region(region, num_points=num_points, rng=np_rng)
                if oracle is not None:
                    try:
                        report = evaluate_curve(oracle, curve)
                    except Exception as exc:
                        print(f"[warn] oracle 复核失败，回退到 polytope margin: {exc}")
                        oracle = None
                        report = _build_region_only_curve_report(region, curve)
                    if bool(report["any_collision"]):
                        continue
                else:
                    report = _build_region_only_curve_report(region, curve)
                report["region_id"] = int(region.region_id)
                report["region_log_det"] = float(region.log_det)
                report["num_points"] = int(num_points)
                curve_reports.append(report)
                break

    if len(curve_reports) < num_curves:
        raise RuntimeError(f"仅成功采样到 {len(curve_reports)} 条安全曲线，少于目标 {num_curves}。")
    return curve_reports[:num_curves]


def main():
    parser = argparse.ArgumentParser(description="从 VCC 凸分解中生成多段随机安全路径并回放录制")
    parser.add_argument("--cover", default=DEFAULT_COVER_JSON, help="cover JSON 路径")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT_JSON, help="experiment JSON 路径，仅用于记录来源")
    parser.add_argument("--num-curves", type=int, default=6, help="随机生成的安全路径条数")
    parser.add_argument("--num-points", type=int, default=80, help="每条曲线的采样点数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--speed", type=float, default=0.08, help="GUI 每帧间隔秒数")
    parser.add_argument("--hold", type=float, default=5.0, help="播放结束后保持窗口秒数")
    parser.add_argument("--between", type=float, default=0.6, help="曲线之间的停顿秒数")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="输出 mp4 路径")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="输出汇总 JSON 路径")
    parser.add_argument("--video-fps", type=int, default=15, help="视频帧率")
    parser.add_argument("--skip-validation", action="store_true", help="不使用 coal oracle 复核，直接按 region 内部曲线输出")
    args = parser.parse_args()

    regions, coverage = _load_regions(args.cover)
    print(f"[load] regions={len(regions)}  cover_ratio={coverage.get('ratio')}")
    print(f"[load] source experiment={args.experiment}")

    oracle = None if args.skip_validation else CoalSelfCollisionOracle(RobotQueryConfig())
    try:
        curve_reports = _sample_safe_curves(
            regions,
            oracle=oracle,
            num_curves=int(args.num_curves),
            num_points=int(args.num_points),
            seed=int(args.seed),
        )
    finally:
        if oracle is not None:
            oracle.close()

    output = {
        "source_cover_json": str(Path(args.cover)),
        "source_experiment_json": str(Path(args.experiment)),
        "num_curves": len(curve_reports),
        "seed": int(args.seed),
        "coverage": coverage,
        "curve_reports": curve_reports,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] 多曲线报告已写入: {output_path}")
    for idx, report in enumerate(curve_reports, start=1):
        print(
            f"[curve {idx}] region={report['region_id']}  "
            f"log_det={report['region_log_det']:+.4f}  "
            f"min_clearance={report['min_clearance']:+.6f}  "
            f"collision={report['any_collision']}"
        )

    playback_multiple_curves_gui(
        RobotQueryConfig(),
        curve_reports,
        sleep_dt=float(args.speed),
        hold_seconds=float(args.hold),
        between_curves_seconds=float(args.between),
        video_output_path=args.video,
        video_fps=int(args.video_fps),
    )


if __name__ == "__main__":
    main()
