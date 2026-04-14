from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.self_collision.self_collision_cspace_hull_evaluation import evaluate_hulls_from_files  # noqa: E402
from CBF_experiment.active.pybullet.self_collision.self_collision_cspace_hulls import (  # noqa: E402
    build_monitored_link_pairs,
    classify_self_collision_sample,
    extract_revolute_metadata,
    extract_self_collision_monitor_metadata,
    monte_carlo_self_collision_hulls,
    sample_revolute_configurations,
)
from CBF_experiment.active.pybullet.main_pipe_line.simulation_module import Robot, load_config, _resolve  # noqa: E402


class BenchmarkParameters:
    CFG_PATH = None
    WORK_DIR = Path(_resolve("artifacts/sdf_exp"))

    HULL_NUM_SAMPLES = 200000
    HULL_SEED = 7
    HULL_VOXEL_SIZE = 0.08
    HULL_MIN_CLUSTER_SIZE = 24
    PENETRATION_THRESH = -0.001
    MIN_INDEX_GAP = 2
    QUERY_DISTANCE = 0.12
    SAMPLE_MODE = "boundary_band"
    BOUNDARY_BAND = 0.02
    PROGRESS_EVERY = 250

    TEST_NUM_SAMPLES = 12000
    TEST_SEED = 17
    TEST_PROGRESS_EVERY = 500

    HULL_OUTPUT_JSON = str(WORK_DIR / "self_collision_cspace_hulls.json")
    HULL_OUTPUT_PNG = str(WORK_DIR / "self_collision_cspace_hulls.png")
    SAMPLE_OUTPUT_JSON = str(WORK_DIR / "self_collision_eval_samples.json")
    EVAL_OUTPUT_JSON = str(WORK_DIR / "self_collision_eval_report.json")
    EVAL_SUMMARY_PNG = str(WORK_DIR / "self_collision_eval_summary.png")


def _default_progress(prefix: str, current: int, total: int, extra: str = "") -> None:
    total = max(int(total), 1)
    frac = current / total
    width = 32
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{prefix}] |{bar}| {100.0 * frac:5.1f}% [{current}/{total}] {extra}", end="", flush=True)
    if current >= total:
        print()


def run_hull_builder(params) -> dict:
    return monte_carlo_self_collision_hulls(
        cfg_path=params.CFG_PATH,
        num_samples=int(params.HULL_NUM_SAMPLES),
        seed=int(params.HULL_SEED),
        voxel_size=float(params.HULL_VOXEL_SIZE),
        min_cluster_size=int(params.HULL_MIN_CLUSTER_SIZE),
        penetration_thresh=float(params.PENETRATION_THRESH),
        min_index_gap=int(params.MIN_INDEX_GAP),
        query_distance=float(params.QUERY_DISTANCE),
        sample_mode=str(params.SAMPLE_MODE),
        boundary_band=float(params.BOUNDARY_BAND),
        output_json=params.HULL_OUTPUT_JSON,
        output_png=params.HULL_OUTPUT_PNG,
        progress_every=int(params.PROGRESS_EVERY),
    )


def generate_evaluation_samples(params) -> dict:
    cfg = load_config(params.CFG_PATH)
    rng = np.random.default_rng(int(params.TEST_SEED))
    created_connection = False
    if not p.isConnected():
        p.connect(p.DIRECT)
        created_connection = True
    try:
        robot = Robot(cfg)
        q_base, dq_base = robot.get_joint_state()
        revolute_ids, _revolute_names, joint_limits, q_indices = extract_revolute_metadata(robot)
        monitored_link_ids, monitored_link_names = extract_self_collision_monitor_metadata(robot)
        monitored_pairs = build_monitored_link_pairs(monitored_link_ids, min_index_gap=int(params.MIN_INDEX_GAP))
        from CBF_experiment.active.pybullet.self_collision.self_collision_backend_coal import (
            build_coal_link_models,
        )
        link_models = build_coal_link_models(robot, monitored_link_ids)
        sampled_q = sample_revolute_configurations(
            q_base,
            q_indices,
            joint_limits,
            num_samples=int(params.TEST_NUM_SAMPLES),
            rng=rng,
        )

        collision_samples = []
        free_samples = []
        collision_distances = []
        free_distances = []
        for idx, q in enumerate(sampled_q, start=1):
            robot.set_joint_state(q, dq=np.zeros_like(q))
            metric = classify_self_collision_sample(
                robot,
                monitored_pairs=monitored_pairs,
                link_models=link_models,
                penetration_thresh=float(params.PENETRATION_THRESH),
            )
            revolute_q = np.asarray(q, dtype=float)[q_indices]
            if metric["is_collision"]:
                collision_samples.append(revolute_q.tolist())
                collision_distances.append(float(metric["min_distance"]))
            else:
                free_samples.append(revolute_q.tolist())
                free_distances.append(float(metric["min_distance"]))
            if idx == int(params.TEST_NUM_SAMPLES) or idx % int(params.TEST_PROGRESS_EVERY) == 0:
                _default_progress(
                    "self-cspace-samples",
                    idx,
                    int(params.TEST_NUM_SAMPLES),
                    extra=f"collision={len(collision_samples)} free={len(free_samples)}",
                )

        payload = {
            "collision_samples": collision_samples,
            "free_samples": free_samples,
            "collision_distances": collision_distances,
            "free_distances": free_distances,
            "dimension": int(len(joint_limits)),
            "num_collision_samples": int(len(collision_samples)),
            "num_free_samples": int(len(free_samples)),
            "joint_indices": [int(j) for j in revolute_ids],
            "monitored_link_indices": [int(j) for j in monitored_link_ids],
            "monitored_link_names": [str(name) for name in monitored_link_names],
        }
        out_path = Path(params.SAMPLE_OUTPUT_JSON)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[self-cspace-benchmark] sample json -> {out_path}")
        robot.set_joint_state(q_base, dq_base)
        return payload
    finally:
        if created_connection and p.isConnected():
            p.disconnect()


def run_hull_evaluation(params) -> dict:
    return evaluate_hulls_from_files(
        hull_json=params.HULL_OUTPUT_JSON,
        sample_json=params.SAMPLE_OUTPUT_JSON,
        output_json=params.EVAL_OUTPUT_JSON,
        boundary_band=float(params.BOUNDARY_BAND),
        space="joint",
    )


def write_evaluation_summary_plot(report: dict, output_png: str | Path) -> Path:
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("coverage", float(report.get("coverage", 0.0))),
        ("miss_rate", float(report.get("miss_rate", 0.0))),
        ("false_positive_rate", float(report.get("false_positive_rate", 0.0))),
        ("boundary_collision_coverage", float(report.get("boundary_collision_coverage", 0.0))),
        ("boundary_free_false_positive_rate", float(report.get("boundary_free_false_positive_rate", 0.0))),
    ]
    labels = [item[0] for item in metrics]
    values = [item[1] for item in metrics]
    fig = plt.figure(figsize=(9.0, 4.8))
    ax = fig.add_subplot(111)
    bars = ax.bar(np.arange(len(values)), values, color=["royalblue", "tomato", "darkorange", "seagreen", "mediumpurple"])
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ratio")
    ax.set_title("Self-collision C-space hull benchmark summary")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, min(value + 0.03, 0.98), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    with open(output_path, "wb") as fp:
        fig.savefig(fp, dpi=140, format="png")
    plt.close(fig)
    return output_path


def run_benchmark(params=BenchmarkParameters) -> dict:
    Path(params.WORK_DIR).mkdir(parents=True, exist_ok=True)
    print("[self-cspace-benchmark] step 1/3 build hulls")
    hull_payload = run_hull_builder(params)
    print("[self-cspace-benchmark] step 2/3 generate evaluation samples")
    sample_payload = generate_evaluation_samples(params)
    print("[self-cspace-benchmark] step 3/3 evaluate hulls")
    eval_report = run_hull_evaluation(params)
    summary_png = write_evaluation_summary_plot(eval_report, params.EVAL_SUMMARY_PNG)
    summary = {
        "work_dir": str(Path(params.WORK_DIR)),
        "hull": {
            "json": str(Path(params.HULL_OUTPUT_JSON)),
            "png": str(Path(params.HULL_OUTPUT_PNG)),
            "num_clusters": int(len(hull_payload.get("clusters", []))),
        },
        "samples": {
            "json": str(Path(params.SAMPLE_OUTPUT_JSON)),
            "num_collision_samples": int(sample_payload.get("num_collision_samples", 0)),
            "num_free_samples": int(sample_payload.get("num_free_samples", 0)),
        },
        "evaluation": eval_report,
        "summary_png": str(summary_png),
    }
    print(
        "[self-cspace-benchmark] "
        f"coverage={eval_report['coverage']:.4f} "
        f"miss_rate={eval_report['miss_rate']:.4f} "
        f"false_positive_rate={eval_report['false_positive_rate']:.4f}"
    )
    print(f"[self-cspace-benchmark] summary png -> {summary_png}")
    return summary


if __name__ == "__main__":
    run_benchmark(BenchmarkParameters)
