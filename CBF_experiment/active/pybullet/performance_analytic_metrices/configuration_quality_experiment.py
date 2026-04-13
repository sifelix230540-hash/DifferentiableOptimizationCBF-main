"""Independent six-axis configuration-quality benchmark around a fixed EE pose."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CBF_experiment.active.pybullet.configuration_metrics import compute_environment_clearance, compute_self_collision_clearance, evaluate_configuration_quality, extract_revolute_joint_limits, rank_configuration_records, summarize_clearance_entries  # noqa: E402
from CBF_experiment.active.pybullet.simulation_module import Robot, Workpiece, load_config, _resolve  # noqa: E402


def sample_joint_neighborhood(
    q_center,
    joint_limits,
    *,
    sigma: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    center = np.asarray(q_center, dtype=float).reshape(-1)
    samples = np.repeat(center.reshape(1, -1), int(n_samples), axis=0)
    for row in range(samples.shape[0]):
        for col, (lo, hi) in enumerate(joint_limits):
            samples[row, col] = float(samples[row, col] + rng.normal(0.0, sigma))
            if hi > lo:
                samples[row, col] = float(np.clip(samples[row, col], lo, hi))
    return samples


def _quat_angle_error(q_a: np.ndarray, q_b: np.ndarray) -> float:
    qa = Rotation.from_quat(np.asarray(q_a, dtype=float).reshape(4))
    qb = Rotation.from_quat(np.asarray(q_b, dtype=float).reshape(4))
    return float(np.linalg.norm((qb * qa.inv()).as_rotvec()))


def run(
    cfg_path: str | Path | None = None,
    *,
    output_json: str | Path | None = None,
    output_png: str | Path | None = None,
    target_pos=None,
    target_quat=None,
):
    cfg = load_config(cfg_path)
    quality_cfg = cfg.get("configuration_quality", {})
    bench_cfg = quality_cfg.get("benchmark", {})
    rng = np.random.default_rng(int(bench_cfg.get("seed", 7)))
    n_samples = int(bench_cfg.get("sample_count", 256))
    sigma = float(bench_cfg.get("sample_std", 0.35))
    motion_component = str(quality_cfg.get("motion_component", "linear"))
    self_query_distance = float(quality_cfg.get("self_collision_query_distance", 0.12))
    env_query_distance = float(bench_cfg.get("environment_query_distance", 0.20))
    ik_pos_tol = float(bench_cfg.get("ik_position_tolerance", 0.02))
    ik_ori_tol = float(bench_cfg.get("ik_orientation_tolerance", 0.10))

    created_connection = False
    if not p.isConnected():
        p.connect(p.DIRECT)
        created_connection = True
    try:
        robot = Robot(cfg)
        workpiece = Workpiece(cfg)
        q_seed, dq_seed = robot.get_joint_state()
        ee_pos_seed, ee_quat_seed = robot.get_ee_pose()
        target_pos = np.asarray(
            ee_pos_seed if target_pos is None else target_pos,
            dtype=float,
        ).reshape(3)
        target_quat = np.asarray(
            ee_quat_seed if target_quat is None else target_quat,
            dtype=float,
        ).reshape(4)
        joint_limits = extract_revolute_joint_limits(robot)
        rest_seed = q_seed[-len(joint_limits):] if joint_limits else np.zeros(0, dtype=float)
        rest_samples = sample_joint_neighborhood(
            rest_seed,
            joint_limits,
            sigma=sigma,
            n_samples=n_samples,
            rng=rng,
        )

        records: list[dict] = []
        for rest_revolute in rest_samples:
            rest_full = q_seed.copy()
            if rest_revolute.size:
                rest_full[-rest_revolute.size :] = rest_revolute
            q_candidate = robot.calculate_ik(
                target_pos.tolist(),
                target_quat.tolist(),
                rest_poses=rest_full.tolist(),
            )
            robot.set_joint_state(q_candidate, dq=np.zeros_like(q_candidate))
            ee_pos, ee_quat = robot.get_ee_pose()
            pos_err = float(np.linalg.norm(target_pos - ee_pos))
            ori_err = _quat_angle_error(ee_quat, target_quat)
            if pos_err > ik_pos_tol or ori_err > ik_ori_tol:
                continue
            self_clearance = compute_self_collision_clearance(
                robot,
                link_indices=robot.revolute_joints,
                min_index_gap=2,
                query_distance=self_query_distance,
            )
            env_clearance = compute_environment_clearance(
                robot,
                obstacle_body_id=workpiece.body_id,
                link_indices=robot.revolute_joints,
                max_distance=env_query_distance,
            )
            quality = evaluate_configuration_quality(
                robot,
                q_candidate,
                dq=dq_seed,
                motion_component=motion_component,
                clearance_summary=summarize_clearance_entries([
                    {"kind": "self_collision", "distance": float(self_clearance)},
                    {"kind": "environment", "distance": float(env_clearance)},
                ]),
                joint_limits=joint_limits,
            )
            records.append({
                "q": np.asarray(q_candidate, dtype=float).tolist(),
                "ik_position_error": pos_err,
                "ik_orientation_error": ori_err,
                "configuration_quality": quality,
            })

        ranked = rank_configuration_records(records, weights=quality_cfg.get("selection_weights"))
        payload = {
            "target_pos": target_pos.astype(float).tolist(),
            "target_quat": target_quat.astype(float).tolist(),
            "num_requested_samples": n_samples,
            "num_feasible_samples": len(ranked),
            "best_candidate": ranked[0] if ranked else None,
            "records": ranked,
            "online_objective": dict(quality_cfg.get("online_objective", {})),
        }

        out_json_path = Path(output_json or _resolve(bench_cfg.get("output_json", "artifacts/sdf_exp/config_quality_benchmark.json")))
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        out_png_path = Path(output_png or _resolve(bench_cfg.get("output_png", "artifacts/sdf_exp/config_quality_benchmark.png")))
        out_png_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(8.5, 5.2))
        ax = fig.add_subplot(111)
        if ranked:
            inv_cond = np.asarray([r["configuration_quality"]["inverse_condition"] for r in ranked], dtype=float)
            self_clearance = np.asarray([r["configuration_quality"]["self_collision_distance"] for r in ranked], dtype=float)
            finite_self = self_clearance[np.isfinite(self_clearance)]
            if finite_self.size:
                self_clearance[~np.isfinite(self_clearance)] = float(np.max(finite_self) * 1.05)
            scores = np.asarray([r["selection_score"] for r in ranked], dtype=float)
            sc = ax.scatter(inv_cond, self_clearance, c=scores, cmap="viridis", s=20, alpha=0.85)
            fig.colorbar(sc, ax=ax, label="selection score")
        else:
            ax.text(0.5, 0.5, "no feasible IK samples", ha="center", va="center")
        ax.set_xlabel("inverse condition")
        ax.set_ylabel("self-collision clearance (m)")
        ax.set_title("Six-axis configuration-quality benchmark")
        fig.tight_layout()
        with open(out_png_path, "wb") as fp:
            fig.savefig(fp, dpi=140, format="png")
        plt.close(fig)

        print(f"[config-quality] feasible {len(ranked)} / {n_samples}")
        print(f"[config-quality] saved -> {out_json_path}")
        print(f"[config-quality] plot -> {out_png_path}")
        return payload
    finally:
        if created_connection and p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    run()
