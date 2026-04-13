from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pair_key(pair) -> str:
    if not pair:
        return "none"
    return f"{int(pair[0])}-{int(pair[1])}"


def plot_coal_validation_report(report_json: str | Path, output_png: str | Path) -> Path:
    report = load_json(report_json)
    samples = list(report.get("samples", []))
    disagreements = list(report.get("disagreements", []))

    matrix_counter = Counter(
        (bool(item.get("pybullet_is_collision", False)), bool(item.get("coal_is_collision", False)))
        for item in samples
    )
    labels = ["TT", "TF", "FT", "FF"]
    values = [
        matrix_counter.get((True, True), 0),
        matrix_counter.get((True, False), 0),
        matrix_counter.get((False, True), 0),
        matrix_counter.get((False, False), 0),
    ]

    disagree_pair_counter = Counter(_pair_key(item.get("pybullet_active_pair")) for item in disagreements)
    top_pairs = disagree_pair_counter.most_common(8)
    pair_labels = [item[0] for item in top_pairs] if top_pairs else ["none"]
    pair_values = [item[1] for item in top_pairs] if top_pairs else [0]

    pybullet_dist = np.asarray([float(item.get("pybullet_min_distance", 0.0)) for item in samples], dtype=float)
    coal_dist = np.asarray([float(item.get("coal_min_distance", 0.0)) for item in samples], dtype=float)

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.5, 8.0))

    ax1 = fig.add_subplot(2, 2, 1)
    bars1 = ax1.bar(labels, values, color=["seagreen", "tomato", "mediumpurple", "royalblue"])
    ax1.set_title("Agreement Matrix")
    ax1.set_ylabel("count")
    ax1.set_xlabel("PyBullet / coal")
    for bar, value in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, value + 0.3, str(value), ha="center", va="bottom", fontsize=9)

    ax2 = fig.add_subplot(2, 2, 2)
    bars2 = ax2.bar(pair_labels, pair_values, color="darkorange")
    ax2.set_title("Disagreement Pair Frequency")
    ax2.set_ylabel("count")
    ax2.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars2, pair_values):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, value + 0.2, str(value), ha="center", va="bottom", fontsize=9)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(pybullet_dist, bins=20, alpha=0.70, color="crimson", label="PyBullet")
    ax3.hist(coal_dist, bins=20, alpha=0.55, color="navy", label="coal")
    ax3.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax3.set_title("Min-distance Distribution")
    ax3.set_xlabel("distance")
    ax3.set_ylabel("count")
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    idx = np.arange(len(samples))
    ax4.scatter(idx, pybullet_dist, s=20, color="crimson", alpha=0.75, label="PyBullet")
    ax4.scatter(idx, coal_dist, s=20, color="navy", alpha=0.75, label="coal")
    ax4.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax4.set_title("Sample-wise Distance Comparison")
    ax4.set_xlabel("sample index")
    ax4.set_ylabel("distance")
    ax4.legend()

    fig.suptitle(
        "Self-collision PyBullet vs coal\n"
        f"samples={report.get('num_samples', 0)}  "
        f"agreement_rate={float(report.get('agreement_rate', 0.0)):.3f}  "
        f"disagreements={int(report.get('disagreement_count', 0))}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=150, format="png")
    plt.close(fig)
    print(f"[self-cspace-coal-plot] saved -> {output_path}")
    return output_path


if __name__ == "__main__":
    report_path = Path("artifacts/sdf_exp/self_collision_coal_validation_medium.json")
    output_path = Path("artifacts/sdf_exp/self_collision_coal_validation_medium.png")
    plot_coal_validation_report(report_path, output_path)
