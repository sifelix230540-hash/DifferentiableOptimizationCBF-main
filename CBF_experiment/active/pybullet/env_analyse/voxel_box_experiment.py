"""Greedy voxel-to-box experiment for occupancy grids.

Input:
    *_udf_occ.npz produced by occupancy bake.

Output:
    - boxes.json: axis-aligned boxes in world coordinates
    - report.json: compression / expansion statistics
    - projections.png: occupancy vs box-cover projections

This experiment prioritizes geometric fit while allowing limited conservative
expansion into empty voxels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class BoxRecord:
    ijk_min: list[int]
    ijk_max: list[int]  # exclusive upper bound
    world_min: list[float]
    world_max: list[float]
    world_center: list[float]
    half_extents: list[float]
    voxel_volume: int
    occupied_count: int
    empty_count: int
    fill_ratio: float
    expansion_ratio: float


def _box_lo_hi(box: BoxRecord) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(box.ijk_min, dtype=np.int64),
        np.asarray(box.ijk_max, dtype=np.int64),
    )


def _load_occ_npz(path: Path) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as data:
        shape = tuple(int(x) for x in data["grid_shape"])
        bits = np.asarray(data["grid"], dtype=np.uint8)
        flat = np.unpackbits(bits)[: int(np.prod(shape))]
        grid = flat.astype(np.bool_).reshape(shape)
        origin = np.asarray(data["origin"], dtype=np.float64).reshape(3)
        spacing = float(np.asarray(data["occ_spacing"]).reshape(()))
        bbox_min = np.asarray(data["bbox_min"], dtype=np.float64).reshape(3)
        bbox_max = np.asarray(data["bbox_max"], dtype=np.float64).reshape(3)
    return grid, origin, spacing, bbox_min, bbox_max


def _region_slices(lo: np.ndarray, hi: np.ndarray) -> tuple[slice, slice, slice]:
    return tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))  # type: ignore[return-value]


def _build_prefix_sum(grid: np.ndarray) -> np.ndarray:
    prefix = grid.astype(np.int64, copy=False)
    prefix = prefix.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    prefix = np.pad(prefix, ((1, 0), (1, 0), (1, 0)), mode="constant")
    return prefix


def _prefix_region_sum(prefix: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> int:
    x0, y0, z0 = [int(v) for v in lo]
    x1, y1, z1 = [int(v) for v in hi]
    return int(
        prefix[x1, y1, z1]
        - prefix[x0, y1, z1]
        - prefix[x1, y0, z1]
        - prefix[x1, y1, z0]
        + prefix[x0, y0, z1]
        + prefix[x0, y1, z0]
        + prefix[x1, y0, z0]
        - prefix[x0, y0, z0]
    )


def _evaluate_region(
    active_occ: np.ndarray,
    claimed: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[int, int, float, float] | None:
    region = _region_slices(lo, hi)
    if claimed[region].any():
        return None
    volume = int(np.prod(hi - lo))
    occupied_count = int(active_occ[region].sum())
    if occupied_count <= 0:
        return None
    empty_count = volume - occupied_count
    fill_ratio = occupied_count / max(volume, 1)
    expansion_ratio = empty_count / max(volume, 1)
    return occupied_count, empty_count, fill_ratio, expansion_ratio


def _evaluate_region_prefix(
    occ_prefix: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[int, int, float, float]:
    volume = int(np.prod(hi - lo))
    occupied_count = _prefix_region_sum(occ_prefix, lo, hi)
    empty_count = volume - occupied_count
    fill_ratio = occupied_count / max(volume, 1)
    expansion_ratio = empty_count / max(volume, 1)
    return occupied_count, empty_count, fill_ratio, expansion_ratio


def _grow_box(
    seed: np.ndarray,
    active_occ: np.ndarray,
    claimed: np.ndarray,
    *,
    min_fill_ratio: float,
    max_expansion_ratio: float,
) -> tuple[np.ndarray, np.ndarray, int, int, float, float]:
    shape = np.asarray(active_occ.shape, dtype=int)
    lo = seed.astype(int).copy()
    hi = lo + 1

    base = _evaluate_region(active_occ, claimed, lo, hi)
    if base is None:
        raise RuntimeError("seed must lie in an uncovered occupied voxel")
    occ_count, empty_count, fill_ratio, expansion_ratio = base

    while True:
        best = None
        best_score = None
        for axis in range(3):
            if hi[axis] >= shape[axis]:
                continue
            cand_lo = lo.copy()
            cand_hi = hi.copy()
            cand_hi[axis] += 1
            stats = _evaluate_region(active_occ, claimed, cand_lo, cand_hi)
            if stats is None:
                continue
            cand_occ, cand_empty, cand_fill, cand_expand = stats
            if cand_fill < min_fill_ratio or cand_expand > max_expansion_ratio:
                continue
            gain_occ = cand_occ - occ_count
            gain_empty = cand_empty - empty_count
            score = (
                cand_fill,
                gain_occ - gain_empty,
                gain_occ,
                -gain_empty,
                -axis,
            )
            if best_score is None or score > best_score:
                best_score = score
                best = (cand_lo, cand_hi, cand_occ, cand_empty, cand_fill, cand_expand)
        if best is None:
            break
        lo, hi, occ_count, empty_count, fill_ratio, expansion_ratio = best

    return lo, hi, occ_count, empty_count, fill_ratio, expansion_ratio


def greedy_voxel_boxes(
    occ_grid: np.ndarray,
    *,
    min_fill_ratio: float,
    max_expansion_ratio: float,
) -> tuple[list[BoxRecord], np.ndarray]:
    active_occ = occ_grid.copy()
    claimed = np.zeros_like(occ_grid, dtype=np.bool_)
    occ_indices = np.argwhere(occ_grid)
    total_occ = int(occ_grid.sum())
    covered_occ = 0
    boxes: list[BoxRecord] = []

    for idx, seed in enumerate(occ_indices):
        seed_t = tuple(int(v) for v in seed)
        if not active_occ[seed_t]:
            continue
        lo, hi, occ_count, empty_count, fill_ratio, expansion_ratio = _grow_box(
            seed.astype(int),
            active_occ,
            claimed,
            min_fill_ratio=min_fill_ratio,
            max_expansion_ratio=max_expansion_ratio,
        )
        region = _region_slices(lo, hi)
        claimed[region] = True
        active_occ[region] = False
        covered_occ += occ_count
        boxes.append(BoxRecord(
            ijk_min=lo.astype(int).tolist(),
            ijk_max=hi.astype(int).tolist(),
            world_min=[],
            world_max=[],
            world_center=[],
            half_extents=[],
            voxel_volume=int(np.prod(hi - lo)),
            occupied_count=occ_count,
            empty_count=empty_count,
            fill_ratio=round(fill_ratio, 6),
            expansion_ratio=round(expansion_ratio, 6),
        ))
        if (len(boxes) % 500 == 0) or (idx == len(occ_indices) - 1):
            print(
                f"[voxel-box] boxes={len(boxes)}  "
                f"covered_occ={covered_occ}/{total_occ} "
                f"({100.0 * covered_occ / max(total_occ, 1):.1f}%)",
                flush=True,
            )

    return boxes, claimed


def finalize_world_geometry(
    boxes: list[BoxRecord],
    origin: np.ndarray,
    spacing: float,
) -> None:
    for box in boxes:
        ijk_min = np.asarray(box.ijk_min, dtype=np.float64)
        ijk_max = np.asarray(box.ijk_max, dtype=np.float64)
        world_min = origin + ijk_min * spacing
        world_max = origin + ijk_max * spacing
        center = 0.5 * (world_min + world_max)
        half_extents = 0.5 * (world_max - world_min)
        box.world_min = np.round(world_min, 6).tolist()
        box.world_max = np.round(world_max, 6).tolist()
        box.world_center = np.round(center, 6).tolist()
        box.half_extents = np.round(half_extents, 6).tolist()


def boxes_to_mask(shape: tuple[int, int, int], boxes: list[BoxRecord]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.bool_)
    for box in boxes:
        lo, hi = _box_lo_hi(box)
        mask[_region_slices(lo, hi)] = True
    return mask


def _ranges_within_gap(
    lo_a: np.ndarray,
    hi_a: np.ndarray,
    lo_b: np.ndarray,
    hi_b: np.ndarray,
    max_gap: int,
) -> bool:
    for ax in range(3):
        gap = max(int(lo_b[ax] - hi_a[ax]), int(lo_a[ax] - hi_b[ax]), 0)
        if gap > max_gap:
            return False
    return True


def merge_boxes(
    boxes: list[BoxRecord],
    occ_grid: np.ndarray,
    *,
    min_fill_ratio: float,
    max_expansion_ratio: float,
    max_gap: int,
) -> tuple[list[BoxRecord], int]:
    if len(boxes) <= 1:
        return boxes, 0

    occ_prefix = _build_prefix_sum(occ_grid)
    boxes_cur = list(boxes)
    merge_count = 0

    while True:
        best_pair: tuple[int, int] | None = None
        best_payload = None
        best_score = None

        for i in range(len(boxes_cur)):
            lo_i, hi_i = _box_lo_hi(boxes_cur[i])
            for j in range(i + 1, len(boxes_cur)):
                lo_j, hi_j = _box_lo_hi(boxes_cur[j])
                if not _ranges_within_gap(lo_i, hi_i, lo_j, hi_j, max_gap):
                    continue
                lo = np.minimum(lo_i, lo_j)
                hi = np.maximum(hi_i, hi_j)
                occ_count, empty_count, fill_ratio, expansion_ratio = _evaluate_region_prefix(
                    occ_prefix, lo, hi,
                )
                if fill_ratio < min_fill_ratio or expansion_ratio > max_expansion_ratio:
                    continue
                old_empty = boxes_cur[i].empty_count + boxes_cur[j].empty_count
                extra_empty = empty_count - old_empty
                score = (
                    -extra_empty,
                    fill_ratio,
                    int(np.prod(hi - lo)),
                    -j,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (i, j)
                    best_payload = (lo, hi, occ_count, empty_count, fill_ratio, expansion_ratio)

        if best_pair is None or best_payload is None:
            break

        i, j = best_pair
        lo, hi, occ_count, empty_count, fill_ratio, expansion_ratio = best_payload
        merged = BoxRecord(
            ijk_min=lo.astype(int).tolist(),
            ijk_max=hi.astype(int).tolist(),
            world_min=[],
            world_max=[],
            world_center=[],
            half_extents=[],
            voxel_volume=int(np.prod(hi - lo)),
            occupied_count=int(occ_count),
            empty_count=int(empty_count),
            fill_ratio=round(float(fill_ratio), 6),
            expansion_ratio=round(float(expansion_ratio), 6),
        )
        boxes_cur.pop(j)
        boxes_cur.pop(i)
        boxes_cur.append(merged)
        merge_count += 1
        if merge_count % 25 == 0:
            print(f"[voxel-box] merge_pass merged={merge_count}  boxes={len(boxes_cur)}", flush=True)

    return boxes_cur, merge_count


def save_projection_png(
    occ_grid: np.ndarray,
    claimed: np.ndarray,
    out_png: Path,
) -> None:
    occ_xy = occ_grid.any(axis=2)
    occ_xz = occ_grid.any(axis=1)
    occ_yz = occ_grid.any(axis=0)
    box_xy = claimed.any(axis=2)
    box_xz = claimed.any(axis=1)
    box_yz = claimed.any(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    views = [
        ("Occupancy XY", occ_xy),
        ("Occupancy XZ", occ_xz),
        ("Occupancy YZ", occ_yz),
        ("Boxes XY", box_xy),
        ("Boxes XZ", box_xz),
        ("Boxes YZ", box_yz),
    ]
    for ax, (title, arr) in zip(axes.ravel(), views):
        ax.imshow(arr.T, origin="lower", interpolation="nearest", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("axis-0")
        ax.set_ylabel("axis-1")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=160)
    plt.close(fig)


def save_plotly_html(
    occ_grid: np.ndarray,
    boxes: list[BoxRecord],
    origin: np.ndarray,
    spacing: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    out_html: Path,
    *,
    occ_max_points: int = 60000,
    box_edge_limit: int = 1200,
) -> bool:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("[voxel-box] plotly not installed; skip 3D html", flush=True)
        return False

    occ_idx = np.argwhere(occ_grid)
    centers = origin + (occ_idx + 0.5) * spacing
    if centers.shape[0] > occ_max_points:
        rng = np.random.default_rng(42)
        sel = rng.choice(centers.shape[0], occ_max_points, replace=False)
        centers = centers[sel]

    traces = [
        go.Scatter3d(
            x=centers[:, 0].tolist(),
            y=centers[:, 1].tolist(),
            z=centers[:, 2].tolist(),
            mode="markers",
            marker=dict(size=2, color="tomato", opacity=0.25),
            name=f"occupied voxels ({centers.shape[0]:,})",
        )
    ]

    edge_boxes = sorted(boxes, key=lambda b: b.voxel_volume, reverse=True)[:box_edge_limit]
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    edge_pairs = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for box in edge_boxes:
        mn = np.asarray(box.world_min, dtype=float)
        mx = np.asarray(box.world_max, dtype=float)
        verts = np.array([
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mx[1], mx[2]],
        ])
        for a, b in edge_pairs:
            xs.extend([float(verts[a, 0]), float(verts[b, 0]), None])
            ys.extend([float(verts[a, 1]), float(verts[b, 1]), None])
            zs.extend([float(verts[a, 2]), float(verts[b, 2]), None])
    traces.append(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="royalblue", width=3),
        name=f"box edges ({len(edge_boxes)} shown)",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text="Voxel Occupancy and Merged Boxes", x=0.5),
        scene=dict(
            xaxis=dict(title="X (m)", range=[float(bbox_min[0]), float(bbox_max[0])]),
            yaxis=dict(title="Y (m)", range=[float(bbox_min[1]), float(bbox_max[1])]),
            zaxis=dict(title="Z (m)", range=[float(bbox_min[2]), float(bbox_max[2])]),
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.75)"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=760,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(out_html), include_plotlyjs="cdn", full_html=True)
    print(f"[voxel-box] 3d html -> {out_html}", flush=True)
    return True


def build_report(
    occ_grid: np.ndarray,
    claimed: np.ndarray,
    boxes: list[BoxRecord],
    occ_path: Path,
    spacing: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    min_fill_ratio: float,
    max_expansion_ratio: float,
    *,
    pre_merge_box_count: int | None = None,
    merge_count: int = 0,
) -> dict:
    occ_count = int(occ_grid.sum())
    claimed_count = int(claimed.sum())
    extra_count = int(np.logical_and(claimed, ~occ_grid).sum())
    uncovered_count = int(np.logical_and(occ_grid, ~claimed).sum())
    fill_ratios = np.asarray([b.fill_ratio for b in boxes], dtype=np.float64) if boxes else np.zeros(0)
    expansion_ratios = np.asarray([b.expansion_ratio for b in boxes], dtype=np.float64) if boxes else np.zeros(0)
    voxel_volumes = np.asarray([b.voxel_volume for b in boxes], dtype=np.int64) if boxes else np.zeros(0, dtype=np.int64)
    return {
        "input_npz": str(occ_path),
        "grid_shape": list(map(int, occ_grid.shape)),
        "spacing": spacing,
        "bbox_min": np.round(bbox_min, 6).tolist(),
        "bbox_max": np.round(bbox_max, 6).tolist(),
        "n_occupied_voxels": occ_count,
        "n_claimed_voxels": claimed_count,
        "n_extra_voxels": extra_count,
        "n_uncovered_voxels": uncovered_count,
        "coverage_ratio": round((occ_count - uncovered_count) / max(occ_count, 1), 6),
        "extra_over_occ_ratio": round(extra_count / max(occ_count, 1), 6),
        "n_boxes": len(boxes),
        "n_boxes_pre_merge": int(pre_merge_box_count if pre_merge_box_count is not None else len(boxes)),
        "merge_count": int(merge_count),
        "min_fill_ratio_cfg": min_fill_ratio,
        "max_expansion_ratio_cfg": max_expansion_ratio,
        "box_fill_ratio_mean": round(float(fill_ratios.mean()), 6) if fill_ratios.size else 0.0,
        "box_fill_ratio_min": round(float(fill_ratios.min()), 6) if fill_ratios.size else 0.0,
        "box_fill_ratio_max": round(float(fill_ratios.max()), 6) if fill_ratios.size else 0.0,
        "box_expansion_ratio_mean": round(float(expansion_ratios.mean()), 6) if expansion_ratios.size else 0.0,
        "box_voxel_volume_mean": round(float(voxel_volumes.mean()), 3) if voxel_volumes.size else 0.0,
        "box_voxel_volume_max": int(voxel_volumes.max()) if voxel_volumes.size else 0,
    }


def parse_args() -> argparse.Namespace:
    default_occ = REPO_ROOT / "assets/cad_exports/model_CAD/scene/urdf/中组立0725(1).stp.SLDASM_udf_occ.npz"
    default_out = REPO_ROOT / "artifacts/voxel_box_exp"
    parser = argparse.ArgumentParser(description="Greedy voxel-to-box experiment")
    parser.add_argument("--occ-npz", type=Path, default=default_occ)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--min-fill-ratio", type=float, default=0.72)
    parser.add_argument("--max-expansion-ratio", type=float, default=0.28)
    parser.add_argument("--merge-fill-ratio", type=float, default=0.66)
    parser.add_argument("--merge-expansion-ratio", type=float, default=0.34)
    parser.add_argument("--merge-gap", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    occ_grid, origin, spacing, bbox_min, bbox_max = _load_occ_npz(args.occ_npz)
    print(
        f"[voxel-box] shape={occ_grid.shape} spacing={spacing:.4f}m "
        f"occupied={int(occ_grid.sum())}",
        flush=True,
    )

    boxes, _claimed_initial = greedy_voxel_boxes(
        occ_grid,
        min_fill_ratio=float(args.min_fill_ratio),
        max_expansion_ratio=float(args.max_expansion_ratio),
    )
    pre_merge_box_count = len(boxes)
    boxes, merge_count = merge_boxes(
        boxes,
        occ_grid,
        min_fill_ratio=float(args.merge_fill_ratio),
        max_expansion_ratio=float(args.merge_expansion_ratio),
        max_gap=int(args.merge_gap),
    )
    finalize_world_geometry(boxes, origin, spacing)
    claimed = boxes_to_mask(occ_grid.shape, boxes)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes_json = out_dir / "boxes.json"
    report_json = out_dir / "report.json"
    projections_png = out_dir / "projections.png"
    html_3d = out_dir / "boxes_3d.html"

    report = build_report(
        occ_grid,
        claimed,
        boxes,
        args.occ_npz,
        spacing,
        bbox_min,
        bbox_max,
        float(args.min_fill_ratio),
        float(args.max_expansion_ratio),
        pre_merge_box_count=pre_merge_box_count,
        merge_count=merge_count,
    )
    boxes_payload = {
        "input_npz": str(args.occ_npz),
        "spacing": spacing,
        "n_boxes": len(boxes),
        "boxes": [asdict(b) for b in boxes],
    }

    boxes_json.write_text(json.dumps(boxes_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_projection_png(occ_grid, claimed, projections_png)
    save_plotly_html(occ_grid, boxes, origin, spacing, bbox_min, bbox_max, html_3d)

    print(f"[voxel-box] boxes  -> {boxes_json}")
    print(f"[voxel-box] report -> {report_json}")
    print(f"[voxel-box] image  -> {projections_png}")
    print(f"[voxel-box] 3d    -> {html_3d}")
    print(
        "[voxel-box] summary: "
        f"boxes={report['n_boxes_pre_merge']}->{report['n_boxes']}  "
        f"coverage={report['coverage_ratio']:.4f}  "
        f"extra/occ={report['extra_over_occ_ratio']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
