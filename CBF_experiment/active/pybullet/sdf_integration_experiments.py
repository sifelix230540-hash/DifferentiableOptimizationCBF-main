from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from CBF_experiment.active.pybullet.welding_320_common import ExperimentConfig
from CBF_experiment.active.pybullet.welding_320_robot import JakaRobot, WorkpieceModel


# =============================================================================
# 用户集中参数区（建议只改这里）
# =============================================================================
#
# 说明：
# 1) 这些参数是各子命令的默认值，CLI 传参会覆盖。
# 2) 与“初始位姿/焊点”相关的默认来源：
#    - 初始位姿：ExperimentConfig.gantry_initial_q
#    - 起始/终末焊点：ExperimentConfig.start_link_name / goal_link_name
# 3) 你可先只改本区，再直接运行命令，不必每次写一堆参数。
#
_CFG = ExperimentConfig()

# ---- 通用路径 ----
DEFAULT_SDF_NPZ = (
    "assets/cad_exports/model_CAD/scene/urdf/中组立0725(1).stp.SLDASM_udf.npz"
)
DEFAULT_ROBOT_URDF_PATH = _CFG.urdf_path
DEFAULT_WORKPIECE_URDF_PATH = _CFG.workpiece_urdf_path

# ---- 工艺语义（来自现有 welding 配置）----
DEFAULT_GANTRY_INITIAL_Q = tuple(float(x) for x in _CFG.gantry_initial_q)
DEFAULT_WELD_START_LINK = str(_CFG.start_link_name)  # 例如 l2
DEFAULT_WELD_GOAL_LINK = str(_CFG.goal_link_name)    # 例如 l3

# ---- 1) align 默认 ----
ALIGN_MODE = "known"  # known=仅使用已知外参；optimize=在已知外参上再优化
ALIGN_KIND = "auto"
ALIGN_PER_LINK_POINTS = 1200
ALIGN_MAX_POINTS = 20000
ALIGN_MAX_ITER = 160
ALIGN_SEED = 1
ALIGN_OUTPUT_JSON = "artifacts/sdf_exp/alignment_report.json"
ALIGN_OUTPUT_PNG = "artifacts/sdf_exp/alignment_error.png"

# ---- 2) nearest-region 默认（薄壁建议）----
NEAR_KIND = "auto"
NEAR_SEARCH_RADIUS_MIN = 0.02
NEAR_SEARCH_RADIUS_MAX = 0.50
NEAR_SEARCH_RINGS = 16
NEAR_SEARCH_STEP = 0.02
NEAR_SAMPLES_PER_RING = 150
NEAR_MIN_CLEARANCE = 0.01
NEAR_MIN_LINE_CLEARANCE = 0.005
NEAR_KERNEL_CLEARANCE = 0.008
NEAR_LINE_SAMPLES = 24
NEAR_TOP_K = 8
NEAR_SEED = 2
NEAR_OUTPUT_JSON = "artifacts/sdf_exp/nearest_region.json"
NEAR_OUTPUT_PNG = "artifacts/sdf_exp/nearest_region.png"
NEAR_SURFACE_NORMAL_EPS = 0.002       # SDF 有限差分步长（米），用于估计表面法线
NEAR_NORMAL_HALF_SPHERE = True        # True=仅搜索法线侧半球
NEAR_NORMAL_CONE_COS = 0.0           # 半球锥体内角余弦阈值（0=半球，0.5=60°锥）
NEAR_LINE_SKIP_RATIO = 0.05          # 连线检查跳过焊点端比例下限
NEAR_REQUIRE_ABOVE_WELD = True        # True=候选Z≥焊点Z（倒置臂）
NEAR_ABOVE_WELD_MIN_DZ = 0.0         # 候选Z至少比焊点高多少（米）

# ---- 3) init-config 默认 ----
INIT_NUM_SAMPLES = 400
INIT_SAMPLE_STD = 0.2
INIT_MIN_CLEARANCE = 0.005
INIT_BBOX_X = 1.2
INIT_BBOX_Y = 1.2
INIT_BBOX_Z = 1.2
INIT_VOXEL = 0.04
INIT_SEED = 3
INIT_OUTPUT_NPZ = "artifacts/sdf_exp/init_kernel.npz"
INIT_OUTPUT_PNG = "artifacts/sdf_exp/init_kernel.png"
INIT_SKIP_EXTERNAL_COLLISION = True
INIT_OUTPUT_JSON = "artifacts/sdf_exp/init_config_report.json"

# ---- 4) plan 默认 ----
PLAN_KIND = "auto"
PLAN_MIN_CLEARANCE = 0.01
PLAN_STEP_SIZE = 0.06
PLAN_NEAR_RADIUS = 0.15
PLAN_GOAL_SAMPLE_PROB = 0.12
PLAN_MAX_ITER = 5000
PLAN_GOAL_TOLERANCE = 0.08
PLAN_EDGE_CHECK_STEP = 0.02
PLAN_SMOOTH_ITERS = 250
PLAN_BOUND_MARGIN = 0.02
PLAN_OUTPUT_JSON = "artifacts/sdf_exp/rrt_star_path.json"
PLAN_OUTPUT_PNG = "artifacts/sdf_exp/rrt_star_path.png"
PLAN_AUTO_FIX_ENDPOINTS = True
PLAN_ENDPOINT_FIX_RADIUS = 0.30
PLAN_ENDPOINT_FIX_STEP = 0.02

PLAN_NEAREST_REGION_AS_GOAL = True    # True=nearest-region 输出直接作为 goal
PLAN_NEAREST_REGION_JSON = NEAR_OUTPUT_JSON  # 自动衔接 nearest-region
PLAN_INIT_CONFIG_NPZ = INIT_OUTPUT_NPZ       # 用于计算 robobase 初始位置

# 可选：若你希望直接用焊点作为规划起终点，改为具体 xyz；留空则自动计算。
PLAN_START = None  # 留空=自动用 robobase 初始世界坐标
PLAN_GOAL = None   # 留空=自动从 nearest-region JSON 读取


def _load_udf_module():
    path = Path(__file__).resolve().parent / "4_1_udf.py"
    spec = importlib.util.spec_from_file_location("udf_module_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclass 在执行时会通过 __module__ 到 sys.modules 反查；需提前注册。
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_vec3(text: str) -> np.ndarray:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"需要 3 个值，收到: {text}")
    return np.asarray(parts, dtype=float)


def _world_to_local(points_world: np.ndarray, world_pos: np.ndarray, world_quat: np.ndarray) -> np.ndarray:
    inv_pos, inv_quat = p.invertTransform(
        np.asarray(world_pos, dtype=float).tolist(),
        np.asarray(world_quat, dtype=float).tolist(),
    )
    rot = np.array(p.getMatrixFromQuaternion(inv_quat), dtype=float).reshape(3, 3)
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    return (rot @ pts.T).T + np.asarray(inv_pos, dtype=float).reshape(1, 3)


def _query_field(field, points: np.ndarray, kind: str) -> np.ndarray:
    vals = field.query(np.asarray(points, dtype=np.float32), kind=kind, clip=True)
    return np.asarray(vals, dtype=float).reshape(-1)


def _estimate_surface_normal(
    field,
    point: np.ndarray,
    kind: str,
    eps: float = 0.002,
) -> np.ndarray:
    """用 SDF 有限差分估计表面法线方向（指向正值/自由空间侧）。"""
    pt = np.asarray(point, dtype=float).reshape(3)
    grad = np.zeros(3, dtype=float)
    for ax in range(3):
        pp = pt.copy(); pp[ax] += eps
        pm = pt.copy(); pm[ax] -= eps
        vp = float(_query_field(field, pp.reshape(1, 3), kind=kind)[0])
        vm = float(_query_field(field, pm.reshape(1, 3), kind=kind)[0])
        grad[ax] = (vp - vm) / (2.0 * eps)
    nrm = np.linalg.norm(grad)
    if nrm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return grad / nrm


def _auto_fix_point_if_infeasible(
    field,
    point: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    kind: str,
    clearance: float,
    search_radius: float,
    search_step: float,
) -> tuple[np.ndarray, bool]:
    p0 = np.asarray(point, dtype=float).reshape(3)
    d0 = float(_query_field(field, p0.reshape(1, 3), kind=kind)[0])
    if d0 > clearance:
        return p0, True
    step = max(float(search_step), 1e-4)
    max_r = max(float(search_radius), step)
    best = None
    best_dist = float("inf")
    for r in np.arange(step, max_r + 0.5 * step, step):
        span = max(1, int(math.ceil(float(r) / step)))
        for ix in range(-span, span + 1):
            for iy in range(-span, span + 1):
                for iz in range(-span, span + 1):
                    off = np.asarray([ix, iy, iz], dtype=float) * step
                    if np.linalg.norm(off) > r + 0.5 * step:
                        continue
                    cand = np.clip(p0 + off, bounds_min, bounds_max)
                    d = float(_query_field(field, cand.reshape(1, 3), kind=kind)[0])
                    if d <= clearance:
                        continue
                    move = float(np.linalg.norm(cand - p0))
                    if move < best_dist:
                        best = cand
                        best_dist = move
        if best is not None:
            break
    if best is None:
        return p0, False
    return np.asarray(best, dtype=float), True


def _best_kind(field) -> str:
    if np.isfinite(np.asarray(field.o3d_sdf_grid)).any():
        return "o3d_sdf"
    if np.isfinite(np.asarray(field.igl_sdf_grid)).any():
        return "igl_sdf"
    return "udf"


def _make_robot_and_workpiece(cfg: ExperimentConfig) -> tuple[JakaRobot, WorkpieceModel]:
    class _SceneStub:
        pass

    robot = JakaRobot(cfg, _SceneStub())
    workpiece = WorkpieceModel(cfg)
    robot.register_surface_obstacle(workpiece.body_id, None)
    return robot, workpiece


def _collect_surface_points(robot: JakaRobot, body_id: int, per_link_points: int = 800) -> np.ndarray:
    clouds = robot.get_surface_visualization_clouds(
        body_id=body_id,
        link_indices=None,
        max_points_per_link=per_link_points,
    )
    if not clouds:
        return np.zeros((0, 3), dtype=float)
    pts = [np.asarray(c["points"], dtype=float).reshape(-1, 3) for c in clouds]
    return np.vstack(pts) if pts else np.zeros((0, 3), dtype=float)


def run_alignment_experiment(args) -> None:
    udf_mod = _load_udf_module()
    field = udf_mod.load_distance_field(args.sdf_npz)
    kind = args.kind if args.kind != "auto" else _best_kind(field)
    rng = np.random.default_rng(args.seed)

    p.connect(p.DIRECT)
    try:
        cfg = ExperimentConfig()
        if args.urdf_path:
            cfg.urdf_path = args.urdf_path
        if args.workpiece_urdf_path:
            cfg.workpiece_urdf_path = args.workpiece_urdf_path
        if args.workpiece_position is not None:
            cfg.workpiece_position = tuple(_parse_vec3(args.workpiece_position).tolist())
        if args.workpiece_orientation_deg is not None:
            cfg.workpiece_orientation_deg = tuple(_parse_vec3(args.workpiece_orientation_deg).tolist())
        robot, workpiece = _make_robot_and_workpiece(cfg)
        pts = _collect_surface_points(robot, workpiece.body_id, per_link_points=args.per_link_points)
        if pts.shape[0] == 0:
            raise RuntimeError("未采样到工件表面点，无法执行对齐实验。")
        if pts.shape[0] > args.max_points:
            idx = rng.choice(pts.shape[0], size=args.max_points, replace=False)
            pts = pts[idx]
    finally:
        p.disconnect()

    def eval_params(x: np.ndarray) -> np.ndarray:
        t = x[:3]
        r = Rotation.from_rotvec(x[3:]).as_matrix()
        pts_sdf = (r @ pts.T).T + t.reshape(1, 3)
        return _query_field(field, pts_sdf, kind=kind)

    def objective(x: np.ndarray) -> float:
        d = np.abs(eval_params(x))
        return float(np.median(d) + 0.25 * np.mean(d))

    # 已知解析变换：PyBullet 中工件是 R_cfg, t_cfg 作用到 URDF 系，
    # 而 SDF 是按 URDF 原始系烘焙，因此 pybullet->sdf = inverse(R_cfg, t_cfg)。
    r_cfg = Rotation.from_euler("xyz", np.asarray(cfg.workpiece_orientation_deg, dtype=float), degrees=True).as_matrix()
    t_cfg = np.asarray(cfg.workpiece_position, dtype=float).reshape(3)
    r_known = r_cfg.T
    t_known = -r_known @ t_cfg
    x_identity = np.zeros(6, dtype=float)
    x_known = np.zeros(6, dtype=float)
    x_known[:3] = t_known
    x_known[3:] = Rotation.from_matrix(r_known).as_rotvec()

    d_identity = np.abs(eval_params(x_identity))
    d_known = np.abs(eval_params(x_known))

    opt = None
    if args.align_mode == "known":
        x_final = x_known.copy()
        d_final = d_known
    elif args.align_mode == "optimize":
        opt = minimize(
            objective,
            x_known,
            method="Powell",
            options={"maxiter": int(args.max_iter), "xtol": 1e-4, "ftol": 1e-4},
        )
        x_final = np.asarray(opt.x, dtype=float)
        d_final = np.abs(eval_params(x_final))
    else:
        raise ValueError(f"未知 align_mode: {args.align_mode}")

    out_json = Path(args.output_json)
    _ensure_parent(out_json)
    report = {
        "align_mode": str(args.align_mode),
        "kind": kind,
        "n_points": int(pts.shape[0]),
        "identity": {
            "mean": float(np.mean(d_identity)),
            "p95": float(np.percentile(d_identity, 95)),
            "max": float(np.max(d_identity)),
        },
        "known": {
            "mean": float(np.mean(d_known)),
            "p95": float(np.percentile(d_known, 95)),
            "max": float(np.max(d_known)),
        },
        "optimized": {
            "mean": float(np.mean(d_final)),
            "p95": float(np.percentile(d_final, 95)),
            "max": float(np.max(d_final)),
        },
        "known_transform_pybullet_to_sdf": {
            "translation": x_known[:3].tolist(),
            "rotvec": x_known[3:].tolist(),
        },
        "transform_pybullet_to_sdf": {
            "translation": x_final[:3].tolist(),
            "rotvec": x_final[3:].tolist(),
        },
        "optimizer": {
            "success": bool(True if opt is None else opt.success),
            "message": "skipped (known mode)" if opt is None else str(opt.message),
            "nit": -1 if opt is None else int(getattr(opt, "nit", -1)),
            "fun": float(objective(x_final)) if opt is None else float(opt.fun),
        },
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(d_identity, bins=60, alpha=0.55, label="identity")
    axes[0].hist(d_known, bins=60, alpha=0.55, label="known")
    if args.align_mode == "optimize":
        axes[0].hist(d_final, bins=60, alpha=0.55, label="optimized")
    axes[0].set_title("Surface distance error histogram")
    axes[0].set_xlabel("|distance| (m)")
    axes[0].legend()

    show_n = min(2000, pts.shape[0])
    axes[1].scatter(pts[:show_n, 0], pts[:show_n, 1], c=d_final[:show_n], s=4, cmap="magma")
    axes[1].set_title("Final alignment error (XY)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].axis("equal")
    fig.tight_layout()
    out_png = Path(args.output_png)
    _ensure_parent(out_png)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    print(f"[align] 完成，结果写入: {out_json}")
    print(f"[align] 可视化写入: {out_png}")


def _load_kernel_offsets(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    with np.load(path) as data:
        if "kernel_offsets" not in data:
            raise KeyError(f"{path} 中缺少 kernel_offsets")
        return np.asarray(data["kernel_offsets"], dtype=float).reshape(-1, 3)


def _is_candidate_feasible(
    field,
    point: np.ndarray,
    weld_point: np.ndarray,
    kind: str,
    min_clearance: float,
    min_line_clearance: float,
    kernel_offsets: np.ndarray | None,
    kernel_clearance: float,
    line_samples: int,
    line_skip_ratio: float = 0.05,
) -> tuple[bool, dict]:
    # --- 层级 1: 候选点自身 SDF ---
    d0 = float(_query_field(field, point.reshape(1, 3), kind=kind)[0])
    if d0 <= min_clearance:
        return False, {"distance": d0, "reason": "point_clearance"}

    # --- 层级 2: 连线检查（跳过焊点端，焊点在表面 SDF≈0 会误报）---
    dist = float(np.linalg.norm(point - weld_point))
    if dist > 1e-6:
        t_skip = max(float(line_skip_ratio), 2.0 * min_line_clearance / dist)
    else:
        t_skip = float(line_skip_ratio)
    t_skip = min(t_skip, 0.5)
    ts = np.linspace(t_skip, 1.0, max(int(line_samples), 2))
    line = weld_point.reshape(1, 3) * (1.0 - ts[:, None]) + point.reshape(1, 3) * ts[:, None]
    d_line = _query_field(field, line, kind=kind)
    if float(np.min(d_line)) <= min_line_clearance:
        return False, {"distance": d0, "reason": "line_cross_wall"}

    # --- 层级 3: AABB 8角快速排除 ---
    min_kernel = None
    if kernel_offsets is not None and kernel_offsets.shape[0] > 0:
        ko = kernel_offsets
        aabb_min = ko.min(axis=0)
        aabb_max = ko.max(axis=0)
        corners = np.array([
            [aabb_min[0], aabb_min[1], aabb_min[2]],
            [aabb_min[0], aabb_min[1], aabb_max[2]],
            [aabb_min[0], aabb_max[1], aabb_min[2]],
            [aabb_min[0], aabb_max[1], aabb_max[2]],
            [aabb_max[0], aabb_min[1], aabb_min[2]],
            [aabb_max[0], aabb_min[1], aabb_max[2]],
            [aabb_max[0], aabb_max[1], aabb_min[2]],
            [aabb_max[0], aabb_max[1], aabb_max[2]],
        ]) + point.reshape(1, 3)
        d_corners = _query_field(field, corners, kind=kind)
        if float(np.min(d_corners)) <= kernel_clearance:
            return False, {"distance": d0, "min_kernel": float(np.min(d_corners)), "reason": "kernel_aabb"}

        # --- 层级 4: 全核检查 ---
        pts = point.reshape(1, 3) + kernel_offsets
        d_kernel = _query_field(field, pts, kind=kind)
        min_kernel = float(np.min(d_kernel))
        if min_kernel <= kernel_clearance:
            return False, {"distance": d0, "min_kernel": min_kernel, "reason": "kernel_collision"}
    return True, {"distance": d0, "min_kernel": min_kernel, "reason": "ok"}


def run_nearest_region_experiment(args) -> None:
    udf_mod = _load_udf_module()
    field = udf_mod.load_distance_field(args.sdf_npz)
    kind = args.kind if args.kind != "auto" else _best_kind(field)
    kernel_offsets = _load_kernel_offsets(args.kernel_npz)

    p.connect(p.DIRECT)
    try:
        cfg = ExperimentConfig()
        if args.workpiece_urdf_path:
            cfg.workpiece_urdf_path = args.workpiece_urdf_path
        workpiece = WorkpieceModel(cfg)
        if args.weld_point:
            weld_point = _parse_vec3(args.weld_point)
        else:
            weld_point, _ = workpiece.get_frame_pose(args.weld_link_name)
            weld_point = np.asarray(weld_point, dtype=float)
    finally:
        p.disconnect()

    # 薄壁结构更需要“按距离递增”的确定性搜索，避免随机漏检最近可行点。
    surface_normal = _estimate_surface_normal(
        field, weld_point, kind,
        eps=float(getattr(args, "surface_normal_eps", NEAR_SURFACE_NORMAL_EPS)),
    )
    print(f"[nearest] surface normal at weld: {surface_normal.tolist()}")

    use_half_sphere = bool(getattr(args, "normal_half_sphere", NEAR_NORMAL_HALF_SPHERE))
    cone_cos = float(getattr(args, "normal_cone_cos", NEAR_NORMAL_CONE_COS))
    require_above = bool(getattr(args, "require_above_weld", NEAR_REQUIRE_ABOVE_WELD))
    above_min_dz = float(getattr(args, "above_weld_min_dz", NEAR_ABOVE_WELD_MIN_DZ))
    line_skip_ratio = float(getattr(args, "line_skip_ratio", NEAR_LINE_SKIP_RATIO))

    rng = np.random.default_rng(args.seed)
    candidates = []
    checked = set()
    bmin = np.asarray(field.bbox_min, dtype=float)
    bmax = np.asarray(field.bbox_max, dtype=float)
    search_step = max(float(args.search_step), 1e-4)

    for r in np.linspace(args.search_radius_min, args.search_radius_max, args.search_rings):
        span = max(1, int(math.ceil(float(r) / search_step)))
        offsets = []
        for ix in range(-span, span + 1):
            for iy in range(-span, span + 1):
                for iz in range(-span, span + 1):
                    if ix == 0 and iy == 0 and iz == 0:
                        continue
                    off = np.asarray([ix, iy, iz], dtype=float) * search_step
                    d = float(np.linalg.norm(off))
                    if d < float(r) - 0.5 * search_step or d > float(r) + 0.5 * search_step:
                        continue
                    if use_half_sphere:
                        off_dir = off / max(d, 1e-12)
                        if float(np.dot(off_dir, surface_normal)) < cone_cos:
                            continue
                    if require_above and off[2] < above_min_dz:
                        continue
                    offsets.append(off)
        if not offsets:
            continue
        rng.shuffle(offsets)
        for off in offsets[: int(args.samples_per_ring)]:
            pt = weld_point + off
            if np.any(pt < bmin) or np.any(pt > bmax):
                continue
            key = tuple(np.round(pt, decimals=5).tolist())
            if key in checked:
                continue
            checked.add(key)
            feasible, meta = _is_candidate_feasible(
                field=field,
                point=np.asarray(pt, dtype=float),
                weld_point=weld_point,
                kind=kind,
                min_clearance=float(args.min_clearance),
                min_line_clearance=float(args.min_line_clearance),
                kernel_offsets=kernel_offsets,
                kernel_clearance=float(args.kernel_clearance),
                line_samples=int(args.line_samples),
                line_skip_ratio=line_skip_ratio,
            )
            if feasible:
                score = float(np.linalg.norm(pt - weld_point))
                candidates.append(
                    {
                        "point": pt.tolist(),
                        "score": score,
                        "distance_value": float(meta["distance"]),
                        "min_kernel_value": None
                        if meta.get("min_kernel") is None
                        else float(meta["min_kernel"]),
                    }
                )
        if len(candidates) >= int(args.top_k):
            # 已找到足够候选，提前结束，保证“迅速”。
            break

    candidates.sort(key=lambda item: item["score"])
    topk = candidates[: max(int(args.top_k), 1)]

    out_json = Path(args.output_json)
    _ensure_parent(out_json)
    payload = {
        "kind": kind,
        "weld_point": weld_point.tolist(),
        "surface_normal": surface_normal.tolist(),
        "total_feasible": len(candidates),
        "top_k": topk,
        "params": {
            "min_clearance": float(args.min_clearance),
            "min_line_clearance": float(args.min_line_clearance),
            "kernel_clearance": float(args.kernel_clearance),
            "line_skip_ratio": line_skip_ratio,
            "normal_half_sphere": use_half_sphere,
            "require_above_weld": require_above,
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([weld_point[0]], [weld_point[1]], [weld_point[2]], c="red", s=60, label="weld")
    if topk:
        pts = np.asarray([x["point"] for x in topk], dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="limegreen", s=28, label="feasible top-k")
    ax.quiver(weld_point[0], weld_point[1], weld_point[2],
              surface_normal[0], surface_normal[1], surface_normal[2],
              length=0.1, color="blue", label="surface normal")
    ax.set_title("Nearest feasible intermediate region")
    ax.legend()
    out_png = Path(args.output_png)
    _ensure_parent(out_png)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[nearest] done -> {out_json}")
    if not topk:
        print("[nearest] WARNING: no feasible point found, try larger radius or relaxed thresholds.")


def _has_self_collision(robot: JakaRobot) -> bool:
    p.performCollisionDetection()
    contacts = p.getContactPoints(bodyA=robot.body_id, bodyB=robot.body_id)
    for c in contacts:
        la = int(c[3])
        lb = int(c[4])
        if la == lb:
            continue
        if abs(la - lb) <= 1:
            continue
        return True
    return False


def _external_collision(robot: JakaRobot, workpiece: WorkpieceModel, check_links: list[int], min_clearance: float) -> bool:
    p.performCollisionDetection()
    for li in check_links:
        closest = robot.get_closest_points_to_obstacle(li, workpiece.body_id, max_dist=max(0.5, min_clearance * 8.0))
        if closest is None:
            continue
        if float(closest["signed_dist"]) < float(min_clearance):
            return True
    return False


def _build_occupancy_kernel(
    robot: JakaRobot,
    selected_links: list[int],
    bbox_half: np.ndarray,
    voxel: float,
) -> tuple[np.ndarray, int]:
    clouds = robot.get_surface_visualization_clouds(
        body_id=robot.body_id,
        link_indices=selected_links,
        max_points_per_link=1500,
    )
    if not clouds:
        return np.zeros((0, 3), dtype=float), 0
    pts_world = np.vstack([np.asarray(c["points"], dtype=float).reshape(-1, 3) for c in clouds])
    base_pos, base_quat = robot.get_robobase_pose()
    pts_local = _world_to_local(pts_world, base_pos, base_quat)

    hh = np.asarray(bbox_half, dtype=float).reshape(1, 3)
    mask = np.all((pts_local >= -hh) & (pts_local <= hh), axis=1)
    pts_local = pts_local[mask]
    if pts_local.shape[0] == 0:
        return np.zeros((0, 3), dtype=float), 0

    dims = np.ceil((2.0 * hh.reshape(3)) / float(voxel)).astype(int)
    idx = np.floor((pts_local + hh) / float(voxel)).astype(int)
    idx = np.clip(idx, 0, dims - 1)
    uniq = np.unique(idx, axis=0)
    centers = (uniq.astype(float) + 0.5) * float(voxel) - hh
    return centers.astype(float), int(uniq.shape[0])


def run_init_config_experiment(args) -> None:
    rng = np.random.default_rng(args.seed)
    p.connect(p.DIRECT)
    try:
        cfg = ExperimentConfig()
        if args.urdf_path:
            cfg.urdf_path = args.urdf_path
        if args.workpiece_urdf_path:
            cfg.workpiece_urdf_path = args.workpiece_urdf_path
        robot, workpiece = _make_robot_and_workpiece(cfg)
        q0, dq0 = robot.get_joint_state()
        active_idx = {j: i for i, j in enumerate(robot.active_joints)}
        third_axis_joint = robot.prismatic_joints[2] if len(robot.prismatic_joints) >= 3 else robot.prismatic_joints[-1]
        kernel_links = sorted(set([int(third_axis_joint)] + [int(x) for x in robot.rear_six_link_indices]))

        mutable_joints = list(robot.revolute_joints)
        mutable_indices = [active_idx[j] for j in mutable_joints if j in active_idx]
        best = None
        feasible_count = 0
        occ_counts = []
        for _ in range(int(args.num_samples)):
            q = np.array(q0, dtype=float)
            for idx in mutable_indices:
                info = p.getJointInfo(robot.body_id, int(robot.active_joints[idx]))
                lo = float(info[8])
                hi = float(info[9])
                delta = float(rng.normal(0.0, args.sample_std))
                q[idx] = q[idx] + delta
                if hi > lo:
                    q[idx] = float(np.clip(q[idx], lo, hi))
            robot.set_joint_state(q, dq=np.zeros_like(q))

            if _has_self_collision(robot):
                continue
            if not bool(args.skip_external_collision):
                if _external_collision(robot, workpiece, kernel_links, min_clearance=float(args.min_clearance)):
                    continue

            kernel_offsets, occ_count = _build_occupancy_kernel(
                robot=robot,
                selected_links=kernel_links,
                bbox_half=np.asarray([args.bbox_x, args.bbox_y, args.bbox_z], dtype=float),
                voxel=float(args.voxel),
            )
            rec = {
                "q": q.tolist(),
                "occupancy_count": int(occ_count),
                "kernel_offsets": kernel_offsets,
            }
            feasible_count += 1
            occ_counts.append(int(occ_count))
            if best is None or occ_count < best["occupancy_count"]:
                best = rec

        if best is None:
            raise RuntimeError("未找到满足条件的初始构型，请放宽采样范围或碰撞阈值。")

        out_npz = Path(args.output_npz)
        _ensure_parent(out_npz)
        np.savez_compressed(
            out_npz,
            q_best=np.asarray(best["q"], dtype=float),
            kernel_offsets=np.asarray(best["kernel_offsets"], dtype=float),
            occupancy_count=np.int32(best["occupancy_count"]),
            kernel_links=np.asarray(kernel_links, dtype=np.int32),
            voxel=np.float32(args.voxel),
            bbox_half=np.asarray([args.bbox_x, args.bbox_y, args.bbox_z], dtype=np.float32),
        )
        print(f"[init-config] 最优构型写入: {out_npz}")
        print(f"[init-config] 可行样本数: {feasible_count} / {args.num_samples}")
        print(f"[init-config] 最小占用体素数: {best['occupancy_count']}")

        q_best = best["q"]
        joint_entries = []
        for j in robot.revolute_joints:
            if j in active_idx:
                idx = active_idx[j]
                name = robot.link_name_by_index.get(int(j), f"joint_{int(j)}")
                rad = float(q_best[idx])
                joint_entries.append({
                    "joint_index": int(j),
                    "name": name,
                    "angle_rad": rad,
                    "angle_deg": float(np.degrees(rad)),
                })
        out_json = Path(args.output_json)
        _ensure_parent(out_json)
        report = {
            "best_joint_config": joint_entries,
            "occupancy_count": int(best["occupancy_count"]),
            "feasible_samples": int(feasible_count),
            "total_samples": int(args.num_samples),
        }
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[init-config] JSON 报告写入: {out_json}")

        # 可视化：最佳体素核点云 + 可行样本占用分布
        fig = plt.figure(figsize=(11, 4.8))
        ax1 = fig.add_subplot(121, projection="3d")
        ko = np.asarray(best["kernel_offsets"], dtype=float).reshape(-1, 3)
        if ko.shape[0] > 0:
            ax1.scatter(ko[:, 0], ko[:, 1], ko[:, 2], s=5, c="royalblue", alpha=0.8)
        ax1.set_title("Best occupancy kernel (robobase frame)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_box_aspect([1, 1, 1])

        ax2 = fig.add_subplot(122)
        if occ_counts:
            ax2.hist(np.asarray(occ_counts, dtype=float), bins=40, color="darkorange", alpha=0.82)
            ax2.axvline(float(best["occupancy_count"]), color="red", linestyle="--", linewidth=1.6, label="best")
            ax2.legend()
        ax2.set_title("Feasible samples occupancy distribution")
        ax2.set_xlabel("occupied voxel count")
        ax2.set_ylabel("count")
        fig.tight_layout()
        out_png = Path(args.output_png)
        _ensure_parent(out_png)
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"[init-config] 可视化写入: {out_png}")

        robot.set_joint_state(q0, dq0)
    finally:
        p.disconnect()


@dataclass
class _Node:
    pos: np.ndarray
    parent: int
    cost: float


def _edge_valid(field, a: np.ndarray, b: np.ndarray, kind: str, clearance: float, step: float) -> bool:
    dist = float(np.linalg.norm(b - a))
    n = max(int(math.ceil(dist / max(step, 1e-4))), 2)
    ts = np.linspace(0.0, 1.0, n)
    pts = a.reshape(1, 3) * (1.0 - ts[:, None]) + b.reshape(1, 3) * ts[:, None]
    d = _query_field(field, pts, kind=kind)
    return bool(np.min(d) > clearance)


def _rrt_star_plan(
    field,
    start: np.ndarray,
    goal: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    kind: str,
    clearance: float,
    step_size: float,
    near_radius: float,
    goal_sample_prob: float,
    max_iter: int,
    edge_step: float,
    goal_tol: float,
) -> list[np.ndarray]:
    nodes: list[_Node] = [_Node(pos=np.asarray(start, dtype=float), parent=-1, cost=0.0)]
    goal_idx = -1
    rng = np.random.default_rng()

    for _ in range(int(max_iter)):
        if rng.random() < goal_sample_prob:
            sample = np.asarray(goal, dtype=float)
        else:
            sample = rng.uniform(bounds_min, bounds_max)

        dists = np.asarray([np.linalg.norm(n.pos - sample) for n in nodes], dtype=float)
        nearest_idx = int(np.argmin(dists))
        nearest = nodes[nearest_idx].pos
        direction = sample - nearest
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        new_pos = nearest + direction / norm * min(step_size, norm)
        new_pos = np.clip(new_pos, bounds_min, bounds_max)

        if not _edge_valid(field, nearest, new_pos, kind, clearance, edge_step):
            continue

        near_ids = [i for i, n in enumerate(nodes) if np.linalg.norm(n.pos - new_pos) <= near_radius]
        best_parent = nearest_idx
        best_cost = nodes[nearest_idx].cost + float(np.linalg.norm(new_pos - nearest))
        for nid in near_ids:
            c = nodes[nid].cost + float(np.linalg.norm(new_pos - nodes[nid].pos))
            if c < best_cost and _edge_valid(field, nodes[nid].pos, new_pos, kind, clearance, edge_step):
                best_cost = c
                best_parent = nid

        new_idx = len(nodes)
        nodes.append(_Node(pos=new_pos, parent=best_parent, cost=best_cost))

        for nid in near_ids:
            c = best_cost + float(np.linalg.norm(nodes[nid].pos - new_pos))
            if c < nodes[nid].cost and _edge_valid(field, nodes[nid].pos, new_pos, kind, clearance, edge_step):
                nodes[nid].parent = new_idx
                nodes[nid].cost = c

        if np.linalg.norm(new_pos - goal) <= goal_tol and _edge_valid(field, new_pos, goal, kind, clearance, edge_step):
            goal_idx = len(nodes)
            nodes.append(_Node(pos=np.asarray(goal, dtype=float), parent=new_idx, cost=best_cost + float(np.linalg.norm(goal - new_pos))))
            break

    if goal_idx < 0:
        return []

    path = []
    cur = goal_idx
    while cur >= 0:
        path.append(nodes[cur].pos.copy())
        cur = nodes[cur].parent
    path.reverse()
    return path


def _shortcut_smooth(path: list[np.ndarray], field, kind: str, clearance: float, edge_step: float, iters: int) -> list[np.ndarray]:
    if len(path) < 3:
        return path
    arr = [np.asarray(p, dtype=float).copy() for p in path]
    rng = random.Random(0)
    for _ in range(max(int(iters), 1)):
        if len(arr) < 3:
            break
        i = rng.randint(0, len(arr) - 3)
        j = rng.randint(i + 2, len(arr) - 1)
        if _edge_valid(field, arr[i], arr[j], kind, clearance, edge_step):
            arr = arr[: i + 1] + arr[j:]
    return arr


def run_planner_experiment(args) -> None:
    udf_mod = _load_udf_module()
    field = udf_mod.load_distance_field(args.sdf_npz)
    kind = args.kind if args.kind != "auto" else _best_kind(field)

    nearest_region_as_goal = bool(getattr(args, "nearest_region_as_goal", PLAN_NEAREST_REGION_AS_GOAL))
    nearest_json_path = getattr(args, "nearest_region_json", None) or PLAN_NEAREST_REGION_JSON

    # --- resolve start: default = robobase initial world position ---
    if args.start:
        start = _parse_vec3(args.start)
    else:
        p.connect(p.DIRECT)
        try:
            cfg = ExperimentConfig()
            robot, _ = _make_robot_and_workpiece(cfg)
            base_pos, _ = robot.get_robobase_pose()
            start = np.asarray(base_pos, dtype=float)
            print(f"[plan] start auto-set to robobase: {start.tolist()}")
        finally:
            p.disconnect()

    # --- resolve goal: default = nearest-region top_k[0] ---
    if args.goal:
        goal = _parse_vec3(args.goal)
    elif nearest_region_as_goal and Path(nearest_json_path).exists():
        nr_data = json.loads(Path(nearest_json_path).read_text(encoding="utf-8"))
        topk_nr = nr_data.get("top_k", [])
        if topk_nr:
            goal = np.asarray(topk_nr[0]["point"], dtype=float)
            print(f"[plan] goal auto-set from nearest-region: {goal.tolist()}")
        else:
            goal = np.asarray(field.bbox_max, dtype=float) - 0.15
            print("[plan] WARNING: nearest-region JSON has no top_k, using bbox default")
    else:
        goal = np.asarray(field.bbox_max, dtype=float) - 0.15

    via = None
    if args.via_point:
        via = _parse_vec3(args.via_point)

    bmin = np.asarray(field.bbox_min, dtype=float) + args.bound_margin
    bmax = np.asarray(field.bbox_max, dtype=float) - args.bound_margin
    start = np.clip(start, bmin, bmax)
    goal = np.clip(goal, bmin, bmax)
    if via is not None:
        via = np.clip(via, bmin, bmax)

    if bool(args.auto_fix_endpoints):
        start, ok_start = _auto_fix_point_if_infeasible(
            field=field,
            point=start,
            bounds_min=bmin,
            bounds_max=bmax,
            kind=kind,
            clearance=float(args.min_clearance),
            search_radius=float(args.endpoint_fix_radius),
            search_step=float(args.endpoint_fix_step),
        )
        goal, ok_goal = _auto_fix_point_if_infeasible(
            field=field,
            point=goal,
            bounds_min=bmin,
            bounds_max=bmax,
            kind=kind,
            clearance=float(args.min_clearance),
            search_radius=float(args.endpoint_fix_radius),
            search_step=float(args.endpoint_fix_step),
        )
        if via is not None:
            via, ok_via = _auto_fix_point_if_infeasible(
                field=field,
                point=via,
                bounds_min=bmin,
                bounds_max=bmax,
                kind=kind,
                clearance=float(args.min_clearance),
                search_radius=float(args.endpoint_fix_radius),
                search_step=float(args.endpoint_fix_step),
            )
        else:
            ok_via = True
        if not (ok_start and ok_goal and ok_via):
            raise RuntimeError("端点清距修复失败：start/goal/via 至少有一个点无法在邻域内满足 SDF 阈值。")

    if via is None:
        path = _rrt_star_plan(
            field=field,
            start=start,
            goal=goal,
            bounds_min=bmin,
            bounds_max=bmax,
            kind=kind,
            clearance=float(args.min_clearance),
            step_size=float(args.step_size),
            near_radius=float(args.near_radius),
            goal_sample_prob=float(args.goal_sample_prob),
            max_iter=int(args.max_iter),
            edge_step=float(args.edge_check_step),
            goal_tol=float(args.goal_tolerance),
        )
    else:
        path_1 = _rrt_star_plan(
            field=field,
            start=start,
            goal=via,
            bounds_min=bmin,
            bounds_max=bmax,
            kind=kind,
            clearance=float(args.min_clearance),
            step_size=float(args.step_size),
            near_radius=float(args.near_radius),
            goal_sample_prob=float(args.goal_sample_prob),
            max_iter=int(args.max_iter),
            edge_step=float(args.edge_check_step),
            goal_tol=float(args.goal_tolerance),
        )
        path_2 = _rrt_star_plan(
            field=field,
            start=via,
            goal=goal,
            bounds_min=bmin,
            bounds_max=bmax,
            kind=kind,
            clearance=float(args.min_clearance),
            step_size=float(args.step_size),
            near_radius=float(args.near_radius),
            goal_sample_prob=float(args.goal_sample_prob),
            max_iter=int(args.max_iter),
            edge_step=float(args.edge_check_step),
            goal_tol=float(args.goal_tolerance),
        )
        path = []
        if path_1 and path_2:
            path = path_1[:-1] + path_2
    if not path:
        raise RuntimeError("RRT* 未找到可行路径，请增大迭代次数或放宽 clearance。")
    path = _shortcut_smooth(
        path,
        field=field,
        kind=kind,
        clearance=float(args.min_clearance),
        edge_step=float(args.edge_check_step),
        iters=int(args.smooth_iters),
    )

    pts = np.asarray(path, dtype=float)
    d_path = _query_field(field, pts, kind=kind)
    if float(np.min(d_path)) <= float(args.min_clearance):
        raise RuntimeError("平滑后路径不满足最小 SDF 阈值。")

    out_json = Path(args.output_json)
    _ensure_parent(out_json)
    payload = {
        "kind": kind,
        "n_waypoints": int(len(path)),
        "min_sdf_on_waypoints": float(np.min(d_path)),
        "start_used": np.asarray(start, dtype=float).tolist(),
        "goal_used": np.asarray(goal, dtype=float).tolist(),
        "via_point": None if via is None else np.asarray(via, dtype=float).tolist(),
        "path": pts.tolist(),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", ms=2.8, lw=1.4, c="dodgerblue")
    ax.scatter([start[0]], [start[1]], [start[2]], c="green", s=50, label="start")
    ax.scatter([goal[0]], [goal[1]], [goal[2]], c="red", s=50, label="goal")
    ax.set_title("SDF-constrained RRT* path")
    ax.legend()
    out_png = Path(args.output_png)
    _ensure_parent(out_png)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[planner] 完成，结果写入: {out_json}")


def build_parser() -> argparse.ArgumentParser:
    p0 = argparse.ArgumentParser(
        description="SDF 与 pybullet 对接实验集（对齐 / 最近可行区 / 初始构型 / RRT*）。"
    )
    sub = p0.add_subparsers(dest="cmd", required=True)

    p_align = sub.add_parser("align", help="对齐 pybullet 与 SDF 坐标系并量化误差。")
    p_align.add_argument("--sdf-npz", default=DEFAULT_SDF_NPZ)
    p_align.add_argument("--align-mode", default=ALIGN_MODE, choices=["known", "optimize"])
    p_align.add_argument("--kind", default=ALIGN_KIND, choices=["auto", "udf", "igl_sdf", "o3d_sdf"])
    p_align.add_argument("--urdf-path", default=DEFAULT_ROBOT_URDF_PATH)
    p_align.add_argument("--workpiece-urdf-path", default=DEFAULT_WORKPIECE_URDF_PATH)
    p_align.add_argument("--workpiece-position", default=None, help="覆盖工件 basePosition: x,y,z")
    p_align.add_argument("--workpiece-orientation-deg", default=None, help="覆盖工件 baseOrientation 欧拉角(度): r,p,y")
    p_align.add_argument("--per-link-points", type=int, default=ALIGN_PER_LINK_POINTS)
    p_align.add_argument("--max-points", type=int, default=ALIGN_MAX_POINTS)
    p_align.add_argument("--max-iter", type=int, default=ALIGN_MAX_ITER)
    p_align.add_argument("--seed", type=int, default=ALIGN_SEED)
    p_align.add_argument("--output-json", default=ALIGN_OUTPUT_JSON)
    p_align.add_argument("--output-png", default=ALIGN_OUTPUT_PNG)

    p_near = sub.add_parser("nearest-region", help="从焊点附近搜索最近可行中间区域。")
    p_near.add_argument("--sdf-npz", default=DEFAULT_SDF_NPZ)
    p_near.add_argument("--kind", default=NEAR_KIND, choices=["auto", "udf", "igl_sdf", "o3d_sdf"])
    p_near.add_argument("--kernel-npz", default=INIT_OUTPUT_NPZ, help="init-config kernel NPZ")
    p_near.add_argument("--workpiece-urdf-path", default=DEFAULT_WORKPIECE_URDF_PATH)
    p_near.add_argument("--weld-point", default=None, help="x,y,z；不填则用 weld_link_name")
    p_near.add_argument("--weld-link-name", default=DEFAULT_WELD_START_LINK)
    p_near.add_argument("--search-radius-min", type=float, default=NEAR_SEARCH_RADIUS_MIN)
    p_near.add_argument("--search-radius-max", type=float, default=NEAR_SEARCH_RADIUS_MAX)
    p_near.add_argument("--search-rings", type=int, default=NEAR_SEARCH_RINGS)
    p_near.add_argument("--samples-per-ring", type=int, default=NEAR_SAMPLES_PER_RING)
    p_near.add_argument("--search-step", type=float, default=NEAR_SEARCH_STEP, help="近邻搜索网格步长（米）")
    p_near.add_argument("--min-clearance", type=float, default=NEAR_MIN_CLEARANCE)
    p_near.add_argument("--min-line-clearance", type=float, default=NEAR_MIN_LINE_CLEARANCE)
    p_near.add_argument("--kernel-clearance", type=float, default=NEAR_KERNEL_CLEARANCE)
    p_near.add_argument("--line-samples", type=int, default=NEAR_LINE_SAMPLES)
    p_near.add_argument("--top-k", type=int, default=NEAR_TOP_K)
    p_near.add_argument("--seed", type=int, default=NEAR_SEED)
    p_near.add_argument("--surface-normal-eps", type=float, default=NEAR_SURFACE_NORMAL_EPS)
    p_near.add_argument("--normal-half-sphere", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=NEAR_NORMAL_HALF_SPHERE)
    p_near.add_argument("--normal-cone-cos", type=float, default=NEAR_NORMAL_CONE_COS)
    p_near.add_argument("--line-skip-ratio", type=float, default=NEAR_LINE_SKIP_RATIO)
    p_near.add_argument("--require-above-weld", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=NEAR_REQUIRE_ABOVE_WELD)
    p_near.add_argument("--above-weld-min-dz", type=float, default=NEAR_ABOVE_WELD_MIN_DZ)
    p_near.add_argument("--output-json", default=NEAR_OUTPUT_JSON)
    p_near.add_argument("--output-png", default=NEAR_OUTPUT_PNG)

    p_init = sub.add_parser("init-config", help="搜索无自碰撞、占用体素最小的初始构型。")
    p_init.add_argument("--urdf-path", default=DEFAULT_ROBOT_URDF_PATH)
    p_init.add_argument("--workpiece-urdf-path", default=DEFAULT_WORKPIECE_URDF_PATH)
    p_init.add_argument("--num-samples", type=int, default=INIT_NUM_SAMPLES)
    p_init.add_argument("--sample-std", type=float, default=INIT_SAMPLE_STD)
    p_init.add_argument("--min-clearance", type=float, default=INIT_MIN_CLEARANCE)
    p_init.add_argument("--bbox-x", type=float, default=INIT_BBOX_X)
    p_init.add_argument("--bbox-y", type=float, default=INIT_BBOX_Y)
    p_init.add_argument("--bbox-z", type=float, default=INIT_BBOX_Z)
    p_init.add_argument("--voxel", type=float, default=INIT_VOXEL)
    p_init.add_argument("--seed", type=int, default=INIT_SEED)
    p_init.add_argument("--output-npz", default=INIT_OUTPUT_NPZ)
    p_init.add_argument("--output-png", default=INIT_OUTPUT_PNG)
    p_init.add_argument("--skip-external-collision", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=INIT_SKIP_EXTERNAL_COLLISION, help="跳过与工件的外部碰撞检测")
    p_init.add_argument("--output-json", default=INIT_OUTPUT_JSON)

    p_plan = sub.add_parser("plan", help="用 SDF clearance 约束进行 RRT* 建图与平滑。")
    p_plan.add_argument("--sdf-npz", default=DEFAULT_SDF_NPZ)
    p_plan.add_argument("--kind", default=PLAN_KIND, choices=["auto", "udf", "igl_sdf", "o3d_sdf"])
    p_plan.add_argument("--start", default=PLAN_START, help="x,y,z; empty=auto robobase")
    p_plan.add_argument("--goal", default=PLAN_GOAL, help="x,y,z; empty=auto nearest-region")
    p_plan.add_argument("--via-point", default=None, help="optional via x,y,z")
    p_plan.add_argument("--nearest-region-json", default=PLAN_NEAREST_REGION_JSON)
    p_plan.add_argument("--nearest-region-as-goal", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=PLAN_NEAREST_REGION_AS_GOAL)
    p_plan.add_argument("--min-clearance", type=float, default=PLAN_MIN_CLEARANCE)
    p_plan.add_argument("--step-size", type=float, default=PLAN_STEP_SIZE)
    p_plan.add_argument("--near-radius", type=float, default=PLAN_NEAR_RADIUS)
    p_plan.add_argument("--goal-sample-prob", type=float, default=PLAN_GOAL_SAMPLE_PROB)
    p_plan.add_argument("--max-iter", type=int, default=PLAN_MAX_ITER)
    p_plan.add_argument("--goal-tolerance", type=float, default=PLAN_GOAL_TOLERANCE)
    p_plan.add_argument("--edge-check-step", type=float, default=PLAN_EDGE_CHECK_STEP)
    p_plan.add_argument("--smooth-iters", type=int, default=PLAN_SMOOTH_ITERS)
    p_plan.add_argument("--bound-margin", type=float, default=PLAN_BOUND_MARGIN)
    p_plan.add_argument("--auto-fix-endpoints", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=PLAN_AUTO_FIX_ENDPOINTS)
    p_plan.add_argument("--endpoint-fix-radius", type=float, default=PLAN_ENDPOINT_FIX_RADIUS)
    p_plan.add_argument("--endpoint-fix-step", type=float, default=PLAN_ENDPOINT_FIX_STEP)
    p_plan.add_argument("--output-json", default=PLAN_OUTPUT_JSON)
    p_plan.add_argument("--output-png", default=PLAN_OUTPUT_PNG)

    return p0


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "align":
        run_alignment_experiment(args)
    elif args.cmd == "nearest-region":
        run_nearest_region_experiment(args)
    elif args.cmd == "init-config":
        run_init_config_experiment(args)
    elif args.cmd == "plan":
        run_planner_experiment(args)
    else:
        raise ValueError(f"未知命令: {args.cmd}")


if __name__ == "__main__":
    main()
