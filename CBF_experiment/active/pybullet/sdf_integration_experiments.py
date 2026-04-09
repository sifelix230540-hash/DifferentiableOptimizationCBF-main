from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from CBF_experiment.active.pybullet.welding_320_common import ExperimentConfig  # noqa: E402
from CBF_experiment.active.pybullet.welding_320_robot import JakaRobot, WorkpieceModel  # noqa: E402

# =============================================================================
# 用户集中参数区（直接 F5 运行，只改这里）
# =============================================================================
#
# RUN_STEPS: 控制要执行哪些实验，按列表顺序依次执行。
#   可选值: "align", "init-config", "nearest-region", "plan"
#   推荐流水线: ["align", "init-config", "nearest-region", "plan"]
#   只想跑某一步？改为例如 ["init-config"]
#
RUN_STEPS = ["align", "init-config", "nearest-region", "plan"]
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

# ---- 2) nearest-region 默认 ----
#   新流程：SDF网格过滤 -> z约束 -> 连通域 -> 最近点
NEAR_KIND = "auto"
NEAR_MIN_CLEARANCE = 0.01            # 基础最小SDF余量（米），实际取 max(此值, bbox_radius+margin)
NEAR_TOP_K = 8                       # 输出最近的 top-k 个可行体素
NEAR_REQUIRE_ABOVE_WELD = True       # True=候选Z≥焊点Z（倒置臂）
NEAR_ABOVE_WELD_MIN_DZ = 0.0        # 候选Z至少比焊点高多少（米）
NEAR_BBOX_MARGIN = 0.10             # 候选点 SDF 需 > bbox_radius + 此值（米）
NEAR_SURFACE_NORMAL_EPS = 0.002      # SDF 有限差分步长（米），用于估计表面法线
NEAR_OUTPUT_JSON = "artifacts/sdf_exp/nearest_region.json"
NEAR_OUTPUT_PNG = "artifacts/sdf_exp/nearest_region.png"

# 以下旧参数保留兼容但不再使用
NEAR_SEARCH_RADIUS_MIN = 0.02
NEAR_SEARCH_RADIUS_MAX = 5
NEAR_SEARCH_RINGS = 16
NEAR_SEARCH_STEP = 0.02
NEAR_SAMPLES_PER_RING = 150
NEAR_MIN_LINE_CLEARANCE = 0.005
NEAR_KERNEL_CLEARANCE = 0.008
NEAR_LINE_SAMPLES = 24
NEAR_SEED = 2
NEAR_NORMAL_HALF_SPHERE = True
NEAR_NORMAL_CONE_COS = 0.0
NEAR_LINE_SKIP_RATIO = 0.05

# ---- 3) init-config 默认 ----
INIT_NUM_SAMPLES = 400
INIT_SAMPLE_STD = 0.2
INIT_MIN_CLEARANCE = 0.005
INIT_VOXEL = 0.04
INIT_SEED = 3
INIT_OUTPUT_NPZ = "artifacts/sdf_exp/init_kernel.npz"
INIT_OUTPUT_PNG = "artifacts/sdf_exp/init_kernel.png"
INIT_SKIP_EXTERNAL_COLLISION = True
INIT_OUTPUT_JSON = "artifacts/sdf_exp/init_config_report.json"

# ---- 4) plan 默认 ----
PLAN_METHOD = "rrt*"             # "astar" 或 "rrt*"
PLAN_KIND = "auto"
PLAN_MIN_CLEARANCE = 0.30         # 过渡路径清距（米）；只需保证紧凑构型横截面不碰工件
PLAN_RESAMPLE_SPACING = 0.05      # 输出路径最大点间距（米），确保密采样
# RRT* 专用参数（仅 PLAN_METHOD="rrt*" 时生效）
PLAN_STEP_SIZE = 0.30
PLAN_NEAR_RADIUS = 0.60
PLAN_GOAL_SAMPLE_PROB = 0.15
PLAN_MAX_ITER = 8000
PLAN_GOAL_TOLERANCE = 0.20
PLAN_EDGE_CHECK_STEP = 0.02
PLAN_SMOOTH_ITERS = 80
PLAN_BOUND_MARGIN = 0.02
PLAN_OUTPUT_JSON = "artifacts/sdf_exp/plan_path.json"
PLAN_OUTPUT_PNG = "artifacts/sdf_exp/plan_path.png"
PLAN_AUTO_FIX_ENDPOINTS = True
PLAN_ENDPOINT_FIX_RADIUS = 0.30
PLAN_ENDPOINT_FIX_STEP = 0.02

PLAN_NEAREST_REGION_AS_GOAL = True    # True=nearest-region 输出直接作为 goal
PLAN_NEAREST_REGION_JSON = NEAR_OUTPUT_JSON  # 自动衔接 nearest-region
PLAN_INIT_CONFIG_NPZ = INIT_OUTPUT_NPZ       # 用于计算 robobase 初始位置

# 可选：若你希望直接用焊点作为规划起终点，改为具体 xyz；留空则自动计算。
PLAN_START = None  # 留空=自动用 robobase 初始世界坐标
PLAN_GOAL = None   # 留空=自动从 nearest-region JSON 读取

# ---- 5) PyBullet GUI 可视化 ----
VIS_PYBULLET = True           # True=每步实验后在 PyBullet GUI 交互式查看
VIS_PYBULLET_STEPS = ["init-config", "nearest-region", "plan"]  # 需要 GUI 的步骤
VIS_CAMERA_DISTANCE = 1.4
VIS_CAMERA_YAW = -215.0
VIS_CAMERA_PITCH = -26.0


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


def _build_pybullet_to_sdf_transform() -> tuple[np.ndarray, np.ndarray]:
    """从 ExperimentConfig 计算 PyBullet→SDF 的仿射变换（平移+旋转矩阵）。

    SDF 场在工件网格自身坐标系，PyBullet 将工件放置在 workpiece_position / orientation。
    SDF_pt = R_inv @ (PB_pt - wp_pos)
    """
    cfg = ExperimentConfig()
    wp_pos = np.asarray(cfg.workpiece_position, dtype=float)
    wp_deg = np.asarray(cfg.workpiece_orientation_deg, dtype=float)
    R = Rotation.from_euler("xyz", wp_deg, degrees=True).as_matrix()
    R_inv = R.T
    return wp_pos, R_inv


_WP_POS, _R_INV = _build_pybullet_to_sdf_transform()


def _pb2sdf(pts_pybullet: np.ndarray) -> np.ndarray:
    """PyBullet 世界坐标 → SDF 场坐标（批量）。"""
    pts = np.asarray(pts_pybullet, dtype=float).reshape(-1, 3)
    return (pts - _WP_POS.reshape(1, 3)) @ _R_INV.T


def _sdf2pb(pts_sdf: np.ndarray) -> np.ndarray:
    """SDF 场坐标 → PyBullet 世界坐标（批量）。"""
    R = _R_INV.T
    R_fwd = np.linalg.inv(R) if abs(np.linalg.det(R) - 1.0) < 1e-6 else R.T
    pts = np.asarray(pts_sdf, dtype=float).reshape(-1, 3)
    return pts @ R_fwd.T + _WP_POS.reshape(1, 3)


def _query_field(field, points_sdf: np.ndarray, kind: str, safe_oob: bool = False) -> np.ndarray:
    """查询 SDF 场。points_sdf 必须已在 SDF 坐标系。
    safe_oob=True 时，利用 query_with_gradient 的线性外推处理 bbox 外点。
    """
    pts = np.asarray(points_sdf, dtype=np.float32).reshape(-1, 3)
    if safe_oob:
        vals, _ = field.query_with_gradient(pts, kind=kind)
        return np.asarray(vals, dtype=float).reshape(-1)
    vals = field.query(pts, kind=kind, clip=True)
    return np.asarray(vals, dtype=float).reshape(-1)


def _query_field_pb(field, points_pybullet: np.ndarray, kind: str) -> np.ndarray:
    """查询 SDF 场，输入为 PyBullet 世界坐标（自动变换）。"""
    return _query_field(field, _pb2sdf(points_pybullet), kind=kind)


def _estimate_surface_normal(
    field,
    point_sdf: np.ndarray,
    kind: str,
    eps: float = 0.002,
) -> np.ndarray:
    """用 SDF 有限差分估计表面法线方向（指向正值/自由空间侧）。point_sdf 在 SDF 坐标系。"""
    pt = np.asarray(point_sdf, dtype=float).reshape(3)
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
    d0 = float(_query_field(field, p0.reshape(1, 3), kind=kind, safe_oob=True)[0])
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


def _local_to_world(pts_local: np.ndarray, world_pos, world_quat) -> np.ndarray:
    """robobase 本体坐标 → 世界坐标（批量）。"""
    rot = np.array(p.getMatrixFromQuaternion(
        np.asarray(world_quat, dtype=float).tolist(),
    ), dtype=float).reshape(3, 3)
    pts = np.asarray(pts_local, dtype=float).reshape(-1, 3)
    return (rot @ pts.T).T + np.asarray(world_pos, dtype=float).reshape(1, 3)


def _pybullet_gui_open(
    cfg: ExperimentConfig, *,
    load_robot: bool = True,
    load_workpiece: bool = True,
    robot_q=None,
    camera_target=None,
) -> tuple:
    """打开 PyBullet GUI 窗口，加载模型，返回 (robot, workpiece)。"""
    import pybullet_data as _pbd
    p.connect(p.GUI, options="--width=1600 --height=900")
    p.setAdditionalSearchPath(_pbd.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(rgbBackground=[1, 1, 1])
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.95, 0.95, 0.95, 1.0])

    class _Stub:
        pass

    robot = workpiece = None
    if load_robot:
        robot = JakaRobot(cfg, _Stub())
        if robot_q is not None:
            robot.set_joint_state(np.asarray(robot_q, dtype=float))
    if load_workpiece:
        workpiece = WorkpieceModel(cfg)
    if robot is not None and workpiece is not None:
        robot.register_surface_obstacle(workpiece.body_id, None)
    ct = list(camera_target) if camera_target is not None else list(cfg.camera_target)
    p.resetDebugVisualizerCamera(
        cameraDistance=VIS_CAMERA_DISTANCE,
        cameraYaw=VIS_CAMERA_YAW,
        cameraPitch=VIS_CAMERA_PITCH,
        cameraTargetPosition=ct,
    )
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    return robot, workpiece


def _pybullet_gui_wait():
    """阻塞直到用户关闭 PyBullet 窗口或按 Ctrl+C。"""
    print("[vis] PyBullet GUI 已打开，关闭窗口或 Ctrl+C 继续下一步...")
    try:
        while p.isConnected():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


def _create_sphere_marker(pos, radius: float, rgba):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=list(rgba))
    bid = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=vis,
        basePosition=np.asarray(pos, dtype=float).tolist(),
    )
    p.setCollisionFilterGroupMask(bid, -1, 0, 0)
    return bid


def _draw_box_wireframe(lo, hi, world_pos, world_quat,
                        color=(1.0, 0.5, 0.0), width=2.0):
    """在世界坐标系中绘制 robobase 本体坐标下的 AABB 线框。"""
    lo_a = np.asarray(lo, dtype=float)
    hi_a = np.asarray(hi, dtype=float)
    c_local = np.array([
        [lo_a[0], lo_a[1], lo_a[2]], [hi_a[0], lo_a[1], lo_a[2]],
        [hi_a[0], hi_a[1], lo_a[2]], [lo_a[0], hi_a[1], lo_a[2]],
        [lo_a[0], lo_a[1], hi_a[2]], [hi_a[0], lo_a[1], hi_a[2]],
        [hi_a[0], hi_a[1], hi_a[2]], [lo_a[0], hi_a[1], hi_a[2]],
    ])
    c_w = _local_to_world(c_local, world_pos, world_quat)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]:
        p.addUserDebugLine(c_w[a].tolist(), c_w[b].tolist(), list(color), lineWidth=width)


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

    if VIS_PYBULLET and "align" in VIS_PYBULLET_STEPS:
        _, _ = _pybullet_gui_open(ExperimentConfig(), load_robot=True, load_workpiece=True)
        max_err = max(float(np.max(d_final)), 1e-6)
        t_err = np.clip(d_final / max_err, 0.0, 1.0)
        colors_err = np.column_stack([t_err, 1.0 - t_err, np.zeros_like(t_err)])
        p.addUserDebugPoints(pts.tolist(), colors_err.tolist(), pointSize=5)
        p.addUserDebugText(
            f"Align: mean={np.mean(d_final)*1000:.2f}mm  "
            f"p95={np.percentile(d_final, 95)*1000:.2f}mm  "
            f"max={np.max(d_final)*1000:.2f}mm",
            [0.0, -0.3, 0.5], [0.1, 0.1, 0.1], textSize=1.2,
        )
        _pybullet_gui_wait()


def _load_kernel_offsets(path: str | None) -> np.ndarray | None:
    """Load kernel offsets and rotate from robobase-local to SDF/world frame.

    robobase has rpy=(-pi, 0, 0) relative to gantry → R_x(-pi) flips Y and Z.
    Since workpiece orientation is identity, SDF frame = PyBullet frame (up to translation).
    """
    if not path:
        return None
    p_path = Path(path)
    if not p_path.exists():
        print(f"[kernel] file not found: {path}, skipping kernel check")
        return None
    with np.load(p_path) as data:
        if "kernel_offsets" not in data:
            raise KeyError(f"{path} 中缺少 kernel_offsets")
        ko_local = np.asarray(data["kernel_offsets"], dtype=float).reshape(-1, 3)
    # R_x(-pi): x stays, y and z negate
    ko_world = ko_local.copy()
    ko_world[:, 1] *= -1.0
    ko_world[:, 2] *= -1.0
    print(f"[kernel] loaded {ko_world.shape[0]} voxels, "
          f"AABB(world) = {ko_world.min(axis=0)} .. {ko_world.max(axis=0)}")
    return ko_world


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

    # --- 层级 2: 连线检查（跳过焊点端附近，焊点在表面 SDF≤0 会误报）---
    # t=0 是焊点端，t=1 是候选点端。
    # 焊点在工件表面（SDF≈0 或略负），焊点附近 SDF 不可靠，需大幅跳过。
    dist = float(np.linalg.norm(point - weld_point))
    if dist > 1e-6:
        skip_dist = max(0.02, dist * float(line_skip_ratio))
        t_skip = min(skip_dist / dist, 0.6)
    else:
        t_skip = 0.6
    ts = np.linspace(t_skip, 1.0, max(int(line_samples), 2))
    line = weld_point.reshape(1, 3) * (1.0 - ts[:, None]) + point.reshape(1, 3) * ts[:, None]
    d_line = _query_field(field, line, kind=kind)
    if float(np.min(d_line)) <= min_line_clearance:
        return False, {"distance": d0, "reason": "line_cross_wall"}

    # --- 层级 3: 全核碰撞检查（向量化批量查询）---
    min_kernel = None
    if kernel_offsets is not None and kernel_offsets.shape[0] > 0:
        pts = point.reshape(1, 3) + kernel_offsets
        d_kernel = _query_field(field, pts, kind=kind)
        min_kernel = float(np.min(d_kernel))
        if min_kernel <= kernel_clearance:
            return False, {"distance": d0, "min_kernel": min_kernel, "reason": "kernel_collision"}
    return True, {"distance": d0, "min_kernel": min_kernel, "reason": "ok"}


def run_nearest_region_experiment(args) -> None:
    from scipy.ndimage import label as ndimage_label

    udf_mod = _load_udf_module()
    field = udf_mod.load_distance_field(args.sdf_npz)
    kind = args.kind if args.kind != "auto" else _best_kind(field)

    p.connect(p.DIRECT)
    try:
        cfg = ExperimentConfig()
        if args.workpiece_urdf_path:
            cfg.workpiece_urdf_path = args.workpiece_urdf_path
        workpiece = WorkpieceModel(cfg)
        if args.weld_point:
            weld_point_pb = _parse_vec3(args.weld_point)
        else:
            weld_point_pb, _ = workpiece.get_frame_pose(args.weld_link_name)
            weld_point_pb = np.asarray(weld_point_pb, dtype=float)
    finally:
        p.disconnect()

    weld_point = _pb2sdf(weld_point_pb).reshape(3)
    print(f"[nearest] weld PB={weld_point_pb.tolist()} -> SDF={weld_point.tolist()}")

    surface_normal = _estimate_surface_normal(
        field, weld_point, kind,
        eps=float(getattr(args, "surface_normal_eps", NEAR_SURFACE_NORMAL_EPS)),
    )
    print(f"[nearest] surface normal at weld: {surface_normal.tolist()}")

    require_above = bool(getattr(args, "require_above_weld", NEAR_REQUIRE_ABOVE_WELD))
    above_min_dz = float(getattr(args, "above_weld_min_dz", NEAR_ABOVE_WELD_MIN_DZ))

    d_weld = float(_query_field(field, weld_point.reshape(1, 3), kind=kind)[0])
    print(f"[nearest] SDF at weld point = {d_weld:.6f} m")

    _bbox_radius = 0.0
    _kernel_npz_path = getattr(args, "kernel_npz", None)
    if _kernel_npz_path and Path(_kernel_npz_path).exists():
        with np.load(_kernel_npz_path) as _d:
            if "bbox_radius" in _d:
                _bbox_radius = float(_d["bbox_radius"])
    bbox_margin = float(getattr(args, "bbox_margin", NEAR_BBOX_MARGIN))
    effective_clearance = float(args.min_clearance)
    if _bbox_radius > 0:
        effective_clearance = max(effective_clearance, _bbox_radius + bbox_margin)
    print(f"[nearest] bbox_radius={_bbox_radius:.4f}m  margin={bbox_margin:.2f}m  "
          f"-> effective_clearance={effective_clearance:.4f}m")

    # ---- Step 1: 直接读取 SDF 3D 网格 ----
    grid = np.asarray(udf_mod._grid_for_kind(field, kind), dtype=float)
    bmin = np.asarray(field.bbox_min, dtype=float)
    bmax = np.asarray(field.bbox_max, dtype=float)
    nx, ny, nz = grid.shape
    xs = np.linspace(bmin[0], bmax[0], nx)
    ys = np.linspace(bmin[1], bmax[1], ny)
    zs = np.linspace(bmin[2], bmax[2], nz)
    print(f"[nearest] grid shape={grid.shape}  "
          f"spacing~{np.round((bmax-bmin)/(np.array(grid.shape)-1), 5).tolist()}")

    # ---- Step 2: 过滤 SDF > effective_clearance ----
    mask = grid > effective_clearance
    n_after_sdf = int(np.sum(mask))
    print(f"[nearest] voxels with SDF>{effective_clearance:.3f}: {n_after_sdf} / {grid.size}")

    # ---- Step 3: 过滤 z >= weld_z（倒置臂，候选在焊点上方）----
    if require_above:
        z_threshold = weld_point[2] + above_min_dz
        z_ok = zs >= z_threshold
        mask[:, :, ~z_ok] = False

    n_feasible = int(np.sum(mask))
    print(f"[nearest] feasible voxels (+ z-filter): {n_feasible}")

    topk = []
    n_components = 0
    total_feasible = 0

    if n_feasible > 0:
        # ---- Step 4: 连通域标记 ----
        labels, n_components = ndimage_label(mask)
        print(f"[nearest] connected components: {n_components}")

        # ---- Step 5: 找到距焊点最近的连通域 ----
        feasible_ijk = np.argwhere(mask)
        feasible_xyz = np.column_stack([
            xs[feasible_ijk[:, 0]],
            ys[feasible_ijk[:, 1]],
            zs[feasible_ijk[:, 2]],
        ])
        feasible_labels = labels[mask]
        dists = np.linalg.norm(feasible_xyz - weld_point.reshape(1, 3), axis=1)

        best_comp = -1
        best_comp_dist = float("inf")
        for comp_id in range(1, n_components + 1):
            comp_mask = feasible_labels == comp_id
            comp_min_dist = float(np.min(dists[comp_mask]))
            if comp_min_dist < best_comp_dist:
                best_comp_dist = comp_min_dist
                best_comp = comp_id

        comp_sel = feasible_labels == best_comp
        comp_xyz = feasible_xyz[comp_sel]
        comp_dists = dists[comp_sel]
        comp_ijk = feasible_ijk[comp_sel]
        total_feasible = int(np.sum(comp_sel))

        sorted_idx = np.argsort(comp_dists)
        top_k_n = min(int(args.top_k), len(sorted_idx))
        for rank in range(top_k_n):
            idx = sorted_idx[rank]
            pt_sdf = comp_xyz[idx]
            pt_pb = _sdf2pb(pt_sdf.reshape(1, 3)).reshape(3)
            sdf_val = float(grid[tuple(comp_ijk[idx])])
            topk.append({
                "point": pt_pb.tolist(),
                "score": float(comp_dists[idx]),
                "distance_value": sdf_val,
                "min_kernel_value": None,
            })
        print(f"[nearest] best component #{best_comp}: "
              f"{total_feasible} voxels, nearest={best_comp_dist:.4f}m")

    # ---- 输出 ----
    out_json = Path(args.output_json)
    _ensure_parent(out_json)
    payload = {
        "kind": kind,
        "weld_point": weld_point_pb.tolist(),
        "weld_point_sdf": weld_point.tolist(),
        "sdf_at_weld": d_weld,
        "surface_normal": surface_normal.tolist(),
        "total_feasible": total_feasible,
        "n_components": n_components,
        "top_k": topk,
        "params": {
            "bbox_radius": _bbox_radius,
            "bbox_margin": bbox_margin,
            "effective_clearance": effective_clearance,
            "require_above_weld": require_above,
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([weld_point_pb[0]], [weld_point_pb[1]], [weld_point_pb[2]], c="red", s=60, label="weld")
    if topk:
        pts_vis = np.asarray([x["point"] for x in topk], dtype=float)
        ax.scatter(pts_vis[:, 0], pts_vis[:, 1], pts_vis[:, 2], c="limegreen", s=28, label="top-k")
    ax.quiver(weld_point_pb[0], weld_point_pb[1], weld_point_pb[2],
              surface_normal[0], surface_normal[1], surface_normal[2],
              length=0.1, color="blue", label="normal")
    ax.set_title("Nearest feasible region (voxel filter)")
    ax.legend()
    out_png = Path(args.output_png)
    _ensure_parent(out_png)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[nearest] done -> {out_json}")
    if not topk:
        print("[nearest] WARNING: no feasible voxel found.")

    if VIS_PYBULLET and "nearest-region" in VIS_PYBULLET_STEPS:
        _, _ = _pybullet_gui_open(
            ExperimentConfig(), load_robot=True, load_workpiece=True,
            camera_target=weld_point_pb.tolist(),
        )
        _create_sphere_marker(weld_point_pb, 0.015, (0.95, 0.15, 0.15, 0.9))
        p.addUserDebugText(
            "weld", (weld_point_pb + np.array([0, 0, 0.03])).tolist(),
            [0.9, 0.1, 0.1], textSize=1.2,
        )
        nl = 0.15
        p.addUserDebugLine(
            weld_point_pb.tolist(),
            (weld_point_pb + nl * surface_normal).tolist(),
            [0.1, 0.3, 1.0], lineWidth=3.0,
        )
        for i, cand in enumerate(topk):
            cp = np.asarray(cand["point"], dtype=float)
            _create_sphere_marker(cp, 0.012, (0.1, 0.9, 0.2, 0.9))
            p.addUserDebugLine(
                weld_point_pb.tolist(), cp.tolist(),
                [0.4, 0.8, 0.2], lineWidth=1.5,
            )
            p.addUserDebugText(
                f"#{i+1} d={cand['score']*1000:.1f}mm",
                (cp + np.array([0, 0, 0.02])).tolist(),
                [0.1, 0.6, 0.1], textSize=0.9,
            )
        p.addUserDebugText(
            f"feasible: {total_feasible} voxels, {n_components} components",
            [0.0, -0.3, 0.5], [0.1, 0.1, 0.1], textSize=1.2,
        )
        _pybullet_gui_wait()


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
    voxel: float,
) -> tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
    """Build voxelized occupancy kernel from robot surface points.

    Returns (kernel_offsets, voxel_count, aabb_volume, aabb_min, aabb_max).
    The AABB is tight-fit to the actual point cloud (no fixed bbox).
    """
    clouds = robot.get_surface_visualization_clouds(
        body_id=robot.body_id,
        link_indices=selected_links,
        max_points_per_link=1500,
    )
    if not clouds:
        z = np.zeros((0, 3), dtype=float)
        return z, 0, 0.0, np.zeros(3), np.zeros(3)
    pts_world = np.vstack([np.asarray(c["points"], dtype=float).reshape(-1, 3) for c in clouds])
    base_pos, base_quat = robot.get_robobase_pose()
    pts_local = _world_to_local(pts_world, base_pos, base_quat)

    aabb_min = pts_local.min(axis=0)
    aabb_max = pts_local.max(axis=0)
    aabb_size = aabb_max - aabb_min
    aabb_volume = float(np.prod(aabb_size))

    hh = (aabb_size / 2.0 + voxel).reshape(1, 3)
    center = ((aabb_min + aabb_max) / 2.0).reshape(1, 3)
    pts_centered = pts_local - center

    dims = np.ceil((2.0 * hh.reshape(3)) / float(voxel)).astype(int)
    idx = np.floor((pts_centered + hh) / float(voxel)).astype(int)
    idx = np.clip(idx, 0, dims - 1)
    uniq = np.unique(idx, axis=0)
    centers = (uniq.astype(float) + 0.5) * float(voxel) - hh + center
    return centers.astype(float), int(uniq.shape[0]), aabb_volume, aabb_min, aabb_max


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
        kernel_links = sorted(set(int(x) for x in robot.rear_six_link_indices))

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

            kernel_offsets, occ_count, aabb_vol, aabb_lo, aabb_hi = _build_occupancy_kernel(
                robot=robot,
                selected_links=kernel_links,
                voxel=float(args.voxel),
            )
            aabb_size = aabb_hi - aabb_lo
            aabb_max_dim = float(np.max(aabb_size))
            half_ext = np.maximum(np.abs(aabb_lo), np.abs(aabb_hi))
            bbox_radius = float(np.linalg.norm(half_ext))
            rec = {
                "q": q.tolist(),
                "occupancy_count": int(occ_count),
                "aabb_volume": float(aabb_vol),
                "aabb_max_dim": aabb_max_dim,
                "aabb_min": aabb_lo.tolist(),
                "aabb_max": aabb_hi.tolist(),
                "bbox_radius": bbox_radius,
                "kernel_offsets": kernel_offsets,
            }
            feasible_count += 1
            occ_counts.append(aabb_max_dim)
            if best is None or aabb_max_dim < best["aabb_max_dim"]:
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
            aabb_volume=np.float64(best["aabb_volume"]),
            aabb_min=np.asarray(best["aabb_min"], dtype=np.float32),
            aabb_max=np.asarray(best["aabb_max"], dtype=np.float32),
            bbox_radius=np.float64(best["bbox_radius"]),
            kernel_links=np.asarray(kernel_links, dtype=np.int32),
            voxel=np.float32(args.voxel),
        )
        print(f"[init-config] best -> {out_npz}")
        print(f"[init-config] feasible: {feasible_count} / {args.num_samples}")
        print(f"[init-config] best AABB max_dim: {best['aabb_max_dim']:.4f} m  "
              f"(volume: {best['aabb_volume']:.6f} m^3, voxels: {best['occupancy_count']})")
        print(f"[init-config] AABB: {best['aabb_min']} .. {best['aabb_max']}")
        print(f"[init-config] bbox_radius (rear-6 circumscribed): {best['bbox_radius']:.4f} m")

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
            "aabb_volume_m3": float(best["aabb_volume"]),
            "aabb_max_dim_m": float(best["aabb_max_dim"]),
            "aabb_min": best["aabb_min"],
            "aabb_max": best["aabb_max"],
            "bbox_radius_m": float(best["bbox_radius"]),
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
            ax2.axvline(float(best["aabb_max_dim"]), color="red", linestyle="--", linewidth=1.6, label="best")
            ax2.legend()
        ax2.set_title("Feasible samples AABB max-dim distribution")
        ax2.set_xlabel("AABB max dimension (m)")
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

    if VIS_PYBULLET and "init-config" in VIS_PYBULLET_STEPS:
        _cfg_v = ExperimentConfig()
        robot_v, _ = _pybullet_gui_open(
            _cfg_v, load_robot=True, load_workpiece=True,
            robot_q=np.asarray(best["q"], dtype=float),
        )
        bp, bq = robot_v.get_robobase_pose()
        ko = np.asarray(best["kernel_offsets"], dtype=float).reshape(-1, 3)
        if ko.shape[0] > 0:
            ko_w = _local_to_world(ko, bp, bq)
            c_blue = np.full((ko_w.shape[0], 3), [0.2, 0.4, 1.0])
            p.addUserDebugPoints(ko_w.tolist(), c_blue.tolist(), pointSize=3)
        _draw_box_wireframe(best["aabb_min"], best["aabb_max"], bp, bq)
        _create_sphere_marker(bp, 0.025, (1.0, 0.3, 0.0, 0.9))
        p.addUserDebugText(
            "robobase", (bp + np.array([0, 0, 0.06])).tolist(),
            [0.8, 0.2, 0.0], textSize=1.2,
        )
        p.addUserDebugText(
            f"AABB max_dim: {best['aabb_max_dim']:.4f}m  r={best['bbox_radius']:.4f}m",
            [0.0, -0.3, 0.5], [0.1, 0.1, 0.1], textSize=1.2,
        )
        _pybullet_gui_wait()


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
    d = _query_field(field, pts, kind=kind, safe_oob=True)
    return bool(np.min(d) > clearance)


def _astar_sdf_plan(field, start: np.ndarray, goal: np.ndarray,
                    kind: str, clearance: float) -> list[np.ndarray]:
    """3D 最短路，基于硬障碍体素搜索。

    关键点：
    1. 使用 field.origin / field.spacing 与 SDF 查询保持一致；
    2. 只允许在 ``grid > clearance`` 的体素中通行；
    3. 起点吸附到 ``goal`` 所在连通域中最近的可行体素，避免吸到错误侧。
    """
    from scipy import ndimage
    try:
        import dijkstra3d
    except ImportError as e:
        raise RuntimeError("A* 需要安装 dijkstra3d：`python -X utf8 -m pip install dijkstra3d`") from e

    udf_mod = sys.modules.get("udf_module_runtime") or _load_udf_module()
    grid = np.asarray(udf_mod._grid_for_kind(field, kind), dtype=float)
    origin = np.asarray(field.origin, dtype=float).reshape(3)
    spacing = float(field.spacing)
    shape = np.array(grid.shape, dtype=int)
    bmin = np.asarray(field.bbox_min, dtype=float).reshape(3)
    bmax = np.asarray(field.bbox_max, dtype=float).reshape(3)

    feasible = grid > clearance
    labels, n_components = ndimage.label(feasible)

    def xyz_to_ijk(xyz: np.ndarray) -> tuple[int, int, int]:
        coord = (np.asarray(xyz, dtype=float) - origin) / spacing - 0.5
        idx = np.clip(np.round(coord).astype(int), 0, shape - 1)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def ijk_to_xyz(ijk: tuple[int, int, int]) -> np.ndarray:
        return origin + (np.asarray(ijk, dtype=float) + 0.5) * spacing

    def _nearest_voxel_in_mask(point_xyz: np.ndarray, mask: np.ndarray) -> tuple[int, int, int] | None:
        vox = np.argwhere(mask)
        if vox.size == 0:
            return None
        vox_xyz = origin.reshape(1, 3) + (vox.astype(float) + 0.5) * spacing
        best_idx = int(np.argmin(np.linalg.norm(vox_xyz - point_xyz.reshape(1, 3), axis=1)))
        out = vox[best_idx]
        return int(out[0]), int(out[1]), int(out[2])

    goal_clip = np.clip(goal, bmin, bmax)
    goal_ijk_guess = xyz_to_ijk(goal_clip)
    goal_ijk = goal_ijk_guess if feasible[goal_ijk_guess] else _nearest_voxel_in_mask(goal_clip, feasible)
    if goal_ijk is None:
        print("[astar] ERROR: 终点附近无可行体素")
        return []

    goal_label = int(labels[goal_ijk])
    if goal_label <= 0:
        print("[astar] ERROR: 终点不在任何可行连通域中")
        return []

    component_mask = labels == goal_label
    start_clip = np.clip(start, bmin, bmax)
    start_ijk = _nearest_voxel_in_mask(start_clip, component_mask)
    if start_ijk is None:
        print("[astar] ERROR: 起点无法连接到终点所在连通域")
        return []

    print(f"[astar] grid={shape.tolist()}, spacing={spacing:.4f}, "
          f"components={n_components}, goal_label={goal_label}, "
          f"component_voxels={int(np.sum(component_mask))}")
    print(f"[astar] start_ijk={start_ijk}, goal_ijk={goal_ijk}, "
          f"start_voxel_sdf={float(grid[start_ijk]):.4f}, goal_voxel_sdf={float(grid[goal_ijk]):.4f}")

    try:
        indices = dijkstra3d.binary_dijkstra(
            component_mask.astype(bool),
            start_ijk,
            goal_ijk,
            connectivity=26,
            background_color=0,
        )
    except Exception as e:
        print(f"[astar] dijkstra3d failed: {e}")
        return []

    path_xyz = [ijk_to_xyz((int(i), int(j), int(k))) for i, j, k in indices]
    print(f"[astar] raw path: {len(path_xyz)} waypoints")
    return path_xyz


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

    for it in range(int(max_iter)):
        if it % 1000 == 0 and it > 0:
            best_to_goal = min(float(np.linalg.norm(n.pos - goal)) for n in nodes)
            print(f"  [rrt*] iter {it}/{max_iter}, nodes={len(nodes)}, "
                  f"best_dist_to_goal={best_to_goal:.3f}m")
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
            print(f"  [rrt*] goal reached at iter {it}, nodes={len(nodes)}")
            break

    if goal_idx < 0:
        best_to_goal = min(float(np.linalg.norm(n.pos - goal)) for n in nodes)
        print(f"  [rrt*] FAILED after {max_iter} iters, nodes={len(nodes)}, "
              f"best_dist_to_goal={best_to_goal:.3f}m")
        return []

    path = []
    cur = goal_idx
    while cur >= 0:
        path.append(nodes[cur].pos.copy())
        cur = nodes[cur].parent
    path.reverse()
    return path



def _resample_path(path: list[np.ndarray], max_spacing: float) -> list[np.ndarray]:
    """在路径点之间插值，使相邻点间距不超过 max_spacing。"""
    if len(path) < 2:
        return path
    result = [path[0].copy()]
    for i in range(1, len(path)):
        seg = path[i] - path[i - 1]
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= max_spacing:
            result.append(path[i].copy())
        else:
            n_sub = int(np.ceil(seg_len / max_spacing))
            for k in range(1, n_sub + 1):
                t = k / n_sub
                result.append(path[i - 1] + t * seg)
    return result

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

    clearance = float(getattr(args, "min_clearance", PLAN_MIN_CLEARANCE))
    print(f"[plan] clearance = {clearance:.4f}m")

    # --- resolve start: default = robobase initial world position ---
    if args.start:
        start = _pb2sdf(_parse_vec3(args.start)).reshape(3)
    else:
        p.connect(p.DIRECT)
        try:
            cfg = ExperimentConfig()
            robot, _ = _make_robot_and_workpiece(cfg)
            base_pos, _ = robot.get_robobase_pose()
            start_pb = np.asarray(base_pos, dtype=float)
            start = _pb2sdf(start_pb).reshape(3)
            print(f"[plan] start robobase PB={start_pb.tolist()} -> SDF={start.tolist()}")
        finally:
            p.disconnect()

    # --- resolve goal: default = nearest-region top_k[0] ---
    if args.goal:
        goal = _pb2sdf(_parse_vec3(args.goal)).reshape(3)
    elif nearest_region_as_goal and Path(nearest_json_path).exists():
        nr_data = json.loads(Path(nearest_json_path).read_text(encoding="utf-8"))
        topk_nr = nr_data.get("top_k", [])
        if topk_nr:
            goal_pb = np.asarray(topk_nr[0]["point"], dtype=float)
            goal = _pb2sdf(goal_pb).reshape(3)
            print(f"[plan] goal from nearest-region PB={goal_pb.tolist()} -> SDF={goal.tolist()}")
        else:
            goal = np.asarray(field.bbox_max, dtype=float) - 0.15
            print("[plan] WARNING: nearest-region JSON has no top_k, using bbox default")
    else:
        goal = np.asarray(field.bbox_max, dtype=float) - 0.15

    via = None
    if args.via_point:
        via = _pb2sdf(_parse_vec3(args.via_point)).reshape(3)

    bmin = np.asarray(field.bbox_min, dtype=float) + args.bound_margin
    bmax = np.asarray(field.bbox_max, dtype=float) - args.bound_margin
    bmin = np.minimum(bmin, start - 0.1)
    bmax = np.maximum(bmax, start + 0.1)
    bmin = np.minimum(bmin, goal - 0.1)
    bmax = np.maximum(bmax, goal + 0.1)
    if via is not None:
        bmin = np.minimum(bmin, via - 0.1)
        bmax = np.maximum(bmax, via + 0.1)

    if bool(args.auto_fix_endpoints):
        start, ok_start = _auto_fix_point_if_infeasible(
            field=field,
            point=start,
            bounds_min=bmin,
            bounds_max=bmax,
            kind=kind,
            clearance=clearance,
            search_radius=float(args.endpoint_fix_radius),
            search_step=float(args.endpoint_fix_step),
        )
        goal, ok_goal = _auto_fix_point_if_infeasible(
            field=field,
            point=goal,
            bounds_min=bmin,
            bounds_max=bmax,
            kind=kind,
            clearance=clearance,
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
                clearance=clearance,
                search_radius=float(args.endpoint_fix_radius),
                search_step=float(args.endpoint_fix_step),
            )
        else:
            ok_via = True
        if not (ok_start and ok_goal and ok_via):
            raise RuntimeError("端点清距修复失败：start/goal/via 至少有一个点无法在邻域内满足 SDF 阈值。")

    s_sdf = float(_query_field(field, start.reshape(1,3), kind=kind, safe_oob=True)[0])
    g_sdf = float(_query_field(field, goal.reshape(1,3), kind=kind, safe_oob=True)[0])
    dist_sg = float(np.linalg.norm(goal - start))
    print(f"[plan] SDF at start={s_sdf:.4f}, at goal={g_sdf:.4f}")
    print(f"[plan] start-goal dist={dist_sg:.3f}m, bounds={bmin.tolist()} .. {bmax.tolist()}")

    method = getattr(args, "method", PLAN_METHOD)
    bmin_f = np.asarray(field.bbox_min, dtype=float)
    bmax_f = np.asarray(field.bbox_max, dtype=float)
    start_oob = np.any(start < bmin_f) or np.any(start > bmax_f)
    goal_oob = np.any(goal < bmin_f) or np.any(goal > bmax_f)
    waypoints = [start] + ([via] if via is not None else []) + [goal]

    if method == "astar":
        print(f"[plan] method=A*")
        path = []
        for seg_i in range(len(waypoints) - 1):
            seg = _astar_sdf_plan(field, waypoints[seg_i], waypoints[seg_i + 1],
                                  kind=kind, clearance=clearance)
            if not seg:
                raise RuntimeError(f"A* 段 {seg_i} 未找到可行路径。")
            path.extend(seg if not path else seg[1:])
    else:
        print(f"[plan] method=RRT*")
        path = []
        for seg_i in range(len(waypoints) - 1):
            seg = _rrt_star_plan(
                field=field,
                start=waypoints[seg_i],
                goal=waypoints[seg_i + 1],
                bounds_min=bmin,
                bounds_max=bmax,
                kind=kind,
                clearance=clearance,
                step_size=float(args.step_size),
                near_radius=float(args.near_radius),
                goal_sample_prob=float(args.goal_sample_prob),
                max_iter=int(args.max_iter),
                edge_step=float(args.edge_check_step),
                goal_tol=float(args.goal_tolerance),
            )
            if not seg:
                raise RuntimeError(f"RRT* 段 {seg_i} 未找到可行路径。")
            path.extend(seg if not path else seg[1:])
        path = _shortcut_smooth(
            path, field=field, kind=kind, clearance=clearance,
            edge_step=float(args.edge_check_step),
            iters=int(args.smooth_iters),
        )

    if not path:
        raise RuntimeError("未找到可行路径。")

    print(f"[plan] raw/smoothed path: {len(path)} waypoints")
    resample_spacing = float(getattr(args, "resample_spacing", PLAN_RESAMPLE_SPACING))
    path = _resample_path(path, resample_spacing)

    pts_sdf = np.asarray(path, dtype=float)
    d_path = _query_field(field, pts_sdf, kind=kind, safe_oob=True)
    min_sdf = float(np.min(d_path))
    print(f"[plan] resampled grid path: {len(path)} waypoints, "
          f"min_sdf={min_sdf:.4f}m, max_spacing={resample_spacing}m")
    if min_sdf <= clearance:
        raise RuntimeError(
            f"路径不满足 SDF 阈值 {clearance:.4f}m (min={min_sdf:.4f}m)。")

    if start_oob:
        path.insert(0, start.copy())
        print(f"[plan] 起点在 SDF bbox 外，已插入原始起点 (总 {len(path)} 点)")
    if goal_oob:
        path.append(goal.copy())
        print(f"[plan] 终点在 SDF bbox 外，已追加原始终点 (总 {len(path)} 点)")

    pts_sdf = np.asarray(path, dtype=float)
    d_path = _query_field(field, pts_sdf, kind=kind, safe_oob=True)
    pts = _sdf2pb(pts_sdf)

    out_json = Path(args.output_json)
    _ensure_parent(out_json)
    payload = {
        "method": method,
        "kind": kind,
        "n_waypoints": int(len(path)),
        "min_sdf_on_waypoints": float(np.min(d_path)),
        "start_used": _sdf2pb(np.asarray(start, dtype=float).reshape(1,3)).reshape(3).tolist(),
        "goal_used": _sdf2pb(np.asarray(goal, dtype=float).reshape(1,3)).reshape(3).tolist(),
        "via_point": None if via is None else _sdf2pb(np.asarray(via, dtype=float).reshape(1,3)).reshape(3).tolist(),
        "path": pts.tolist(),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", ms=2.8, lw=1.4, c="dodgerblue")
    ax.scatter([start[0]], [start[1]], [start[2]], c="green", s=50, label="start")
    ax.scatter([goal[0]], [goal[1]], [goal[2]], c="red", s=50, label="goal")
    ax.set_title(f"SDF-constrained {method.upper()} path")
    ax.legend()
    out_png = Path(args.output_png)
    _ensure_parent(out_png)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[planner] 完成，结果写入: {out_json}")

    if VIS_PYBULLET and "plan" in VIS_PYBULLET_STEPS:
        _, _ = _pybullet_gui_open(ExperimentConfig(), load_robot=True, load_workpiece=True)
        for i in range(len(pts) - 1):
            p.addUserDebugLine(
                pts[i].tolist(), pts[i + 1].tolist(),
                [0.1, 0.5, 1.0], lineWidth=2.5,
            )
        if pts.shape[0] > 0:
            c_path = np.full((pts.shape[0], 3), [0.1, 0.5, 1.0])
            p.addUserDebugPoints(pts.tolist(), c_path.tolist(), pointSize=5)
        start_pb = _sdf2pb(start.reshape(1, 3)).reshape(3)
        goal_pb = _sdf2pb(goal.reshape(1, 3)).reshape(3)
        _create_sphere_marker(start_pb, 0.02, (0.1, 0.9, 0.2, 0.9))
        _create_sphere_marker(goal_pb, 0.02, (0.95, 0.15, 0.15, 0.9))
        p.addUserDebugText(
            "start", (start_pb + np.array([0, 0, 0.04])).tolist(),
            [0.1, 0.7, 0.1], textSize=1.2,
        )
        p.addUserDebugText(
            "goal", (goal_pb + np.array([0, 0, 0.04])).tolist(),
            [0.9, 0.1, 0.1], textSize=1.2,
        )
        p.addUserDebugText(
            f"{method.upper()} path: {len(path)} pts, min SDF={float(np.min(d_path))*1000:.1f}mm",
            [0.0, -0.3, 0.5], [0.1, 0.1, 0.1], textSize=1.2,
        )
        _pybullet_gui_wait()


def _default_align_args() -> argparse.Namespace:
    return argparse.Namespace(
        sdf_npz=DEFAULT_SDF_NPZ,
        align_mode=ALIGN_MODE,
        kind=ALIGN_KIND,
        urdf_path=DEFAULT_ROBOT_URDF_PATH,
        workpiece_urdf_path=DEFAULT_WORKPIECE_URDF_PATH,
        workpiece_position=None,
        workpiece_orientation_deg=None,
        per_link_points=ALIGN_PER_LINK_POINTS,
        max_points=ALIGN_MAX_POINTS,
        max_iter=ALIGN_MAX_ITER,
        seed=ALIGN_SEED,
        output_json=ALIGN_OUTPUT_JSON,
        output_png=ALIGN_OUTPUT_PNG,
    )


def _default_nearest_region_args() -> argparse.Namespace:
    return argparse.Namespace(
        sdf_npz=DEFAULT_SDF_NPZ,
        kind=NEAR_KIND,
        kernel_npz=INIT_OUTPUT_NPZ,
        workpiece_urdf_path=DEFAULT_WORKPIECE_URDF_PATH,
        weld_point=None,
        weld_link_name=DEFAULT_WELD_START_LINK,
        search_radius_min=NEAR_SEARCH_RADIUS_MIN,
        search_radius_max=NEAR_SEARCH_RADIUS_MAX,
        search_rings=NEAR_SEARCH_RINGS,
        samples_per_ring=NEAR_SAMPLES_PER_RING,
        search_step=NEAR_SEARCH_STEP,
        min_clearance=NEAR_MIN_CLEARANCE,
        min_line_clearance=NEAR_MIN_LINE_CLEARANCE,
        kernel_clearance=NEAR_KERNEL_CLEARANCE,
        line_samples=NEAR_LINE_SAMPLES,
        top_k=NEAR_TOP_K,
        seed=NEAR_SEED,
        surface_normal_eps=NEAR_SURFACE_NORMAL_EPS,
        normal_half_sphere=NEAR_NORMAL_HALF_SPHERE,
        normal_cone_cos=NEAR_NORMAL_CONE_COS,
        line_skip_ratio=NEAR_LINE_SKIP_RATIO,
        require_above_weld=NEAR_REQUIRE_ABOVE_WELD,
        above_weld_min_dz=NEAR_ABOVE_WELD_MIN_DZ,
        bbox_margin=NEAR_BBOX_MARGIN,
        output_json=NEAR_OUTPUT_JSON,
        output_png=NEAR_OUTPUT_PNG,
    )


def _default_init_config_args() -> argparse.Namespace:
    return argparse.Namespace(
        urdf_path=DEFAULT_ROBOT_URDF_PATH,
        workpiece_urdf_path=DEFAULT_WORKPIECE_URDF_PATH,
        num_samples=INIT_NUM_SAMPLES,
        sample_std=INIT_SAMPLE_STD,
        min_clearance=INIT_MIN_CLEARANCE,
        voxel=INIT_VOXEL,
        seed=INIT_SEED,
        output_npz=INIT_OUTPUT_NPZ,
        output_png=INIT_OUTPUT_PNG,
        skip_external_collision=INIT_SKIP_EXTERNAL_COLLISION,
        output_json=INIT_OUTPUT_JSON,
    )


def _default_plan_args() -> argparse.Namespace:
    return argparse.Namespace(
        method=PLAN_METHOD,
        sdf_npz=DEFAULT_SDF_NPZ,
        kind=PLAN_KIND,
        start=PLAN_START,
        goal=PLAN_GOAL,
        via_point=None,
        nearest_region_json=PLAN_NEAREST_REGION_JSON,
        nearest_region_as_goal=PLAN_NEAREST_REGION_AS_GOAL,
        min_clearance=PLAN_MIN_CLEARANCE,
        step_size=PLAN_STEP_SIZE,
        near_radius=PLAN_NEAR_RADIUS,
        goal_sample_prob=PLAN_GOAL_SAMPLE_PROB,
        max_iter=PLAN_MAX_ITER,
        goal_tolerance=PLAN_GOAL_TOLERANCE,
        edge_check_step=PLAN_EDGE_CHECK_STEP,
        smooth_iters=PLAN_SMOOTH_ITERS,
        resample_spacing=PLAN_RESAMPLE_SPACING,
        bound_margin=PLAN_BOUND_MARGIN,
        auto_fix_endpoints=PLAN_AUTO_FIX_ENDPOINTS,
        endpoint_fix_radius=PLAN_ENDPOINT_FIX_RADIUS,
        endpoint_fix_step=PLAN_ENDPOINT_FIX_STEP,
        init_config_npz=PLAN_INIT_CONFIG_NPZ,
        output_json=PLAN_OUTPUT_JSON,
        output_png=PLAN_OUTPUT_PNG,
    )


_STEP_DISPATCH = {
    "align": (run_alignment_experiment, _default_align_args),
    "init-config": (run_init_config_experiment, _default_init_config_args),
    "nearest-region": (run_nearest_region_experiment, _default_nearest_region_args),
    "plan": (run_planner_experiment, _default_plan_args),
}


if __name__ == "__main__":
    print("=" * 60)
    print(f"  SDF Integration Experiments")
    print(f"  Steps: {RUN_STEPS}")
    print("=" * 60)
    for step_name in RUN_STEPS:
        if step_name not in _STEP_DISPATCH:
            raise ValueError(f"unknown step: {step_name!r}, valid: {list(_STEP_DISPATCH)}")
        run_fn, args_fn = _STEP_DISPATCH[step_name]
        print(f"\n{'='*60}")
        print(f"  >>> Running: {step_name}")
        print(f"{'='*60}\n")
        run_fn(args_fn())
