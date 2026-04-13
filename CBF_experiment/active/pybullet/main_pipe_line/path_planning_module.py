from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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

from CBF_experiment.active.pybullet.configuration_metrics import compute_self_collision_clearance, evaluate_configuration_quality, rank_configuration_records, summarize_clearance_entries  # noqa: E402
from CBF_experiment.active.pybullet.simulation_module import ExperimentConfig, JakaRobot, WorkpieceModel  # noqa: E402


def _rank_init_config_candidates(records: list[dict], selection_weights: dict | None = None) -> list[dict]:
    return rank_configuration_records(records, weights=selection_weights)


class ExperimentParameters:
    """用户集中参数区（直接 F5 运行，优先改这里）。"""

    _CFG = ExperimentConfig()
    _QUALITY_CFG = dict(_CFG._cfg.get("configuration_quality", {}))

    RUN_STEPS = ["align", "init-config", "nearest-region", "plan"]
    #RUN_STEPS = ["nearest-region", "plan"]
    DEFAULT_SDF_NPZ = (
        "assets/cad_exports/model_CAD/scene/urdf/中组立0725(1).stp.SLDASM_udf.npz"
    )
    DEFAULT_ROBOT_URDF_PATH = _CFG.urdf_path
    DEFAULT_WORKPIECE_URDF_PATH = _CFG.workpiece_urdf_path

    DEFAULT_GANTRY_INITIAL_Q = tuple(float(x) for x in _CFG.gantry_initial_q)
    DEFAULT_WELD_START_LINK = str(_CFG.start_link_name)
    DEFAULT_WELD_GOAL_LINK = str(_CFG.goal_link_name)

    ALIGN_MODE = "known"
    ALIGN_KIND = "auto"
    ALIGN_PER_LINK_POINTS = 1200
    ALIGN_MAX_POINTS = 20000
    ALIGN_MAX_ITER = 160
    ALIGN_SEED = 1
    ALIGN_OUTPUT_JSON = "artifacts/sdf_exp/alignment_report.json"
    ALIGN_OUTPUT_PNG = "artifacts/sdf_exp/alignment_error.png"

    NEAR_KIND = "auto"
    NEAR_MIN_CLEARANCE = 0.01
    NEAR_TOP_K = 8
    NEAR_CANDIDATE_POOL = 256
    NEAR_REQUIRE_ABOVE_WELD = True
    NEAR_ABOVE_WELD_MIN_DZ = 0.3
    NEAR_BBOX_MARGIN = 0.10
    NEAR_SURFACE_NORMAL_EPS = 0.002
    NEAR_WELD_GOAL_POINT = None
    NEAR_WELD_GOAL_LINK_NAME = DEFAULT_WELD_GOAL_LINK
    NEAR_SEAM_SAMPLES = 9
    NEAR_OCCUPANCY_NPZ = None
    NEAR_OCCUPANCY_MARGIN = 0.0
    NEAR_RAY_STEP = 0.02
    NEAR_GRAD_SIDE_DOT_EPS = 0.0
    NEAR_GRAD_SIDE_MIN_RATIO = 0.6
    NEAR_REPRESENTATIVE_STRATEGY = "centroid"
    NEAR_MAX_RAYS_TO_VIS = 12
    NEAR_TOPK_RAY_SAMPLE_STRIDE = 2
    NEAR_OUTPUT_JSON = "artifacts/sdf_exp/nearest_region.json"
    NEAR_OUTPUT_PNG = "artifacts/sdf_exp/nearest_region.png"

    INIT_NUM_SAMPLES = 2000
    INIT_SAMPLE_STD = 0.8
    INIT_REFINE_TOP_K = 10
    INIT_REFINE_SAMPLES = 300
    INIT_REFINE_STD = 0.15
    INIT_MIN_CLEARANCE = 0.005
    INIT_VOXEL = 0.04
    INIT_SEED = 3
    INIT_OUTPUT_NPZ = "artifacts/sdf_exp/init_kernel.npz"
    INIT_OUTPUT_PNG = "artifacts/sdf_exp/init_kernel.png"
    INIT_SKIP_EXTERNAL_COLLISION = True
    INIT_OUTPUT_JSON = "artifacts/sdf_exp/init_config_report.json"
    INIT_SELECTION_WEIGHTS = dict(_QUALITY_CFG.get("selection_weights", {}))
    INIT_MOTION_COMPONENT = str(_QUALITY_CFG.get("motion_component", "linear"))
    INIT_SELF_COLLISION_QUERY_DISTANCE = float(_QUALITY_CFG.get("self_collision_query_distance", 0.12))

    PLAN_METHOD = "rrt*"
    PLAN_KIND = "auto"
    PLAN_MIN_CLEARANCE = 0.30
    PLAN_RESAMPLE_SPACING = 0.05
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
    PLAN_NEAREST_REGION_AS_GOAL = True
    PLAN_NEAREST_REGION_JSON = NEAR_OUTPUT_JSON
    PLAN_INIT_CONFIG_NPZ = INIT_OUTPUT_NPZ
    PLAN_START = None
    PLAN_GOAL = None
    PLAN_EE_MIN_CLEARANCE = 0.05
    PLAN_EE_TARGET_SDF = 0.10
    PLAN_EE_BACKTRACK_INIT_STEP = 0.20
    PLAN_EE_BACKTRACK_MIN_STEP = 0.005
    PLAN_EE_BACKTRACK_SHRINK = 0.5
    PLAN_EE_BACKTRACK_MAX_ITERS = 32
    PLAN_EE_BACKTRACK_CURV_EPS = 0.01
    PLAN_EE_BACKTRACK_ARMIJO_C1 = 0.05
    PLAN_EE_STEP_SIZE = 0.08
    PLAN_EE_NEAR_RADIUS = 0.18
    PLAN_EE_GOAL_TOLERANCE = 0.05
    PLAN_EE_EDGE_CHECK_STEP = 0.01
    PLAN_EE_RESAMPLE_SPACING = 0.03
    PLAN_EE_MAX_ITER = 5000
    PLAN_EE_BEZIER_SAMPLES_PER_SEG = 40
    PLAN_EE_BEZIER_APPROACH_MIN_SDF = -1e-4

    VIS_PYBULLET = True
    VIS_PYBULLET_STEPS = ["init-config", "nearest-region", "plan"]
    VIS_CAMERA_DISTANCE = 1.4
    VIS_CAMERA_YAW = -215.0
    VIS_CAMERA_PITCH = -26.0

    @classmethod
    def alignment_settings(cls) -> SimpleNamespace:
        return SimpleNamespace(
            sdf_npz=cls.DEFAULT_SDF_NPZ,
            align_mode=cls.ALIGN_MODE,
            kind=cls.ALIGN_KIND,
            urdf_path=cls.DEFAULT_ROBOT_URDF_PATH,
            workpiece_urdf_path=cls.DEFAULT_WORKPIECE_URDF_PATH,
            workpiece_position=None,
            workpiece_orientation_deg=None,
            per_link_points=cls.ALIGN_PER_LINK_POINTS,
            max_points=cls.ALIGN_MAX_POINTS,
            max_iter=cls.ALIGN_MAX_ITER,
            seed=cls.ALIGN_SEED,
            output_json=cls.ALIGN_OUTPUT_JSON,
            output_png=cls.ALIGN_OUTPUT_PNG,
        )

    @classmethod
    def nearest_region_settings(cls) -> SimpleNamespace:
        return SimpleNamespace(
            sdf_npz=cls.DEFAULT_SDF_NPZ,
            kind=cls.NEAR_KIND,
            kernel_npz=cls.INIT_OUTPUT_NPZ,
            workpiece_urdf_path=cls.DEFAULT_WORKPIECE_URDF_PATH,
            weld_point=None,
            weld_link_name=cls.DEFAULT_WELD_START_LINK,
            weld_goal_point=cls.NEAR_WELD_GOAL_POINT,
            weld_goal_link_name=cls.NEAR_WELD_GOAL_LINK_NAME,
            seam_samples=cls.NEAR_SEAM_SAMPLES,
            min_clearance=cls.NEAR_MIN_CLEARANCE,
            top_k=cls.NEAR_TOP_K,
            candidate_pool=cls.NEAR_CANDIDATE_POOL,
            surface_normal_eps=cls.NEAR_SURFACE_NORMAL_EPS,
            require_above_weld=cls.NEAR_REQUIRE_ABOVE_WELD,
            above_weld_min_dz=cls.NEAR_ABOVE_WELD_MIN_DZ,
            bbox_margin=cls.NEAR_BBOX_MARGIN,
            occupancy_npz=cls.NEAR_OCCUPANCY_NPZ,
            occupancy_margin=cls.NEAR_OCCUPANCY_MARGIN,
            ray_step=cls.NEAR_RAY_STEP,
            grad_side_dot_eps=cls.NEAR_GRAD_SIDE_DOT_EPS,
            grad_side_min_ratio=cls.NEAR_GRAD_SIDE_MIN_RATIO,
            representative_strategy=cls.NEAR_REPRESENTATIVE_STRATEGY,
            max_rays_to_vis=cls.NEAR_MAX_RAYS_TO_VIS,
            topk_ray_sample_stride=cls.NEAR_TOPK_RAY_SAMPLE_STRIDE,
            output_json=cls.NEAR_OUTPUT_JSON,
            output_png=cls.NEAR_OUTPUT_PNG,
        )

    @classmethod
    def init_config_settings(cls) -> SimpleNamespace:
        return SimpleNamespace(
            urdf_path=cls.DEFAULT_ROBOT_URDF_PATH,
            workpiece_urdf_path=cls.DEFAULT_WORKPIECE_URDF_PATH,
            num_samples=cls.INIT_NUM_SAMPLES,
            sample_std=cls.INIT_SAMPLE_STD,
            refine_top_k=cls.INIT_REFINE_TOP_K,
            refine_samples=cls.INIT_REFINE_SAMPLES,
            refine_std=cls.INIT_REFINE_STD,
            min_clearance=cls.INIT_MIN_CLEARANCE,
            voxel=cls.INIT_VOXEL,
            seed=cls.INIT_SEED,
            output_npz=cls.INIT_OUTPUT_NPZ,
            output_png=cls.INIT_OUTPUT_PNG,
            skip_external_collision=cls.INIT_SKIP_EXTERNAL_COLLISION,
            output_json=cls.INIT_OUTPUT_JSON,
            selection_weights=cls.INIT_SELECTION_WEIGHTS,
            motion_component=cls.INIT_MOTION_COMPONENT,
            self_collision_query_distance=cls.INIT_SELF_COLLISION_QUERY_DISTANCE,
        )

    @classmethod
    def plan_settings(cls) -> SimpleNamespace:
        return SimpleNamespace(
            method=cls.PLAN_METHOD,
            sdf_npz=cls.DEFAULT_SDF_NPZ,
            kind=cls.PLAN_KIND,
            start=cls.PLAN_START,
            goal=cls.PLAN_GOAL,
            via_point=None,
            nearest_region_json=cls.PLAN_NEAREST_REGION_JSON,
            nearest_region_as_goal=cls.PLAN_NEAREST_REGION_AS_GOAL,
            min_clearance=cls.PLAN_MIN_CLEARANCE,
            step_size=cls.PLAN_STEP_SIZE,
            near_radius=cls.PLAN_NEAR_RADIUS,
            goal_sample_prob=cls.PLAN_GOAL_SAMPLE_PROB,
            max_iter=cls.PLAN_MAX_ITER,
            goal_tolerance=cls.PLAN_GOAL_TOLERANCE,
            edge_check_step=cls.PLAN_EDGE_CHECK_STEP,
            smooth_iters=cls.PLAN_SMOOTH_ITERS,
            resample_spacing=cls.PLAN_RESAMPLE_SPACING,
            bound_margin=cls.PLAN_BOUND_MARGIN,
            auto_fix_endpoints=cls.PLAN_AUTO_FIX_ENDPOINTS,
            endpoint_fix_radius=cls.PLAN_ENDPOINT_FIX_RADIUS,
            endpoint_fix_step=cls.PLAN_ENDPOINT_FIX_STEP,
            init_config_npz=cls.PLAN_INIT_CONFIG_NPZ,
            ee_min_clearance=cls.PLAN_EE_MIN_CLEARANCE,
            ee_target_sdf=cls.PLAN_EE_TARGET_SDF,
            ee_backtrack_init_step=cls.PLAN_EE_BACKTRACK_INIT_STEP,
            ee_backtrack_min_step=cls.PLAN_EE_BACKTRACK_MIN_STEP,
            ee_backtrack_shrink=cls.PLAN_EE_BACKTRACK_SHRINK,
            ee_backtrack_max_iters=cls.PLAN_EE_BACKTRACK_MAX_ITERS,
            ee_backtrack_curv_eps=cls.PLAN_EE_BACKTRACK_CURV_EPS,
            ee_backtrack_armijo_c1=cls.PLAN_EE_BACKTRACK_ARMIJO_C1,
            ee_step_size=cls.PLAN_EE_STEP_SIZE,
            ee_near_radius=cls.PLAN_EE_NEAR_RADIUS,
            ee_goal_tolerance=cls.PLAN_EE_GOAL_TOLERANCE,
            ee_edge_check_step=cls.PLAN_EE_EDGE_CHECK_STEP,
            ee_resample_spacing=cls.PLAN_EE_RESAMPLE_SPACING,
            ee_max_iter=cls.PLAN_EE_MAX_ITER,
            ee_bezier_samples_per_seg=cls.PLAN_EE_BEZIER_SAMPLES_PER_SEG,
            ee_bezier_approach_min_sdf=cls.PLAN_EE_BEZIER_APPROACH_MIN_SDF,
            output_json=cls.PLAN_OUTPUT_JSON,
            output_png=cls.PLAN_OUTPUT_PNG,
        )


class ExperimentRunner:
    def __init__(self) -> None:
        self._udf_module = None
        self.align_settings = ExperimentParameters.alignment_settings()
        self.nearest_region_settings = ExperimentParameters.nearest_region_settings()
        self.init_config_settings = ExperimentParameters.init_config_settings()
        self.plan_settings = ExperimentParameters.plan_settings()
        self._wp_pos, self._r_inv = self._build_pybullet_to_sdf_transform()

    @staticmethod
    def _load_udf_module():
        path = Path(__file__).resolve().parent / "4_1_udf.py"
        spec = importlib.util.spec_from_file_location("udf_module_runtime", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载模块: {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        if not path.is_absolute():
            return (REPO_ROOT / path).resolve()
        return path

    @staticmethod
    def _ensure_parent(path: Path) -> str:
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    @staticmethod
    def _parse_vec3(text: str) -> np.ndarray:
        parts = [float(x.strip()) for x in text.split(",")]
        if len(parts) != 3:
            raise ValueError(f"需要 3 个值，收到: {text}")
        return np.asarray(parts, dtype=float)

    @staticmethod
    def _world_to_local(points_world: np.ndarray, world_pos: np.ndarray, world_quat: np.ndarray) -> np.ndarray:
        inv_pos, inv_quat = p.invertTransform(
            np.asarray(world_pos, dtype=float).tolist(),
            np.asarray(world_quat, dtype=float).tolist(),
        )
        rot = np.array(p.getMatrixFromQuaternion(inv_quat), dtype=float).reshape(3, 3)
        pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
        return (rot @ pts.T).T + np.asarray(inv_pos, dtype=float).reshape(1, 3)

    @staticmethod
    def _local_to_world(pts_local: np.ndarray, world_pos, world_quat) -> np.ndarray:
        rot = np.array(
            p.getMatrixFromQuaternion(np.asarray(world_quat, dtype=float).tolist()),
            dtype=float,
        ).reshape(3, 3)
        pts = np.asarray(pts_local, dtype=float).reshape(-1, 3)
        return (rot @ pts.T).T + np.asarray(world_pos, dtype=float).reshape(1, 3)

    @staticmethod
    def _best_kind(field) -> str:
        if np.isfinite(np.asarray(field.o3d_sdf_grid)).any():
            return "o3d_sdf"
        if np.isfinite(np.asarray(field.igl_sdf_grid)).any():
            return "igl_sdf"
        return "udf"

    @staticmethod
    def _build_pybullet_to_sdf_transform() -> tuple[np.ndarray, np.ndarray]:
        cfg = ExperimentConfig()
        wp_pos = np.asarray(cfg.workpiece_position, dtype=float)
        wp_deg = np.asarray(cfg.workpiece_orientation_deg, dtype=float)
        r_inv = Rotation.from_euler("xyz", wp_deg, degrees=True).as_matrix().T
        return wp_pos, r_inv

    def load_udf_module(self):
        if self._udf_module is None:
            self._udf_module = self._load_udf_module()
        return self._udf_module

    def load_field(self, sdf_npz: str):
        return self.load_udf_module().load_distance_field(sdf_npz)

    @staticmethod
    def _default_occupancy_npz_path(sdf_npz: str) -> Path:
        p_npz = Path(sdf_npz)
        return p_npz.with_name(f"{p_npz.stem}_occ.npz")

    def resolve_kind(self, field, kind: str) -> str:
        return kind if kind != "auto" else self._best_kind(field)

    def ensure_parent(self, path: Path) -> str:
        return self._ensure_parent(path)

    def parse_vec3(self, text: str) -> np.ndarray:
        return self._parse_vec3(text)

    def load_init_config(self, npz_path: str | None = None) -> dict | None:
        path = Path(npz_path or self.init_config_settings.output_npz)
        if not path.exists():
            return None
        with np.load(path) as data:
            payload = {}
            for key in data.files:
                value = data[key]
                if isinstance(value, np.ndarray):
                    payload[key] = np.array(value)
                else:
                    payload[key] = value
        payload["path"] = str(path)
        return payload

    def move_robot_base_to_position(
        self,
        robot: JakaRobot,
        target_base_pb: np.ndarray,
        q_seed: np.ndarray | None = None,
        max_iters: int = 6,
    ) -> np.ndarray:
        if q_seed is None:
            q, _ = robot.get_joint_state()
        else:
            q = np.asarray(q_seed, dtype=float).copy()
            robot.set_joint_state(q, dq=np.zeros_like(q))
        n_base = min(3, len(robot.prismatic_joints), q.shape[0])
        if n_base <= 0:
            return q
        target = np.asarray(target_base_pb, dtype=float).reshape(3)
        for _ in range(max(int(max_iters), 1)):
            base_pos, _ = robot.get_robobase_pose()
            delta = target - np.asarray(base_pos, dtype=float).reshape(3)
            if np.linalg.norm(delta) < 1e-4:
                break
            for i in range(n_base):
                q[i] = float(q[i] + delta[i])
                joint_index = int(robot.active_joints[i])
                info = p.getJointInfo(robot.body_id, joint_index)
                lo = float(info[8])
                hi = float(info[9])
                if hi > lo:
                    q[i] = float(np.clip(q[i], lo, hi))
            robot.set_joint_state(q, dq=np.zeros_like(q))
        return q

    def pb2sdf(self, pts_pybullet: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_pybullet, dtype=float).reshape(-1, 3)
        return (pts - self._wp_pos.reshape(1, 3)) @ self._r_inv.T

    def sdf2pb(self, pts_sdf: np.ndarray) -> np.ndarray:
        r = self._r_inv.T
        r_fwd = np.linalg.inv(r) if abs(np.linalg.det(r) - 1.0) < 1e-6 else r.T
        pts = np.asarray(pts_sdf, dtype=float).reshape(-1, 3)
        return pts @ r_fwd.T + self._wp_pos.reshape(1, 3)

    def sdf_dirs_to_pb(self, dirs_sdf: np.ndarray) -> np.ndarray:
        dirs = np.asarray(dirs_sdf, dtype=float).reshape(-1, 3)
        return dirs @ self._r_inv

    def query_field(self, field, points_sdf: np.ndarray, kind: str, safe_oob: bool = False) -> np.ndarray:
        pts = np.asarray(points_sdf, dtype=np.float32).reshape(-1, 3)
        if safe_oob:
            vals, _ = field.query_with_gradient(pts, kind=kind)
            return np.asarray(vals, dtype=float).reshape(-1)
        vals = field.query(pts, kind=kind, clip=True)
        return np.asarray(vals, dtype=float).reshape(-1)

    def query_field_pb(self, field, points_pybullet: np.ndarray, kind: str) -> np.ndarray:
        return self.query_field(field, self.pb2sdf(points_pybullet), kind=kind)

    def load_or_bake_occupancy_field(
        self,
        field,
        *,
        sdf_npz: str,
        workpiece_urdf_path: str,
        occupancy_npz: str | None,
        occupancy_margin: float = 0.0,
    ):
        udf_mod = self.load_udf_module()
        occ_path = (
            Path(occupancy_npz)
            if occupancy_npz
            else self._default_occupancy_npz_path(sdf_npz)
        )
        if occ_path.exists():
            occ_field = udf_mod.load_occupancy_field(occ_path)
            same_shape = tuple(int(x) for x in occ_field.grid.shape) == tuple(int(x) for x in np.asarray(field.udf_grid).shape)
            same_spacing = abs(float(occ_field.occ_spacing) - float(field.spacing)) < 1e-9
            same_origin = np.allclose(np.asarray(occ_field.origin), np.asarray(field.origin), atol=1e-6)
            if same_shape and same_spacing and same_origin:
                print(f"[occupancy] loaded cache: {occ_path}")
                return occ_field, occ_path
            print(f"[occupancy] cache mismatch, rebaking: {occ_path}")

        assy = udf_mod.load_assembly_from_urdf(workpiece_urdf_path)
        occ_grid, origin, _ = udf_mod.bake_occupancy_grid(
            assy.triangles,
            assy.bbox_min,
            assy.bbox_max,
            spacing=float(field.spacing),
            margin=float(field.build_config.get("margin", occupancy_margin)) if field.build_config else float(occupancy_margin),
        )
        occ_field = udf_mod.OccupancyField(
            origin=np.asarray(origin, dtype=np.float32),
            spacing=float(field.spacing),
            grid=np.asarray(occ_grid, dtype=np.bool_),
            bbox_min=np.asarray(assy.bbox_min, dtype=np.float32),
            bbox_max=np.asarray(assy.bbox_max, dtype=np.float32),
            occ_spacing=float(field.spacing),
        )
        occ_path = self.ensure_parent(occ_path)
        udf_mod.save_occupancy_field(occ_path, occ_field)
        return occ_field, occ_path

    def estimate_surface_normal(self, field, point_sdf: np.ndarray, kind: str, eps: float) -> np.ndarray:
        pt = np.asarray(point_sdf, dtype=float).reshape(3)
        grad = np.zeros(3, dtype=float)
        for ax in range(3):
            pp = pt.copy()
            pm = pt.copy()
            pp[ax] += eps
            pm[ax] -= eps
            vp = float(self.query_field(field, pp.reshape(1, 3), kind=kind)[0])
            vm = float(self.query_field(field, pm.reshape(1, 3), kind=kind)[0])
            grad[ax] = (vp - vm) / (2.0 * eps)
        nrm = np.linalg.norm(grad)
        if nrm < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return grad / nrm

    def auto_fix_point_if_infeasible(
        self,
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
        d0 = float(self.query_field(field, p0.reshape(1, 3), kind=kind, safe_oob=True)[0])
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
                        d = float(self.query_field(field, cand.reshape(1, 3), kind=kind)[0])
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

    def make_robot_and_workpiece(self, cfg: ExperimentConfig) -> tuple[JakaRobot, WorkpieceModel]:
        class _SceneStub:
            pass

        robot = JakaRobot(cfg, _SceneStub())
        workpiece = WorkpieceModel(cfg)
        robot.register_surface_obstacle(workpiece.body_id, None)
        return robot, workpiece

    def collect_surface_points(self, robot: JakaRobot, body_id: int, per_link_points: int = 800) -> np.ndarray:
        clouds = robot.get_surface_visualization_clouds(
            body_id=body_id,
            link_indices=None,
            max_points_per_link=per_link_points,
        )
        if not clouds:
            return np.zeros((0, 3), dtype=float)
        pts = [np.asarray(c["points"], dtype=float).reshape(-1, 3) for c in clouds]
        return np.vstack(pts) if pts else np.zeros((0, 3), dtype=float)

    def open_scene(
        self,
        cfg: ExperimentConfig,
        *,
        load_robot: bool = True,
        load_workpiece: bool = True,
        robot_q: np.ndarray | None = None,
        camera_target: list[float] | None = None,
    ):
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
            cameraDistance=ExperimentParameters.VIS_CAMERA_DISTANCE,
            cameraYaw=ExperimentParameters.VIS_CAMERA_YAW,
            cameraPitch=ExperimentParameters.VIS_CAMERA_PITCH,
            cameraTargetPosition=ct,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return robot, workpiece

    @staticmethod
    def wait() -> None:
        print("[vis] PyBullet GUI 已打开，关闭窗口或 Ctrl+C 继续下一步...")
        try:
            while p.isConnected():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            if p.isConnected():
                p.disconnect()

    @staticmethod
    def create_sphere_marker(pos, radius: float, rgba):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=list(rgba))
        bid = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=np.asarray(pos, dtype=float).tolist(),
        )
        p.setCollisionFilterGroupMask(bid, -1, 0, 0)
        return bid

    def draw_box_wireframe(self, lo, hi, world_pos, world_quat, color=(1.0, 0.5, 0.0), width=2.0):
        lo_a = np.asarray(lo, dtype=float)
        hi_a = np.asarray(hi, dtype=float)
        c_local = np.array([
            [lo_a[0], lo_a[1], lo_a[2]], [hi_a[0], lo_a[1], lo_a[2]],
            [hi_a[0], hi_a[1], lo_a[2]], [lo_a[0], hi_a[1], lo_a[2]],
            [lo_a[0], lo_a[1], hi_a[2]], [hi_a[0], lo_a[1], hi_a[2]],
            [hi_a[0], hi_a[1], hi_a[2]], [lo_a[0], hi_a[1], hi_a[2]],
        ])
        c_w = self._local_to_world(c_local, world_pos, world_quat)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0),
                     (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]:
            p.addUserDebugLine(c_w[a].tolist(), c_w[b].tolist(), list(color), lineWidth=width)

    def show_alignment_points(self, pts: np.ndarray, d_final: np.ndarray) -> None:
        if not (ExperimentParameters.VIS_PYBULLET and "align" in ExperimentParameters.VIS_PYBULLET_STEPS):
            return
        self.open_scene(ExperimentConfig(), load_robot=True, load_workpiece=True)
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
        self.wait()

    def show_nearest_region(
        self,
        weld_points_pb: np.ndarray,
        surface_normals_pb: np.ndarray,
        component_summaries: list[dict],
        selected_component_id: int | None,
        topk: list[dict],
        ray_segments: list[dict],
        total_feasible: int,
        n_components: int,
    ) -> None:
        if not (ExperimentParameters.VIS_PYBULLET and "nearest-region" in ExperimentParameters.VIS_PYBULLET_STEPS):
            return
        weld_point_pb = np.asarray(weld_points_pb[0], dtype=float)
        init_cfg = self.load_init_config()
        robot_q = None if init_cfg is None else np.asarray(init_cfg.get("q_best"), dtype=float).reshape(-1)
        self.open_scene(
            ExperimentConfig(),
            load_robot=True,
            load_workpiece=True,
            robot_q=robot_q,
            camera_target=weld_point_pb.tolist(),
        )
        for idx, wp in enumerate(np.asarray(weld_points_pb, dtype=float).reshape(-1, 3)):
            self.create_sphere_marker(wp, 0.012 if idx else 0.015, (0.95, 0.15, 0.15, 0.9))
        for idx, (wp, nrm) in enumerate(zip(np.asarray(weld_points_pb, dtype=float), np.asarray(surface_normals_pb, dtype=float))):
            p.addUserDebugLine(
                wp.tolist(),
                (wp + 0.10 * nrm).tolist(),
                [0.1, 0.3, 1.0], lineWidth=2.0 if idx else 3.0,
            )
        p.addUserDebugText(
            "seam", (weld_point_pb + np.array([0, 0, 0.03])).tolist(),
            [0.9, 0.1, 0.1], textSize=1.2,
        )
        for comp in component_summaries:
            rep = np.asarray(comp["representative_pb"], dtype=float)
            selected = selected_component_id is not None and int(comp["component_id"]) == int(selected_component_id)
            rgba = (1.0, 0.7, 0.0, 0.95) if selected else (0.75, 0.75, 0.75, 0.6)
            self.create_sphere_marker(rep, 0.014 if selected else 0.01, rgba)
            p.addUserDebugText(
                f"C{comp['component_id']} v={comp['votes']}",
                (rep + np.array([0, 0, 0.02])).tolist(),
                [0.4, 0.2, 0.1] if selected else [0.3, 0.3, 0.3],
                textSize=0.8,
            )
        for seg in ray_segments:
            a = np.asarray(seg["start"], dtype=float)
            b = np.asarray(seg["end"], dtype=float)
            occ_len = float(seg["occupied_length"])
            col = [0.2, 0.8, 0.2] if occ_len <= 1e-6 else [0.9, 0.2, 0.2]
            p.addUserDebugLine(a.tolist(), b.tolist(), col, lineWidth=2.0)
        for i, cand in enumerate(topk):
            cp = np.asarray(cand["point"], dtype=float)
            self.create_sphere_marker(cp, 0.012, (0.1, 0.9, 0.2, 0.9))
            p.addUserDebugLine(
                weld_point_pb.tolist(), cp.tolist(),
                [0.4, 0.8, 0.2], lineWidth=1.5,
            )
            p.addUserDebugText(
                f"#{i+1} score={cand['score']:.3f}",
                (cp + np.array([0, 0, 0.02])).tolist(),
                [0.1, 0.6, 0.1], textSize=0.9,
            )
        p.addUserDebugText(
            f"feasible: {total_feasible} voxels, {n_components} components, selected={selected_component_id}",
            [0.0, -0.3, 0.5], [0.1, 0.1, 0.1], textSize=1.2,
        )
        self.wait()

    def show_init_config(self, best: dict) -> None:
        if not (ExperimentParameters.VIS_PYBULLET and "init-config" in ExperimentParameters.VIS_PYBULLET_STEPS):
            return
        cfg_v = ExperimentConfig()
        robot_v, _ = self.open_scene(
            cfg_v,
            load_robot=True,
            load_workpiece=True,
            robot_q=np.asarray(best["q"], dtype=float),
        )
        bp, bq = robot_v.get_robobase_pose()
        ko = np.asarray(best["kernel_offsets"], dtype=float).reshape(-1, 3)
        if ko.shape[0] > 0:
            ko_w = self._local_to_world(ko, bp, bq)
            c_blue = np.full((ko_w.shape[0], 3), [0.2, 0.4, 1.0])
            p.addUserDebugPoints(ko_w.tolist(), c_blue.tolist(), pointSize=3)
        self.draw_box_wireframe(best["aabb_min"], best["aabb_max"], bp, bq)
        self.create_sphere_marker(bp, 0.025, (1.0, 0.3, 0.0, 0.9))
        p.addUserDebugText(
            "robobase", (bp + np.array([0, 0, 0.06])).tolist(),
            [0.8, 0.2, 0.0], textSize=1.2,
        )
        p.addUserDebugText(
            f"AABB max_dim: {best['aabb_max_dim']:.4f}m  r={best['bbox_radius']:.4f}m",
            [0.0, -0.3, 0.5], [0.1, 0.1, 0.1], textSize=1.2,
        )
        self.wait()

    def show_plan_path(
        self,
        base_pts_pb: np.ndarray,
        start_pb: np.ndarray,
        goal_pb: np.ndarray,
        method: str,
        d_path: np.ndarray,
        path: list[np.ndarray],
        *,
        robot_q: np.ndarray | None = None,
        ee_pts_pb: np.ndarray | None = None,
        ee_d_path: np.ndarray | None = None,
        ee_smoothed_pts_pb: np.ndarray | None = None,
        ee_smoothed_d_path: np.ndarray | None = None,
        ee_control_points_pb: np.ndarray | None = None,
        weld_start_pb: np.ndarray | None = None,
        weld_goal_pb: np.ndarray | None = None,
        retreat_pts_pb: np.ndarray | None = None,
        retreat_goal_pb: np.ndarray | None = None,
        approach_smoothed_pts_pb: np.ndarray | None = None,
        approach_smoothed_d_path: np.ndarray | None = None,
        approach_control_points_pb: np.ndarray | None = None,
        approach_line_pts_pb: np.ndarray | None = None,
        approach_line_d_path: np.ndarray | None = None,
        approach_line_feasible: bool | None = None,
        end_retreat_pts_pb: np.ndarray | None = None,
        end_retreat_goal_pb: np.ndarray | None = None,
        end_ee_pts_pb: np.ndarray | None = None,
        end_ee_smoothed_pts_pb: np.ndarray | None = None,
        end_approach_smoothed_pts_pb: np.ndarray | None = None,
        end_approach_control_points_pb: np.ndarray | None = None,
        end_approach_line_pts_pb: np.ndarray | None = None,
        end_approach_line_feasible: bool | None = None,
    ) -> None:
        if not (ExperimentParameters.VIS_PYBULLET and "plan" in ExperimentParameters.VIS_PYBULLET_STEPS):
            return
        robot, _ = self.open_scene(
            ExperimentConfig(),
            load_robot=True,
            load_workpiece=True,
            robot_q=None if robot_q is None else np.asarray(robot_q, dtype=float),
            camera_target=np.asarray(goal_pb, dtype=float).reshape(3).tolist(),
        )
        pts = np.asarray(base_pts_pb, dtype=float).reshape(-1, 3)
        for i in range(len(pts) - 1):
            p.addUserDebugLine(
                pts[i].tolist(), pts[i + 1].tolist(),
                [0.1, 0.5, 1.0], lineWidth=2.5,
            )
        if pts.shape[0] > 0:
            c_path = np.full((pts.shape[0], 3), [0.1, 0.5, 1.0])
            p.addUserDebugPoints(pts.tolist(), c_path.tolist(), pointSize=5)
        start_pb = np.asarray(start_pb, dtype=float).reshape(3)
        goal_pb = np.asarray(goal_pb, dtype=float).reshape(3)
        self.create_sphere_marker(start_pb, 0.02, (0.1, 0.9, 0.2, 0.9))
        self.create_sphere_marker(goal_pb, 0.02, (0.95, 0.15, 0.15, 0.9))
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
        if robot is not None:
            base_pos, _ = robot.get_robobase_pose()
            self.create_sphere_marker(base_pos, 0.025, (1.0, 0.5, 0.0, 0.9))
            p.addUserDebugText(
                "robobase@goal", (base_pos + np.array([0, 0, 0.06])).tolist(),
                [0.8, 0.35, 0.0], textSize=1.1,
            )
        if weld_start_pb is not None:
            weld_start_pb = np.asarray(weld_start_pb, dtype=float).reshape(3)
            self.create_sphere_marker(weld_start_pb, 0.014, (0.95, 0.1, 0.1, 0.9))
            p.addUserDebugText(
                "weld-start", (weld_start_pb + np.array([0, 0, 0.03])).tolist(),
                [0.8, 0.1, 0.1], textSize=0.95,
            )
        if retreat_pts_pb is not None:
            retreat_pts = np.asarray(retreat_pts_pb, dtype=float).reshape(-1, 3)
            for i in range(len(retreat_pts) - 1):
                p.addUserDebugLine(
                    retreat_pts[i].tolist(), retreat_pts[i + 1].tolist(),
                    [0.8, 0.2, 0.8], lineWidth=2.0,
                )
            p.addUserDebugPoints(
                retreat_pts.tolist(),
                np.full((retreat_pts.shape[0], 3), [0.8, 0.2, 0.8]).tolist(),
                pointSize=5,
            )
        if retreat_goal_pb is not None:
            retreat_goal_pb = np.asarray(retreat_goal_pb, dtype=float).reshape(3)
            self.create_sphere_marker(retreat_goal_pb, 0.016, (0.7, 0.0, 0.9, 0.95))
            p.addUserDebugText(
                "retreat-goal", (retreat_goal_pb + np.array([0, 0, 0.03])).tolist(),
                [0.5, 0.0, 0.7], textSize=0.95,
            )
        if ee_pts_pb is not None:
            ee_pts = np.asarray(ee_pts_pb, dtype=float).reshape(-1, 3)
            for i in range(len(ee_pts) - 1):
                p.addUserDebugLine(
                    ee_pts[i].tolist(), ee_pts[i + 1].tolist(),
                    [0.95, 0.75, 0.1], lineWidth=2.4,
                )
            p.addUserDebugPoints(
                ee_pts.tolist(),
                np.full((ee_pts.shape[0], 3), [0.95, 0.75, 0.1]).tolist(),
                pointSize=5,
            )
            if ee_d_path is not None and len(ee_d_path) > 0:
                p.addUserDebugText(
                    f"EE path: {ee_pts.shape[0]} pts, min SDF={float(np.min(ee_d_path))*1000:.1f}mm",
                    [0.0, -0.36, 0.45], [0.1, 0.1, 0.1], textSize=1.0,
                )
        if ee_smoothed_pts_pb is not None:
            ee_smooth = np.asarray(ee_smoothed_pts_pb, dtype=float).reshape(-1, 3)
            for i in range(len(ee_smooth) - 1):
                p.addUserDebugLine(
                    ee_smooth[i].tolist(), ee_smooth[i + 1].tolist(),
                    [0.0, 0.8, 0.8], lineWidth=2.2,
                )
            if ee_control_points_pb is not None:
                cps = np.asarray(ee_control_points_pb, dtype=float).reshape(-1, 3)
                p.addUserDebugPoints(
                    cps.tolist(),
                    np.full((cps.shape[0], 3), [0.0, 0.8, 0.8]).tolist(),
                    pointSize=7,
                )
            if ee_smoothed_d_path is not None and len(ee_smoothed_d_path) > 0:
                p.addUserDebugText(
                    f"EE bezier: {ee_smooth.shape[0]} pts, min SDF={float(np.min(ee_smoothed_d_path))*1000:.1f}mm",
                    [0.0, -0.41, 0.40], [0.1, 0.1, 0.1], textSize=1.0,
                )
        app_smooth = None
        if approach_smoothed_pts_pb is not None:
            app_smooth = np.asarray(approach_smoothed_pts_pb, dtype=float).reshape(-1, 3)
            for i in range(len(app_smooth) - 1):
                p.addUserDebugLine(
                    app_smooth[i].tolist(), app_smooth[i + 1].tolist(),
                    [0.95, 0.45, 0.0], lineWidth=2.0,
                )
            if approach_control_points_pb is not None:
                cps = np.asarray(approach_control_points_pb, dtype=float).reshape(-1, 3)
                p.addUserDebugPoints(
                    cps.tolist(),
                    np.full((cps.shape[0], 3), [0.95, 0.45, 0.0]).tolist(),
                    pointSize=7,
                )
            if approach_smoothed_d_path is not None and len(approach_smoothed_d_path) > 0:
                p.addUserDebugText(
                        f"Approach line: {app_smooth.shape[0]} pts, min SDF={float(np.min(approach_smoothed_d_path))*1000:.1f}mm",
                    [0.0, -0.46, 0.35], [0.1, 0.1, 0.1], textSize=0.95,
                )
        if approach_line_pts_pb is not None:
            app_line = np.asarray(approach_line_pts_pb, dtype=float).reshape(-1, 3)
            same_as_actual = (
                app_smooth is not None
                and app_smooth.shape == app_line.shape
                and np.allclose(app_smooth, app_line)
            )
            if not same_as_actual:
                line_color = [1.0, 0.82, 0.15] if bool(approach_line_feasible) else [1.0, 0.55, 0.1]
                for i in range(len(app_line) - 1):
                    p.addUserDebugLine(
                        app_line[i].tolist(), app_line[i + 1].tolist(),
                        line_color, lineWidth=1.8,
                    )
                p.addUserDebugPoints(
                    app_line.tolist(),
                    np.full((app_line.shape[0], 3), line_color).tolist(),
                    pointSize=4,
                )
                if approach_line_d_path is not None and len(approach_line_d_path) > 0:
                    tag = "feasible" if bool(approach_line_feasible) else "candidate only"
                    p.addUserDebugText(
                        f"Approach line: {app_line.shape[0]} pts, min SDF={float(np.min(approach_line_d_path))*1000:.1f}mm ({tag})",
                        [0.0, -0.51, 0.30], [0.1, 0.1, 0.1], textSize=0.92,
                    )
        if weld_start_pb is not None and weld_goal_pb is not None:
            ws = np.asarray(weld_start_pb, dtype=float).reshape(3)
            wg = np.asarray(weld_goal_pb, dtype=float).reshape(3)
            p.addUserDebugLine(ws.tolist(), wg.tolist(), [0.95, 0.1, 0.1], lineWidth=3.5)
            self.create_sphere_marker(wg, 0.014, (0.6, 0.0, 0.85, 0.9))
            p.addUserDebugText(
                "weld-goal", (wg + np.array([0, 0, 0.03])).tolist(),
                [0.5, 0.0, 0.7], textSize=0.95,
            )
        if end_retreat_pts_pb is not None:
            end_retreat = np.asarray(end_retreat_pts_pb, dtype=float).reshape(-1, 3)
            for i in range(len(end_retreat) - 1):
                p.addUserDebugLine(
                    end_retreat[i].tolist(), end_retreat[i + 1].tolist(),
                    [0.55, 0.2, 0.8], lineWidth=2.0,
                )
            p.addUserDebugPoints(
                end_retreat.tolist(),
                np.full((end_retreat.shape[0], 3), [0.55, 0.2, 0.8]).tolist(),
                pointSize=5,
            )
        if end_retreat_goal_pb is not None:
            erg = np.asarray(end_retreat_goal_pb, dtype=float).reshape(3)
            self.create_sphere_marker(erg, 0.016, (0.45, 0.0, 0.75, 0.95))
            p.addUserDebugText(
                "end-retreat-goal", (erg + np.array([0, 0, 0.03])).tolist(),
                [0.35, 0.0, 0.6], textSize=0.9,
            )
        if end_ee_pts_pb is not None:
            end_ee = np.asarray(end_ee_pts_pb, dtype=float).reshape(-1, 3)
            for i in range(len(end_ee) - 1):
                p.addUserDebugLine(
                    end_ee[i].tolist(), end_ee[i + 1].tolist(),
                    [0.7, 0.5, 0.95], lineWidth=2.2,
                )
            p.addUserDebugPoints(
                end_ee.tolist(),
                np.full((end_ee.shape[0], 3), [0.7, 0.5, 0.95]).tolist(),
                pointSize=5,
            )
        if end_ee_smoothed_pts_pb is not None:
            end_ee_smooth = np.asarray(end_ee_smoothed_pts_pb, dtype=float).reshape(-1, 3)
            for i in range(len(end_ee_smooth) - 1):
                p.addUserDebugLine(
                    end_ee_smooth[i].tolist(), end_ee_smooth[i + 1].tolist(),
                    [0.4, 0.3, 0.85], lineWidth=2.4,
                )
        end_app_smooth_vis = None
        if end_approach_smoothed_pts_pb is not None:
            end_app_smooth_vis = np.asarray(end_approach_smoothed_pts_pb, dtype=float).reshape(-1, 3)
            for i in range(len(end_app_smooth_vis) - 1):
                p.addUserDebugLine(
                    end_app_smooth_vis[i].tolist(), end_app_smooth_vis[i + 1].tolist(),
                    [0.9, 0.35, 0.2], lineWidth=2.0,
                )
            if end_approach_control_points_pb is not None:
                ecps = np.asarray(end_approach_control_points_pb, dtype=float).reshape(-1, 3)
                p.addUserDebugPoints(
                    ecps.tolist(),
                    np.full((ecps.shape[0], 3), [0.9, 0.35, 0.2]).tolist(),
                    pointSize=7,
                )
        if end_approach_line_pts_pb is not None:
            end_app_line = np.asarray(end_approach_line_pts_pb, dtype=float).reshape(-1, 3)
            end_same_vis = (
                end_app_smooth_vis is not None
                and end_app_smooth_vis.shape == end_app_line.shape
                and np.allclose(end_app_smooth_vis, end_app_line)
            )
            if not end_same_vis:
                elc = [0.95, 0.6, 0.15] if bool(end_approach_line_feasible) else [0.95, 0.35, 0.1]
                for i in range(len(end_app_line) - 1):
                    p.addUserDebugLine(
                        end_app_line[i].tolist(), end_app_line[i + 1].tolist(),
                        elc, lineWidth=1.8,
                    )
                p.addUserDebugPoints(
                    end_app_line.tolist(),
                    np.full((end_app_line.shape[0], 3), elc).tolist(),
                    pointSize=4,
                )
        self.wait()

    def run_step(self, step_name: str) -> None:
        experiments = {
            "align": AlignmentExperiment(self, self.align_settings),
            "init-config": InitConfigExperiment(self, self.init_config_settings),
            "nearest-region": NearestRegionExperiment(self, self.nearest_region_settings),
            "plan": PlannerExperiment(self, self.plan_settings),
        }
        if step_name not in experiments:
            raise ValueError(f"unknown step: {step_name!r}, valid: {list(experiments)}")
        experiments[step_name].run()

    def run_steps(self, steps: list[str]) -> None:
        print("=" * 60)
        print("  SDF Integration Experiments")
        print(f"  Steps: {steps}")
        print("=" * 60)
        for step_name in steps:
            print(f"\n{'='*60}")
            print(f"  >>> Running: {step_name}")
            print(f"{'='*60}\n")
            self.run_step(step_name)


class AlignmentExperiment:
    def __init__(self, runner: ExperimentRunner, settings: SimpleNamespace) -> None:
        self.runner = runner
        self.settings = settings

    def run(self) -> None:
        args = self.settings
        field = self.runner.load_field(args.sdf_npz)
        kind = self.runner.resolve_kind(field, args.kind)
        rng = np.random.default_rng(args.seed)

        p.connect(p.DIRECT)
        try:
            cfg = ExperimentConfig()
            if args.urdf_path:
                cfg.urdf_path = args.urdf_path
            if args.workpiece_urdf_path:
                cfg.workpiece_urdf_path = args.workpiece_urdf_path
            if args.workpiece_position is not None:
                cfg.workpiece_position = tuple(self.runner.parse_vec3(args.workpiece_position).tolist())
            if args.workpiece_orientation_deg is not None:
                cfg.workpiece_orientation_deg = tuple(self.runner.parse_vec3(args.workpiece_orientation_deg).tolist())
            robot, workpiece = self.runner.make_robot_and_workpiece(cfg)
            pts = self.runner.collect_surface_points(robot, workpiece.body_id, per_link_points=args.per_link_points)
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
            return self.runner.query_field(field, pts_sdf, kind=kind)

        def objective(x: np.ndarray) -> float:
            d = np.abs(eval_params(x))
            return float(np.median(d) + 0.25 * np.mean(d))

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

        out_json = self.runner.ensure_parent(Path(args.output_json))
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
        with open(out_json, "w", encoding="utf-8") as _f:
            _f.write(json.dumps(report, ensure_ascii=False, indent=2))

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
        out_png = self.runner.ensure_parent(Path(args.output_png))
        with open(out_png, "wb") as _fp:
            fig.savefig(_fp, dpi=140, format="png")
        plt.close(fig)

        print(f"[align] 完成，结果写入: {out_json}")
        print(f"[align] 可视化写入: {out_png}")
        self.runner.show_alignment_points(pts, d_final)


class NearestRegionExperiment:
    def __init__(self, runner: ExperimentRunner, settings: SimpleNamespace) -> None:
        self.runner = runner
        self.settings = settings

    @staticmethod
    def _sample_segment(start: np.ndarray, goal: np.ndarray, num_samples: int) -> np.ndarray:
        p0 = np.asarray(start, dtype=float).reshape(3)
        p1 = np.asarray(goal, dtype=float).reshape(3)
        n = max(2, int(num_samples))
        if np.linalg.norm(p1 - p0) < 1e-9:
            return np.repeat(p0.reshape(1, 3), n, axis=0)
        t = np.linspace(0.0, 1.0, n, dtype=float)
        return (1.0 - t[:, None]) * p0.reshape(1, 3) + t[:, None] * p1.reshape(1, 3)

    @staticmethod
    def _point_to_segment_distance(points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        a = np.asarray(seg_a, dtype=float).reshape(1, 3)
        b = np.asarray(seg_b, dtype=float).reshape(1, 3)
        ab = b - a
        denom = float(np.dot(ab.ravel(), ab.ravel()))
        if denom < 1e-12:
            return np.linalg.norm(pts - a, axis=1)
        t = np.sum((pts - a) * ab, axis=1) / denom
        t = np.clip(t, 0.0, 1.0)
        proj = a + t[:, None] * ab
        return np.linalg.norm(pts - proj, axis=1)

    @staticmethod
    def _world_points_from_indices(field, ijk: np.ndarray) -> np.ndarray:
        idx = np.asarray(ijk, dtype=float).reshape(-1, 3)
        origin = np.asarray(field.origin, dtype=float).reshape(1, 3)
        sp = float(field.spacing)
        return origin + (idx + 0.5) * sp

    @staticmethod
    def _nearest_voxel_to_point(points: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int]:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        tgt = np.asarray(target, dtype=float).reshape(1, 3)
        d = np.linalg.norm(pts - tgt, axis=1)
        idx = int(np.argmin(d))
        return pts[idx], idx

    def _resolve_weld_endpoint_pb(self, workpiece: WorkpieceModel, point_text: str | None, link_name: str | None) -> np.ndarray:
        if point_text:
            return self.runner.parse_vec3(point_text).reshape(3)
        if not link_name:
            raise ValueError("weld endpoint requires either explicit point or link name")
        pt, _ = workpiece.get_frame_pose(link_name)
        return np.asarray(pt, dtype=float).reshape(3)

    def _ray_occupied_length(self, occ_field, start: np.ndarray, end: np.ndarray, step: float) -> float:
        p0 = np.asarray(start, dtype=float).reshape(3)
        p1 = np.asarray(end, dtype=float).reshape(3)
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-12:
            return 0.0
        n_seg = max(1, int(math.ceil(seg_len / max(float(step), 1e-4))))
        seg_step = seg_len / n_seg
        occ_len = 0.0
        for i in range(n_seg):
            t_mid = (i + 0.5) / n_seg
            pmid = p0 + t_mid * seg
            if occ_field.is_occupied(pmid):
                occ_len += seg_step
        return float(occ_len)

    def _component_summary(
        self,
        field,
        comp_id: int,
        comp_ijk: np.ndarray,
        comp_xyz: np.ndarray,
        comp_vals: np.ndarray,
    ) -> dict:
        centroid = np.mean(comp_xyz, axis=0)
        rep_sdf, rep_local_idx = self._nearest_voxel_to_point(comp_xyz, centroid)
        rep_ijk = np.asarray(comp_ijk[int(rep_local_idx)], dtype=int)
        bbox_min = np.min(comp_xyz, axis=0)
        bbox_max = np.max(comp_xyz, axis=0)
        return {
            "component_id": int(comp_id),
            "voxel_count": int(comp_xyz.shape[0]),
            "representative_sdf": rep_sdf.tolist(),
            "representative_pb": self.runner.sdf2pb(rep_sdf.reshape(1, 3)).reshape(3).tolist(),
            "representative_ijk": rep_ijk.tolist(),
            "centroid_sdf": centroid.tolist(),
            "bbox_min_sdf": bbox_min.tolist(),
            "bbox_max_sdf": bbox_max.tolist(),
            "bbox_min_pb": self.runner.sdf2pb(bbox_min.reshape(1, 3)).reshape(3).tolist(),
            "bbox_max_pb": self.runner.sdf2pb(bbox_max.reshape(1, 3)).reshape(3).tolist(),
            "clearance_min": float(np.min(comp_vals)),
            "clearance_mean": float(np.mean(comp_vals)),
            "clearance_max": float(np.max(comp_vals)),
            "votes": 0,
            "total_occlusion_length": 0.0,
            "mean_rep_distance_to_seam": 0.0,
            "same_side_positive_ratio": 0.0,
            "same_side_mean_dot": 0.0,
            "same_side_pass": True,
        }

    @staticmethod
    def _normalize_rows(v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=float).reshape(-1, 3)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)

    def _component_same_side_stats(
        self,
        rep_sdf: np.ndarray,
        seam_points: np.ndarray,
        seam_gradients: np.ndarray,
        *,
        dot_eps: float,
        min_ratio: float,
    ) -> dict:
        rep = np.asarray(rep_sdf, dtype=float).reshape(1, 3)
        seam_pts = np.asarray(seam_points, dtype=float).reshape(-1, 3)
        grads = self._normalize_rows(seam_gradients)
        dirs = self._normalize_rows(rep - seam_pts)
        dots = np.sum(dirs * grads, axis=1)
        positive = dots > float(dot_eps)
        positive_ratio = float(np.mean(positive)) if dots.size else 0.0
        return {
            "dots": dots.tolist(),
            "positive_mask": positive.astype(int).tolist(),
            "positive_ratio": positive_ratio,
            "mean_dot": float(np.mean(dots)) if dots.size else 0.0,
            "passed": bool(positive_ratio >= float(min_ratio)),
        }

    def _rank_topk_points(
        self,
        comp_xyz: np.ndarray,
        comp_vals: np.ndarray,
        seam_points: np.ndarray,
        seam_start: np.ndarray,
        seam_goal: np.ndarray,
        occ_field,
        args: SimpleNamespace,
    ) -> list[dict]:
        if comp_xyz.size == 0:
            return []
        seam_dist = self._point_to_segment_distance(comp_xyz, seam_start, seam_goal)
        stride = max(1, int(getattr(args, "topk_ray_sample_stride", 2)))
        seam_ray_pts = seam_points[::stride]
        if seam_ray_pts.shape[0] == 0 or not np.allclose(seam_ray_pts[-1], seam_points[-1]):
            seam_ray_pts = np.vstack([seam_ray_pts, seam_points[-1]])
        candidate_pool = min(int(getattr(args, "candidate_pool", 256)), comp_xyz.shape[0])
        prelim_score = seam_dist - 0.20 * np.asarray(comp_vals, dtype=float)
        prelim_idx = np.argsort(prelim_score)[:candidate_pool]
        ranked: list[dict] = []
        for idx in prelim_idx:
            pt = np.asarray(comp_xyz[idx], dtype=float)
            clearance = float(comp_vals[idx])
            ray_costs = [
                self._ray_occupied_length(occ_field, wp, pt, float(args.ray_step))
                for wp in seam_ray_pts
            ]
            avg_ray = float(np.mean(ray_costs)) if ray_costs else 0.0
            score = float(seam_dist[idx] + 3.0 * avg_ray - 0.30 * clearance)
            ranked.append(
                {
                    "point_sdf": pt.tolist(),
                    "point": self.runner.sdf2pb(pt.reshape(1, 3)).reshape(3).tolist(),
                    "score": score,
                    "distance_to_seam": float(seam_dist[idx]),
                    "distance_value": clearance,
                    "avg_occlusion_length": avg_ray,
                    "min_kernel_value": None,
                }
            )
        ranked.sort(key=lambda item: (item["score"], item["avg_occlusion_length"], item["distance_to_seam"], -item["distance_value"]))
        return ranked[: max(1, int(args.top_k))]

    def run(self) -> None:
        from scipy.ndimage import label as ndimage_label

        args = self.settings
        udf_mod = self.runner.load_udf_module()
        field = self.runner.load_field(args.sdf_npz)
        kind = self.runner.resolve_kind(field, args.kind)

        p.connect(p.DIRECT)
        try:
            cfg = ExperimentConfig()
            if args.workpiece_urdf_path:
                cfg.workpiece_urdf_path = args.workpiece_urdf_path
            workpiece = WorkpieceModel(cfg)
            weld_start_pb = self._resolve_weld_endpoint_pb(workpiece, args.weld_point, args.weld_link_name)
            weld_goal_pb = self._resolve_weld_endpoint_pb(
                workpiece,
                getattr(args, "weld_goal_point", None),
                getattr(args, "weld_goal_link_name", None),
            )
        finally:
            p.disconnect()

        seam_points_pb = self._sample_segment(
            weld_start_pb,
            weld_goal_pb,
            int(getattr(args, "seam_samples", ExperimentParameters.NEAR_SEAM_SAMPLES)),
        )
        seam_points = self.runner.pb2sdf(seam_points_pb)
        d_seam, seam_gradients = field.query_with_gradient(
            np.asarray(seam_points, dtype=np.float32),
            kind=kind,
        )
        d_seam = np.asarray(d_seam, dtype=float).reshape(-1)
        seam_gradients = np.asarray(seam_gradients, dtype=float).reshape(-1, 3)
        surface_normals = self._normalize_rows(seam_gradients)
        surface_normals_pb = self.runner.sdf_dirs_to_pb(surface_normals)
        surface_normals_pb = self._normalize_rows(surface_normals_pb)
        print(
            f"[nearest] seam start/end PB={weld_start_pb.tolist()} -> {weld_goal_pb.tolist()}  "
            f"samples={seam_points.shape[0]}"
        )
        print(
            f"[nearest] seam SDF range = "
            f"[{float(np.min(d_seam)):.6f}, {float(np.max(d_seam)):.6f}] m"
        )

        require_above = bool(getattr(args, "require_above_weld", ExperimentParameters.NEAR_REQUIRE_ABOVE_WELD))

        bbox_radius = 0.0
        kernel_npz_path = getattr(args, "kernel_npz", None)
        if kernel_npz_path and Path(kernel_npz_path).exists():
            with np.load(kernel_npz_path) as data:
                if "bbox_radius" in data:
                    bbox_radius = float(data["bbox_radius"])
        bbox_margin = float(getattr(args, "bbox_margin", ExperimentParameters.NEAR_BBOX_MARGIN))
        effective_clearance = float(args.min_clearance)
        if bbox_radius > 0:
            effective_clearance = max(effective_clearance, bbox_radius + bbox_margin)
        print(
            f"[nearest] bbox_radius={bbox_radius:.4f}m  margin={bbox_margin:.2f}m  "
            f"-> effective_clearance={effective_clearance:.4f}m"
        )

        occ_field, occ_path = self.runner.load_or_bake_occupancy_field(
            field,
            sdf_npz=args.sdf_npz,
            workpiece_urdf_path=args.workpiece_urdf_path,
            occupancy_npz=getattr(args, "occupancy_npz", None),
            occupancy_margin=float(getattr(args, "occupancy_margin", 0.0)),
        )
        print(f"[nearest] occupancy cache = {occ_path}")

        grid = np.asarray(udf_mod._grid_for_kind(field, kind), dtype=float)
        nx, ny, nz = grid.shape
        origin = np.asarray(field.origin, dtype=float)
        spacing = float(field.spacing)
        xs = origin[0] + (np.arange(nx, dtype=float) + 0.5) * spacing
        ys = origin[1] + (np.arange(ny, dtype=float) + 0.5) * spacing
        zs = origin[2] + (np.arange(nz, dtype=float) + 0.5) * spacing
        print(f"[nearest] grid shape={grid.shape}  spacing={spacing:.5f}m")

        mask = np.isfinite(grid) & (grid > effective_clearance)
        print(f"[nearest] voxels with SDF>{effective_clearance:.3f}: {int(np.sum(mask))} / {grid.size}")

        n_feasible = int(np.sum(mask))
        print(f"[nearest] feasible voxels: {n_feasible}")

        topk = []
        n_components = 0
        total_feasible = 0
        selected_component_id: int | None = None
        component_summaries: list[dict] = []
        ray_segments: list[dict] = []
        vote_candidates: list[dict] = []
        if n_feasible > 0:
            labels, n_components = ndimage_label(mask)
            print(f"[nearest] connected components: {n_components}")

            feasible_ijk = np.argwhere(mask)
            feasible_xyz = self._world_points_from_indices(field, feasible_ijk)
            feasible_labels = labels[mask]
            seam_start = seam_points[0]
            seam_goal = seam_points[-1]
            comp_arrays: dict[int, dict[str, np.ndarray]] = {}
            for comp_id in range(1, n_components + 1):
                comp_sel = feasible_labels == comp_id
                comp_xyz = feasible_xyz[comp_sel]
                comp_ijk = feasible_ijk[comp_sel]
                comp_vals = grid[tuple(comp_ijk.T)]
                comp_arrays[int(comp_id)] = {
                    "xyz": comp_xyz,
                    "ijk": comp_ijk,
                    "vals": np.asarray(comp_vals, dtype=float),
                }
                summary = self._component_summary(field, comp_id, comp_ijk, comp_xyz, comp_vals)
                summary["mean_rep_distance_to_seam"] = float(
                    self._point_to_segment_distance(
                        np.asarray(summary["representative_sdf"], dtype=float).reshape(1, 3),
                        seam_start,
                        seam_goal,
                    )[0]
                )
                side_stats = self._component_same_side_stats(
                    np.asarray(summary["representative_sdf"], dtype=float),
                    seam_points,
                    seam_gradients,
                    dot_eps=float(getattr(args, "grad_side_dot_eps", ExperimentParameters.NEAR_GRAD_SIDE_DOT_EPS)),
                    min_ratio=float(getattr(args, "grad_side_min_ratio", ExperimentParameters.NEAR_GRAD_SIDE_MIN_RATIO)),
                )
                summary["same_side_positive_ratio"] = float(side_stats["positive_ratio"])
                summary["same_side_mean_dot"] = float(side_stats["mean_dot"])
                summary["same_side_positive_mask"] = side_stats["positive_mask"]
                summary["same_side_dots"] = side_stats["dots"]
                summary["same_side_pass"] = bool(side_stats["passed"])
                component_summaries.append(summary)

            passed_components = [comp for comp in component_summaries if bool(comp["same_side_pass"])]
            vote_candidates = passed_components if passed_components else component_summaries
            if passed_components:
                print(f"[nearest] gradient same-side filter kept {len(passed_components)} / {len(component_summaries)} components")
            else:
                print("[nearest] gradient same-side filter rejected all components, fallback to all")

            vote_stats: dict[int, dict[str, float]] = {
                int(comp["component_id"]): {
                    "votes": 0,
                    "total_occlusion_length": 0.0,
                    "total_distance": 0.0,
                    "per_sample_occlusion_lengths": [],
                    "per_sample_total_costs": [],
                }
                for comp in component_summaries
            }
            for sample_idx, seam_pt in enumerate(seam_points):
                best_id = None
                best_key = None
                best_occ = None
                best_rep = None
                for comp in vote_candidates:
                    comp_id = int(comp["component_id"])
                    rep = np.asarray(comp["representative_sdf"], dtype=float)
                    occ_len = self._ray_occupied_length(occ_field, seam_pt, rep, float(getattr(args, "ray_step", 0.02)))
                    dist = float(np.linalg.norm(rep - seam_pt))
                    total_cost = occ_len
                    vote_stats[comp_id]["per_sample_occlusion_lengths"].append(float(occ_len))
                    vote_stats[comp_id]["per_sample_total_costs"].append(float(total_cost))
                    key = (total_cost, dist)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_id = comp_id
                        best_occ = occ_len
                        best_rep = rep
                if best_id is None:
                    continue
                vote_stats[best_id]["votes"] += 1
                vote_stats[best_id]["total_occlusion_length"] += float(best_occ)
                vote_stats[best_id]["total_distance"] += float(np.linalg.norm(best_rep - seam_pt))
                ray_segments.append(
                    {
                        "sample_index": int(sample_idx),
                        "component_id": int(best_id),
                        "start": self.runner.sdf2pb(seam_pt.reshape(1, 3)).reshape(3).tolist(),
                        "end": self.runner.sdf2pb(np.asarray(best_rep, dtype=float).reshape(1, 3)).reshape(3).tolist(),
                        "occupied_length": float(best_occ),
                    }
                )

            component_summaries.sort(
                key=lambda comp: (
                    0 if bool(comp.get("same_side_pass", True)) else 1,
                    -vote_stats[int(comp["component_id"])]["votes"],
                    vote_stats[int(comp["component_id"])]["total_occlusion_length"],
                    vote_stats[int(comp["component_id"])]["total_distance"],
                )
            )
            for comp in component_summaries:
                cid = int(comp["component_id"])
                comp["votes"] = int(vote_stats[cid]["votes"])
                comp["total_occlusion_length"] = float(vote_stats[cid]["total_occlusion_length"])
                n_vote = max(1, int(vote_stats[cid]["votes"]))
                comp["mean_vote_distance"] = float(vote_stats[cid]["total_distance"] / n_vote)
                comp["mean_total_cost"] = float(np.mean(vote_stats[cid]["per_sample_total_costs"])) if vote_stats[cid]["per_sample_total_costs"] else 0.0

            if component_summaries:
                selected_component_id = int(component_summaries[0]["component_id"])
                sel = comp_arrays[selected_component_id]
                total_feasible = int(sel["xyz"].shape[0])
                topk = self._rank_topk_points(
                    sel["xyz"],
                    sel["vals"],
                    seam_points,
                    seam_start,
                    seam_goal,
                    occ_field,
                    args,
                )
                print(
                    f"[nearest] selected component #{selected_component_id}: "
                    f"voxels={total_feasible}, votes={component_summaries[0]['votes']}, "
                    f"occ={component_summaries[0]['total_occlusion_length']:.4f}m"
                )

        out_json = self.runner.ensure_parent(Path(args.output_json))
        payload = {
            "kind": kind,
            "weld_start_point": weld_start_pb.tolist(),
            "weld_goal_point": weld_goal_pb.tolist(),
            "sampled_weld_points": seam_points_pb.tolist(),
            "sampled_weld_points_sdf": seam_points.tolist(),
            "sampled_sdf_values": np.asarray(d_seam, dtype=float).tolist(),
            "sampled_gradients_sdf": seam_gradients.tolist(),
            "surface_normals_sdf": surface_normals.tolist(),
            "surface_normals_pb": surface_normals_pb.tolist(),
            "total_feasible": total_feasible,
            "n_components": n_components,
            "components_summary": component_summaries,
            "gradient_side_filter": {
                "dot_eps": float(getattr(args, "grad_side_dot_eps", ExperimentParameters.NEAR_GRAD_SIDE_DOT_EPS)),
                "min_ratio": float(getattr(args, "grad_side_min_ratio", ExperimentParameters.NEAR_GRAD_SIDE_MIN_RATIO)),
                "passed_component_ids": [int(comp["component_id"]) for comp in vote_candidates],
            },
            "component_votes": [
                {
                    "component_id": int(comp["component_id"]),
                    "votes": int(comp["votes"]),
                    "total_occlusion_length": float(comp["total_occlusion_length"]),
                    "mean_rep_distance_to_seam": float(comp["mean_rep_distance_to_seam"]),
                    "mean_total_cost": float(comp.get("mean_total_cost", 0.0)),
                    "same_side_positive_ratio": float(comp.get("same_side_positive_ratio", 0.0)),
                    "same_side_mean_dot": float(comp.get("same_side_mean_dot", 0.0)),
                    "same_side_pass": bool(comp.get("same_side_pass", True)),
                }
                for comp in component_summaries
            ],
            "component_occlusion_costs": [
                {
                    "component_id": int(comp["component_id"]),
                    "per_sample_occlusion_lengths": [
                        float(x) for x in vote_stats[int(comp["component_id"])]["per_sample_occlusion_lengths"]
                    ],
                    "per_sample_total_costs": [
                        float(x) for x in vote_stats[int(comp["component_id"])]["per_sample_total_costs"]
                    ],
                }
                for comp in component_summaries
            ],
            "selected_component_id": selected_component_id,
            "top_k": topk,
            "params": {
                "bbox_radius": bbox_radius,
                "bbox_margin": bbox_margin,
                "effective_clearance": effective_clearance,
                "require_above_weld": require_above,
                "occupancy_npz": str(occ_path),
                "ray_step": float(getattr(args, "ray_step", 0.02)),
                "grad_side_dot_eps": float(getattr(args, "grad_side_dot_eps", ExperimentParameters.NEAR_GRAD_SIDE_DOT_EPS)),
                "grad_side_min_ratio": float(getattr(args, "grad_side_min_ratio", ExperimentParameters.NEAR_GRAD_SIDE_MIN_RATIO)),
            },
        }
        with open(out_json, "w", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, ensure_ascii=False, indent=2))

        fig = plt.figure(figsize=(11.5, 5.4))
        ax = fig.add_subplot(121, projection="3d")
        ax.scatter(seam_points_pb[:, 0], seam_points_pb[:, 1], seam_points_pb[:, 2], c="red", s=36, label="seam samples")
        for wp, nrm in zip(seam_points_pb, surface_normals_pb):
            ax.quiver(wp[0], wp[1], wp[2], nrm[0], nrm[1], nrm[2], length=0.08, color="royalblue")
        for comp in component_summaries:
            rep_pb = np.asarray(comp["representative_pb"], dtype=float)
            selected = selected_component_id is not None and int(comp["component_id"]) == int(selected_component_id)
            passed = bool(comp.get("same_side_pass", True))
            ax.scatter(
                [rep_pb[0]], [rep_pb[1]], [rep_pb[2]],
                c="orange" if selected else ("gray" if passed else "crimson"),
                s=60 if selected else 26,
                alpha=0.95 if selected else 0.6,
            )
        for seg in ray_segments[: max(1, int(getattr(args, "max_rays_to_vis", 12)))]:
            a = np.asarray(seg["start"], dtype=float)
            b = np.asarray(seg["end"], dtype=float)
            occ_len = float(seg["occupied_length"])
            ax.plot(
                [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                c="green" if occ_len <= 1e-6 else "crimson",
                lw=1.2,
                alpha=0.85,
            )
        if topk:
            pts_vis = np.asarray([x["point"] for x in topk], dtype=float)
            ax.scatter(pts_vis[:, 0], pts_vis[:, 1], pts_vis[:, 2], c="limegreen", s=28, label="top-k")
        ax.set_title("Seam-region decision (3D)")
        ax.legend(loc="upper left")

        ax_bar = fig.add_subplot(122)
        if component_summaries:
            labels_bar = [f"C{comp['component_id']}" for comp in component_summaries]
            votes_bar = [float(comp["votes"]) for comp in component_summaries]
            occ_bar = [float(comp["total_occlusion_length"]) for comp in component_summaries]
            ratio_bar = [float(comp.get("same_side_positive_ratio", 0.0)) for comp in component_summaries]
            y = np.arange(len(labels_bar))
            ax_bar.barh(y, votes_bar, color="steelblue", alpha=0.85)
            for yi, (v, occ, ratio) in enumerate(zip(votes_bar, occ_bar, ratio_bar)):
                ax_bar.text(v + 0.05, yi, f"occ={occ:.3f}m  side={ratio:.2f}", va="center", fontsize=8)
            ax_bar.set_yticks(y, labels_bar)
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel("votes")
            ax_bar.set_title("Component votes / occlusion")
        else:
            ax_bar.text(0.5, 0.5, "no feasible components", ha="center", va="center")
            ax_bar.set_axis_off()

        fig.tight_layout()
        out_png = self.runner.ensure_parent(Path(args.output_png))
        with open(out_png, "wb") as _fp:
            fig.savefig(_fp, dpi=140, format="png")
        plt.close(fig)
        print(f"[nearest] done -> {out_json}")
        if not topk:
            print("[nearest] WARNING: no feasible voxel found.")
        self.runner.show_nearest_region(
            seam_points_pb,
            surface_normals_pb,
            component_summaries,
            selected_component_id,
            topk,
            ray_segments[: max(1, int(getattr(args, "max_rays_to_vis", 12)))],
            total_feasible,
            n_components,
        )


class InitConfigExperiment:
    def __init__(self, runner: ExperimentRunner, settings: SimpleNamespace) -> None:
        self.runner = runner
        self.settings = settings

    @staticmethod
    def _has_self_collision(robot: JakaRobot, penetration_thresh: float = -0.001) -> bool:
        """检测机器人自碰撞（含龙门架与臂体之间的碰撞）。
        penetration_thresh: 穿透距离阈值，负值表示允许极小穿透噪声。"""
        p.performCollisionDetection()
        contacts = p.getContactPoints(bodyA=robot.body_id, bodyB=robot.body_id)
        for c in contacts:
            la = int(c[3])
            lb = int(c[4])
            dist = float(c[8])
            if la == lb:
                continue
            if abs(la - lb) <= 1:
                continue
            if dist < penetration_thresh:
                return True
        return False

    @staticmethod
    def _external_collision(robot: JakaRobot, workpiece: WorkpieceModel, check_links: list[int], min_clearance: float) -> bool:
        p.performCollisionDetection()
        for li in check_links:
            closest = robot.get_closest_points_to_obstacle(li, workpiece.body_id, max_dist=max(0.5, min_clearance * 8.0))
            if closest is None:
                continue
            if float(closest["signed_dist"]) < float(min_clearance):
                return True
        return False

    @staticmethod
    def _minimum_external_clearance(robot: JakaRobot, workpiece: WorkpieceModel, check_links: list[int], min_clearance: float) -> float:
        p.performCollisionDetection()
        best = float("inf")
        for li in check_links:
            closest = robot.get_closest_points_to_obstacle(li, workpiece.body_id, max_dist=max(0.5, min_clearance * 8.0))
            if closest is None:
                continue
            best = min(best, float(closest["signed_dist"]))
        return best

    def _build_occupancy_kernel(
        self,
        robot: JakaRobot,
        selected_links: list[int],
        voxel: float,
    ) -> tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
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
        pts_local = self.runner._world_to_local(pts_world, base_pos, base_quat)

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

    def run(self) -> None:
        args = self.settings
        rng = np.random.default_rng(args.seed)
        p.connect(p.DIRECT)
        try:
            cfg = ExperimentConfig()
            if args.urdf_path:
                cfg.urdf_path = args.urdf_path
            if args.workpiece_urdf_path:
                cfg.workpiece_urdf_path = args.workpiece_urdf_path
            robot, workpiece = self.runner.make_robot_and_workpiece(cfg)
            q0, dq0 = robot.get_joint_state()
            active_idx = {j: i for i, j in enumerate(robot.active_joints)}
            kernel_links = sorted(set(int(x) for x in robot.rear_six_link_indices))

            mutable_joints = list(robot.revolute_joints)
            mutable_indices = [active_idx[j] for j in mutable_joints if j in active_idx]

            joint_limits = []
            for idx in mutable_indices:
                info = p.getJointInfo(robot.body_id, int(robot.active_joints[idx]))
                joint_limits.append((float(info[8]), float(info[9])))

            def _sample_and_eval(q_center, sigma, n_samples):
                results = []
                for _ in range(n_samples):
                    q = np.array(q_center, dtype=float)
                    for i, idx in enumerate(mutable_indices):
                        lo, hi = joint_limits[i]
                        q[idx] = q[idx] + float(rng.normal(0.0, sigma))
                        if hi > lo:
                            q[idx] = float(np.clip(q[idx], lo, hi))
                    robot.set_joint_state(q, dq=np.zeros_like(q))
                    if self._has_self_collision(robot):
                        continue
                    env_clearance = float("inf")
                    if not bool(args.skip_external_collision):
                        env_clearance = self._minimum_external_clearance(
                            robot,
                            workpiece,
                            kernel_links,
                            min_clearance=float(args.min_clearance),
                        )
                        if env_clearance < float(args.min_clearance):
                            continue
                    kernel_offsets, occ_count, aabb_vol, aabb_lo, aabb_hi = self._build_occupancy_kernel(
                        robot=robot, selected_links=kernel_links, voxel=float(args.voxel),
                    )
                    aabb_size = aabb_hi - aabb_lo
                    aabb_max_dim = float(np.max(aabb_size))
                    half_ext = np.maximum(np.abs(aabb_lo), np.abs(aabb_hi))
                    bbox_radius = float(np.linalg.norm(half_ext))
                    self_clearance = compute_self_collision_clearance(
                        robot,
                        link_indices=robot.revolute_joints,
                        min_index_gap=2,
                        query_distance=float(getattr(args, "self_collision_query_distance", 0.12)),
                    )
                    clearance_entries = [{"kind": "self_collision", "distance": float(self_clearance)}]
                    if np.isfinite(env_clearance):
                        clearance_entries.append({"kind": "environment", "distance": float(env_clearance)})
                    quality_metrics = evaluate_configuration_quality(
                        robot,
                        q,
                        dq=np.zeros_like(q),
                        motion_component=str(getattr(args, "motion_component", "linear")),
                        clearance_summary=summarize_clearance_entries(clearance_entries),
                        joint_limits=joint_limits,
                    )
                    results.append({
                        "q": q.tolist(),
                        "occupancy_count": int(occ_count),
                        "aabb_volume": float(aabb_vol),
                        "aabb_max_dim": aabb_max_dim,
                        "aabb_min": aabb_lo.tolist(),
                        "aabb_max": aabb_hi.tolist(),
                        "bbox_radius": bbox_radius,
                        "kernel_offsets": kernel_offsets,
                        "configuration_quality": quality_metrics,
                        "inverse_condition": float(quality_metrics["inverse_condition"]),
                        "self_collision_distance": float(quality_metrics["self_collision_distance"]),
                        "joint_limit_margin": float(quality_metrics["joint_limit_margin"]),
                        "environment_distance": float(quality_metrics["environment_distance"]),
                    })
                return results

            # Phase 1: wide search
            print(f"[init-config] Phase 1: {args.num_samples} samples, sigma={args.sample_std:.2f}")
            phase1 = _sample_and_eval(q0, float(args.sample_std), int(args.num_samples))
            feasible_count = len(phase1)
            occ_counts = [r["aabb_max_dim"] for r in phase1]
            print(f"[init-config] Phase 1 feasible: {feasible_count} / {args.num_samples}")

            # Phase 2: local refinement around top-K candidates
            refine_top_k = int(getattr(args, "refine_top_k", 10))
            refine_samples = int(getattr(args, "refine_samples", 300))
            refine_std = float(getattr(args, "refine_std", 0.15))
            if phase1 and refine_top_k > 0 and refine_samples > 0:
                phase1_sorted = _rank_init_config_candidates(
                    phase1,
                    selection_weights=getattr(args, "selection_weights", None),
                )
                top_k = phase1_sorted[:refine_top_k]
                per_seed = max(refine_samples // refine_top_k, 1)
                print(f"[init-config] Phase 2: refine top-{refine_top_k}, {per_seed} samples/seed, sigma={refine_std:.2f}")
                for seed_rec in top_k:
                    q_seed = np.array(seed_rec["q"], dtype=float)
                    local = _sample_and_eval(q_seed, refine_std, per_seed)
                    phase1.extend(local)
                    occ_counts.extend([r["aabb_max_dim"] for r in local])
                    feasible_count += len(local)
                print(f"[init-config] Phase 2 added {feasible_count - len(phase1_sorted)} feasible samples")

            if not phase1:
                raise RuntimeError("未找到满足条件的初始构型，请放宽采样范围或碰撞阈值。")

            ranked_candidates = _rank_init_config_candidates(
                phase1,
                selection_weights=getattr(args, "selection_weights", None),
            )
            best = ranked_candidates[0]

            out_npz = self.runner.ensure_parent(Path(args.output_npz))
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
            print(
                f"[init-config] best AABB max_dim: {best['aabb_max_dim']:.4f} m  "
                f"(volume: {best['aabb_volume']:.6f} m^3, voxels: {best['occupancy_count']})"
            )
            print(f"[init-config] AABB: {best['aabb_min']} .. {best['aabb_max']}")
            print(f"[init-config] bbox_radius (rear-6 circumscribed): {best['bbox_radius']:.4f} m")
            print(
                "[init-config] quality: "
                f"inverse_condition={best['configuration_quality']['inverse_condition']:.4f}, "
                f"joint_margin={best['configuration_quality']['joint_limit_margin']:.4f}, "
                f"self_clearance={best['configuration_quality']['self_collision_distance']:.4f} m, "
                f"score={best['selection_score']:.4f}"
            )

            q_best = best["q"]
            joint_entries = []
            for j in robot.revolute_joints:
                if j in active_idx:
                    idx = active_idx[j]
                    name = robot.link_name_by_index.get(int(j), f"joint_{int(j)}")
                    rad = float(q_best[idx])
                    joint_entries.append(
                        {
                            "joint_index": int(j),
                            "name": name,
                            "angle_rad": rad,
                            "angle_deg": float(np.degrees(rad)),
                        }
                    )
            out_json = self.runner.ensure_parent(Path(args.output_json))
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
                "selection_score": float(best["selection_score"]),
                "selection_score_components": best["selection_score_components"],
                "configuration_quality": best["configuration_quality"],
            }
            with open(out_json, "w", encoding="utf-8") as _f:
                _f.write(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"[init-config] JSON 报告写入: {out_json}")

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
            out_png = self.runner.ensure_parent(Path(args.output_png))
            with open(out_png, "wb") as _fp:
                fig.savefig(_fp, dpi=140, format="png")
            plt.close(fig)
            print(f"[init-config] 可视化写入: {out_png}")

            robot.set_joint_state(q0, dq0)
        finally:
            p.disconnect()

        self.runner.show_init_config(best)


class PlannerExperiment:
    def __init__(self, runner: ExperimentRunner, settings: SimpleNamespace) -> None:
        self.runner = runner
        self.settings = settings

    @staticmethod
    def _edge_valid(field, a: np.ndarray, b: np.ndarray, kind: str, clearance: float, step: float) -> bool:
        dist = float(np.linalg.norm(b - a))
        n = max(int(math.ceil(dist / max(step, 1e-4))), 2)
        ts = np.linspace(0.0, 1.0, n)
        pts = a.reshape(1, 3) * (1.0 - ts[:, None]) + b.reshape(1, 3) * ts[:, None]
        vals, _ = field.query_with_gradient(np.asarray(pts, dtype=np.float32), kind=kind)
        d = np.asarray(vals, dtype=float).reshape(-1)
        return bool(np.min(d) > clearance)

    @staticmethod
    def _astar_sdf_plan(field, start: np.ndarray, goal: np.ndarray, kind: str, clearance: float) -> list[np.ndarray]:
        from scipy import ndimage
        try:
            import dijkstra3d  # type: ignore[reportMissingImports]
        except ImportError as e:
            raise RuntimeError("A* 需要安装 dijkstra3d：`python -X utf8 -m pip install dijkstra3d`") from e

        udf_mod = sys.modules.get("udf_module_runtime") or ExperimentRunner._load_udf_module()
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

        def nearest_voxel_in_mask(point_xyz: np.ndarray, mask: np.ndarray) -> tuple[int, int, int] | None:
            vox = np.argwhere(mask)
            if vox.size == 0:
                return None
            vox_xyz = origin.reshape(1, 3) + (vox.astype(float) + 0.5) * spacing
            best_idx = int(np.argmin(np.linalg.norm(vox_xyz - point_xyz.reshape(1, 3), axis=1)))
            out = vox[best_idx]
            return int(out[0]), int(out[1]), int(out[2])

        goal_clip = np.clip(goal, bmin, bmax)
        goal_ijk_guess = xyz_to_ijk(goal_clip)
        goal_ijk = goal_ijk_guess if feasible[goal_ijk_guess] else nearest_voxel_in_mask(goal_clip, feasible)
        if goal_ijk is None:
            print("[astar] ERROR: 终点附近无可行体素")
            return []

        goal_label = int(labels[goal_ijk])
        if goal_label <= 0:
            print("[astar] ERROR: 终点不在任何可行连通域中")
            return []

        component_mask = labels == goal_label
        start_clip = np.clip(start, bmin, bmax)
        start_ijk = nearest_voxel_in_mask(start_clip, component_mask)
        if start_ijk is None:
            print("[astar] ERROR: 起点无法连接到终点所在连通域")
            return []

        print(
            f"[astar] grid={shape.tolist()}, spacing={spacing:.4f}, "
            f"components={n_components}, goal_label={goal_label}, "
            f"component_voxels={int(np.sum(component_mask))}"
        )
        print(
            f"[astar] start_ijk={start_ijk}, goal_ijk={goal_ijk}, "
            f"start_voxel_sdf={float(grid[start_ijk]):.4f}, goal_voxel_sdf={float(grid[goal_ijk]):.4f}"
        )

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
        self,
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
        nodes: list[SimpleNamespace] = [SimpleNamespace(pos=np.asarray(start, dtype=float), parent=-1, cost=0.0)]
        goal_idx = -1
        rng = np.random.default_rng()

        for it in range(int(max_iter)):
            if it % 1000 == 0 and it > 0:
                best_to_goal = min(float(np.linalg.norm(n.pos - goal)) for n in nodes)
                print(f"  [rrt*] iter {it}/{max_iter}, nodes={len(nodes)}, best_dist_to_goal={best_to_goal:.3f}m")
            sample = np.asarray(goal, dtype=float) if rng.random() < goal_sample_prob else rng.uniform(bounds_min, bounds_max)

            dists = np.asarray([np.linalg.norm(n.pos - sample) for n in nodes], dtype=float)
            nearest_idx = int(np.argmin(dists))
            nearest = nodes[nearest_idx].pos
            direction = sample - nearest
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                continue
            new_pos = nearest + direction / norm * min(step_size, norm)
            new_pos = np.clip(new_pos, bounds_min, bounds_max)

            if not self._edge_valid(field, nearest, new_pos, kind, clearance, edge_step):
                continue

            near_ids = [i for i, n in enumerate(nodes) if np.linalg.norm(n.pos - new_pos) <= near_radius]
            best_parent = nearest_idx
            best_cost = nodes[nearest_idx].cost + float(np.linalg.norm(new_pos - nearest))
            for nid in near_ids:
                c = nodes[nid].cost + float(np.linalg.norm(new_pos - nodes[nid].pos))
                if c < best_cost and self._edge_valid(field, nodes[nid].pos, new_pos, kind, clearance, edge_step):
                    best_cost = c
                    best_parent = nid

            new_idx = len(nodes)
            nodes.append(SimpleNamespace(pos=new_pos, parent=best_parent, cost=best_cost))

            for nid in near_ids:
                c = best_cost + float(np.linalg.norm(nodes[nid].pos - new_pos))
                if c < nodes[nid].cost and self._edge_valid(field, nodes[nid].pos, new_pos, kind, clearance, edge_step):
                    nodes[nid].parent = new_idx
                    nodes[nid].cost = c

            if np.linalg.norm(new_pos - goal) <= goal_tol and self._edge_valid(field, new_pos, goal, kind, clearance, edge_step):
                goal_idx = len(nodes)
                nodes.append(
                    SimpleNamespace(
                        pos=np.asarray(goal, dtype=float),
                        parent=new_idx,
                        cost=best_cost + float(np.linalg.norm(goal - new_pos)),
                    )
                )
                print(f"  [rrt*] goal reached at iter {it}, nodes={len(nodes)}")
                break

        if goal_idx < 0:
            best_to_goal = min(float(np.linalg.norm(n.pos - goal)) for n in nodes)
            print(f"  [rrt*] FAILED after {max_iter} iters, nodes={len(nodes)}, best_dist_to_goal={best_to_goal:.3f}m")
            return []

        path = []
        cur = goal_idx
        while cur >= 0:
            path.append(nodes[cur].pos.copy())
            cur = nodes[cur].parent
        path.reverse()
        return path

    @staticmethod
    def _resample_path(path: list[np.ndarray], max_spacing: float) -> list[np.ndarray]:
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

    def _shortcut_smooth(self, path: list[np.ndarray], field, kind: str, clearance: float, edge_step: float, iters: int) -> list[np.ndarray]:
        if len(path) < 3:
            return path
        arr = [np.asarray(p, dtype=float).copy() for p in path]
        rng = random.Random(0)
        for _ in range(max(int(iters), 1)):
            if len(arr) < 3:
                break
            i = rng.randint(0, len(arr) - 3)
            j = rng.randint(i + 2, len(arr) - 1)
            if self._edge_valid(field, arr[i], arr[j], kind, clearance, edge_step):
                arr = arr[: i + 1] + arr[j:]
        return arr

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=float).reshape(3)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-12:
            return np.zeros(3, dtype=float)
        return arr / norm

    def _gradient_backtrack_escape(
        self,
        field,
        start_sdf: np.ndarray,
        kind: str,
        *,
        target_sdf: float,
        init_step: float,
        min_step: float,
        shrink: float,
        max_iters: int,
        curv_eps: float,
        armijo_c1: float,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        def _directional_second_derivative(point: np.ndarray, direction: np.ndarray, eps: float) -> float:
            eps_use = max(float(eps), 1e-4)
            p_plus = point + eps_use * direction
            p_minus = point - eps_use * direction
            _, grads_pm = field.query_with_gradient(
                np.vstack([p_plus, p_minus]).astype(np.float32),
                kind=kind,
            )
            grads_pm = np.asarray(grads_pm, dtype=float).reshape(-1, 3)
            gp = float(np.dot(grads_pm[0], direction))
            gm = float(np.dot(grads_pm[1], direction))
            return (gp - gm) / (2.0 * eps_use)

        def _predict_step(delta_target: float, g_dir: float, h_dir: float, fallback_step: float) -> float:
            if delta_target <= 0.0:
                return max(float(min_step), 0.0)
            g_eff = max(float(g_dir), 1e-8)
            h_eff = float(h_dir)
            if abs(h_eff) < 1e-8:
                return max(float(min_step), delta_target / g_eff)
            disc = g_eff * g_eff + 2.0 * h_eff * delta_target
            if disc > 0.0:
                sqrt_disc = math.sqrt(disc)
                roots = [
                    (-g_eff + sqrt_disc) / h_eff,
                    (-g_eff - sqrt_disc) / h_eff,
                ]
                positive_roots = [r for r in roots if r > float(min_step)]
                if positive_roots:
                    return min(positive_roots)
            first_order = delta_target / g_eff
            return max(float(min_step), min(float(fallback_step), first_order))

        cur = np.asarray(start_sdf, dtype=float).reshape(3)
        vals, grads = field.query_with_gradient(cur.reshape(1, 3).astype(np.float32), kind=kind)
        cur_val = float(np.asarray(vals, dtype=float).reshape(-1)[0])
        cur_grad = np.asarray(grads, dtype=float).reshape(-1, 3)[0]
        path = [cur.copy()]
        val_hist = [cur_val]
        if cur_val >= float(target_sdf):
            return path, np.asarray(val_hist, dtype=float), cur.copy()
        for _ in range(max(int(max_iters), 1)):
            if cur_val >= float(target_sdf):
                break
            direction = self._normalize(cur_grad)
            if np.linalg.norm(direction) < 1e-12:
                direction = self._normalize(self.runner.estimate_surface_normal(field, cur, kind, eps=0.002))
            if np.linalg.norm(direction) < 1e-12:
                break
            g_dir = float(np.dot(cur_grad, direction))
            if g_dir <= 1e-9:
                break
            h_dir = _directional_second_derivative(cur, direction, float(curv_eps))
            step = _predict_step(
                float(target_sdf) - cur_val,
                g_dir,
                h_dir,
                float(init_step),
            )
            accepted = False
            while step >= float(min_step):
                cand = cur + step * direction
                vals_c, grads_c = field.query_with_gradient(cand.reshape(1, 3).astype(np.float32), kind=kind)
                cand_val = float(np.asarray(vals_c, dtype=float).reshape(-1)[0])
                sufficient_increase = cur_val + float(armijo_c1) * step * g_dir
                if cand_val >= float(target_sdf) or cand_val >= sufficient_increase:
                    cur = cand
                    cur_val = cand_val
                    cur_grad = np.asarray(grads_c, dtype=float).reshape(-1, 3)[0]
                    path.append(cur.copy())
                    val_hist.append(cur_val)
                    accepted = True
                    break
                step *= float(shrink)
            if not accepted:
                break
        return path, np.asarray(val_hist, dtype=float), cur.copy()

    @staticmethod
    def _polyline_length(points: np.ndarray) -> float:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if pts.shape[0] < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    @staticmethod
    def _sample_polyline_fraction(points: np.ndarray, fraction: float) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if pts.shape[0] == 0:
            return np.zeros(3, dtype=float)
        if pts.shape[0] == 1:
            return pts[0].copy()
        frac = float(np.clip(fraction, 0.0, 1.0))
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(np.sum(seg))
        if total < 1e-12:
            return pts[0].copy()
        target = frac * total
        acc = 0.0
        for i, s in enumerate(seg):
            if acc + float(s) >= target:
                local = (target - acc) / max(float(s), 1e-12)
                return pts[i] + local * (pts[i + 1] - pts[i])
            acc += float(s)
        return pts[-1].copy()

    @staticmethod
    def _eval_bezier(control_points: np.ndarray, n_samples: int) -> np.ndarray:
        cps = np.asarray(control_points, dtype=float).reshape(-1, 3)
        degree = cps.shape[0] - 1
        ts = np.linspace(0.0, 1.0, max(int(n_samples), 2))
        out = np.zeros((ts.shape[0], 3), dtype=float)
        for i in range(degree + 1):
            coeff = math.comb(degree, i) * ((1.0 - ts) ** (degree - i)) * (ts ** i)
            out += coeff[:, None] * cps[i].reshape(1, 3)
        return out

    def _build_cubic_bezier_controls(
        self,
        path_pts: np.ndarray,
        *,
        end_tangent: np.ndarray | None = None,
    ) -> np.ndarray:
        pts = np.asarray(path_pts, dtype=float).reshape(-1, 3)
        p0 = pts[0].copy()
        p3 = pts[-1].copy()
        if pts.shape[0] < 3:
            chord = p3 - p0
            if end_tangent is not None:
                tan = np.asarray(end_tangent, dtype=float).reshape(3)
                return np.vstack([p0, p0 + chord / 3.0, p3 - tan / 3.0, p3])
            return np.vstack([p0, p0 + chord / 3.0, p0 + 2.0 * chord / 3.0, p3])
        p1 = self._sample_polyline_fraction(pts, 0.25)
        if end_tangent is not None:
            p2 = p3 - np.asarray(end_tangent, dtype=float).reshape(3) / 3.0
        else:
            p2 = self._sample_polyline_fraction(pts, 0.75)
        return np.vstack([p0, p1, p2, p3])

    def _smooth_ee_paths(
        self,
        field,
        kind: str,
        ee_path_sdf_arr: np.ndarray,
        retreat_path_sdf_arr: np.ndarray,
        *,
        ee_clearance: float,
        approach_min_sdf: float,
        n_samples_per_seg: int,
    ) -> dict:
        def _line_resample(start: np.ndarray, goal: np.ndarray, n_samples: int) -> np.ndarray:
            s = np.asarray(start, dtype=float).reshape(3)
            g = np.asarray(goal, dtype=float).reshape(3)
            ts = np.linspace(0.0, 1.0, max(int(n_samples), 2))
            return (1.0 - ts[:, None]) * s.reshape(1, 3) + ts[:, None] * g.reshape(1, 3)

        result = {
            "ee_controls_sdf": None,
            "ee_smoothed_sdf": ee_path_sdf_arr.copy(),
            "ee_smoothed_d": self.runner.query_field(field, ee_path_sdf_arr, kind=kind, safe_oob=True),
            "approach_controls_sdf": None,
            "approach_smoothed_sdf": retreat_path_sdf_arr[::-1].copy(),
            "approach_smoothed_d": self.runner.query_field(
                field,
                retreat_path_sdf_arr[::-1],
                kind=kind,
                safe_oob=True,
            ),
            "approach_line_sdf": None,
            "approach_line_d": None,
            "approach_line_feasible": None,
        }
        ee_raw = np.asarray(ee_path_sdf_arr, dtype=float).reshape(-1, 3)
        approach_raw = np.asarray(retreat_path_sdf_arr[::-1], dtype=float).reshape(-1, 3)
        line_tangent = None
        if approach_raw.shape[0] >= 2:
            line_tangent = approach_raw[-1] - approach_raw[0]
        if ee_raw.shape[0] >= 2:
            ee_controls = self._build_cubic_bezier_controls(ee_raw, end_tangent=line_tangent)
            ee_smooth = self._eval_bezier(ee_controls, n_samples_per_seg)
            ee_d = self.runner.query_field(field, ee_smooth, kind=kind, safe_oob=True)
            if float(np.min(ee_d)) >= float(ee_clearance):
                result["ee_controls_sdf"] = ee_controls
                result["ee_smoothed_sdf"] = ee_smooth
                result["ee_smoothed_d"] = ee_d

        if approach_raw.shape[0] >= 2:
            app_smooth = _line_resample(approach_raw[0], approach_raw[-1], n_samples_per_seg)
            app_d = self.runner.query_field(field, app_smooth, kind=kind, safe_oob=True)
            result["approach_line_sdf"] = app_smooth
            result["approach_line_d"] = app_d
            result["approach_line_feasible"] = bool(float(np.min(app_d)) >= float(approach_min_sdf))
            result["approach_controls_sdf"] = np.vstack([approach_raw[0], approach_raw[-1]])
            result["approach_smoothed_sdf"] = app_smooth
            result["approach_smoothed_d"] = app_d
        return result

    def _plan_escape_bundle(
        self,
        *,
        field,
        kind: str,
        args,
        weld_point_pb: np.ndarray | None,
        ee_start_pb: np.ndarray,
        field_bbox_min: np.ndarray,
        field_bbox_max: np.ndarray,
    ) -> dict:
        empty = {
            "weld_point_pb": None,
            "retreat_path_pb": None,
            "retreat_vals": None,
            "retreat_goal_pb": None,
            "ee_path_pb": None,
            "ee_d_path": None,
            "ee_smoothed_pb": None,
            "ee_smoothed_d": None,
            "ee_control_points_pb": None,
            "approach_smoothed_pb": None,
            "approach_smoothed_d": None,
            "approach_control_points_pb": None,
            "approach_line_pb": None,
            "approach_line_d": None,
            "approach_line_feasible": None,
        }
        if weld_point_pb is None:
            return empty

        weld_point_pb = np.asarray(weld_point_pb, dtype=float).reshape(3)
        weld_point_sdf = self.runner.pb2sdf(weld_point_pb).reshape(3)
        retreat_path_sdf, retreat_vals, retreat_goal_sdf = self._gradient_backtrack_escape(
            field,
            weld_point_sdf,
            kind,
            target_sdf=float(getattr(args, "ee_target_sdf", ExperimentParameters.PLAN_EE_TARGET_SDF)),
            init_step=float(getattr(args, "ee_backtrack_init_step", ExperimentParameters.PLAN_EE_BACKTRACK_INIT_STEP)),
            min_step=float(getattr(args, "ee_backtrack_min_step", ExperimentParameters.PLAN_EE_BACKTRACK_MIN_STEP)),
            shrink=float(getattr(args, "ee_backtrack_shrink", ExperimentParameters.PLAN_EE_BACKTRACK_SHRINK)),
            max_iters=int(getattr(args, "ee_backtrack_max_iters", ExperimentParameters.PLAN_EE_BACKTRACK_MAX_ITERS)),
            curv_eps=float(getattr(args, "ee_backtrack_curv_eps", ExperimentParameters.PLAN_EE_BACKTRACK_CURV_EPS)),
            armijo_c1=float(getattr(args, "ee_backtrack_armijo_c1", ExperimentParameters.PLAN_EE_BACKTRACK_ARMIJO_C1)),
        )
        retreat_path_sdf_arr = np.asarray(retreat_path_sdf, dtype=float)
        retreat_path_pb = self.runner.sdf2pb(retreat_path_sdf_arr)
        retreat_goal_pb = self.runner.sdf2pb(np.asarray(retreat_goal_sdf, dtype=float).reshape(1, 3)).reshape(3)

        ee_start_sdf = self.runner.pb2sdf(ee_start_pb).reshape(3)
        ee_search_goal = np.asarray(retreat_goal_sdf, dtype=float).reshape(3)
        ee_clearance = float(getattr(args, "ee_min_clearance", ExperimentParameters.PLAN_EE_MIN_CLEARANCE))
        bmin_ee = np.maximum(
            np.minimum(ee_start_sdf, ee_search_goal) - 0.20,
            np.asarray(field_bbox_min, dtype=float) + float(args.bound_margin),
        )
        bmax_ee = np.minimum(
            np.maximum(ee_start_sdf, ee_search_goal) + 0.20,
            np.asarray(field_bbox_max, dtype=float) - float(args.bound_margin),
        )
        ee_rrt_start = ee_start_sdf.copy()
        ee_rrt_goal = ee_search_goal.copy()
        if bool(args.auto_fix_endpoints):
            ee_rrt_start, _ = self.runner.auto_fix_point_if_infeasible(
                field,
                ee_rrt_start,
                bmin_ee,
                bmax_ee,
                kind,
                ee_clearance,
                float(args.endpoint_fix_radius),
                float(args.endpoint_fix_step),
            )
            ee_rrt_goal, ok_goal_ee = self.runner.auto_fix_point_if_infeasible(
                field,
                ee_rrt_goal,
                bmin_ee,
                bmax_ee,
                kind,
                ee_clearance,
                float(args.endpoint_fix_radius),
                float(args.endpoint_fix_step),
            )
            if not ok_goal_ee:
                raise RuntimeError("末端 retreat 目标点在邻域内无法满足 EE SDF 阈值。")

        ee_path_sdf = self._rrt_star_plan(
            field=field,
            start=ee_rrt_start,
            goal=ee_rrt_goal,
            bounds_min=bmin_ee,
            bounds_max=bmax_ee,
            kind=kind,
            clearance=ee_clearance,
            step_size=float(getattr(args, "ee_step_size", ExperimentParameters.PLAN_EE_STEP_SIZE)),
            near_radius=float(getattr(args, "ee_near_radius", ExperimentParameters.PLAN_EE_NEAR_RADIUS)),
            goal_sample_prob=float(args.goal_sample_prob),
            max_iter=int(getattr(args, "ee_max_iter", ExperimentParameters.PLAN_EE_MAX_ITER)),
            edge_step=float(getattr(args, "ee_edge_check_step", ExperimentParameters.PLAN_EE_EDGE_CHECK_STEP)),
            goal_tol=float(getattr(args, "ee_goal_tolerance", ExperimentParameters.PLAN_EE_GOAL_TOLERANCE)),
        )
        if not ee_path_sdf:
            raise RuntimeError("末端 RRT 路径未找到。")
        ee_path_sdf = self._shortcut_smooth(
            ee_path_sdf,
            field=field,
            kind=kind,
            clearance=ee_clearance,
            edge_step=float(getattr(args, "ee_edge_check_step", ExperimentParameters.PLAN_EE_EDGE_CHECK_STEP)),
            iters=int(args.smooth_iters),
        )
        ee_path_sdf = self._resample_path(
            ee_path_sdf,
            float(getattr(args, "ee_resample_spacing", ExperimentParameters.PLAN_EE_RESAMPLE_SPACING)),
        )
        ee_path_sdf_arr = np.asarray(ee_path_sdf, dtype=float)
        ee_d_path = self.runner.query_field(field, ee_path_sdf_arr, kind=kind, safe_oob=True)
        ee_path_pb = self.runner.sdf2pb(ee_path_sdf_arr)

        smooth_info = self._smooth_ee_paths(
            field,
            kind,
            ee_path_sdf_arr,
            retreat_path_sdf_arr,
            ee_clearance=ee_clearance,
            approach_min_sdf=float(
                getattr(
                    args,
                    "ee_bezier_approach_min_sdf",
                    ExperimentParameters.PLAN_EE_BEZIER_APPROACH_MIN_SDF,
                )
            ),
            n_samples_per_seg=int(
                getattr(
                    args,
                    "ee_bezier_samples_per_seg",
                    ExperimentParameters.PLAN_EE_BEZIER_SAMPLES_PER_SEG,
                )
            ),
        )
        ee_smoothed_pb = self.runner.sdf2pb(np.asarray(smooth_info["ee_smoothed_sdf"], dtype=float))
        ee_smoothed_d = np.asarray(smooth_info["ee_smoothed_d"], dtype=float)
        ee_control_points_pb = None
        if smooth_info["ee_controls_sdf"] is not None:
            ee_control_points_pb = self.runner.sdf2pb(np.asarray(smooth_info["ee_controls_sdf"], dtype=float))

        approach_smoothed_pb = self.runner.sdf2pb(np.asarray(smooth_info["approach_smoothed_sdf"], dtype=float))
        approach_smoothed_d = np.asarray(smooth_info["approach_smoothed_d"], dtype=float)
        approach_control_points_pb = None
        if smooth_info["approach_controls_sdf"] is not None:
            approach_control_points_pb = self.runner.sdf2pb(np.asarray(smooth_info["approach_controls_sdf"], dtype=float))

        approach_line_pb = None
        approach_line_d = None
        approach_line_feasible = None
        if smooth_info["approach_line_sdf"] is not None:
            approach_line_d = np.asarray(smooth_info["approach_line_d"], dtype=float)
            approach_line_pb = self.runner.sdf2pb(np.asarray(smooth_info["approach_line_sdf"], dtype=float))
            approach_line_feasible = bool(smooth_info["approach_line_feasible"])

        return {
            "weld_point_pb": weld_point_pb,
            "retreat_path_pb": retreat_path_pb,
            "retreat_vals": np.asarray(retreat_vals, dtype=float),
            "retreat_goal_pb": retreat_goal_pb,
            "ee_path_pb": ee_path_pb,
            "ee_d_path": np.asarray(ee_d_path, dtype=float),
            "ee_smoothed_pb": ee_smoothed_pb,
            "ee_smoothed_d": ee_smoothed_d,
            "ee_control_points_pb": ee_control_points_pb,
            "approach_smoothed_pb": approach_smoothed_pb,
            "approach_smoothed_d": approach_smoothed_d,
            "approach_control_points_pb": approach_control_points_pb,
            "approach_line_pb": approach_line_pb,
            "approach_line_d": approach_line_d,
            "approach_line_feasible": approach_line_feasible,
        }

    def run(self) -> None:
        args = self.settings
        field = self.runner.load_field(args.sdf_npz)
        kind = self.runner.resolve_kind(field, args.kind)
        init_cfg = self.runner.load_init_config(getattr(args, "init_config_npz", None))
        q_best = None if init_cfg is None else np.asarray(init_cfg.get("q_best"), dtype=float).reshape(-1)
        if q_best is not None:
            print(f"[plan] using init-config from {init_cfg.get('path')}")

        nearest_region_as_goal = bool(getattr(args, "nearest_region_as_goal", ExperimentParameters.PLAN_NEAREST_REGION_AS_GOAL))
        nearest_json_path = getattr(args, "nearest_region_json", None) or ExperimentParameters.PLAN_NEAREST_REGION_JSON
        clearance = float(getattr(args, "min_clearance", ExperimentParameters.PLAN_MIN_CLEARANCE))
        print(f"[plan] clearance = {clearance:.4f}m")

        # goal first — needed to compute EE-based start
        if args.goal:
            goal = self.runner.pb2sdf(self.runner.parse_vec3(args.goal)).reshape(3)
        elif nearest_region_as_goal and Path(nearest_json_path).exists():
            nr_data = json.loads(Path(nearest_json_path).read_text(encoding="utf-8"))
            topk_nr = nr_data.get("top_k", [])
            if topk_nr:
                goal_pb = np.asarray(topk_nr[0]["point"], dtype=float)
                goal = self.runner.pb2sdf(goal_pb).reshape(3)
                print(f"[plan] goal from nearest-region PB={goal_pb.tolist()} -> SDF={goal.tolist()}")
            else:
                goal = np.asarray(field.bbox_max, dtype=float) - 0.15
                print("[plan] WARNING: nearest-region JSON has no top_k, using bbox default")
        else:
            goal = np.asarray(field.bbox_max, dtype=float) - 0.15

        if args.start:
            start = self.runner.pb2sdf(self.runner.parse_vec3(args.start)).reshape(3)
        else:
            p.connect(p.DIRECT)
            try:
                cfg = ExperimentConfig()
                robot, _ = self.runner.make_robot_and_workpiece(cfg)
                if q_best is not None:
                    q_default, _ = robot.get_joint_state()
                    q_init = q_default.copy()
                    q_init[robot.n_pris:] = q_best[robot.n_pris:]
                    robot.set_joint_state(q_init, np.zeros_like(q_init))
                base_pos, _ = robot.get_robobase_pose()
                start_pb = np.asarray(base_pos, dtype=float)
                print(f"[plan] start robobase (initial gantry) PB={start_pb.tolist()}")
                start = self.runner.pb2sdf(start_pb).reshape(3)
                print(f"[plan] start SDF={start.tolist()}")
            finally:
                p.disconnect()

        via = None
        if args.via_point:
            via = self.runner.pb2sdf(self.runner.parse_vec3(args.via_point)).reshape(3)

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
            start, ok_start = self.runner.auto_fix_point_if_infeasible(field, start, bmin, bmax, kind, clearance, float(args.endpoint_fix_radius), float(args.endpoint_fix_step))
            goal, ok_goal = self.runner.auto_fix_point_if_infeasible(field, goal, bmin, bmax, kind, clearance, float(args.endpoint_fix_radius), float(args.endpoint_fix_step))
            if via is not None:
                via, ok_via = self.runner.auto_fix_point_if_infeasible(field, via, bmin, bmax, kind, clearance, float(args.endpoint_fix_radius), float(args.endpoint_fix_step))
            else:
                ok_via = True
            if not (ok_start and ok_goal and ok_via):
                raise RuntimeError("端点清距修复失败：start/goal/via 至少有一个点无法在邻域内满足 SDF 阈值。")

        s_sdf = float(self.runner.query_field(field, start.reshape(1, 3), kind=kind, safe_oob=True)[0])
        g_sdf = float(self.runner.query_field(field, goal.reshape(1, 3), kind=kind, safe_oob=True)[0])
        dist_sg = float(np.linalg.norm(goal - start))
        print(f"[plan] SDF at start={s_sdf:.4f}, at goal={g_sdf:.4f}")
        print(f"[plan] start-goal dist={dist_sg:.3f}m, bounds={bmin.tolist()} .. {bmax.tolist()}")

        method = getattr(args, "method", ExperimentParameters.PLAN_METHOD)
        bmin_f = np.asarray(field.bbox_min, dtype=float)
        bmax_f = np.asarray(field.bbox_max, dtype=float)
        start_oob = np.any(start < bmin_f) or np.any(start > bmax_f)
        goal_oob = np.any(goal < bmin_f) or np.any(goal > bmax_f)
        waypoints = [start] + ([via] if via is not None else []) + [goal]

        if method == "astar":
            print("[plan] method=A*")
            path = []
            for seg_i in range(len(waypoints) - 1):
                seg = self._astar_sdf_plan(field, waypoints[seg_i], waypoints[seg_i + 1], kind=kind, clearance=clearance)
                if not seg:
                    raise RuntimeError(f"A* 段 {seg_i} 未找到可行路径。")
                path.extend(seg if not path else seg[1:])
        else:
            print("[plan] method=RRT*")
            path = []
            for seg_i in range(len(waypoints) - 1):
                seg = self._rrt_star_plan(
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
            path = self._shortcut_smooth(path, field=field, kind=kind, clearance=clearance, edge_step=float(args.edge_check_step), iters=int(args.smooth_iters))

        if not path:
            raise RuntimeError("未找到可行路径。")

        print(f"[plan] raw/smoothed path: {len(path)} waypoints")
        resample_spacing = float(getattr(args, "resample_spacing", ExperimentParameters.PLAN_RESAMPLE_SPACING))
        path = self._resample_path(path, resample_spacing)

        pts_sdf = np.asarray(path, dtype=float)
        d_path = self.runner.query_field(field, pts_sdf, kind=kind, safe_oob=True)
        min_sdf = float(np.min(d_path))
        print(f"[plan] resampled grid path: {len(path)} waypoints, min_sdf={min_sdf:.4f}m, max_spacing={resample_spacing}m")
        if min_sdf <= clearance:
            raise RuntimeError(f"路径不满足 SDF 阈值 {clearance:.4f}m (min={min_sdf:.4f}m)。")

        if start_oob:
            path.insert(0, start.copy())
            print(f"[plan] 起点在 SDF bbox 外，已插入原始起点 (总 {len(path)} 点)")
        if goal_oob:
            path.append(goal.copy())
            print(f"[plan] 终点在 SDF bbox 外，已追加原始终点 (总 {len(path)} 点)")

        pts_sdf = np.asarray(path, dtype=float)
        d_path = self.runner.query_field(field, pts_sdf, kind=kind, safe_oob=True)
        pts = self.runner.sdf2pb(pts_sdf)
        start_pb_used = self.runner.sdf2pb(np.asarray(start, dtype=float).reshape(1, 3)).reshape(3)
        goal_pb_used = self.runner.sdf2pb(np.asarray(goal, dtype=float).reshape(1, 3)).reshape(3)

        q_plan_final = None
        weld_start_pb = None
        weld_goal_pb = None
        start_bundle = self._plan_escape_bundle(
            field=field,
            kind=kind,
            args=args,
            weld_point_pb=None,
            ee_start_pb=np.zeros(3, dtype=float),
            field_bbox_min=np.asarray(field.bbox_min, dtype=float),
            field_bbox_max=np.asarray(field.bbox_max, dtype=float),
        )
        end_bundle = self._plan_escape_bundle(
            field=field,
            kind=kind,
            args=args,
            weld_point_pb=None,
            ee_start_pb=np.zeros(3, dtype=float),
            field_bbox_min=np.asarray(field.bbox_min, dtype=float),
            field_bbox_max=np.asarray(field.bbox_max, dtype=float),
        )
        if Path(nearest_json_path).exists():
            nr_data = json.loads(Path(nearest_json_path).read_text(encoding="utf-8"))
            weld_start_raw = nr_data.get("weld_start_point")
            if weld_start_raw is not None:
                weld_start_pb = np.asarray(weld_start_raw, dtype=float).reshape(3)
            weld_goal_raw = nr_data.get("weld_goal_point")
            if weld_goal_raw is not None:
                weld_goal_pb = np.asarray(weld_goal_raw, dtype=float).reshape(3)
        mid_robobase_pb = None
        mid_ee_pb = None
        if q_best is not None:
            p.connect(p.DIRECT)
            try:
                cfg = ExperimentConfig()
                robot, workpiece = self.runner.make_robot_and_workpiece(cfg)
                q_plan_final = self.runner.move_robot_base_to_position(robot, goal_pb_used, q_seed=q_best)
                base_after_pb, _ = robot.get_robobase_pose()
                ee_start_pb, _ = robot.get_ee_pose()
                mid_robobase_pb = np.asarray(base_after_pb, dtype=float)
                mid_ee_pb = np.asarray(ee_start_pb, dtype=float)
                print(f"[plan] mid robobase PB={mid_robobase_pb.tolist()}")
                print(f"[plan] mid EE      PB={mid_ee_pb.tolist()}")
                if weld_start_pb is not None:
                    start_bundle = self._plan_escape_bundle(
                        field=field,
                        kind=kind,
                        args=args,
                        weld_point_pb=weld_start_pb,
                        ee_start_pb=ee_start_pb,
                        field_bbox_min=np.asarray(field.bbox_min, dtype=float),
                        field_bbox_max=np.asarray(field.bbox_max, dtype=float),
                    )
                    print(
                        f"[plan] weld-start escape: start_sdf={float(start_bundle['retreat_vals'][0]):.4f}m -> "
                        f"end_sdf={float(start_bundle['retreat_vals'][-1]):.4f}m, "
                        f"steps={len(start_bundle['retreat_path_pb'])}"
                    )
                    print(
                        f"[plan] weld-start ee path: {len(start_bundle['ee_path_pb'])} waypoints, "
                        f"min_sdf={float(np.min(start_bundle['ee_d_path'])):.4f}m, "
                        f"goal={np.asarray(start_bundle['retreat_goal_pb'], dtype=float).tolist()}"
                    )
                    print(
                        f"[plan] weld-start smooth: ee_min={float(np.min(start_bundle['ee_smoothed_d'])):.4f}m, "
                        f"approach_min={float(np.min(start_bundle['approach_smoothed_d'])):.4f}m"
                    )
                if weld_goal_pb is not None:
                    end_bundle = self._plan_escape_bundle(
                        field=field,
                        kind=kind,
                        args=args,
                        weld_point_pb=weld_goal_pb,
                        ee_start_pb=ee_start_pb,
                        field_bbox_min=np.asarray(field.bbox_min, dtype=float),
                        field_bbox_max=np.asarray(field.bbox_max, dtype=float),
                    )
                    print(
                        f"[plan] weld-goal escape: start_sdf={float(end_bundle['retreat_vals'][0]):.4f}m -> "
                        f"end_sdf={float(end_bundle['retreat_vals'][-1]):.4f}m, "
                        f"steps={len(end_bundle['retreat_path_pb'])}"
                    )
                    print(
                        f"[plan] weld-goal ee path: {len(end_bundle['ee_path_pb'])} waypoints, "
                        f"min_sdf={float(np.min(end_bundle['ee_d_path'])):.4f}m, "
                        f"goal={np.asarray(end_bundle['retreat_goal_pb'], dtype=float).tolist()}"
                    )
                    print(
                        f"[plan] weld-goal smooth: ee_min={float(np.min(end_bundle['ee_smoothed_d'])):.4f}m, "
                        f"approach_min={float(np.min(end_bundle['approach_smoothed_d'])):.4f}m"
                    )
                _ = workpiece
            finally:
                p.disconnect()

        out_json = self.runner.ensure_parent(Path(args.output_json))
        payload = {
            "method": method,
            "kind": kind,
            "n_waypoints": int(len(path)),
            "min_sdf_on_waypoints": float(np.min(d_path)),
            "start_used": start_pb_used.tolist(),
            "start_is_robobase": True,
            "goal_used": goal_pb_used.tolist(),
            "mid_robobase_position": None if mid_robobase_pb is None else mid_robobase_pb.tolist(),
            "mid_ee_position": None if mid_ee_pb is None else mid_ee_pb.tolist(),
            "via_point": None if via is None else self.runner.sdf2pb(np.asarray(via, dtype=float).reshape(1, 3)).reshape(3).tolist(),
            "path": pts.tolist(),
            "path_is_robobase": True,
            "init_config_used": None if q_best is None else {
                "path": str(init_cfg.get("path")),
                "q_best": q_best.tolist(),
            },
            "weld_start_point": None if start_bundle["weld_point_pb"] is None else np.asarray(start_bundle["weld_point_pb"], dtype=float).tolist(),
            "retreat_path": None if start_bundle["retreat_path_pb"] is None else np.asarray(start_bundle["retreat_path_pb"], dtype=float).tolist(),
            "retreat_sdf_values": None if start_bundle["retreat_vals"] is None else np.asarray(start_bundle["retreat_vals"], dtype=float).tolist(),
            "retreat_goal": None if start_bundle["retreat_goal_pb"] is None else np.asarray(start_bundle["retreat_goal_pb"], dtype=float).tolist(),
            "ee_path": None if start_bundle["ee_path_pb"] is None else np.asarray(start_bundle["ee_path_pb"], dtype=float).tolist(),
            "ee_min_sdf_on_waypoints": None if start_bundle["ee_d_path"] is None else float(np.min(start_bundle["ee_d_path"])),
            "ee_bezier_control_points": None if start_bundle["ee_control_points_pb"] is None else np.asarray(start_bundle["ee_control_points_pb"], dtype=float).tolist(),
            "ee_bezier_path": None if start_bundle["ee_smoothed_pb"] is None else np.asarray(start_bundle["ee_smoothed_pb"], dtype=float).tolist(),
            "ee_bezier_min_sdf": None if start_bundle["ee_smoothed_d"] is None else float(np.min(start_bundle["ee_smoothed_d"])),
            "approach_bezier_control_points": None if start_bundle["approach_control_points_pb"] is None else np.asarray(start_bundle["approach_control_points_pb"], dtype=float).tolist(),
            "approach_bezier_path": None if start_bundle["approach_smoothed_pb"] is None else np.asarray(start_bundle["approach_smoothed_pb"], dtype=float).tolist(),
            "approach_bezier_min_sdf": None if start_bundle["approach_smoothed_d"] is None else float(np.min(start_bundle["approach_smoothed_d"])),
            "approach_line_path": None if start_bundle["approach_line_pb"] is None else np.asarray(start_bundle["approach_line_pb"], dtype=float).tolist(),
            "approach_line_min_sdf": None if start_bundle["approach_line_d"] is None else float(np.min(start_bundle["approach_line_d"])),
            "approach_line_feasible": None if start_bundle["approach_line_feasible"] is None else bool(start_bundle["approach_line_feasible"]),
            "weld_goal_point": None if end_bundle["weld_point_pb"] is None else np.asarray(end_bundle["weld_point_pb"], dtype=float).tolist(),
            "retreat_end_path": None if end_bundle["retreat_path_pb"] is None else np.asarray(end_bundle["retreat_path_pb"], dtype=float).tolist(),
            "retreat_end_sdf_values": None if end_bundle["retreat_vals"] is None else np.asarray(end_bundle["retreat_vals"], dtype=float).tolist(),
            "retreat_end_goal": None if end_bundle["retreat_goal_pb"] is None else np.asarray(end_bundle["retreat_goal_pb"], dtype=float).tolist(),
            "ee_path_return": None if end_bundle["ee_path_pb"] is None else np.asarray(end_bundle["ee_path_pb"], dtype=float).tolist(),
            "ee_return_min_sdf_on_waypoints": None if end_bundle["ee_d_path"] is None else float(np.min(end_bundle["ee_d_path"])),
            "ee_bezier_control_points_return": None if end_bundle["ee_control_points_pb"] is None else np.asarray(end_bundle["ee_control_points_pb"], dtype=float).tolist(),
            "ee_bezier_path_return": None if end_bundle["ee_smoothed_pb"] is None else np.asarray(end_bundle["ee_smoothed_pb"], dtype=float).tolist(),
            "ee_bezier_min_sdf_return": None if end_bundle["ee_smoothed_d"] is None else float(np.min(end_bundle["ee_smoothed_d"])),
            "approach_end_bezier_control_points": None if end_bundle["approach_control_points_pb"] is None else np.asarray(end_bundle["approach_control_points_pb"], dtype=float).tolist(),
            "approach_end_bezier_path": None if end_bundle["approach_smoothed_pb"] is None else np.asarray(end_bundle["approach_smoothed_pb"], dtype=float).tolist(),
            "approach_end_bezier_min_sdf": None if end_bundle["approach_smoothed_d"] is None else float(np.min(end_bundle["approach_smoothed_d"])),
            "approach_end_line_path": None if end_bundle["approach_line_pb"] is None else np.asarray(end_bundle["approach_line_pb"], dtype=float).tolist(),
            "approach_end_line_min_sdf": None if end_bundle["approach_line_d"] is None else float(np.min(end_bundle["approach_line_d"])),
            "approach_end_line_feasible": None if end_bundle["approach_line_feasible"] is None else bool(end_bundle["approach_line_feasible"]),
            "robot_q_at_goal": None if q_plan_final is None else np.asarray(q_plan_final, dtype=float).tolist(),
        }
        with open(out_json, "w", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, ensure_ascii=False, indent=2))

        fig = plt.figure(figsize=(7, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", ms=2.8, lw=1.4, c="dodgerblue", label="robobase RRT path")
        ax.scatter([start_pb_used[0]], [start_pb_used[1]], [start_pb_used[2]], c="green", s=50, label="robobase start")
        ax.scatter([goal_pb_used[0]], [goal_pb_used[1]], [goal_pb_used[2]], c="red", s=50, label="robobase goal (mid)")
        if mid_ee_pb is not None:
            ax.scatter([mid_ee_pb[0]], [mid_ee_pb[1]], [mid_ee_pb[2]], c="cyan", s=50, marker="^", label="mid EE")
        if weld_start_pb is not None:
            ax.scatter([weld_start_pb[0]], [weld_start_pb[1]], [weld_start_pb[2]], c="crimson", s=42, label="weld start")
        if weld_goal_pb is not None:
            ax.scatter([weld_goal_pb[0]], [weld_goal_pb[1]], [weld_goal_pb[2]], c="purple", s=42, label="weld goal")
        if start_bundle["retreat_path_pb"] is not None:
            retreat_arr = np.asarray(start_bundle["retreat_path_pb"], dtype=float)
            ax.plot(retreat_arr[:, 0], retreat_arr[:, 1], retreat_arr[:, 2], "-o", ms=2.2, lw=1.2, c="orchid", label="gradient retreat")
        if start_bundle["ee_path_pb"] is not None:
            ee_arr = np.asarray(start_bundle["ee_path_pb"], dtype=float)
            ax.plot(ee_arr[:, 0], ee_arr[:, 1], ee_arr[:, 2], "-o", ms=2.0, lw=1.2, c="goldenrod", label="EE RRT")
        if start_bundle["ee_smoothed_pb"] is not None:
            ee_smooth_arr = np.asarray(start_bundle["ee_smoothed_pb"], dtype=float)
            ax.plot(
                ee_smooth_arr[:, 0], ee_smooth_arr[:, 1], ee_smooth_arr[:, 2],
                "-", lw=2.2, c="teal", label="EE bezier",
            )
        if start_bundle["approach_line_pb"] is not None:
            app_line_arr = np.asarray(start_bundle["approach_line_pb"], dtype=float)
            same_as_actual = (
                start_bundle["approach_smoothed_pb"] is not None
                and np.asarray(start_bundle["approach_smoothed_pb"], dtype=float).shape == app_line_arr.shape
                and np.allclose(np.asarray(start_bundle["approach_smoothed_pb"], dtype=float), app_line_arr)
            )
            if not same_as_actual:
                ax.plot(
                    app_line_arr[:, 0], app_line_arr[:, 1], app_line_arr[:, 2],
                    "--", lw=1.8, c="orange", label="approach line",
                )
        if start_bundle["approach_smoothed_pb"] is not None:
            app_smooth_arr = np.asarray(start_bundle["approach_smoothed_pb"], dtype=float)
            ax.plot(
                app_smooth_arr[:, 0], app_smooth_arr[:, 1], app_smooth_arr[:, 2],
                "-", lw=2.0, c="darkorange", label="approach line",
            )
        if weld_start_pb is not None and weld_goal_pb is not None:
            weld_seg = np.vstack([weld_start_pb.reshape(1, 3), weld_goal_pb.reshape(1, 3)])
            ax.plot(
                weld_seg[:, 0], weld_seg[:, 1], weld_seg[:, 2],
                "-s", ms=5, lw=2.8, c="red", label="weld path",
            )
        if end_bundle["retreat_path_pb"] is not None:
            retreat_end_arr = np.asarray(end_bundle["retreat_path_pb"], dtype=float)
            ax.plot(
                retreat_end_arr[:, 0], retreat_end_arr[:, 1], retreat_end_arr[:, 2],
                "-o", ms=2.0, lw=1.1, c="mediumpurple", label="goal retreat",
            )
        if end_bundle["ee_path_pb"] is not None:
            ee_end_arr = np.asarray(end_bundle["ee_path_pb"], dtype=float)
            ax.plot(
                ee_end_arr[:, 0], ee_end_arr[:, 1], ee_end_arr[:, 2],
                "-o", ms=2.0, lw=1.2, c="plum", label="end EE RRT",
            )
        if end_bundle["ee_smoothed_pb"] is not None:
            ee_return_arr = np.asarray(end_bundle["ee_smoothed_pb"], dtype=float)
            ax.plot(
                ee_return_arr[:, 0], ee_return_arr[:, 1], ee_return_arr[:, 2],
                "-", lw=1.8, c="slateblue", label="return bezier",
            )
        if end_bundle["approach_line_pb"] is not None:
            end_app_line_arr = np.asarray(end_bundle["approach_line_pb"], dtype=float)
            end_same = (
                end_bundle["approach_smoothed_pb"] is not None
                and np.asarray(end_bundle["approach_smoothed_pb"], dtype=float).shape == end_app_line_arr.shape
                and np.allclose(np.asarray(end_bundle["approach_smoothed_pb"], dtype=float), end_app_line_arr)
            )
            if not end_same:
                ax.plot(
                    end_app_line_arr[:, 0], end_app_line_arr[:, 1], end_app_line_arr[:, 2],
                    "--", lw=1.6, c="salmon", label="end approach line",
                )
        if end_bundle["approach_smoothed_pb"] is not None:
            end_app_smooth_arr = np.asarray(end_bundle["approach_smoothed_pb"], dtype=float)
            ax.plot(
                end_app_smooth_arr[:, 0], end_app_smooth_arr[:, 1], end_app_smooth_arr[:, 2],
                "-", lw=1.8, c="tomato", label="end approach smooth",
            )
        ax.set_title(f"SDF-constrained {method.upper()} path (full cycle)")
        ax.legend()
        out_png = self.runner.ensure_parent(Path(args.output_png))
        with open(out_png, "wb") as _fp:
            fig.savefig(_fp, dpi=140, format="png")
        plt.close(fig)
        print(f"[planner] 完成，结果写入: {out_json}")
        self.runner.show_plan_path(
            pts,
            start_pb_used,
            goal_pb_used,
            method,
            d_path,
            path,
            robot_q=q_plan_final,
            ee_pts_pb=start_bundle["ee_path_pb"],
            ee_d_path=start_bundle["ee_d_path"],
            ee_smoothed_pts_pb=start_bundle["ee_smoothed_pb"],
            ee_smoothed_d_path=start_bundle["ee_smoothed_d"],
            ee_control_points_pb=start_bundle["ee_control_points_pb"],
            weld_start_pb=weld_start_pb,
            weld_goal_pb=weld_goal_pb,
            retreat_pts_pb=start_bundle["retreat_path_pb"],
            retreat_goal_pb=start_bundle["retreat_goal_pb"],
            approach_smoothed_pts_pb=start_bundle["approach_smoothed_pb"],
            approach_smoothed_d_path=start_bundle["approach_smoothed_d"],
            approach_control_points_pb=start_bundle["approach_control_points_pb"],
            approach_line_pts_pb=start_bundle["approach_line_pb"],
            approach_line_d_path=start_bundle["approach_line_d"],
            approach_line_feasible=start_bundle["approach_line_feasible"],
            end_retreat_pts_pb=end_bundle["retreat_path_pb"],
            end_retreat_goal_pb=end_bundle["retreat_goal_pb"],
            end_ee_pts_pb=end_bundle["ee_path_pb"],
            end_ee_smoothed_pts_pb=end_bundle["ee_smoothed_pb"],
            end_approach_smoothed_pts_pb=end_bundle["approach_smoothed_pb"],
            end_approach_control_points_pb=end_bundle["approach_control_points_pb"],
            end_approach_line_pts_pb=end_bundle["approach_line_pb"],
            end_approach_line_feasible=end_bundle["approach_line_feasible"],
        )


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_steps(ExperimentParameters.RUN_STEPS)
