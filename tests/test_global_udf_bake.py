"""Tests for global UDF bake helpers (unittest runner)."""

from __future__ import annotations

import importlib
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None  # type: ignore[misc, assignment]

udf_mod = importlib.import_module("CBF_experiment.active.pybullet.4_1_udf")


class TestVoxelCenters(unittest.TestCase):
    def test_voxel_centers_follow_origin_plus_half_spacing(self):
        origin = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        spacing = 0.5

        centers = udf_mod.compute_voxel_centers(origin, spacing, shape=(2, 1, 1))

        self.assertTrue(
            np.allclose(
                centers[:, 0, 0],
                [[1.25, 2.25, 3.25], [1.75, 2.25, 3.25]],
            )
        )


def _sample_field(
    *,
    udf_grid: np.ndarray | None = None,
    igl: np.ndarray | None = None,
    o3d: np.ndarray | None = None,
    bbox_max: np.ndarray | None = None,
) -> "udf_mod.DistanceField":
    if udf_grid is None:
        udf_grid = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    if igl is None:
        igl = (udf_grid + 10.0).astype(np.float32)
    if o3d is None:
        o3d = (udf_grid + 100.0).astype(np.float32)
    if bbox_max is None:
        bbox_max = np.ones(3, dtype=np.float32)
    return udf_mod.DistanceField(
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        spacing=np.float32(1.0),
        udf_grid=udf_grid,
        igl_sdf_grid=igl,
        o3d_sdf_grid=o3d,
        bbox_min=np.zeros(3, dtype=np.float32),
        bbox_max=bbox_max,
        status_flags={"udf_ok": True, "igl_ok": False, "o3d_ok": True},
        failure_reasons=["a", "b"],
        build_config={"version": 1},
    )


class TestDistanceFieldSaveLoad(unittest.TestCase):
    def test_roundtrip_saves_required_arrays_and_metadata(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.npz"
            udf_mod.save_distance_field(path, field)
            loaded = udf_mod.load_distance_field(path)

        self.assertTrue(np.allclose(loaded.origin, field.origin))
        self.assertEqual(float(loaded.spacing), float(field.spacing))
        self.assertEqual(loaded.udf_grid.shape, field.udf_grid.shape)
        self.assertTrue(np.allclose(loaded.udf_grid, field.udf_grid))
        self.assertTrue(np.allclose(loaded.igl_sdf_grid, field.igl_sdf_grid))
        self.assertTrue(np.allclose(loaded.o3d_sdf_grid, field.o3d_sdf_grid))
        self.assertTrue(np.allclose(loaded.bbox_min, field.bbox_min))
        self.assertTrue(np.allclose(loaded.bbox_max, field.bbox_max))
        self.assertEqual(loaded.status_flags, field.status_flags)
        self.assertEqual(loaded.failure_reasons, field.failure_reasons)
        self.assertEqual(loaded.build_config, field.build_config)

    def test_save_raises_when_grid_shapes_do_not_match(self):
        field = _sample_field(igl=np.zeros((3, 2, 2), dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.npz"
            with self.assertRaises(ValueError):
                udf_mod.save_distance_field(path, field)

    def test_load_raises_when_shape_mismatches_sdf_grid(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.npz"
            udf_mod.save_distance_field(path, field)
            with np.load(path, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
            payload["igl_sdf_grid"] = np.zeros((3, 2, 2), dtype=np.float32)
            np.savez_compressed(path, **payload)

            with self.assertRaises(ValueError):
                udf_mod.load_distance_field(path)

    def test_load_raises_on_invalid_json(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.npz"
            udf_mod.save_distance_field(path, field)
            with np.load(path, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
            payload["failure_reasons"] = np.array("{not valid json", dtype=np.str_)
            np.savez_compressed(path, **payload)

            with self.assertRaises(ValueError):
                udf_mod.load_distance_field(path)

    def test_load_raises_on_invalid_json_types(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.npz"
            udf_mod.save_distance_field(path, field)
            with np.load(path, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
            payload["status_flags"] = np.array('["not", "a", "dict"]', dtype=np.str_)
            np.savez_compressed(path, **payload)

            with self.assertRaises(ValueError):
                udf_mod.load_distance_field(path)

    def test_load_raises_when_required_key_is_missing(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.npz"
            udf_mod.save_distance_field(path, field)
            with np.load(path, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files if key != "bbox_max"}
            np.savez_compressed(path, **payload)

            with self.assertRaises(ValueError):
                udf_mod.load_distance_field(path)


class TestDistanceFieldQuery(unittest.TestCase):
    def test_trilinear_query_returns_expected_value(self):
        field = udf_mod.DistanceField(
            origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            spacing=np.float32(1.0),
            udf_grid=np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            igl_sdf_grid=np.zeros((2, 2, 2), dtype=np.float32),
            o3d_sdf_grid=np.zeros((2, 2, 2), dtype=np.float32),
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.ones(3, dtype=np.float32),
        )
        value = field.query_single(
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            kind="udf",
            clip=True,
        )
        self.assertAlmostEqual(float(value), 3.5, places=6)

    def test_query_batch_and_single_kinds(self):
        field = _sample_field()
        p1 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        batch = np.stack([p1, p1], axis=0)

        v_udf = field.query(p1, kind="udf", clip=False)
        self.assertEqual(v_udf.shape, ())
        self.assertAlmostEqual(float(v_udf), 3.5, places=5)

        v_igl = field.query(p1, kind="igl_sdf", clip=False)
        self.assertAlmostEqual(float(v_igl), 13.5, places=5)

        v_o3d = field.query(p1, kind="o3d_sdf", clip=False)
        self.assertAlmostEqual(float(v_o3d), 103.5, places=5)

        vb = field.query(batch, kind="udf", clip=False)
        self.assertEqual(vb.shape, (2,))
        self.assertTrue(np.allclose(vb, np.array([3.5, 3.5], dtype=np.float32)))

    def test_query_out_of_bounds_raises_without_clip(self):
        field = _sample_field(bbox_max=np.ones(3, dtype=np.float32) * 0.5)
        p = np.array([1.0, 0.2, 0.2], dtype=np.float32)
        with self.assertRaises(udf_mod.DistanceFieldQueryOutOfBoundsError):
            field.query(p, kind="udf", clip=False)

    def test_query_clip_true_returns_finite_inside_bbox(self):
        field = _sample_field(bbox_max=np.ones(3, dtype=np.float32) * 0.5)
        p = np.array([1.0, 0.2, 0.2], dtype=np.float32)
        v = field.query(p, kind="udf", clip=True)
        self.assertTrue(np.isfinite(float(v)))

    def test_nan_in_stencil_returns_nan(self):
        g = np.arange(8, dtype=np.float32).reshape(2, 2, 2).copy()
        g[0, 0, 0] = np.nan
        field = _sample_field(udf_grid=g)
        p = np.array([0.25, 0.25, 0.25], dtype=np.float32)
        v = field.query(p, kind="udf", clip=True)
        self.assertTrue(np.isnan(float(v)))

    def test_invalid_kind_raises(self):
        field = _sample_field()
        p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        with self.assertRaises(ValueError):
            field.query(p, kind="not_a_kind", clip=True)

    def test_invalid_points_shape_raises(self):
        field = _sample_field()
        with self.assertRaises(ValueError):
            field.query(np.zeros((2, 2), dtype=np.float32), kind="udf", clip=True)

    def test_single_voxel_axis_no_negative_wrap(self):
        udf = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
        field = udf_mod.DistanceField(
            origin=np.zeros(3, dtype=np.float32),
            spacing=np.float32(1.0),
            udf_grid=udf,
            igl_sdf_grid=np.zeros_like(udf),
            o3d_sdf_grid=np.zeros_like(udf),
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        v = field.query(p, kind="udf", clip=True)
        self.assertTrue(np.isfinite(float(v)))


@unittest.skipIf(trimesh is None, "trimesh required for URDF assembly tests")
class TestUrdfAssembly(unittest.TestCase):
    def _write_toy_scene(self, tmp: Path, *, child_xyz: str, mesh_scale: str | None = None) -> Path:
        """scene/urdf/toy.urdf + scene/meshes/{base,child}.stl (identical small boxes)."""
        scene = tmp / "scene"
        urdf_dir = scene / "urdf"
        mesh_dir = scene / "meshes"
        urdf_dir.mkdir(parents=True)
        mesh_dir.mkdir(parents=True)

        box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        base_stl = mesh_dir / "base.stl"
        child_stl = mesh_dir / "child.stl"
        box.export(str(base_stl))
        box.export(str(child_stl))

        scale_attr = ""
        if mesh_scale is not None:
            scale_attr = f' scale="{mesh_scale}"'

        urdf_text = f"""<?xml version="1.0"?>
<robot name="toy">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="package://toy/meshes/base.stl"{scale_attr}/></geometry>
    </visual>
  </link>
  <link name="child_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="package://toy/meshes/child.stl"{scale_attr}/></geometry>
    </visual>
  </link>
  <joint name="j1" type="fixed">
    <origin xyz="{child_xyz}" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="child_link"/>
  </joint>
</robot>
"""
        urdf_path = urdf_dir / "toy.urdf"
        urdf_path.write_text(urdf_text, encoding="utf-8")
        return urdf_path

    def test_fixed_joint_translation_bbox(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_path = self._write_toy_scene(tmp, child_xyz="1 0 0")
            assy = udf_mod.load_assembly_from_urdf(urdf_path)

        self.assertIsInstance(assy, udf_mod.UrdfAssembly)
        self.assertEqual(assy.triangles.ndim, 3)
        self.assertEqual(assy.triangles.shape[-1], 3)
        self.assertTrue(len(assy.link_names) >= 2)
        self.assertEqual(len(assy.mesh_paths), len(assy.link_names))
        # base ~[-0.05,0.05], child shifted +1 in x -> ~[0.95,1.05]
        np.testing.assert_allclose(assy.bbox_min[0], -0.05, atol=1e-3, rtol=0)
        np.testing.assert_allclose(assy.bbox_max[0], 1.05, atol=1e-3, rtol=0)

    def test_package_uri_resolves_next_to_scene_meshes(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_path = self._write_toy_scene(tmp, child_xyz="0 0 0")
            resolved = udf_mod.resolve_urdf_mesh_uri(urdf_path, "package://any/meshes/base.stl")
            self.assertTrue(resolved.is_file())
            self.assertEqual(resolved.parent.name, "meshes")

    def test_mesh_scale_inflates_local_geometry(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_unscaled = self._write_toy_scene(tmp, child_xyz="0 0 0", mesh_scale=None)
            assy_unscaled = udf_mod.load_assembly_from_urdf(urdf_unscaled)
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_scaled = self._write_toy_scene(tmp, child_xyz="0 0 0", mesh_scale="2 2 2")
            assy_scaled = udf_mod.load_assembly_from_urdf(urdf_scaled)
        # Same joint layout (identity); scale 2 should double half-extents ~0.05 -> ~0.1
        self.assertGreater(float(assy_scaled.bbox_max[0]), float(assy_unscaled.bbox_max[0]) + 0.04)

    def test_visual_and_collision_origin_are_order_insensitive(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            scene = tmp / "scene"
            urdf_dir = scene / "urdf"
            mesh_dir = scene / "meshes"
            urdf_dir.mkdir(parents=True)
            mesh_dir.mkdir(parents=True)

            box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
            box.export(str(mesh_dir / "base.stl"))
            box.export(str(mesh_dir / "child.stl"))

            urdf_path = urdf_dir / "toy_reordered.urdf"
            urdf_path.write_text(
                """<?xml version="1.0"?>
<robot name="toy">
  <link name="base_link">
    <visual>
      <geometry><mesh filename="package://toy/meshes/base.stl"/></geometry>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
    </visual>
  </link>
  <link name="child_link">
    <collision>
      <geometry><mesh filename="package://toy/meshes/child.stl"/></geometry>
      <origin xyz="0.25 0 0" rpy="0 0 0"/>
    </collision>
  </link>
  <joint name="j1" type="fixed">
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="child_link"/>
  </joint>
</robot>
""",
                encoding="utf-8",
            )

            assy = udf_mod.load_assembly_from_urdf(urdf_path)

        np.testing.assert_allclose(assy.bbox_min[0], 0.45, atol=1e-3, rtol=0)
        np.testing.assert_allclose(assy.bbox_max[0], 1.30, atol=1e-3, rtol=0)

    def test_malformed_joint_raises_value_error(self):
        bad_urdfs = [
            """<?xml version="1.0"?>
<robot name="toy">
  <link name="base_link"/>
  <link name="child_link"/>
  <joint name="j1" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <child link="child_link"/>
  </joint>
</robot>
""",
            """<?xml version="1.0"?>
<robot name="toy">
  <link name="base_link"/>
  <link name="child_link"/>
  <joint name="j1" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base_link"/>
  </joint>
</robot>
""",
            """<?xml version="1.0"?>
<robot name="toy">
  <link name="base_link"/>
  <link name="child_link"/>
  <joint name="j1" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link=""/>
    <child link="child_link"/>
  </joint>
</robot>
""",
            """<?xml version="1.0"?>
<robot name="toy">
  <link name="base_link"/>
  <link name="child_link"/>
  <joint name="j1" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link=""/>
  </joint>
</robot>
""",
        ]
        for i, xml_text in enumerate(bad_urdfs):
            with self.subTest(case=i):
                with tempfile.TemporaryDirectory() as raw:
                    tmp = Path(raw)
                    urdf_path = tmp / f"bad_{i}.urdf"
                    urdf_path.write_text(xml_text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        udf_mod.load_assembly_from_urdf(urdf_path)

    def test_parse_args_minimal_flags(self):
        args = udf_mod.parse_args(
            [
                "--urdf",
                "foo.urdf",
                "--inspect-only",
                "--spacing",
                "0.02",
                "--margin",
                "0.1",
                "--output",
                "x.npz",
                "--max-points-per-batch",
                "321",
            ]
        )
        self.assertEqual(args.urdf, "foo.urdf")
        self.assertTrue(args.inspect_only)
        self.assertAlmostEqual(args.spacing, 0.02)
        self.assertAlmostEqual(args.margin, 0.1)
        self.assertEqual(args.output, "x.npz")
        self.assertEqual(args.max_points_per_batch, 321)
        self.assertEqual(args.max_memory_mb, udf_mod.DEFAULT_MAX_MEMORY_MB)
        self.assertIsNone(getattr(args, "load_path", None))
        self.assertFalse(getattr(args, "render", False))
        self.assertIsNone(getattr(args, "artifact_dir", None))


class TestUrdfAssemblyParseArgsAlias(unittest.TestCase):
    def test_output_long_flag(self):
        args = udf_mod.parse_args(["--urdf", "a.urdf", "--output", "y.npz"])
        self.assertEqual(args.output, "y.npz")


REAL_URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "cad_exports"
    / "model_CAD"
    / "scene"
    / "urdf"
    / "中组立0725(1).stp.SLDASM.urdf"
)


@unittest.skipIf(trimesh is None, "trimesh required for real URDF smoke test")
@unittest.skipUnless(REAL_URDF_PATH.is_file(), "real CAD URDF not available in repository")
class TestRealUrdfSmoke(unittest.TestCase):
    def test_load_assembly_from_real_urdf(self):
        assy = udf_mod.load_assembly_from_urdf(REAL_URDF_PATH)

        self.assertIn("base_link", assy.link_names)
        self.assertIn("l2", assy.link_names)
        self.assertIn("l3", assy.link_names)
        self.assertEqual(len(assy.mesh_paths), 3)
        for mesh_path in assy.mesh_paths:
            self.assertTrue(Path(mesh_path).is_file(), msg=mesh_path)
        self.assertTrue(np.all(assy.bbox_max > assy.bbox_min))


class TestEstimateGridMemory(unittest.TestCase):
    def test_float32_three_grids(self):
        shape = (10, 20, 30)
        nbytes = udf_mod.estimate_grid_memory(shape, num_enabled_grids=3, dtype=np.float32)
        self.assertEqual(nbytes, 10 * 20 * 30 * 3 * 4)

    def test_dtype_itemsize_respected(self):
        shape = (2, 2, 2)
        n64 = udf_mod.estimate_grid_memory(shape, 1, dtype=np.float64)
        n32 = udf_mod.estimate_grid_memory(shape, 1, dtype=np.float32)
        self.assertEqual(n64, 2 * 2 * 2 * 8)
        self.assertEqual(n32, 2 * 2 * 2 * 4)


class TestComputeGridDomain(unittest.TestCase):
    def test_origin_shape_cover_expanded_bbox_with_margin(self):
        bbox_min = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        bbox_max = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        spacing = 0.5
        margin = 0.25
        origin, shape = udf_mod.compute_grid_domain(bbox_min, bbox_max, spacing, margin)
        exp_min = bbox_min - margin
        exp_max = bbox_max + margin
        np.testing.assert_allclose(origin, exp_min, rtol=0, atol=1e-6)
        nx, ny, nz = shape
        self.assertEqual(
            tuple(shape),
            (
                int(np.ceil((exp_max[0] - exp_min[0]) / spacing)),
                int(np.ceil((exp_max[1] - exp_min[1]) / spacing)),
                int(np.ceil((exp_max[2] - exp_min[2]) / spacing)),
            ),
        )
        for ax in range(3):
            self.assertGreaterEqual(origin[ax] + shape[ax] * spacing, exp_max[ax] - 1e-5)

    def test_bbox_min_max_unchanged_semantics(self):
        """Field bbox stays the raw assembly AABB; grid origin uses expanded domain only."""
        bbox_min = np.zeros(3, dtype=np.float32)
        bbox_max = np.ones(3, dtype=np.float32)
        origin, shape = udf_mod.compute_grid_domain(bbox_min, bbox_max, 0.25, 0.1)
        self.assertTrue(np.allclose(origin, np.full(3, -0.1)))
        centers = udf_mod.compute_voxel_centers(origin, 0.25, shape)
        self.assertEqual(centers.shape, (*shape, 3))
        self.assertTrue(np.allclose(centers[0, 0, 0], origin + 0.125))


class TestBakeIglOpen3dSdfGrid(unittest.TestCase):
    def _tiny_grid_params(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], float]:
        tri = np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32
        )
        bbox_min = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        bbox_max = np.array([1.5, 1.5, 0.5], dtype=np.float32)
        spacing = 0.5
        margin = 0.0
        origin, shape = udf_mod.compute_grid_domain(bbox_min, bbox_max, spacing, margin)
        return tri, origin, shape, spacing

    def test_bake_igl_returns_nan_when_import_fails(self):
        tri, origin, shape, spacing = self._tiny_grid_params()
        with mock.patch.object(udf_mod, "_import_igl_module", side_effect=ImportError("no igl")):
            grid, reason = udf_mod.bake_igl_sdf_grid(tri, origin, shape, spacing)
        self.assertTrue(np.all(np.isnan(grid)))
        self.assertEqual(grid.shape, shape)
        self.assertIsNotNone(reason)
        self.assertIn("igl", (reason or "").lower())

    def test_bake_open3d_returns_nan_when_import_fails(self):
        tri, origin, shape, spacing = self._tiny_grid_params()
        with mock.patch.object(udf_mod, "_import_open3d_module", side_effect=ImportError("no o3d")):
            grid, reason = udf_mod.bake_open3d_sdf_grid(tri, origin, shape, spacing)
        self.assertTrue(np.all(np.isnan(grid)))
        self.assertEqual(grid.shape, shape)
        self.assertIsNotNone(reason)
        self.assertIn("open3d", (reason or "").lower())

    def test_bake_igl_returns_nan_when_import_raises_oserror(self):
        tri, origin, shape, spacing = self._tiny_grid_params()
        with mock.patch.object(
            udf_mod, "_import_igl_module", side_effect=OSError("dll load failed")
        ):
            grid, reason = udf_mod.bake_igl_sdf_grid(tri, origin, shape, spacing)
        self.assertTrue(np.all(np.isnan(grid)))
        self.assertEqual(grid.shape, shape)
        self.assertIsNotNone(reason)
        self.assertIn("backend unavailable", reason or "")
        self.assertIn("oserror", (reason or "").lower())

    def test_bake_open3d_finite_for_simple_triangle_when_installed(self):
        try:
            udf_mod._import_open3d_module()
        except ImportError:
            self.skipTest("open3d not installed")
        tri, origin, shape, spacing = self._tiny_grid_params()
        grid, reason = udf_mod.bake_open3d_sdf_grid(tri, origin, shape, spacing)
        self.assertIsNone(reason, msg=reason)
        self.assertEqual(grid.shape, shape)
        self.assertTrue(np.all(np.isfinite(grid)))


class TestBakeUdfGrid(unittest.TestCase):
    def test_bake_udf_grid_returns_full_grid_for_bbox(self):
        tri = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        udf_grid = udf_mod.bake_udf_grid(
            tri.reshape(1, 3, 3),
            np.array([0.0, 0.0, -0.5], dtype=np.float32),
            np.array([1.0, 1.0, 0.5], dtype=np.float32),
            spacing=0.5,
            margin=0.0,
        )
        self.assertEqual(udf_grid.shape, (2, 2, 2))

    def test_single_triangle_center_on_plane_near_zero(self):
        tri = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        triangles = tri.reshape(1, 3, 3)
        bbox_min = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        bbox_max = np.array([1.5, 1.5, 0.5], dtype=np.float32)
        spacing = 0.25
        margin = 0.0
        origin, shape = udf_mod.compute_grid_domain(bbox_min, bbox_max, spacing, margin)
        udf_grid = udf_mod.bake_udf_grid(triangles, bbox_min, bbox_max, spacing, margin)
        self.assertEqual(udf_grid.shape, shape)
        self.assertTrue(np.all(np.isfinite(udf_grid)))
        self.assertGreater(float(np.max(udf_grid)), 0.5)
        self.assertLess(float(np.min(udf_grid)), 0.16)

    def test_bake_raises_when_memory_budget_exceeded(self):
        tri = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        triangles = tri.reshape(1, 3, 3)
        bbox_min = np.zeros(3, dtype=np.float32)
        bbox_max = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        with self.assertRaises(udf_mod.GridMemoryBudgetError):
            udf_mod.bake_udf_grid(
                triangles,
                bbox_min,
                bbox_max,
                spacing=0.01,
                margin=0.0,
                max_memory_bytes=100,
            )


@unittest.skipIf(trimesh is None, "trimesh required for bake_distance_field integration")
class TestBakeDistanceField(unittest.TestCase):
    def _toy_urdf(self, tmp: Path) -> Path:
        scene = tmp / "scene"
        urdf_dir = scene / "urdf"
        mesh_dir = scene / "meshes"
        urdf_dir.mkdir(parents=True)
        mesh_dir.mkdir(parents=True)
        tri_mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        tri_mesh.export(str(mesh_dir / "b.stl"))
        urdf_dir.joinpath("t.urdf").write_text(
            """<?xml version="1.0"?>
<robot name="t">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="package://p/meshes/b.stl"/></geometry>
    </visual>
  </link>
</robot>
""",
            encoding="utf-8",
        )
        return urdf_dir / "t.urdf"

    def test_udf_finite_igl_o3d_nan(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_path = self._toy_urdf(tmp)
            assy = udf_mod.load_assembly_from_urdf(urdf_path)
            args = udf_mod.parse_args(
                ["--urdf", str(urdf_path), "--spacing", "0.15", "--margin", "0.05"]
            )
            field = udf_mod.bake_distance_field(assy, args)
        self.assertTrue(np.all(np.isfinite(field.udf_grid)))
        sf = field.status_flags or {}
        self.assertTrue(sf.get("udf_ok"))
        self.assertIn("igl_ok", sf)
        self.assertIn("o3d_ok", sf)
        if sf.get("igl_ok"):
            self.assertTrue(np.all(np.isfinite(field.igl_sdf_grid)))
        else:
            self.assertTrue(np.all(np.isnan(field.igl_sdf_grid)))
            self.assertTrue(
                any("igl" in str(r).lower() for r in (field.failure_reasons or [])),
                msg=field.failure_reasons,
            )
        if sf.get("o3d_ok"):
            self.assertTrue(np.all(np.isfinite(field.o3d_sdf_grid)))
        else:
            self.assertTrue(np.all(np.isnan(field.o3d_sdf_grid)))
            self.assertTrue(
                any("open3d" in str(r).lower() for r in (field.failure_reasons or [])),
                msg=field.failure_reasons,
            )
        self.assertTrue(np.allclose(field.bbox_min, assy.bbox_min))
        self.assertTrue(np.allclose(field.bbox_max, assy.bbox_max))
        self.assertIsInstance(field.build_config, dict)

    def test_bake_save_and_load_roundtrip_keeps_udf_finite_and_sdf_placeholders_nan(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_path = self._toy_urdf(tmp)
            assy = udf_mod.load_assembly_from_urdf(urdf_path)
            args = udf_mod.parse_args(
                ["--urdf", str(urdf_path), "--spacing", "0.15", "--margin", "0.05"]
            )
            field = udf_mod.bake_distance_field(assy, args)
            out_path = tmp / "toy_field.npz"
            udf_mod.save_distance_field(out_path, field)
            loaded = udf_mod.load_distance_field(out_path)

        self.assertTrue(np.all(np.isfinite(loaded.udf_grid)))
        self.assertFalse(np.isnan(loaded.udf_grid).any())
        lsf = loaded.status_flags or {}
        if lsf.get("igl_ok"):
            self.assertTrue(np.all(np.isfinite(loaded.igl_sdf_grid)))
        else:
            self.assertTrue(np.all(np.isnan(loaded.igl_sdf_grid)))
        if lsf.get("o3d_ok"):
            self.assertTrue(np.all(np.isfinite(loaded.o3d_sdf_grid)))
        else:
            self.assertTrue(np.all(np.isnan(loaded.o3d_sdf_grid)))

    def test_bake_distance_field_succeeds_when_both_sdf_imports_fail(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_path = self._toy_urdf(tmp)
            assy = udf_mod.load_assembly_from_urdf(urdf_path)
            args = udf_mod.parse_args(
                ["--urdf", str(urdf_path), "--spacing", "0.15", "--margin", "0.05"]
            )
            with mock.patch.object(udf_mod, "_import_igl_module", side_effect=ImportError("no igl")):
                with mock.patch.object(
                    udf_mod, "_import_open3d_module", side_effect=ImportError("no o3d")
                ):
                    field = udf_mod.bake_distance_field(assy, args)
        self.assertTrue(np.all(np.isfinite(field.udf_grid)))
        sf = field.status_flags or {}
        self.assertTrue(sf.get("udf_ok"))
        self.assertFalse(sf.get("igl_ok"))
        self.assertFalse(sf.get("o3d_ok"))
        self.assertTrue(np.all(np.isnan(field.igl_sdf_grid)))
        self.assertTrue(np.all(np.isnan(field.o3d_sdf_grid)))
        self.assertGreaterEqual(len(field.failure_reasons or []), 2)

    def test_bake_distance_field_udf_only_skips_optional_backends(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_path = self._toy_urdf(tmp)
            assy = udf_mod.load_assembly_from_urdf(urdf_path)
            args = udf_mod.parse_args(
                [
                    "--urdf",
                    str(urdf_path),
                    "--spacing",
                    "0.05",
                    "--margin",
                    "0.05",
                    "--max-memory-mb",
                    "0.002",
                    "--udf-only",
                ]
            )
            with mock.patch.object(
                udf_mod, "_import_igl_module", side_effect=AssertionError("should not import igl")
            ):
                with mock.patch.object(
                    udf_mod,
                    "_import_open3d_module",
                    side_effect=AssertionError("should not import open3d"),
                ):
                    field = udf_mod.bake_distance_field(assy, args)
        self.assertTrue(np.all(np.isfinite(field.udf_grid)))
        self.assertTrue(np.all(np.isnan(field.igl_sdf_grid)))
        self.assertTrue(np.all(np.isnan(field.o3d_sdf_grid)))
        sf = field.status_flags or {}
        self.assertTrue(sf.get("udf_ok"))
        self.assertFalse(sf.get("igl_ok"))
        self.assertFalse(sf.get("o3d_ok"))
        self.assertIn("igl_sdf: skipped (--udf-only)", field.failure_reasons or [])
        self.assertIn("open3d_sdf: skipped (--udf-only)", field.failure_reasons or [])
        self.assertEqual((field.build_config or {}).get("udf_only"), True)

    def test_bake_distance_field_raises_when_three_grid_budget_exceeded(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_path = self._toy_urdf(tmp)
            assy = udf_mod.load_assembly_from_urdf(urdf_path)
            args = udf_mod.parse_args(
                [
                    "--urdf",
                    str(urdf_path),
                    "--spacing",
                    "0.05",
                    "--margin",
                    "0.05",
                    "--max-memory-mb",
                    "0.0001",
                ]
            )
            with self.assertRaises(udf_mod.GridMemoryBudgetError):
                udf_mod.bake_distance_field(assy, args)

    def test_main_udf_only_writes_npz_with_finite_udf(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_path = self._toy_urdf(tmp)
            out_path = tmp / "cli_field.npz"
            exit_code = udf_mod.main(
                [
                    "--urdf",
                    str(urdf_path),
                    "--spacing",
                    "0.15",
                    "--margin",
                    "0.05",
                    "--udf-only",
                    "--output",
                    str(out_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_path.is_file())
            loaded = udf_mod.load_distance_field(out_path)

        self.assertTrue(np.all(np.isfinite(loaded.udf_grid)))
        self.assertFalse(np.isnan(loaded.udf_grid).any())
        self.assertTrue(np.all(np.isnan(loaded.igl_sdf_grid)))
        self.assertTrue(np.all(np.isnan(loaded.o3d_sdf_grid)))
        lsf = loaded.status_flags or {}
        self.assertTrue(lsf.get("udf_ok"))
        self.assertFalse(lsf.get("igl_ok"))
        self.assertFalse(lsf.get("o3d_ok"))
        self.assertIn("igl_sdf: skipped (--udf-only)", loaded.failure_reasons or [])
        self.assertIn("open3d_sdf: skipped (--udf-only)", loaded.failure_reasons or [])


class TestPrepareSliceForPlot(unittest.TestCase):
    def test_prepare_slice_for_plot_masks_nan(self):
        g = np.ones((3, 3, 3), dtype=np.float32)
        g[1, 1, 1] = np.nan
        sl = udf_mod.prepare_slice_for_plot(g, axis=2, index=1)
        self.assertIsInstance(sl, np.ma.MaskedArray)
        self.assertTrue(bool(sl.mask[1, 1]))
        self.assertFalse(bool(sl.mask[0, 0]))

    def test_prepare_slice_for_plot_masks_inf(self):
        g = np.zeros((2, 2, 2), dtype=np.float32)
        g[0, 0, 0] = np.inf
        sl = udf_mod.prepare_slice_for_plot(g, axis=0, index=0)
        self.assertIsInstance(sl, np.ma.MaskedArray)
        self.assertTrue(bool(sl.mask[0, 0]))


class TestFieldVisualizationRenders(unittest.TestCase):
    def test_render_slice_comparison_writes_pngs(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            paths = udf_mod.render_slice_comparison(field, out, axis=2, index=0)
            self.assertGreater(len(paths), 0)
            for p in paths:
                self.assertTrue(p.is_file(), msg=str(p))
                self.assertEqual(p.suffix.lower(), ".png")

    def test_render_sample_point_comparison_writes_png(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            paths = udf_mod.render_sample_point_comparison(
                field, out, n_samples=32, seed=1
            )
            self.assertGreaterEqual(len(paths), 1)
            for p in paths:
                self.assertTrue(p.is_file(), msg=str(p))
                self.assertEqual(p.suffix.lower(), ".png")

    def test_render_sign_diagnostics_writes_pngs_minimal_field(self):
        udf = np.full((2, 2, 2), 0.5, dtype=np.float32)
        igl = np.array(
            [[[-1.0, 1.0], [1.0, -1.0]], [[1.0, -1.0], [-1.0, 1.0]]],
            dtype=np.float32,
        )
        o3d = -igl.copy()
        field = udf_mod.DistanceField(
            origin=np.zeros(3, dtype=np.float32),
            spacing=np.float32(0.1),
            udf_grid=udf,
            igl_sdf_grid=igl,
            o3d_sdf_grid=o3d,
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.ones(3, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            paths = udf_mod.render_sign_diagnostics(
                field, out, udf_far_threshold=0.15
            )
            self.assertGreaterEqual(len(paths), 2)
            for p in paths:
                self.assertTrue(p.is_file(), msg=str(p))
                self.assertEqual(p.suffix.lower(), ".png")

    def test_render_sign_diagnostics_default_threshold_uses_twice_spacing(self):
        udf = np.full((2, 2, 2), 1.0, dtype=np.float32)
        igl = np.full((2, 2, 2), -0.01, dtype=np.float32)
        o3d = np.full((2, 2, 2), np.nan, dtype=np.float32)
        field = udf_mod.DistanceField(
            origin=np.zeros(3, dtype=np.float32),
            spacing=np.float32(0.05),
            udf_grid=udf,
            igl_sdf_grid=igl,
            o3d_sdf_grid=o3d,
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.ones(3, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            paths = udf_mod.render_sign_diagnostics(field, out, udf_far_threshold=None)
            self.assertGreaterEqual(len(paths), 1)
            for p in paths:
                self.assertTrue(p.is_file(), msg=str(p))

    def test_render_plots_cli_smoke_from_disk_npz(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            npz_path = root / "global_field_compare.npz"
            out = root / "plots"
            udf_mod.save_distance_field(npz_path, field)

            rc = udf_mod.main(
                [
                    "--render-plots-npz",
                    str(npz_path),
                    "--render-plots-out",
                    str(out),
                ]
            )

            self.assertEqual(rc, 0)
            names = sorted(p.name for p in out.glob("*.png"))
            self.assertGreaterEqual(len(names), 7)
            self.assertIn("sample_points_xy_comparison.png", names)
            self.assertIn("sign_diag_igl_o3d_mismatch_z1.png", names)
            self.assertIn("sign_diag_neg_sdf_large_udf_z1.png", names)
            self.assertIn("slice_axis2_z_idx1_udf.png", names)
            self.assertIn("slice_axis2_z_idx1_abs_diff_igl_udf.png", names)
            self.assertIn("slice_axis2_z_idx1_abs_diff_o3d_udf.png", names)

    def test_sign_mismatch_ignores_zero_vs_nonzero(self):
        igl = np.array([[[0.0, 1.0]]], dtype=np.float32)
        o3d = np.array([[[1.0, -1.0]]], dtype=np.float32)
        mismatch = udf_mod._sign_mismatch_mask(igl, o3d)
        expected = np.array([[[False, True]]], dtype=bool)
        self.assertTrue(np.array_equal(mismatch, expected))

    def test_render_sign_diagnostics_title_counts_current_slice_only(self):
        titles: list[str] = []

        class _FakeAxes:
            def imshow(self, *_args, **_kwargs):
                return object()

            def set_title(self, title):
                titles.append(title)

        class _FakeFigure:
            def __init__(self):
                self.ax = _FakeAxes()

            def colorbar(self, *_args, **_kwargs):
                return None

            def tight_layout(self):
                return None

            def savefig(self, path, **_kwargs):
                Path(path).write_bytes(b"fake")

        class _FakePlt:
            def subplots(self, *args, **_kwargs):
                fig = _FakeFigure()
                return fig, fig.ax

            def close(self, *_args, **_kwargs):
                return None

        # z=0 slice: exactly one sign disagreement; full volume has more (regression: title
        # must count slice voxels, not np.sum over the whole volume).
        udf = np.ones((2, 2, 2), dtype=np.float32)
        igl = np.ones((2, 2, 2), dtype=np.float32)
        o3d = np.ones((2, 2, 2), dtype=np.float32)
        igl[:, :, 1] = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
        o3d[:, :, 0] = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
        o3d[:, :, 1] = np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
        field = udf_mod.DistanceField(
            origin=np.zeros(3, dtype=np.float32),
            spacing=np.float32(0.1),
            udf_grid=udf,
            igl_sdf_grid=igl,
            o3d_sdf_grid=o3d,
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.ones(3, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(udf_mod, "_matplotlib_pyplot_agg", return_value=_FakePlt()):
                udf_mod.render_sign_diagnostics(field, Path(raw), axis=2, index=0)

        counts = [
            int(m.group(1))
            for title in titles
            if (m := re.search(r"voxels=(\d+)", title)) is not None
        ]
        self.assertIn(1, counts, msg=titles)
        # Full 3D sign-mismatch count is 5; titles must use the z=0 slice (1), not the volume.
        self.assertNotIn(5, counts, msg=titles)

    def test_render_sample_point_comparison_filters_nonfinite_udf_values(self):
        scatter_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        class _FakeAxes:
            def scatter(self, x, y, c, **_kwargs):
                scatter_calls.append((np.asarray(x), np.asarray(y), np.asarray(c)))
                return object()

            def set_title(self, _title):
                return None

            def set_aspect(self, *_args, **_kwargs):
                return None

            def set_xlabel(self, *_args, **_kwargs):
                return None

            def set_ylabel(self, *_args, **_kwargs):
                return None

        class _AxesGrid:
            def __init__(self):
                self._items = [[_FakeAxes(), _FakeAxes()], [_FakeAxes(), _FakeAxes()]]

            def __getitem__(self, item):
                row, col = item
                return self._items[row][col]

            def ravel(self):
                return [ax for row in self._items for ax in row]

        class _FakeFigure:
            def colorbar(self, *_args, **_kwargs):
                return None

            def tight_layout(self):
                return None

            def savefig(self, path, **_kwargs):
                Path(path).write_bytes(b"fake")

        class _FakePlt:
            def subplots(self, *args, **_kwargs):
                return _FakeFigure(), _AxesGrid()

            def close(self, *_args, **_kwargs):
                return None

        field = udf_mod.DistanceField(
            origin=np.zeros(3, dtype=np.float32),
            spacing=np.float32(1.0),
            udf_grid=np.full((2, 2, 2), np.nan, dtype=np.float32),
            igl_sdf_grid=np.zeros((2, 2, 2), dtype=np.float32),
            o3d_sdf_grid=np.zeros((2, 2, 2), dtype=np.float32),
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.ones(3, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(udf_mod, "_matplotlib_pyplot_agg", return_value=_FakePlt()):
                paths = udf_mod.render_sample_point_comparison(
                    field, Path(raw), n_samples=8, seed=0
                )

        self.assertEqual(len(paths), 1)
        udf_x, udf_y, udf_c = scatter_calls[0]
        self.assertEqual(int(udf_x.size), 0)
        self.assertEqual(int(udf_y.size), 0)
        self.assertEqual(int(udf_c.size), 0)


class TestDistanceFieldFailureReasonsValidation(unittest.TestCase):
    def test_load_raises_when_failure_reasons_contains_non_string_item(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.npz"
            udf_mod.save_distance_field(path, field)
            with np.load(path, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
            payload["failure_reasons"] = np.array('["ok", 123]', dtype=np.str_)
            np.savez_compressed(path, **payload)

            with self.assertRaises(ValueError):
                udf_mod.load_distance_field(path)


class TestParseArgsRenderPlots(unittest.TestCase):
    def test_parse_args_accepts_render_plots_flags_without_urdf(self):
        args = udf_mod.parse_args(
            ["--render-plots-npz", "field.npz", "--render-plots-out", "plots"]
        )
        self.assertEqual(args.urdf, udf_mod.DEFAULT_URDF_ARG)
        self.assertEqual(args.render_plots_npz, "field.npz")
        self.assertEqual(args.render_plots_out, "plots")

    def test_parse_args_uses_default_scene_urdf_when_available(self):
        args = udf_mod.parse_args([])
        self.assertEqual(args.urdf, udf_mod.DEFAULT_URDF_ARG)


@unittest.skipIf(trimesh is None, "trimesh required for CLI integration tests")
class TestCliUnifiedTask7(unittest.TestCase):
    def _toy_urdf(self, tmp: Path) -> Path:
        scene = tmp / "scene"
        urdf_dir = scene / "urdf"
        mesh_dir = scene / "meshes"
        urdf_dir.mkdir(parents=True)
        mesh_dir.mkdir(parents=True)
        tri_mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        tri_mesh.export(str(mesh_dir / "b.stl"))
        urdf_dir.joinpath("t.urdf").write_text(
            """<?xml version="1.0"?>
<robot name="t">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="package://p/meshes/b.stl"/></geometry>
    </visual>
  </link>
</robot>
""",
            encoding="utf-8",
        )
        return urdf_dir / "t.urdf"

    def test_parse_args_supports_bake_and_render_modes(self):
        args = udf_mod.parse_args(
            [
                "--urdf",
                "scene.urdf",
                "--load",
                "other.npz",
                "--spacing",
                "0.2",
                "--margin",
                "0.05",
                "--output",
                "out.npz",
                "--render",
                "--artifact-dir",
                "my_artifacts",
            ]
        )
        self.assertEqual(args.urdf, "scene.urdf")
        self.assertEqual(args.load_path, "other.npz")
        self.assertAlmostEqual(args.spacing, 0.2)
        self.assertAlmostEqual(args.margin, 0.05)
        self.assertEqual(args.output, "out.npz")
        self.assertTrue(args.render)
        self.assertEqual(args.artifact_dir, "my_artifacts")

    def test_main_load_and_render_writes_pngs(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            npz_path = root / "loaded.npz"
            udf_mod.save_distance_field(npz_path, field)
            rc = udf_mod.main(
                ["--load", str(npz_path), "--render"]
            )
            self.assertEqual(rc, 0)
            default_dir = root / "loaded_plots"
            self.assertTrue(default_dir.is_dir())
            pngs = list(default_dir.glob("*.png"))
            self.assertGreaterEqual(len(pngs), 1)

    def test_main_urdf_output_and_render_writes_npz_and_pngs(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_path = self._toy_urdf(tmp)
            out_npz = tmp / "baked.npz"
            rc = udf_mod.main(
                [
                    "--urdf",
                    str(urdf_path),
                    "--spacing",
                    "0.15",
                    "--margin",
                    "0.05",
                    "--udf-only",
                    "--output",
                    str(out_npz),
                    "--render",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_npz.is_file())
            plot_dir = tmp / "baked_plots"
            self.assertTrue(plot_dir.is_dir())
            self.assertGreaterEqual(len(list(plot_dir.glob("*.png"))), 1)

    def test_main_render_respects_explicit_artifact_dir(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            npz_path = root / "f.npz"
            udf_mod.save_distance_field(npz_path, field)
            explicit = root / "custom_out"
            rc = udf_mod.main(
                [
                    "--load",
                    str(npz_path),
                    "--render",
                    "--artifact-dir",
                    str(explicit),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(explicit.is_dir())
            self.assertGreaterEqual(len(list(explicit.glob("*.png"))), 1)
            self.assertFalse((root / "f_plots").exists())

    def test_main_load_path_does_not_require_urdf(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            npz_path = root / "only.npz"
            udf_mod.save_distance_field(npz_path, field)
            rc = udf_mod.main(["--load", str(npz_path)])
            self.assertEqual(rc, 0)

    def test_main_urdf_only_no_action_returns_error(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            urdf_path = self._toy_urdf(tmp)
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                rc = udf_mod.main(["--urdf", str(urdf_path)])
            self.assertEqual(rc, 2)
            self.assertIn("--output", stderr.getvalue())
            self.assertIn("--render", stderr.getvalue())

    def test_main_render_default_artifact_dir_prefers_output_over_load(self):
        field = _sample_field()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            load_npz = root / "loaded.npz"
            out_npz = root / "fresh_output.npz"
            udf_mod.save_distance_field(load_npz, field)
            rc = udf_mod.main(
                [
                    "--load",
                    str(load_npz),
                    "--output",
                    str(out_npz),
                    "--render",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((root / "fresh_output_plots").is_dir())
            self.assertGreaterEqual(len(list((root / "fresh_output_plots").glob("*.png"))), 1)
            self.assertFalse((root / "loaded_plots").exists())

    def test_main_explicit_load_wins_over_render_plots_npz_alias(self):
        field_a = _sample_field()
        field_b = _sample_field(
            udf_grid=np.full((2, 2, 2), 7.0, dtype=np.float32),
            igl=np.full((2, 2, 2), 17.0, dtype=np.float32),
            o3d=np.full((2, 2, 2), 27.0, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            explicit_npz = root / "explicit.npz"
            alias_npz = root / "alias.npz"
            artifact_dir = root / "chosen"
            udf_mod.save_distance_field(explicit_npz, field_a)
            udf_mod.save_distance_field(alias_npz, field_b)
            rc = udf_mod.main(
                [
                    "--load",
                    str(explicit_npz),
                    "--render-plots-npz",
                    str(alias_npz),
                    "--artifact-dir",
                    str(artifact_dir),
                    "--render",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(artifact_dir.is_dir())
            self.assertGreaterEqual(len(list(artifact_dir.glob("*.png"))), 1)
            self.assertFalse((root / "alias_plots").exists())


if __name__ == "__main__":
    unittest.main()
