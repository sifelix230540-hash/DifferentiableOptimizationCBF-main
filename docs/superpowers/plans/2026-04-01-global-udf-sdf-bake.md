# Global UDF/SDF Bake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为中组立 URDF 装配体离线生成覆盖全局包围盒的统一 `.npz` 距离场文件，其中 `UDF` 作为主结果，同时额外保存 `libigl SDF` 与 `Open3D SDF` 用于符号可靠性对比，并提供三线性插值查询与基础可视化。

**Architecture:** 以 `CBF_experiment/active/pybullet/4_1_udf.py` 作为单文件入口脚本，内部按“URDF 装配 -> 规则网格生成 -> UDF 烘焙 -> 可选 SDF 烘焙 -> `.npz` 保存/加载 -> 查询 -> 可视化”组织。测试放在 `tests/test_global_udf_bake.py`，优先覆盖纯 Python 逻辑与文件格式，不依赖大型真实网格完成单元验证。

**Tech Stack:** Python, `numpy`, `trimesh`, `matplotlib`, optional `libigl`, optional `open3d`, `unittest`, `pytest`

---

## File Structure

- Modify: `CBF_experiment/active/pybullet/4_1_udf.py`
  - 负责命令行入口、URDF 装配、网格场烘焙、`.npz` 保存/加载、查询接口、可视化输出
- Create: `tests/test_global_udf_bake.py`
  - 负责 UDF 文件格式、网格坐标定义、三线性插值、越界行为、`NaN` 传播等测试
- Read/Use: `assets/cad_exports/model_CAD/scene/urdf/中组立0725(1).stp.SLDASM.urdf`
  - 真实装配输入
- Runtime output only: `artifacts/`
  - 保存 `.npz`、切片图、采样点对比图；不作为计划中的源码文件

## Task 1: 搭建最小测试骨架

**Files:**
- Modify: `CBF_experiment/active/pybullet/4_1_udf.py`
- Create: `tests/test_global_udf_bake.py`

- [ ] **Step 1: 写失败测试，锁定体素中心和插值约定**

```python
import importlib
import numpy as np

udf_mod = importlib.import_module("CBF_experiment.active.pybullet.4_1_udf")


def test_voxel_centers_follow_origin_plus_half_spacing():
    origin = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    spacing = 0.5

    centers = udf_mod.compute_voxel_centers(origin, spacing, shape=(2, 1, 1))

    assert np.allclose(centers[:, 0, 0], [[1.25, 2.25, 3.25], [1.75, 2.25, 3.25]])
```

- [ ] **Step 2: 运行单测并确认失败**

Run: `pytest tests/test_global_udf_bake.py -v`

Expected: FAIL，提示目标函数或模块尚不存在

- [ ] **Step 3: 在 `4_1_udf.py` 加入最小纯函数框架**

```python
def compute_voxel_centers(origin, spacing, shape):
    nx, ny, nz = shape
    grid = np.indices((nx, ny, nz), dtype=np.float32)
    centers = origin.reshape(3, 1, 1, 1) + (grid + 0.5) * float(spacing)
    return np.moveaxis(centers, 0, -1)
```

- [ ] **Step 4: 再补一条插值失败测试**

```python
def test_trilinear_query_returns_expected_value():
    field = udf_mod.DistanceField(
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        spacing=np.float32(1.0),
        udf_grid=np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        igl_sdf_grid=np.zeros((2, 2, 2), dtype=np.float32),
        o3d_sdf_grid=np.zeros((2, 2, 2), dtype=np.float32),
        bbox_min=np.zeros(3, dtype=np.float32),
        bbox_max=np.ones(3, dtype=np.float32),
    )
    value = field.query_single(np.array([1.0, 1.0, 1.0], dtype=np.float32), kind="udf", clip=True)
    assert np.isfinite(value)
```

- [ ] **Step 5: 运行测试并确认仍失败在预期点**

Run: `pytest tests/test_global_udf_bake.py -v`

Expected: FAIL，缺少 `DistanceField` 或 `query_single`

## Task 2: 实现场文件数据结构与查询接口

**Files:**
- Modify: `CBF_experiment/active/pybullet/4_1_udf.py`
- Test: `tests/test_global_udf_bake.py`

- [ ] **Step 1: 写失败测试，覆盖保存/加载字段完整性**

```python
def test_save_and_load_distance_field_round_trips_required_arrays(tmp_path):
    field = udf_mod.DistanceField(
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        spacing=np.float32(0.25),
        udf_grid=np.ones((2, 2, 2), dtype=np.float32),
        igl_sdf_grid=np.full((2, 2, 2), np.nan, dtype=np.float32),
        o3d_sdf_grid=np.full((2, 2, 2), np.nan, dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        bbox_max=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    out_path = tmp_path / "field.npz"
    udf_mod.save_distance_field(out_path, field)
    loaded = udf_mod.load_distance_field(out_path)
    assert loaded.udf_grid.shape == (2, 2, 2)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_global_udf_bake.py::test_save_and_load_distance_field_round_trips_required_arrays -v`

Expected: FAIL，缺少保存/加载接口

- [ ] **Step 3: 实现 `DistanceField`、保存/加载、查询与越界行为**

```python
@dataclass
class DistanceField:
    origin: np.ndarray
    spacing: np.float32
    udf_grid: np.ndarray
    igl_sdf_grid: np.ndarray
    o3d_sdf_grid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    status_flags: dict | None = None
    failure_reasons: dict | None = None
```

- [ ] **Step 4: 运行测试并补充 `NaN` 邻域传播、越界和元数据 round-trip 测试**

Run: `pytest tests/test_global_udf_bake.py -v`

Expected: PASS 新增测试；若失败，仅修正最小实现

- [ ] **Step 5: 提交这个最小可用查询层**

```bash
git add CBF_experiment/active/pybullet/4_1_udf.py tests/test_global_udf_bake.py
git commit -m "feat: add distance field file format and query helpers"
```

## Task 3: 实现 URDF 装配与全局包围盒计算

**Files:**
- Modify: `CBF_experiment/active/pybullet/4_1_udf.py`
- Test: `tests/test_global_udf_bake.py`

- [ ] **Step 1: 写失败测试，验证简单 fixed-joint 装配结果**

```python
def test_build_assembly_mesh_applies_fixed_joint_translation(tmp_path):
    urdf_path = tmp_path / "toy.urdf"
    # 写入一个 base + child fixed joint 的最小 URDF
    assembly = udf_mod.load_assembly_from_urdf(urdf_path)
    assert assembly["bbox_max"][0] > assembly["bbox_min"][0]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_global_udf_bake.py::test_build_assembly_mesh_applies_fixed_joint_translation -v`

Expected: FAIL，缺少 URDF 装配逻辑

- [ ] **Step 3: 实现最小装配逻辑**

```python
def load_assembly_from_urdf(urdf_path):
    # 解析 link mesh、scale、fixed joint origin
    # 递归/拓扑排序累积根到各 link 的刚体变换
    # 读取并变换各 mesh，返回 triangles / bbox / metadata
    ...
```

- [ ] **Step 4: 同步加入最小 CLI 桩，支持从本任务开始使用真实命令**

```python
def parse_args(argv=None):
    # 至少先支持 --urdf、--inspect-only、--spacing、--margin、--output
    ...
```

- [ ] **Step 5: 用真实 URDF 做一次只读冒烟检查**

Run: `python CBF_experiment/active/pybullet/4_1_udf.py --urdf "assets/cad_exports/model_CAD/scene/urdf/中组立0725(1).stp.SLDASM.urdf" --inspect-only`

Expected: 输出 link 名称、mesh 路径、全局 `bbox_min/max`，不报错

- [ ] **Step 6: 提交装配层**

```bash
git add CBF_experiment/active/pybullet/4_1_udf.py tests/test_global_udf_bake.py
git commit -m "feat: assemble scene meshes from urdf for field baking"
```

## Task 4: 实现全局 UDF 主链路

**Files:**
- Modify: `CBF_experiment/active/pybullet/4_1_udf.py`
- Test: `tests/test_global_udf_bake.py`

- [ ] **Step 1: 写失败测试，锁定规则网格和 UDF 输出形状**

```python
def test_bake_udf_grid_returns_full_grid_for_bbox():
    triangles = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ], dtype=np.float32)
    result = udf_mod.bake_udf_grid(
        triangles=triangles,
        bbox_min=np.array([0.0, 0.0, -0.5], dtype=np.float32),
        bbox_max=np.array([1.0, 1.0, 0.5], dtype=np.float32),
        spacing=0.5,
        margin=0.0,
    )
    assert result.shape == (2, 2, 2)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_global_udf_bake.py::test_bake_udf_grid_returns_full_grid_for_bbox -v`

Expected: FAIL，缺少 `bake_udf_grid`

- [ ] **Step 3: 实现最小 UDF 烘焙与内存预估**

```python
def estimate_grid_memory(shape, num_enabled_grids, dtype=np.float32):
    return int(np.prod(shape)) * np.dtype(dtype).itemsize * num_enabled_grids


def bake_udf_grid(...):
    # 生成体素中心
    # 分 batch 计算点到三角面的 unsigned distance
    # 返回 float32 的 udf_grid
    ...
```

- [ ] **Step 4: 实现顶层编排函数，把网格、主结果和元数据拼成统一场对象**

```python
def bake_distance_field(assembly, args):
    # 1. 由 bbox/margin/spacing 生成规则网格定义
    # 2. 调用 bake_udf_grid
    # 3. 先用 NaN grid 占位 igl/o3d
    # 4. 组装 DistanceField 与 status_flags/failure_reasons/build_config
    ...
```

- [ ] **Step 5: 运行整组测试，再做一次真实小分辨率烘焙**

Run: `pytest tests/test_global_udf_bake.py -v`

Run: `python CBF_experiment/active/pybullet/4_1_udf.py --urdf "assets/cad_exports/model_CAD/scene/urdf/中组立0725(1).stp.SLDASM.urdf" --spacing 0.2 --margin 0.1 --output "artifacts/global_udf_debug.npz" --udf-only`

Expected: 生成 `.npz`，且 `udf_grid` 无 `NaN`

- [ ] **Step 6: 提交主链路**

```bash
git add CBF_experiment/active/pybullet/4_1_udf.py tests/test_global_udf_bake.py
git commit -m "feat: bake global udf grid for assembly meshes"
```

## Task 5: 增加可选的 libigl / Open3D SDF 烘焙

**Files:**
- Modify: `CBF_experiment/active/pybullet/4_1_udf.py`
- Test: `tests/test_global_udf_bake.py`

- [ ] **Step 1: 写失败测试，锁定缺依赖时的降级行为**

```python
def test_missing_sdf_backend_falls_back_to_nan_grid():
    grid, reason = udf_mod.build_nan_grid(shape=(2, 2, 2), reason="missing backend")
    assert np.isnan(grid).all()
    assert "missing" in reason
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_global_udf_bake.py::test_missing_sdf_backend_falls_back_to_nan_grid -v`

Expected: FAIL，缺少降级辅助函数

- [ ] **Step 3: 实现两个可选后端**

```python
def bake_igl_sdf_grid(...):
    # try import igl
    # 成功则优先用 pseudonormal 模式批量 signed_distance
    # 若接口支持，再单独暴露 winding-number 选项
    # 失败则返回 NaN grid + failure reason
    ...


def bake_open3d_sdf_grid(...):
    # try import open3d as o3d
    # 用 RaycastingScene.compute_signed_distance
    # 失败则返回 NaN grid + failure reason
    ...
```

- [ ] **Step 4: 扩展顶层编排函数，把两类 SDF 接入统一输出**

```python
def bake_distance_field(assembly, args):
    udf_grid = bake_udf_grid(...)
    igl_sdf_grid, igl_reason = bake_igl_sdf_grid(...)
    o3d_sdf_grid, o3d_reason = bake_open3d_sdf_grid(...)
    # 更新 status_flags / failure_reasons
    ...
```

- [ ] **Step 5: 运行测试并做一次真实后端探测**

Run: `pytest tests/test_global_udf_bake.py -v`

Run: `python CBF_experiment/active/pybullet/4_1_udf.py --urdf "assets/cad_exports/model_CAD/scene/urdf/中组立0725(1).stp.SLDASM.urdf" --spacing 0.2 --margin 0.1 --output "artifacts/global_field_compare.npz"`

Expected: 无论后端是否安装都能输出 `.npz`；若失败则 `status_flags` 为 false 且 `failure_reasons` 有说明

- [ ] **Step 6: 提交可选 SDF 后端**

```bash
git add CBF_experiment/active/pybullet/4_1_udf.py tests/test_global_udf_bake.py
git commit -m "feat: add optional sdf backends for field comparison"
```

## Task 6: 增加切片图和采样点可视化

**Files:**
- Modify: `CBF_experiment/active/pybullet/4_1_udf.py`
- Test: `tests/test_global_udf_bake.py`

- [ ] **Step 1: 写失败测试，锁定 `NaN` 掩膜不会被当零值绘制**

```python
def test_prepare_slice_plot_masks_nan_values():
    grid = np.array([[[1.0], [np.nan]]], dtype=np.float32)
    masked = udf_mod.prepare_slice_for_plot(grid[:, :, 0])
    assert np.ma.isMaskedArray(masked)
    assert masked.mask[0, 1]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_global_udf_bake.py::test_prepare_slice_plot_masks_nan_values -v`

Expected: FAIL，缺少绘图预处理

- [ ] **Step 3: 实现最小可视化输出**

```python
def render_slice_comparison(field, output_dir):
    # udf / igl / o3d / abs diff 切片图
    ...


def render_sample_point_comparison(field, output_dir, num_points=256):
    # 采样点数值散点或直方图
    ...
```

- [ ] **Step 4: 增加符号稳定性诊断图，和 spec 保持一致**

```python
def render_sign_diagnostics(field, output_dir, udf_far_threshold=None):
    # 标出 igl/o3d 符号不一致区域
    # 标出 sdf < 0 且 udf > threshold 的异常区域
    ...
```

- [ ] **Step 5: 用真实 `.npz` 生成图片**

Run: `python CBF_experiment/active/pybullet/4_1_udf.py --load "artifacts/global_field_compare.npz" --render`

Expected: 在 `artifacts/` 下生成切片图、采样点对比图和符号诊断图，`NaN` 区域有独立颜色或掩膜

- [ ] **Step 6: 提交可视化层**

```bash
git add CBF_experiment/active/pybullet/4_1_udf.py tests/test_global_udf_bake.py
git commit -m "feat: add distance field comparison visualizations"
```

## Task 7: 打通 CLI 与最终验收

**Files:**
- Modify: `CBF_experiment/active/pybullet/4_1_udf.py`
- Test: `tests/test_global_udf_bake.py`

- [ ] **Step 1: 写失败测试，锁定命令行参数解析最小行为**

```python
def test_parse_args_supports_bake_and_render_modes():
    args = udf_mod.parse_args([
        "--urdf", "scene.urdf",
        "--spacing", "0.1",
        "--output", "field.npz",
        "--artifact-dir", "artifacts",
    ])
    assert args.urdf == "scene.urdf"
    assert args.output.endswith("field.npz")
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_global_udf_bake.py::test_parse_args_supports_bake_and_render_modes -v`

Expected: FAIL，缺少 `parse_args`

- [ ] **Step 3: 实现 CLI 主流程**

```python
def main():
    args = parse_args()
    if args.load:
        field = load_distance_field(args.load)
    else:
        assembly = load_assembly_from_urdf(args.urdf)
        field = bake_distance_field(assembly, args)
        save_distance_field(args.output, field)
    if args.render:
        render_slice_comparison(field, args.artifact_dir)
```

- [ ] **Step 4: 做最终验证**

Run: `pytest tests/test_global_udf_bake.py -v`

Run: `python CBF_experiment/active/pybullet/4_1_udf.py --urdf "assets/cad_exports/model_CAD/scene/urdf/中组立0725(1).stp.SLDASM.urdf" --spacing 0.2 --margin 0.1 --output "artifacts/global_field_compare.npz" --render`

Expected:
- 测试通过
- 生成 `.npz`
- `.npz` 至少含 `origin`、`spacing`、`shape`、`udf_grid`、`igl_sdf_grid`、`o3d_sdf_grid`、`bbox_min`、`bbox_max`
- 生成至少一张切片图

- [ ] **Step 5: 提交最终 CLI 验收版本**

```bash
git add CBF_experiment/active/pybullet/4_1_udf.py tests/test_global_udf_bake.py
git commit -m "feat: add global distance field bake cli for assembly meshes"
```

## Notes For Execution

- 先保证 `UDF` 主链路通，再接 `libigl`/`Open3D`
- 所有真实装配验证都用较粗分辨率起步，避免首次运行就内存爆炸
- 默认内存阈值和报错文案要在实现时写死，例如超限时明确提示用户调大 `spacing` 或减小 `margin`
- 若 `4_1_udf.py` 变得过大，再在执行阶段评估是否拆出辅助模块，但不要在计划阶段提前做无必要重构
- 本项目已有 `unittest`/`pytest` 风格测试，优先保持与现有 `tests/` 一致的轻量纯 Python 测试
