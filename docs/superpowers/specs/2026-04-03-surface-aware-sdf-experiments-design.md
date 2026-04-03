# Surface-Aware SDF Experiments Design

## Context

9轴倒置龙门焊接机器人（3棱柱+6旋转），焊点在工件表面（SDF≈0）。
执行流程：`init-config → nearest-region → plan`。

## Changes

### 5a/5b: Vectorized SDF batch query (`4_1_udf.py`)
- Add `query_batch_vectorized()` using numpy vectorized trilinear interpolation
- `sdf_integration_experiments.py` 的 `_query_field` 优先调用向量化版本

### 2a: `init-config` 只变动6旋转关节
- `mutable_indices` 从 `revolute_joints` 构建，不含第3棱柱轴
- `kernel_links` 仍包含第3轴+后六轴（碰撞核完整）

### 2b: `init-config` 去掉外碰撞检查
- 新增参数 `INIT_SKIP_EXTERNAL_COLLISION = True`
- 为 True 时跳过 `_external_collision`，只做自碰撞

### 2c: `init-config` JSON摘要
- 新增 `INIT_OUTPUT_JSON`，输出关节名+角度+体素数

### 3a: `nearest-region` 连线检查跳过焊点端
- `t ∈ [t_skip, 1.0]`，`t_skip = max(2*min_line_clearance/dist, NEAR_LINE_SKIP_RATIO)`

### 3b: `nearest-region` kernel-npz 默认衔接
- `--kernel-npz` 默认值改为 `INIT_OUTPUT_NPZ`

### 3c: 法线引导半球搜索 + 分层早停
- `_estimate_surface_normal()` 用有限差分或 `query_with_gradient`
- 候选点过滤：`dot(offset, normal) > cos_threshold`
- 分层：point → AABB 8角 → 全核

### 3d: 倒置臂Z约束 + 法线输出
- `NEAR_REQUIRE_ABOVE_WELD = True`：候选Z ≥ 焊点Z
- JSON输出增加 `surface_normal` 字段

### 4a: `plan` nearest-region 作为 goal
- 新增 `--nearest-region-as-goal` 标志
- `PLAN_NEAREST_REGION_AS_GOAL = True`

### 4b: `plan` start 默认 robobase 位置
- 通过 PyBullet FK 计算 robobase 初始世界坐标作为默认 start

## New Parameters (集中参数区)

```python
INIT_SKIP_EXTERNAL_COLLISION = True
INIT_OUTPUT_JSON = "artifacts/sdf_exp/init_config_report.json"
NEAR_DEFAULT_KERNEL_NPZ = INIT_OUTPUT_NPZ
NEAR_SURFACE_NORMAL_EPS = 0.002
NEAR_NORMAL_HALF_SPHERE = True
NEAR_NORMAL_CONE_COS = 0.0
NEAR_LINE_SKIP_RATIO = 0.05
NEAR_REQUIRE_ABOVE_WELD = True
NEAR_ABOVE_WELD_MIN_DZ = 0.0
PLAN_NEAREST_REGION_AS_GOAL = True
PLAN_NEAREST_REGION_JSON = NEAR_OUTPUT_JSON
PLAN_INIT_CONFIG_NPZ = INIT_OUTPUT_NPZ
```
