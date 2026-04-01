# 全局 UDF/SDF 场文件设计

## 目标

为 `assets/cad_exports/model_CAD/scene/urdf/中组立0725(1).stp.SLDASM.urdf` 对应的中组立装配体离线烘焙一个覆盖全局包围盒的统一距离场文件，而不是仅对单个查询点临时计算最近距离。

主输出是一个稳定可用的全局 `UDF` 网格；同时额外烘焙两份仅用于对比符号可靠性的 `libigl SDF` 与 `Open3D SDF` 网格。三者统一保存到一个 `.npz` 文件中，并提供三线性插值查询接口。

## 设计范围

实现内容限定在：

- 读取 URDF 和其引用的 `base_link`、`l2`、`l3` 网格并拼装到统一世界坐标系
- 计算整个工件的全局包围盒，并按给定分辨率与 margin 构造规则体素网格
- 生成 `udf_grid`
- 尝试生成 `igl_sdf_grid`
- 尝试生成 `o3d_sdf_grid`
- 保存统一 `.npz` 场文件
- 提供单点或批量查询函数，基于保存的网格做三线性插值
- 提供最小但直观的可视化对比，帮助判断两类 SDF 的符号是否可靠

不在本次范围内：

- 直接接入现有 CBF/QP 控制器
- 对机械臂自身做全链路 SDF 建模
- 自动修复非 watertight 网格
- 大规模性能优化或 GPU 加速

## 核心原则

### 1. 主结果以 UDF 为准

装配体 mesh 可能非闭合、自交、薄片化或存在缝隙，这会导致 signed distance 的符号不稳定。为保证“整个工件包围盒内都有效且可查”，本次以 unsigned distance field 作为真正可用的主结果。

### 2. SDF 仅作对比

`libigl` 与 `Open3D` 生成的 SDF 网格不会被视为唯一真值，而是作为辅助诊断：

- `libigl` 用于对比 `pseudonormal` 或可用时的 winding-number 符号判定
- `Open3D` 用于对比其对 watertight 场景下的 signed distance 表现

### 3. 文件优先于在线几何查询

输出必须是一个统一的、可持久化的全局场文件，而不是运行时反复对 mesh 执行最近点计算。后续任何查询都应优先从该文件中读取并插值。

## 输入与装配

### 输入来源

- 主输入：`中组立0725(1).stp.SLDASM.urdf`
- 网格输入：`base_link`、`l2`、`l3` 的 `obj/stl`

优先按 URDF 的 fixed joint 变换将 3 个 link 装配成一个统一 mesh。保留直接读取单独 mesh 的能力，便于排查单个 link 的几何问题。

### 坐标系与单位

统一以 URDF 装配后的世界坐标系作为烘焙坐标系：

- 轴向与右手系约定完全沿用 URDF / PyBullet 的世界系
- 所有距离量默认单位为米
- 若 mesh 文件单位不是米，必须在装配阶段统一缩放到米后再进入后续流程
- 若 URDF 中声明了 mesh `scale`，必须显式应用

### 装配流程

1. 解析 URDF 中各 link 的 mesh 文件路径
2. 解析 fixed joint 的父子关系与 `origin`
3. 将每个 link mesh 变换到统一世界坐标系
4. 合并为一个全局三角面集合
5. 记录全局 `bbox_min`、`bbox_max`

如果装配后存在明显重复面、共面内部面或双层薄壳，本次实现原则上不自动重建“外壳真值”，而是保留原始装配体表面并在结果说明中明确这一点。这意味着 UDF 表示的是“到装配后三角面集合的最近距离”，不是经过拓扑修复后的理想实体外表面距离。

## 网格场烘焙

### 体素网格定义

对全局包围盒外扩 `margin` 后生成规则三维网格，保存：

- `origin`: 体素网格最小角点坐标
- `spacing`: 体素边长，可为标量或三个方向一致的向量
- `shape`: `(nx, ny, nz)`
- `bbox_min`
- `bbox_max`

其中：

- `bbox_min`、`bbox_max` 始终指工件本身的原始 AABB，不包含外扩 margin
- `origin` 指外扩后规则网格域的最小角点
- 第 `(i, j, k)` 个体素中心坐标定义为 `origin + ([i, j, k] + 0.5) * spacing`
- 三线性插值也严格基于这一体素中心定义

默认使用各向同性体素，即三个方向共用一个标量 `spacing`。

建议默认允许从命令行指定：

- `--spacing`
- `--margin`
- `--max-points-per-batch`
- `--output`

在真正分配网格前，先估算内存：

- `num_voxels = nx * ny * nz`
- `bytes_per_grid = num_voxels * dtype_bytes`
- `total_bytes = bytes_per_grid * num_enabled_grids`

若超过预设阈值，则直接报错并提示用户调大 `spacing` 或减小 `margin`，而不是在运行中途耗尽内存。

### UDF 烘焙

对每个体素中心计算到装配体三角面的最小欧氏距离，结果保存为 `udf_grid`。

这是主结果，必须满足：

- 覆盖整个全局包围盒
- 所有体素都有有效数值
- 不依赖 inside/outside 定义

### libigl SDF 烘焙

若环境可用，则基于统一装配后的三角网格对每个体素中心调用 `libigl.signed_distance`，优先使用 `pseudonormal`；若接口与环境支持，再补充 winding-number 相关选项。

结果保存为 `igl_sdf_grid`。若依赖缺失或计算失败，则以 `NaN` 网格占位，并在元数据中记录失败原因。

符号约定统一为：负值表示“方法判定的内部”，正值表示“外部”。由于非 watertight 装配体上该判定可能不稳定，因此该网格只作为对比诊断，不作为主约束来源。

### Open3D SDF 烘焙

若环境可用，则使用 `Open3D RaycastingScene.compute_signed_distance` 对体素中心批量求值，结果保存为 `o3d_sdf_grid`。

同样地，若依赖缺失、mesh 不适用或计算失败，则以 `NaN` 网格占位并记录原因。

其符号解释也统一为负值表示内部、正值表示外部；若 Open3D 在非 watertight 情况下给出看似连续但明显不合理的符号区域，应通过与 UDF 及 `libigl` 的对比来识别，而不是直接信任。

## 输出文件格式

统一输出为单个 `.npz` 文件，至少包含：

- `origin`
- `spacing`
- `shape`
- `udf_grid`
- `igl_sdf_grid`
- `o3d_sdf_grid`
- `bbox_min`
- `bbox_max`

建议额外保存：

- `link_names`
- `mesh_paths`
- `build_config`
- `status_flags`
- `failure_reasons`

其中 `status_flags` 用于标记：

- `udf_ok`
- `igl_ok`
- `o3d_ok`

建议默认使用 `float32` 存储三份网格，以控制文件体积；必要时允许通过参数切换到 `float64`。

## 查询接口

提供一个查询函数，输入任意点坐标，输出指定场的三线性插值结果。

建议接口形式：

- `load_distance_field(path) -> field`
- `field.query(points, kind="udf")`
- `field.query_single(point, kind="udf")`

行为定义：

- `kind="udf"` 返回主结果
- `kind="igl_sdf"` 返回 `igl_sdf_grid` 的插值
- `kind="o3d_sdf"` 返回 `o3d_sdf_grid` 的插值
- `points` 支持形状 `(3,)` 或 `(N, 3)`
- 默认输入输出 dtype 为 `float32`
- 点超出网格范围时，默认抛出显式异常；可选支持 `clip` 模式
- 若目标网格包含 `NaN` 邻域，则返回 `NaN` 并提示该方法在该区域无有效场值

## 可视化

至少提供以下对比图：

### 1. 切片热力图

在一个或多个固定 `z` 截面上显示：

- `udf_grid`
- `igl_sdf_grid`
- `o3d_sdf_grid`
- `abs(igl_sdf_grid - udf_grid)`
- `abs(o3d_sdf_grid - udf_grid)`

若某方法整张或局部为 `NaN`，可视化中必须用单独颜色或掩膜明确标示“无效区域”，而不是静默按零值处理。

### 2. 采样点对比

在包围盒内采样若干空间点，显示：

- UDF 数值
- 两类 SDF 数值
- 与 UDF 的偏差

### 3. 符号稳定性诊断

重点标出：

- `igl_sdf_grid` 与 `o3d_sdf_grid` 符号不一致区域
- SDF 为负但 UDF 远离表面的异常区域

其中“远离表面”阈值应可配置，例如默认取 `udf > 2 * spacing`。

可视化既可以保存为静态图片，也可以额外输出一个简单浏览器页面用于交互查看。

## 错误处理

需要显式处理以下情况：

- URDF 路径解析失败
- mesh 文件缺失
- 单个 link 读取失败
- `libigl` 未安装
- `Open3D` 未安装
- SDF 方法因非 watertight 或接口限制失败
- 体素分辨率过高导致内存超限

策略：

- `UDF` 失败则整体任务失败
- `igl` 或 `Open3D` 失败不阻塞主流程，但必须在输出元数据与终端摘要中说明

## 测试与验证

本仓库没有现成测试体系，因此本次验证以脚本级检查为主：

- 确认 `.npz` 文件中的字段完整
- 对若干已知点执行插值查询，检查返回值形状与数值合理性
- 确认 `udf_grid` 无 `NaN` 或 `inf`
- 若 `igl` 或 `Open3D` 成功，检查其场值维度与 `udf_grid` 一致
- 生成至少一组切片图用于人工判断

## 实现拆分

建议把 `CBF_experiment/active/pybullet/4_1_udf.py` 组织为以下部分：

1. URDF 与 mesh 装配
2. 全局包围盒与规则网格生成
3. UDF 烘焙
4. libigl SDF 烘焙
5. Open3D SDF 烘焙
6. `.npz` 保存与加载
7. 三线性插值查询
8. 可视化与终端摘要

## 推荐执行顺序

先做最小可用主链路：

1. URDF 装配
2. 全局 UDF 烘焙
3. `.npz` 保存
4. 查询函数

再加诊断增强：

5. libigl SDF
6. Open3D SDF
7. 切片图与对比可视化

这样即使两种 SDF 因 mesh 性质表现不稳定，你仍然已经得到一个真正可用的全局 UDF 总表。
