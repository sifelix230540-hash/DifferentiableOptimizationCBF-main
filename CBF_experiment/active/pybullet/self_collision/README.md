# Self Collision 使用说明

本目录下的自碰撞实验现在按职责拆成 4 组。
现在扩展为 6 组。

## 推荐结构

### `backends/`

- `coal_backend.py`
- 作用：统一的 `coal` 自碰撞检测后端。
- 推荐用途：所有新实验都优先从这里拿碰撞判定接口。

### `cspace_hulls/`

- `hulls.py`
- `evaluation.py`
- `benchmark.py`
- `sample_visualizer.py`
- 作用：经典“采样 -> 碰撞区域拟合 -> 评估 -> GUI 检查”主线。

### `validation/`

- `coal_validation.py`
- `coal_validation_plot.py`
- `coal_comparison_visualizer.py`
- 作用：对比诊断 PyBullet 与 `coal` 的差异，不是最终主算法。

### `boundary_learning/`

- `boundary_learner.py`
- 作用：用 `coal` 数据集训练隐式边界 `h(q)`，再提取切平面作为显式约束。

### `safe_cover/`

- `iris_safe_cover.py`
- 作用：借鉴 IRIS 思路，在 6R 安全集采样上迭代生成“最大内接球 + 切平面”形式的安全多面体覆盖。

### `vcc_iris/`

- `pipeline.py`
- `coal_oracle.py`
- `visibility.py`
- `clique_cover.py`
- `iris_zo.py`
- `polytope_sampling.py`
- `statistical_test.py`
- 作用：完全不依赖 Drake、只基于 `coal` 的 `VCC + IRIS-ZO` 实验主线。`VCC` 负责产生 seed 与初始 metric，`IRIS-ZO` 负责 region growth，`hit-and-run + Chernoff/union-bound` 负责内层采样与统计终止。

## 当前兼容策略

顶层旧文件例如：

- `self_collision_cspace_hulls.py`
- `self_collision_cspace_benchmark.py`
- `self_collision_coal_validation.py`

目前仍然保留，避免旧引用立即失效。

新代码建议优先使用新的分组路径，例如：

```python
from CBF_experiment.active.pybullet.self_collision.backends.coal_backend import (
    build_coal_link_models,
    classify_self_collision_sample,
)

from CBF_experiment.active.pybullet.self_collision.cspace_hulls.benchmark import (
    run_benchmark,
)
```

## 典型用途

### 1. 经典凸包主线

一键跑完整流程：

```python
from CBF_experiment.active.pybullet.self_collision.cspace_hulls.benchmark import (
    BenchmarkParameters,
    run_benchmark,
)

summary = run_benchmark(BenchmarkParameters)
```

单独构建碰撞凸包：

```python
from CBF_experiment.active.pybullet.self_collision.cspace_hulls.hulls import (
    monte_carlo_self_collision_hulls,
)

payload = monte_carlo_self_collision_hulls(num_samples=20000)
```

### 2. `coal` 对比验证

生成对比报告：

```python
from CBF_experiment.active.pybullet.self_collision.validation.coal_validation import (
    CoalValidationParameters,
    run_coal_validation,
)

report = run_coal_validation(CoalValidationParameters)
```

GUI 查看分歧样本：

```python
from CBF_experiment.active.pybullet.self_collision.validation.coal_comparison_visualizer import (
    ComparisonVisualizationParameters,
    visualize_comparison_gui,
)

visualize_comparison_gui(ComparisonVisualizationParameters)
```

### 3. 学习型边界

边界学习脚本分 4 个阶段：

```bash
python ".../self_collision_boundary_learner.py" dataset --num-samples 200000
python ".../self_collision_boundary_learner.py" train --epochs 200
python ".../self_collision_boundary_learner.py" tangent --num-boundary 2000
python ".../self_collision_boundary_learner.py" vis
```

推荐新导入路径：

```python
from CBF_experiment.active.pybullet.self_collision.boundary_learning.boundary_learner import (
    generate_dataset,
    train_boundary_mlp,
    extract_tangent_planes,
    visualize_boundary_gui,
)
```

### 4. IRIS 风格安全覆盖

```python
from CBF_experiment.active.pybullet.self_collision.safe_cover import (
    IrisSafeCoverConfig,
    run_iris_safe_cover,
)

cover = run_iris_safe_cover(IrisSafeCoverConfig())
```

## 文件关系

### 核心底座

- `backends/coal_backend.py`

### 主算法

- `cspace_hulls/hulls.py`
- `cspace_hulls/evaluation.py`
- `cspace_hulls/benchmark.py`

### 可视化/诊断

- `cspace_hulls/sample_visualizer.py`
- `validation/coal_validation_plot.py`
- `validation/coal_comparison_visualizer.py`

### 学习实验

- `boundary_learning/boundary_learner.py`

### 安全覆盖

- `safe_cover/iris_safe_cover.py`

### VCC + IRIS-ZO

- `vcc_iris/pipeline.py`
- `vcc_iris/coal_oracle.py`
- `vcc_iris/visibility.py`
- `vcc_iris/clique_cover.py`
- `vcc_iris/iris_zo.py`
- `vcc_iris/polytope_sampling.py`
- `vcc_iris/statistical_test.py`

## 建议

- 如果你的目标是“经典几何建模”，优先看 `cspace_hulls/`
- 如果你的目标是“检查 `coal` 和旧方法差异”，优先看 `validation/`
- 如果你的目标是“得到可微边界函数和切平面约束”，优先看 `boundary_learning/`
- 如果你的目标是“直接得到一组可用于规划筛选的安全多面体覆盖”，优先看 `safe_cover/`
- 如果你的目标是“完全基于 `coal` 做 visibility clique cover + IRIS-ZO 覆盖”，优先看 `vcc_iris/`
- 新实验不要再直接往顶层旧文件里继续堆，优先放到对应分组目录下

