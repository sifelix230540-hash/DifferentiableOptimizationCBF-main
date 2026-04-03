# SDF Integration Experiments 流程图

## 总体流水线

```mermaid
graph TD
    A["<b>① align</b><br/>SDF↔PyBullet 坐标系对齐"] --> B
    B["<b>③ init-config</b><br/>搜索最优6轴初始配置"] --> C
    C["<b>② nearest-region</b><br/>搜索最近可行中间点"] --> D
    D["<b>④ plan</b><br/>RRT* 路径规划"]

    A -->|alignment_report.json| A1["误差统计 + PNG"]
    B -->|init_kernel.npz<br/>init_config_report.json| B1["最优关节角 + 占用核"]
    C -->|nearest_region.json| C1["可行点坐标 + 表面法线"]
    D -->|rrt_star_path.json| D1["平滑无碰撞路径"]

    B1 -->|kernel_offsets| C
    C1 -->|top_k#91;0#93; as goal| D

    style A fill:#e3f2fd,stroke:#1565c0
    style B fill:#e8f5e9,stroke:#2e7d32
    style C fill:#fff3e0,stroke:#ef6c00
    style D fill:#fce4ec,stroke:#c62828
```

## ① align — 坐标系对齐

```mermaid
flowchart TD
    A1[加载 SDF/UDF npz] --> A2[PyBullet 加载工件 URDF]
    A2 --> A3[采样工件表面点<br/>per_link_points × links]
    A3 --> A4{align_mode?}
    A4 -->|known| A5[使用 URDF 已知外参<br/>workpiece_position + orientation]
    A4 -->|optimize| A6[已知外参为初值<br/>Powell 优化 6D 刚体变换]
    A5 --> A7[计算各点 SDF 误差]
    A6 --> A7
    A7 --> A8["输出 JSON + 误差分布 PNG"]
```

## ③ init-config — 最优初始配置搜索

```mermaid
flowchart TD
    I1[加载 URDF<br/>获取关节状态 q0] --> I2[确定 kernel_links<br/>=第3棱柱轴 + 后6轴]
    I2 --> I3["确定 mutable_joints<br/>=仅6个旋转关节"]
    I3 --> I4{循环 num_samples 次}

    I4 --> I5[高斯扰动<br/>mutable_indices 对应关节]
    I5 --> I6{自碰撞检查?}
    I6 -->|碰撞| I4
    I6 -->|无碰撞| I7{skip_external_collision?}
    I7 -->|false| I8{外碰撞检查?}
    I8 -->|碰撞| I4
    I8 -->|通过| I9
    I7 -->|true 跳过| I9

    I9["构建占用核<br/>_build_occupancy_kernel<br/>(kernel_links → 体素化)"]
    I9 --> I10{占用体素数 < 当前最优?}
    I10 -->|是| I11[更新 best]
    I10 -->|否| I4
    I11 --> I4

    I4 -->|采样结束| I12["输出 NPZ<br/>(q_best, kernel_offsets, occupancy_count)"]
    I12 --> I13["输出 JSON 报告<br/>(关节名+角度+体素数)"]
    I13 --> I14["可视化 PNG<br/>(3D核 + 直方图)"]
```

## ② nearest-region — 最近可行中间点搜索

```mermaid
flowchart TD
    N1[加载 SDF + 占用核 NPZ] --> N2[PyBullet 获取焊点位置]
    N2 --> N3["估计表面法线<br/>_estimate_surface_normal<br/>(SDF 有限差分)"]
    N3 --> N4{逐层球壳搜索<br/>r: radius_min → radius_max}

    N4 --> N5[网格生成候选偏移]
    N5 --> N6{半球过滤?<br/>dot·normal ≥ cone_cos}
    N6 -->|法线背侧| N5
    N6 -->|通过| N7{Z约束?<br/>off_z ≥ min_dz<br/>倒置臂}
    N7 -->|不满足| N5
    N7 -->|通过| N8["分层可行性检查<br/>_is_candidate_feasible"]

    N8 --> N81["层级1: 候选点 SDF > min_clearance"]
    N81 -->|fail| N5
    N81 -->|pass| N82["层级2: 连线检查<br/>t∈[t_skip, 1.0]<br/>跳过焊点端SDF≈0"]
    N82 -->|穿墙| N5
    N82 -->|pass| N83["层级3: AABB 8角<br/>快速排除"]
    N83 -->|fail| N5
    N83 -->|pass| N84["层级4: 全核碰撞检查<br/>query_batch_vectorized"]
    N84 -->|碰撞| N5
    N84 -->|通过| N9[记录候选]

    N9 --> N10{已找到 ≥ top_k?}
    N10 -->|是| N11[提前结束]
    N10 -->|否| N4
    N4 -->|搜完| N11

    N11 --> N12[按距离排序取 top_k]
    N12 --> N13["输出 JSON<br/>(weld_point, surface_normal, top_k)"]
    N13 --> N14["可视化 PNG<br/>(焊点 + 候选点 + 法线箭头)"]
```

## ④ plan — RRT* 路径规划

```mermaid
flowchart TD
    P1[加载 SDF] --> P2{start 指定?}
    P2 -->|是| P3[解析 x,y,z]
    P2 -->|否| P4["PyBullet FK → robobase 世界坐标"]
    P3 --> P5
    P4 --> P5

    P5{goal 指定?}
    P5 -->|是| P6[解析 x,y,z]
    P5 -->|否| P7{"读取 nearest_region.json<br/>top_k[0] 作为 goal"}
    P6 --> P8
    P7 --> P8

    P8{auto_fix_endpoints?} -->|是| P9["邻域搜索修复<br/>确保 SDF > clearance"]
    P8 -->|否| P10
    P9 --> P10

    P10{有 via point?}
    P10 -->|是| P11["分段 RRT*<br/>start→via + via→goal"]
    P10 -->|否| P12["单段 RRT*<br/>start→goal"]
    P11 --> P13
    P12 --> P13

    P13["Shortcut 平滑<br/>_shortcut_smooth"] --> P14["输出 JSON<br/>(路径点 + 最小SDF值)"]
    P14 --> P15["可视化 PNG<br/>(3D路径 + SDF热力图)"]
```

## 核心性能优化：向量化 SDF 查询

```mermaid
flowchart LR
    Q1["field.query(pts)"] --> Q2{pts.ndim == 2?}
    Q2 -->|单点 shape=(3,)| Q3["query_single<br/>逐点三线性插值"]
    Q2 -->|批量 shape=(N,3)| Q4["query_batch_vectorized<br/>NumPy 向量化"]
    Q4 --> Q5["一次性计算 N 个体素索引<br/>ix0, iy0, iz0"]
    Q5 --> Q6["高级索引取 8 角值<br/>grid[ix0, iy0, iz0] ..."]
    Q6 --> Q7["向量化三线性插值<br/>返回 (N,) float32"]
```
