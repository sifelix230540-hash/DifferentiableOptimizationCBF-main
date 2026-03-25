"""Drake 九轴机械臂 + 场景 可视化（Step 1：配置、导入、可视化）

模块化类结构：
  DrakeConfig          — 全局路径与参数配置
  URDFPackageResolver  — 将 package:// 名称映射到本地目录
  MeshConverter        — 批量将 STL 转换为 OBJ（Drake / MeshCat 需要）
  URDFPreprocessor     — 读取 URDF，将 mesh 路径中的 .STL 替换为 .obj
  DrakeEnvironment     — DiagramBuilder / MultibodyPlant / SceneGraph / Meshcat
  RobotLoader          — 加载九轴机器人 URDF
  SceneLoader          — 加载场景 URDF（工件/障碍物）
  Visualizer           — 组装并启动 MeshCat 可视化
"""

from __future__ import annotations

import re
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── 路径设置 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSETS        = ROOT / "assets"
ROBOT_PKG_DIR = ASSETS / "robots" / "9_axis"
SCENE_PKG_DIR = ASSETS / "cad_exports" / "model_CAD" / "scene"
ROBOT_URDF    = ROBOT_PKG_DIR / "urdf" / "9_axis.urdf"
SCENE_URDF    = SCENE_PKG_DIR / "urdf" / "中组立0725(1).stp.SLDASM.urdf"

# package:// 名称 → 本地目录（URDF 中 filename="package://XXX/…" 的 XXX 部分）
ROBOT_PKG_NAME = "9_axis"
SCENE_PKG_NAME = "中组立0725(1).stp.SLDASM"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DrakeConfig — 配置数据类
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class DrakeConfig:
    """全局仿真 / 可视化参数。"""

    # 物理步长（0 = 纯运动学，无动力学积分）
    time_step: float = 0.0

    # 机器人初始关节位置（顺序同 URDF：pris01/02/03 + revo04~09）
    robot_initial_q: list[float] = field(
        default_factory=lambda: [10.0, 0.0, 0.0,   # 龙门三轴 (m)
                                  0.0,  0.0, 0.0,   # revo04/05/06 (rad)
                                  0.0,  0.0, 0.0]   # revo07/08/09 (rad)
    )

    # 场景在世界坐标系中的平移偏置 (m)
    scene_translation: list[float] = field(default_factory=lambda: [0.0, 4.0, 0.0])

    # 机器人基座在世界坐标系中的平移偏置 (m)
    robot_translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # 是否自动在浏览器中打开 MeshCat 页面
    auto_open_browser: bool = True

    # 坐标系三轴可视化参数
    frame_axis_length: float = 0.35
    frame_axis_radius: float = 0.01
    frame_axis_opacity: float = 0.9


# ═══════════════════════════════════════════════════════════════════════════════
# 2. URDFPackageResolver — package:// 路径注册
# ═══════════════════════════════════════════════════════════════════════════════
class URDFPackageResolver:
    """将 URDF 中的 package:// 包名注册到 Drake Parser 的 PackageMap 中。"""

    def __init__(self) -> None:
        self._mappings: dict[str, Path] = {}

    def add(self, package_name: str, directory: Path) -> "URDFPackageResolver":
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"包目录不存在：{directory}")
        self._mappings[package_name] = directory
        return self

    def apply_to_parser(self, parser) -> None:
        for name, path in self._mappings.items():
            parser.package_map().Add(name, str(path))

    @classmethod
    def default_for_project(cls) -> "URDFPackageResolver":
        r = cls()
        r.add(ROBOT_PKG_NAME, ROBOT_PKG_DIR)
        r.add(SCENE_PKG_NAME, SCENE_PKG_DIR)
        return r


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MeshConverter — STL → OBJ 批量转换
# ═══════════════════════════════════════════════════════════════════════════════
class MeshConverter:
    """
    使用 trimesh 将指定目录下所有 .STL 文件转换为同名 .obj 文件。

    Drake 碰撞几何要求 .obj / .vtk / .gltf，视觉几何则支持 .STL。
    转换结果保存在同一 meshes/ 目录，不影响原始文件。
    """

    def __init__(self, pkg_dirs: list[Path]) -> None:
        self.pkg_dirs = [Path(d) for d in pkg_dirs]

    def convert_all(self, force: bool = False) -> dict[Path, Path]:
        """
        遍历所有包目录下的 meshes/ 子目录，将 .STL 转为 .obj。

        Args:
            force: True 时即使 .obj 已存在也重新转换。

        Returns:
            {stl_path: obj_path} 映射字典。
        """
        import trimesh

        results: dict[Path, Path] = {}
        for pkg_dir in self.pkg_dirs:
            mesh_dir = pkg_dir / "meshes"
            if not mesh_dir.is_dir():
                print(f"[MeshConverter] 警告：meshes 目录不存在 {mesh_dir}")
                continue
            for stl_path in sorted(mesh_dir.glob("*.STL")):
                obj_path = stl_path.with_suffix(".obj")
                if obj_path.exists() and not force:
                    results[stl_path] = obj_path
                    continue
                try:
                    mesh = trimesh.load(str(stl_path), force="mesh")
                    mesh.export(str(obj_path))
                    results[stl_path] = obj_path
                    print(f"[MeshConverter] {stl_path.name} → {obj_path.name}")
                except Exception as e:
                    print(f"[MeshConverter] 转换失败 {stl_path.name}：{e}")
        return results

    @classmethod
    def default_for_project(cls) -> "MeshConverter":
        return cls([ROBOT_PKG_DIR, SCENE_PKG_DIR])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. URDFPreprocessor — 修正 URDF 碰撞几何路径（.STL → .obj）
# ═══════════════════════════════════════════════════════════════════════════════
class URDFPreprocessor:
    """
    读取 URDF XML，将全部 mesh filename 的 .STL 后缀改为 .obj。

    MeshCat 对可视 mesh 的 `.stl` 也不会显示，因此这里统一改为 `.obj`。
    返回修改后的 XML 字符串，供 parser.AddModelsFromString() 使用。
    """

    @staticmethod
    def process(urdf_path: Path) -> str:
        """读取并处理 URDF，返回修改后的 XML 字符串。"""
        content = urdf_path.read_text(encoding="utf-8")
        return re.sub(r'\.STL\b', '.obj', content, flags=re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DrakeEnvironment — Builder / Plant / SceneGraph / Meshcat
# ═══════════════════════════════════════════════════════════════════════════════
class DrakeEnvironment:
    """持有 Drake 仿真核心对象，负责构建和编译 Diagram。"""

    def __init__(self, config: DrakeConfig) -> None:
        from pydrake.all import (
            AddMultibodyPlantSceneGraph,
            DiagramBuilder,
            MeshcatVisualizer,
            StartMeshcat,
        )

        self.config = config
        self.meshcat = StartMeshcat()
        print(f"[DrakeEnvironment] MeshCat 地址：{self.meshcat.web_url()}")

        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=config.time_step
        )
        self.meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, self.meshcat
        )

        self.diagram: Optional[object] = None
        self.context: Optional[object] = None
        self.plant_context: Optional[object] = None

    def finalize(self) -> None:
        """锁定 Plant 并编译 Diagram。所有模型加载完成后调用。"""
        self.plant.Finalize()
        self.diagram = self.builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RobotLoader — 九轴机器人加载
# ═══════════════════════════════════════════════════════════════════════════════
class RobotLoader:
    """从 URDF 加载九轴机器人，并可设置初始关节位置。"""

    def __init__(self, env: DrakeEnvironment, resolver: URDFPackageResolver) -> None:
        from pydrake.all import Parser, RigidTransform

        self.env = env
        if not ROBOT_URDF.is_file():
            raise FileNotFoundError(f"机器人 URDF 不存在：{ROBOT_URDF}")

        urdf_str = URDFPreprocessor.process(ROBOT_URDF)
        parser = Parser(env.plant)
        resolver.apply_to_parser(parser)
        self.model_instance = parser.AddModelsFromString(urdf_str, "urdf")[0]
        print(f"[RobotLoader] 已加载：{ROBOT_URDF.name}  (instance={self.model_instance})")

        # 将机器人根链接焊到世界，避免 Drake 自动创建 floating base。
        t = np.array(self.env.config.robot_translation, dtype=float)
        base_body = env.plant.GetBodyByName("base_link", self.model_instance)
        env.plant.WeldFrames(
            env.plant.world_frame(),
            base_body.body_frame(),
            RigidTransform(t),
        )

    def set_initial_positions(self) -> None:
        """在 env.finalize() 之后将机器人摆到初始位置。"""
        q_init = np.array(self.env.config.robot_initial_q, dtype=float)
        n_pos = self.env.plant.num_positions(self.model_instance)
        if len(q_init) != n_pos:
            print(f"[RobotLoader] 警告：q_init 长度 {len(q_init)} ≠ num_positions {n_pos}，改用零位")
            q_init = np.zeros(n_pos)
        self.env.plant.SetPositions(self.env.plant_context, self.model_instance, q_init)

    def print_joint_info(self) -> None:
        """打印关节名称、类型及位置范围。"""
        print("\n── 九轴机器人关节信息 ──")
        for idx in self.env.plant.GetJointIndices(self.model_instance):
            j = self.env.plant.get_joint(idx)
            print(f"  {j.name():<26} [{j.type_name()}]", end="")
            try:
                lo = j.position_lower_limits()
                hi = j.position_upper_limits()
                print(f"  lower={np.array2string(lo, precision=3)}"
                      f"  upper={np.array2string(hi, precision=3)}")
            except Exception:
                print("  (fixed)")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SceneLoader — 工件场景加载
# ═══════════════════════════════════════════════════════════════════════════════
class SceneLoader:
    """从 URDF 加载静态场景（工件、障碍物等），并固联到世界坐标系。"""

    def __init__(
        self,
        env: DrakeEnvironment,
        resolver: URDFPackageResolver,
        translation: Optional[list[float]] = None,
    ) -> None:
        from pydrake.all import Parser, RigidTransform

        self.env = env
        if not SCENE_URDF.is_file():
            raise FileNotFoundError(f"场景 URDF 不存在：{SCENE_URDF}")

        urdf_str = URDFPreprocessor.process(SCENE_URDF)
        parser = Parser(env.plant)
        resolver.apply_to_parser(parser)
        self.model_instance = parser.AddModelsFromString(urdf_str, "urdf")[0]
        print(f"[SceneLoader] 已加载：{SCENE_URDF.name}  (instance={self.model_instance})")

        # 将场景根 body 固联到世界坐标系（可附加平移偏置）
        t = np.array(translation if translation else [0.0, 0.0, 0.0], dtype=float)
        base_body = env.plant.GetBodyByName("base_link", self.model_instance)
        env.plant.WeldFrames(
            env.plant.world_frame(),
            base_body.body_frame(),
            RigidTransform(t),
        )

    def get_body_names(self) -> list[str]:
        return [
            self.env.plant.get_body(idx).name()
            for idx in self.env.plant.GetBodyIndices(self.model_instance)
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Visualizer — 主可视化控制器
# ═══════════════════════════════════════════════════════════════════════════════
class Visualizer:
    """组装所有模块，启动 MeshCat 可视化并持续运行。"""

    def __init__(self, config: Optional[DrakeConfig] = None) -> None:
        self.config = config or DrakeConfig()

        # ① 格式转换：STL → OBJ（碰撞几何用）
        converter = MeshConverter.default_for_project()
        converter.convert_all(force=False)

        # ② package:// 解析器
        self.resolver = URDFPackageResolver.default_for_project()

        # ③ Drake 核心环境
        self.env = DrakeEnvironment(self.config)

        # ④ 加载机器人与场景
        self.robot = RobotLoader(self.env, self.resolver)
        self.scene = SceneLoader(
            self.env, self.resolver,
            translation=self.config.scene_translation,
        )

        # ⑤ 添加关键坐标系三轴，便于对齐机器人 / 场景 / 焊点
        self.add_frame_visualization()

        # ⑥ 锁定 Plant，编译 Diagram
        self.env.finalize()

        # ⑦ 设置初始关节位置
        self.robot.set_initial_positions()
        self.robot.print_joint_info()

    def add_frame_visualization(self) -> None:
        """向 MeshCat 添加几个关键坐标系三轴。"""
        from pydrake.all import AddFrameTriadIllustration

        length = self.config.frame_axis_length
        radius = self.config.frame_axis_radius
        opacity = self.config.frame_axis_opacity
        plant = self.env.plant
        scene_graph = self.env.scene_graph

        AddFrameTriadIllustration(
            scene_graph=scene_graph,
            plant=plant,
            frame=plant.world_frame(),
            name="world_frame",
            length=length * 1.2,
            radius=radius * 1.2,
            opacity=opacity,
        )

        AddFrameTriadIllustration(
            scene_graph=scene_graph,
            plant=plant,
            body=plant.GetBodyByName("base_link", self.robot.model_instance),
            name="robot_base_frame",
            length=length,
            radius=radius,
            opacity=opacity,
        )

        AddFrameTriadIllustration(
            scene_graph=scene_graph,
            plant=plant,
            body=plant.GetBodyByName("weld_point", self.robot.model_instance),
            name="weld_point_frame",
            length=length * 0.8,
            radius=radius * 0.8,
            opacity=opacity,
        )

        AddFrameTriadIllustration(
            scene_graph=scene_graph,
            plant=plant,
            body=plant.GetBodyByName("base_link", self.scene.model_instance),
            name="scene_base_frame",
            length=length,
            radius=radius,
            opacity=opacity,
        )

        print("[Visualizer] 已添加 world / robot_base / weld_point / scene_base 坐标系")

    def publish(self) -> None:
        """将当前 context 推送到 MeshCat，刷新 3D 画面。"""
        self.env.diagram.ForcedPublish(self.env.context)

    def run(self, duration: float = 0.0) -> None:
        """
        启动可视化并保持运行。

        Args:
            duration: 运行秒数；0 表示永久运行直到 Ctrl+C。
        """
        self.publish()

        url = self.env.meshcat.web_url()
        print(f"\n[Visualizer] 可视化已就绪 → {url}")
        print("[Visualizer] 在浏览器中打开上方链接查看 3D 模型")

        if self.config.auto_open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass

        if duration > 0:
            print(f"[Visualizer] 将在 {duration:.0f} 秒后自动退出...")
            time.sleep(duration)
        else:
            print("[Visualizer] 按 Ctrl+C 退出")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\n[Visualizer] 已退出")


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    config = DrakeConfig()
    viz = Visualizer(config)
    viz.run(duration=0)


if __name__ == "__main__":
    main()
