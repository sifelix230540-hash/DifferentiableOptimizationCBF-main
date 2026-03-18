"""3_18_import.py
9_axis URDF 静态可视化：加载机器人，按部件着色，保持窗口打开供查看。
"""
import os
import shutil
import tempfile
import time

import pybullet as p
import pybullet_data

# ── 复制 URDF 到无中文临时目录 ────────────────────────────────────────────
URDF_SRC = (
    r"C:\Users\12049\OneDrive\Desktop\科研相关\博一春季\免示教焊接轨迹规划"
    r"\相关资料\CBF_grad_optim_on_trajPlanning"
    r"\DifferentiableOptimizationCBF-main\9_axis"
)
tmp_root = os.path.join(tempfile.gettempdir(), "pybullet_urdf")
tmp_pkg  = os.path.join(tmp_root, "9_axis")
if os.path.exists(tmp_pkg):
    shutil.rmtree(tmp_pkg, ignore_errors=True)
    for _ in range(30):
        if not os.path.exists(tmp_pkg):
            break
        time.sleep(0.1)
shutil.copytree(URDF_SRC, tmp_pkg, dirs_exist_ok=True)

URDF_PATH = os.path.join(tmp_pkg, "urdf", "9_axis.urdf")

# ── 启动 PyBullet GUI ─────────────────────────────────────────────────────
p.connect(p.GUI)
p.setAdditionalSearchPath(tmp_root)
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

plane_id = p.loadURDF(pybullet_data.getDataPath() + "/plane.urdf")
p.changeVisualShape(plane_id, -1, rgbaColor=[0.15, 0.15, 0.18, 1])

# ── 加载机器人 ────────────────────────────────────────────────────────────
uid = p.loadURDF(URDF_PATH, basePosition=[0, -7, 0],
                 baseOrientation=[0, 0, 0, 1], useFixedBase=True)

# ── 颜色方案 ──────────────────────────────────────────────────────────────
LINK_COLORS = {
    "base_link": [0.22, 0.33, 0.47, 1.0],
    "link_01":   [0.18, 0.45, 0.62, 1.0],
    "link_02":   [0.12, 0.55, 0.55, 1.0],
    "link_03":   [0.08, 0.62, 0.42, 1.0],
    "robobase":  [0.20, 0.20, 0.22, 1.0],
    "link04":    [0.88, 0.90, 0.95, 1.0],
    "link05":    [0.95, 0.55, 0.10, 1.0],
    "link06":    [0.88, 0.90, 0.95, 1.0],
    "link07":    [0.95, 0.55, 0.10, 1.0],
    "link08":    [0.88, 0.90, 0.95, 1.0],
    "link09":    [0.85, 0.20, 0.15, 1.0],
}

# ── 初始位姿：X轴前移一段 ─────────────────────────────────────────────────
p.resetJointState(uid, 0, 12)   # pris01 沿X前移5m
p.changeVisualShape(uid, -1, rgbaColor=LINK_COLORS["base_link"])
for i in range(p.getNumJoints(uid)):
    clink = p.getJointInfo(uid, i)[12].decode()
    color = LINK_COLORS.get(clink)
    if color:
        p.changeVisualShape(uid, i, rgbaColor=color)

# ── 相机 ──────────────────────────────────────────────────────────────────
p.resetDebugVisualizerCamera(
    cameraDistance=18, cameraYaw=-135, cameraPitch=-35,
    cameraTargetPosition=[2, 2, 1])

print("窗口已打开，关闭窗口或 Ctrl+C 退出。")

try:
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1/60)
except KeyboardInterrupt:
    pass
finally:
    if p.isConnected():
        p.disconnect()
