"""3_18_import.py
9_axis URDF 静态可视化：加载机器人，按部件着色，保持窗口打开供查看。
"""
import os
import shutil
import tempfile
import time

import numpy as np
import pybullet as p
import pybullet_data

# ── 复制 URDF 到无中文临时目录 ────────────────────────────────────────────
URDF_SRC = (
    r"C:\Users\12049\OneDrive\Desktop\科研相关\博一春季\免示教焊接轨迹规划"
    r"\相关资料\CBF_grad_optim_on_trajPlanning"
    r"\DifferentiableOptimizationCBF-main\assets\robots\9_axis"
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
p.connect(p.GUI, options="--width=1920 --height=1080")
p.setAdditionalSearchPath(tmp_root)
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
p.configureDebugVisualizer(rgbBackground=[1, 1, 1])

plane_id = p.loadURDF(pybullet_data.getDataPath() + "/plane.urdf")
p.changeVisualShape(plane_id, -1, rgbaColor=[0.95, 0.95, 0.95, 1])

# ── 加载机器人 ────────────────────────────────────────────────────────────
uid = p.loadURDF(URDF_PATH, basePosition=[0, -7, 0],
                 baseOrientation=[0, 0, 0, 1], useFixedBase=True)

# ── 颜色方案 ──────────────────────────────────────────────────────────────
LINK_COLORS = {
    "base_link": [0.93, 0.93, 0.95, 1.0],   # 龙门架底座 浅白
    "link_01":   [0.95, 0.92, 0.75, 1.0],   # X轴 浅黄
    "link_02":   [0.80, 0.80, 0.82, 1.0],   # Y轴 浅灰
    "link_03":   [0.70, 0.82, 0.95, 1.0],   # Z轴 浅蓝
    "robobase":  [0.15, 0.15, 0.17, 1.0],   # 机械臂底座 炭黑
    "link04":    [0.85, 0.25, 0.25, 1.0],   # 臂1 红
    "link05":    [0.85, 0.85, 0.87, 1.0],   # 关节1 浅灰
    "link06":    [0.85, 0.25, 0.25, 1.0],   # 臂2 红
    "link07":    [0.85, 0.85, 0.87, 1.0],   # 关节2 浅灰
    "link08":    [0.85, 0.25, 0.25, 1.0],   # 臂3 红
    "link09":    [0.85, 0.85, 0.87, 1.0],   # 末端 浅灰
}

# ── 镜面反射（金属拉丝质感）──────────────────────────────────────────────
SPECULAR = {
    "base_link": [0.6, 0.6, 0.65],
    "link_01":   [0.6, 0.6, 0.65],
    "link_02":   [0.6, 0.6, 0.65],
    "link_03":   [0.5, 0.55, 0.7],
    "robobase":  [0.4, 0.4, 0.45],
    "link04":    [0.7, 0.3, 0.3],
    "link05":    [0.5, 0.5, 0.55],
    "link06":    [0.7, 0.3, 0.3],
    "link07":    [0.5, 0.5, 0.55],
    "link08":    [0.7, 0.3, 0.3],
    "link09":    [0.5, 0.5, 0.55],
}

# ── 初始位姿：X轴前移一段 ─────────────────────────────────────────────────
p.resetJointState(uid, 0, 12)   # pris01 沿X前移5m
p.changeVisualShape(uid, -1, rgbaColor=LINK_COLORS["base_link"],
                    specularColor=SPECULAR.get("base_link", [0.5]*3))
for i in range(p.getNumJoints(uid)):
    clink = p.getJointInfo(uid, i)[12].decode()
    color = LINK_COLORS.get(clink)
    spec  = SPECULAR.get(clink, [0.5]*3)
    if color:
        p.changeVisualShape(uid, i, rgbaColor=color, specularColor=spec)

# ── 相机 ──────────────────────────────────────────────────────────────────
p.resetDebugVisualizerCamera(
    cameraDistance=18, cameraYaw=-135, cameraPitch=-35,
    cameraTargetPosition=[2, 2, 1])

# ── 实时坐标系绘制 ──────────────────────────────────────────────────────
AXIS_COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
AXIS_LABELS = ["X", "Y", "Z"]

def draw_link_frame(body_id, link_index, axis_len=0.3, line_width=4,
                    prev_ids=None):
    """绘制 link 的 XYZ 坐标轴，返回 debug item id 列表以便下帧删除"""
    if prev_ids:
        for item_id in prev_ids:
            p.removeUserDebugItem(item_id)

    state = p.getLinkState(body_id, link_index)
    pos = np.array(state[4])
    rot = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)

    ids = []
    for i in range(3):
        end = pos + rot[:, i] * axis_len
        ids.append(p.addUserDebugLine(
            pos.tolist(), end.tolist(),
            lineColorRGB=AXIS_COLORS[i], lineWidth=line_width))
        ids.append(p.addUserDebugText(
            AXIS_LABELS[i], end.tolist(),
            textColorRGB=AXIS_COLORS[i], textSize=1.5))
    return ids

# ── 查找需要绘制坐标系的 link ──────────────────────────────────────────
frame_targets = {}
for i in range(p.getNumJoints(uid)):
    name = p.getJointInfo(uid, i)[12].decode()
    if name == "link09":
        frame_targets[name] = {"idx": i, "len": 0.15, "width": 5, "ids": None}
    elif name == "welding_gun_base":
        frame_targets[name] = {"idx": i, "len": 0.10, "width": 4, "ids": None}
    elif name == "weld_point":
        frame_targets[name] = {"idx": i, "len": 0.08, "width": 3, "ids": None}

# ── 关节交互滑块 ──────────────────────────────────────────────────────
sliders = []
for i in range(p.getNumJoints(uid)):
    info = p.getJointInfo(uid, i)
    j_name = info[1].decode()
    j_type = info[2]
    if j_type == p.JOINT_FIXED:
        continue
    lo, hi = info[8], info[9]
    init_val = p.getJointState(uid, i)[0]
    sid = p.addUserDebugParameter(j_name, lo, hi, init_val)
    sliders.append((i, sid))

print("窗口已打开，拖动左侧滑块控制关节，坐标系实时跟随。")
print("关闭窗口或 Ctrl+C 退出。")

try:
    while p.isConnected():
        for j_idx, sid in sliders:
            val = p.readUserDebugParameter(sid)
            p.resetJointState(uid, j_idx, val)

        for name, cfg in frame_targets.items():
            cfg["ids"] = draw_link_frame(
                uid, cfg["idx"], cfg["len"], cfg["width"], cfg["ids"])

        p.stepSimulation()
        time.sleep(1/60)
except KeyboardInterrupt:
    pass
finally:
    if p.isConnected():
        p.disconnect()
