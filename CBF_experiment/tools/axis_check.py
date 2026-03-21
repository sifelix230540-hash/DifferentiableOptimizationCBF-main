"""axis_check4.py
直接在 PyBullet 运行时修改关节轴向（通过 p.changeConstraint 不行，
但可以用 p.resetJointState + 测量相邻 link 的世界旋转来反推实际旋转轴）。

核心思路：
  对关节 ji，把它从 0 转到 +0.01 rad（小角度），
  取 child link 的世界旋转四元数差 dq，
  从 dq 提取旋转轴 → 这就是该关节在世界系中的实际旋转轴。
  无需改 URDF，直接用原始加载的 uid。
"""
import os, shutil, tempfile, time
import numpy as np
import pybullet as p

URDF_SRC = (
    r"C:\Users\12049\OneDrive\Desktop\科研相关\博一春季\免示教焊接轨迹规划"
    r"\相关资料\CBF_grad_optim_on_trajPlanning"
    r"\DifferentiableOptimizationCBF-main\assets\robots\9_axis"
)
tmp_root = os.path.join(tempfile.gettempdir(), "pb_axchk4")
tmp_pkg  = os.path.join(tmp_root, "9_axis")
if os.path.exists(tmp_pkg):
    shutil.rmtree(tmp_pkg, ignore_errors=True)
    for _ in range(30):
        if not os.path.exists(tmp_pkg):
            break
        time.sleep(0.1)
shutil.copytree(URDF_SRC, tmp_pkg, dirs_exist_ok=True)
URDF_PATH = os.path.join(tmp_pkg, "urdf", "9_axis.urdf")

p.connect(p.DIRECT)
p.setAdditionalSearchPath(tmp_root)

uid = p.loadURDF(URDF_PATH, useFixedBase=True)
n   = p.getNumJoints(uid)
rids = [i for i in range(n) if p.getJointInfo(uid,i)[2]==0]

def quat_to_axis_angle(q):
    """四元数 (x,y,z,w) → (axis, angle_rad)"""
    x, y, z, w = q
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    s = np.sqrt(max(1 - w*w, 1e-12))
    if s < 1e-6:
        return np.array([0,0,1]), 0.0
    return np.array([x,y,z]) / s, angle

def get_world_orn(uid, link_idx):
    state = p.getLinkState(uid, link_idx, computeForwardKinematics=True)
    return np.array(state[5])  # worldLinkFrameOrientation (xyzw)

def quat_mul(q1, q2):
    """q1 * q2，均为 xyzw"""
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_inv(q):
    x,y,z,w = q
    return np.array([-x,-y,-z,w])

print("各 revolute 关节在世界系中的实际旋转轴（从小角度旋转反推）：\n")
print(f"{'joint':>3}  {'name':<8}  {'世界系旋转轴 (x,y,z)':35}  {'旋转角(deg)':>12}")
print("-" * 65)

DELTA = 0.05  # rad，小角度

for ji in rids:
    jname = p.getJointInfo(uid, ji)[1].decode()

    # 零位姿态
    p.resetJointState(uid, ji, 0.0)
    orn0 = get_world_orn(uid, ji)

    # 小角度旋转
    p.resetJointState(uid, ji, DELTA)
    orn1 = get_world_orn(uid, ji)

    # 相对旋转 = orn0^{-1} * orn1（在 child frame）
    # 在世界系: dq_world = orn1 * orn0^{-1}
    dq = quat_mul(orn1, quat_inv(orn0))
    axis_world, angle = quat_to_axis_angle(dq)

    # 如果角度接近 DELTA，方向是正的
    if angle < 0:
        axis_world = -axis_world

    print(f"{ji:>3}  {jname:<8}  {str(np.round(axis_world,4)):35}  {np.degrees(angle):>12.4f}°")

    # 还原
    p.resetJointState(uid, ji, 0.0)

print("\n说明：这是各关节旋转轴在世界坐标系中的方向（零位下）。")
print("JAKA 期望：J1(腰部)≈Z轴旋转，J2~J5≈各自臂平面法向，J6≈工具轴")
p.disconnect()
