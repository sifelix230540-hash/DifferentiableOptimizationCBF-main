"""对比 6×6 全 Jacobian vs 3×6 平移 Jacobian 的可操作度分布差异。"""
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.manipulability_oracle import (
    _yoshikawa_manipulability, _condition_number,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.robot_model import (
    compose_full_q, load_robot_metadata, sample_joint_box,
)
import numpy as np

cfg = RobotQueryConfig()
robot, metadata, created = load_robot_metadata(cfg)
q_indices = np.array(metadata.q_indices, dtype=int)

rng = np.random.default_rng(0)
N = 5000
samples = sample_joint_box(metadata, rng, N)

manips_full = []
manips_pos = []
for q6 in samples:
    q_full = compose_full_q(metadata, q6)
    dq_zero = np.zeros_like(q_full)
    robot.set_joint_state(q_full, dq_zero)
    J_full = robot.get_ee_jacobian(q_full, dq_zero)
    J = J_full[:, q_indices]
    manips_full.append(_yoshikawa_manipulability(J))
    manips_pos.append(_yoshikawa_manipulability(J[:3, :]))

manips_full = np.array(manips_full)
manips_pos = np.array(manips_pos)

print(f"采样数: {N}\n")

for name, arr in [("6×6 全 Jacobian", manips_full), ("3×6 平移 Jacobian", manips_pos)]:
    print(f"=== {name} ===")
    print(f"  min={arr.min():.6f}  max={arr.max():.6f}  mean={arr.mean():.6f}  median={np.median(arr):.6f}")
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
    for t in thresholds:
        ratio = np.mean(arr >= t)
        print(f"    w >= {t:.3f}:  {ratio:.1%}")
    print()

print("=== 线段连通性抽样估计 ===")
rng2 = np.random.default_rng(42)
n_seg = 2000
for name, arr, thresh_list in [
    ("6×6", manips_full, [0.01, 0.02]),
    ("3×6 pos", manips_pos, [0.05, 0.10]),
]:
    good_idx = np.where(arr >= thresh_list[0])[0]
    if len(good_idx) < 10:
        continue
    for thresh in thresh_list:
        good = np.where(arr >= thresh)[0]
        if len(good) < 10:
            continue
        visible = 0
        total = min(n_seg, len(good) * (len(good) - 1) // 2)
        for _ in range(total):
            i, j = rng2.choice(good, 2, replace=False)
            seg_ok = True
            for alpha in np.linspace(0, 1, 13):
                q_mid = (1 - alpha) * samples[i] + alpha * samples[j]
                q_full_mid = compose_full_q(metadata, q_mid)
                dq_zero = np.zeros_like(q_full_mid)
                robot.set_joint_state(q_full_mid, dq_zero)
                J_mid = robot.get_ee_jacobian(q_full_mid, dq_zero)
                J_mid = J_mid[:, q_indices]
                if name == "3×6 pos":
                    w = _yoshikawa_manipulability(J_mid[:3, :])
                else:
                    w = _yoshikawa_manipulability(J_mid)
                if w < thresh:
                    seg_ok = False
                    break
            if seg_ok:
                visible += 1
        print(f"  {name} w>={thresh}: visibility = {visible}/{total} = {visible/max(total,1):.1%}")

import pybullet as p
robot.set_joint_state(metadata.q_base, metadata.dq_base)
if created and p.isConnected():
    p.disconnect()
print("\nDone.")
