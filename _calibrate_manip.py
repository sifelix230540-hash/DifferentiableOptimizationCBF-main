"""标定 JAKA 6-DOF 臂在关节空间中的 Yoshikawa 可操作度分布。"""
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.manipulability_oracle import (
    ManipulabilityOracle, _yoshikawa_manipulability, _condition_number,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.robot_model import (
    compose_full_q, load_robot_metadata, sample_joint_box,
)
import numpy as np

cfg = RobotQueryConfig()
robot, metadata, created = load_robot_metadata(cfg)

rng = np.random.default_rng(0)
N = 5000
samples = sample_joint_box(metadata, rng, N)

manips = []
conds = []
for q6 in samples:
    q_full = compose_full_q(metadata, q6)
    dq_zero = np.zeros_like(q_full)
    robot.set_joint_state(q_full, dq_zero)
    J_full = robot.get_ee_jacobian(q_full, dq_zero)
    q_indices = np.array(metadata.q_indices, dtype=int)
    J = J_full[:, q_indices]
    manips.append(_yoshikawa_manipulability(J))
    conds.append(_condition_number(J))

manips = np.array(manips)
conds = np.array(conds)

print(f"采样数: {N}")
print(f"\n=== Yoshikawa 可操作度 w(q) ===")
print(f"  min:    {manips.min():.6f}")
print(f"  max:    {manips.max():.6f}")
print(f"  mean:   {manips.mean():.6f}")
print(f"  median: {np.median(manips):.6f}")
print(f"  std:    {manips.std():.6f}")

percentiles = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95]
print(f"\n  百分位数:")
for pct in percentiles:
    val = np.percentile(manips, pct)
    print(f"    P{pct:>2d} = {val:.6f}")

print(f"\n=== 条件数 cond(J) ===")
finite_conds = conds[np.isfinite(conds)]
print(f"  有限比例: {len(finite_conds)}/{N} ({len(finite_conds)/N:.1%})")
if len(finite_conds) > 0:
    print(f"  min:    {finite_conds.min():.2f}")
    print(f"  max:    {finite_conds.max():.2f}")
    print(f"  mean:   {finite_conds.mean():.2f}")
    print(f"  median: {np.median(finite_conds):.2f}")

print(f"\n=== 各阈值下的可行比例 ===")
thresholds = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
for t in thresholds:
    ratio = np.mean(manips >= t)
    print(f"  w >= {t:.3f}:  {ratio:.1%}  ({int(ratio*N)}/{N})")

import pybullet as p
robot.set_joint_state(metadata.q_base, metadata.dq_base)
if created and p.isConnected():
    p.disconnect()
print("\nDone.")
