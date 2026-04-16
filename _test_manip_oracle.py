"""Quick smoke test for ManipulabilityOracle."""
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.manipulability_oracle import ManipulabilityOracle
from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
import numpy as np

oracle = ManipulabilityOracle(
    RobotQueryConfig(),
    manipulability_threshold=0.005,
    use_position_only=False,
)
print(f"dim = {oracle.dim}")
print(f"joint_limits = {oracle.metadata.joint_limits}")

rng = np.random.default_rng(42)
lo = np.array([l for l, _ in oracle.metadata.joint_limits])
hi = np.array([h for _, h in oracle.metadata.joint_limits])

n_total, n_good, n_bad = 0, 0, 0
for _ in range(200):
    q = rng.uniform(lo, hi)
    n_total += 1
    if oracle.is_self_collision(q):
        n_bad += 1
    else:
        n_good += 1
        if n_good <= 3:
            m = oracle.query(q)
            manip = m["manipulability"]
            cond = m["condition_number"]
            clr = m["min_clearance"]
            print(f"  good q: manip={manip:.6f}  cond={cond:.2f}  clearance={clr:.6f}")

print(f"\nResults: total={n_total}  good(manip>=thresh)={n_good}  bad(manip<thresh)={n_bad}")
print(f"Feasible ratio: {n_good/n_total:.2%}")

seg_ok = oracle.segment_is_collision_free(
    rng.uniform(lo, hi), rng.uniform(lo, hi), num_steps=8
)
print(f"segment_is_collision_free test: {seg_ok}")

oracle.close()
print("Done.")
