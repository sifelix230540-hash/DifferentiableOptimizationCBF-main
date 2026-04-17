"""benchmark hybrid GPU oracle vs CPU coal oracle (相同 monitored pairs)。"""
import CBF_experiment.active.pybullet.drm.gpu  # noqa
import numpy as np
import time

from CBF_experiment.active.pybullet.self_collision.self_collision_backend_coal import (
    is_any_pair_collision_fast,
)
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.robot_model import (
    compose_full_q, sample_joint_box,
)
from CBF_experiment.active.pybullet.drm.gpu.gpu_oracle import HybridGPUSelfCollisionOracle


def main():
    oracle = HybridGPUSelfCollisionOracle(
        sphere_pitch=0.035, max_spheres_per_link=50,
        spheres_cache="artifacts/drm_data/link_spheres.json",
        rebuild_spheres=True,
    )
    print(f"monitored links: {oracle.monitored_links}")
    print(f"all pairs ({len(oracle.all_pairs)}): {oracle.all_pairs}")
    print(f"dropped pairs ({len(oracle.dropped_pairs)}): {sorted(oracle.dropped_pairs)}")
    print(f"gpu pairs ({len(oracle.gpu_pairs)})")
    print(f"total spheres: {sum(s.n for s in oracle.link_spheres.values())}")

    rng = np.random.default_rng(0)
    N = 5000
    q = sample_joint_box(oracle.metadata, rng, N).astype(np.float32)

    # warm
    oracle.is_self_collision_batch(q[:10])

    t0 = time.time()
    q_free, stats = oracle.filter_free_batch(q)
    t_hybrid = time.time() - t0
    print(f"\nHybrid filter {N} configs: {t_hybrid*1000:.1f}ms")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # CPU coal baseline：完全相同的 link_models + all_pairs，纯 CPU 串行
    print(f"\nCPU coal baseline (same monitored_pairs)...")
    t0 = time.time()
    coal_collide = np.zeros(N, dtype=bool)
    for i, qq in enumerate(q):
        q_full = compose_full_q(oracle.metadata, qq)
        oracle.robot.set_joint_state(q_full, dq=np.zeros_like(q_full))
        coal_collide[i] = is_any_pair_collision_fast(
            oracle.robot, link_models=oracle.link_models,
            monitored_pairs=oracle.all_pairs,
        )
    t_coal = time.time() - t0
    n_free_coal = int(np.sum(~coal_collide))
    print(f"CPU coal {N}: {t_coal*1000:.1f}ms, free={n_free_coal}")
    speedup = t_coal / t_hybrid
    print(f"speedup vs CPU coal (single thread): {speedup:.2f}x")

    # Correctness：hybrid 结果应等价 coal（保守：hybrid 报 collide 是允许的，
    # 但绝不能 hybrid 报 free 而 coal 报 collide）
    hybrid_collide = oracle.is_self_collision_batch(q)
    n_fp = int(np.sum(hybrid_collide & ~coal_collide))   # hybrid 报 collide, coal free
    n_fn = int(np.sum(~hybrid_collide & coal_collide))   # hybrid 报 free, coal collide  ←危险
    print(f"\nCorrectness vs coal:")
    print(f"  hybrid collide ∩ coal free  (over-conservative): {n_fp}")
    print(f"  hybrid free    ∩ coal collide (UNSAFE FN!):     {n_fn}")
    oracle.close()


if __name__ == "__main__":
    main()
