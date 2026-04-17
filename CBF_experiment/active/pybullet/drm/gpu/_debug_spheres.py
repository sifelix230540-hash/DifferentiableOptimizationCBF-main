"""debug: 看球链在 free 配置上为什么仍然报碰撞"""
import CBF_experiment.active.pybullet.drm.gpu  # noqa
import numpy as np
import torch

from CBF_experiment.active.pybullet.self_collision.vcc_iris.data.config import RobotQueryConfig
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.coal_oracle import CoalSelfCollisionOracle
from CBF_experiment.active.pybullet.self_collision.vcc_iris.robot.robot_model import sample_joint_box
from CBF_experiment.active.pybullet.drm.gpu.gpu_oracle import HybridGPUSelfCollisionOracle
from CBF_experiment.active.pybullet.drm.gpu.batch_fk import batch_fk_link_world
from CBF_experiment.active.pybullet.drm.gpu.sphere_collision import transform_spheres_world


def main():
    oracle = HybridGPUSelfCollisionOracle(
        sphere_pitch=0.035, max_spheres_per_link=40,
        spheres_cache="artifacts/drm_data/link_spheres.json",
    )
    sm = oracle.sm
    gk = oracle.gk

    print("monitored pairs (PyBullet link ids):")
    for i, (a, b) in enumerate(zip(sm.pair_a.cpu().numpy(), sm.pair_b.cpu().numpy())):
        la = sm.link_indices[a]
        lb = sm.link_indices[b]
        print(f"  pair {i:2d}: link {la} <-> link {lb}")

    # 用零位测试
    rng = np.random.default_rng(0)
    q_zero = np.zeros(oracle.dim, dtype=np.float32)
    print(f"\n零位 q: {q_zero}, coal collision: {oracle.coal.is_self_collision(q_zero)}")

    q_t = torch.tensor(q_zero.reshape(1, -1), device="cuda", dtype=torch.float32)
    q_full = sm.compose_full_q(q_t)
    pos, R = batch_fk_link_world(gk, q_full, sm.link_indices)
    pts = transform_spheres_world(pos, R, sm.centers_local)  # (1, L, N, 3)

    # 看每对 link 的最小球-球距离
    pa = pts[:, sm.pair_a, :, :]   # (1, P, N, 3)
    pb = pts[:, sm.pair_b, :, :]
    diff = pa.unsqueeze(3) - pb.unsqueeze(2)
    dist = diff.norm(dim=-1)        # (1, P, N_a, N_b)
    rsum = sm.pair_radii_sum.unsqueeze(0)  # (1, P, N_a, N_b)
    valid = sm.pair_valid.unsqueeze(0)
    sep = dist - rsum
    sep_masked = torch.where(valid, sep, torch.tensor(float("inf"), device="cuda"))
    pair_min_sep = sep_masked.amin(dim=(-1, -2)).cpu().numpy()[0]   # (P,)

    print("\n零位下各对 (min_球距 - 半径和)，负值=球链相交:")
    for i, (a, b) in enumerate(zip(sm.pair_a.cpu().numpy(), sm.pair_b.cpu().numpy())):
        la = sm.link_indices[a]
        lb = sm.link_indices[b]
        s = pair_min_sep[i]
        marker = "  COLLIDE" if s < 0 else ""
        print(f"  link {la} <-> link {lb}: sep = {s*1000:+.1f} mm{marker}")

    # 找一个 coal-free 的 q 看球链怎么样
    q = sample_joint_box(oracle.metadata, rng, 1000).astype(np.float32)
    free_mask = np.array([not oracle.coal.is_self_collision(qq) for qq in q])
    q_free = q[free_mask][:5]
    print(f"\n抽 5 个 coal-free 配置（共 {free_mask.sum()}/{len(q)} free）测球链:")
    for i, qq in enumerate(q_free):
        q_t = torch.tensor(qq.reshape(1, -1), device="cuda", dtype=torch.float32)
        q_full = sm.compose_full_q(q_t)
        pos, R = batch_fk_link_world(gk, q_full, sm.link_indices)
        pts = transform_spheres_world(pos, R, sm.centers_local)
        pa = pts[:, sm.pair_a, :, :]
        pb = pts[:, sm.pair_b, :, :]
        diff = pa.unsqueeze(3) - pb.unsqueeze(2)
        dist = diff.norm(dim=-1)
        rsum = sm.pair_radii_sum.unsqueeze(0)
        valid = sm.pair_valid.unsqueeze(0)
        sep = torch.where(valid, dist - rsum, torch.tensor(float("inf"), device="cuda"))
        worst = sep.amin().item()
        worst_pair = sep.amin(dim=(-1, -2))[0].argmin().item()
        a, b = sm.pair_a[worst_pair].item(), sm.pair_b[worst_pair].item()
        la = sm.link_indices[a]
        lb = sm.link_indices[b]
        print(f"  q={i} 最坏 link {la}<->{lb}: sep = {worst*1000:+.1f} mm")
    oracle.close()


if __name__ == "__main__":
    main()
