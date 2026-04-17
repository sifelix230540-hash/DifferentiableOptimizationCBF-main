"""GPU 批量正运动学：基于 pytorch_kinematics 从 URDF 直接构建 chain。

提供与 PyBullet `getLinkState` 一致的世界 link frame 位姿。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
import torch
import pytorch_kinematics as pk


@dataclass
class GPUKinematics:
    """GPU FK 包装器。"""
    chain: object                          # pytorch_kinematics SerialChain or Chain
    joint_names: list[str]                 # chain 内部使用的 joint 名顺序
    active_joint_idx_in_chain: list[int]   # active_joints[i] 对应 chain joint_names 中的索引
    link_names: list[str]                  # 全部 link 名（chain 内部）
    pb_link_idx_to_chain: dict[int, str]   # PyBullet link 索引 → chain 中的 link 名
    device: torch.device


def build_gpu_kinematics(
    urdf_path: str | Path,
    body_id: int,
    active_joints: list[int],
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> GPUKinematics:
    """从 URDF 文件构建 GPU FK chain，并建立 active joint / link 索引映射。

    Parameters
    ----------
    urdf_path     : URDF 文件路径
    body_id       : 已加载到 PyBullet 的机器人 body id（用于读 joint/link 名）
    active_joints : Robot.active_joints 中的 PyBullet joint 索引顺序
    """
    urdf_bytes = Path(urdf_path).read_bytes()
    chain = pk.build_chain_from_urdf(urdf_bytes)
    chain = chain.to(dtype=dtype, device=device)

    chain_joint_names = chain.get_joint_parameter_names(exclude_fixed=True)
    chain_link_names = chain.get_link_names()

    pb_link_idx_to_chain: dict[int, str] = {}
    pb_joint_to_chain_idx: dict[int, int] = {}

    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        link_name = info[12].decode()
        joint_name = info[1].decode()
        if link_name in chain_link_names:
            pb_link_idx_to_chain[i] = link_name
        if joint_name in chain_joint_names:
            pb_joint_to_chain_idx[i] = chain_joint_names.index(joint_name)

    active_joint_idx_in_chain = [pb_joint_to_chain_idx.get(int(j), -1) for j in active_joints]

    return GPUKinematics(
        chain=chain,
        joint_names=chain_joint_names,
        active_joint_idx_in_chain=active_joint_idx_in_chain,
        link_names=chain_link_names,
        pb_link_idx_to_chain=pb_link_idx_to_chain,
        device=torch.device(device),
    )


def batch_fk_link_world(
    gk: GPUKinematics,
    q_batch: torch.Tensor,
    pb_link_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """批量计算指定 PyBullet link 的世界变换。

    Parameters
    ----------
    q_batch          : (B, dof) 关节配置（按 active_joints 顺序）
    pb_link_indices  : 要查询的 PyBullet link 索引列表

    Returns
    -------
    pos_batch : (B, L, 3) 世界位置
    R_batch   : (B, L, 3, 3) 世界旋转
    """
    B, dof = q_batch.shape
    n_chain_joints = len(gk.joint_names)

    # 重新映射到 chain 的 joint 顺序
    q_chain = torch.zeros(B, n_chain_joints, device=q_batch.device, dtype=q_batch.dtype)
    for i, ci in enumerate(gk.active_joint_idx_in_chain):
        if ci >= 0:
            q_chain[:, ci] = q_batch[:, i]

    fk_dict = gk.chain.forward_kinematics(q_chain)

    pos_list = []
    R_list = []
    for li in pb_link_indices:
        link_name = gk.pb_link_idx_to_chain.get(int(li))
        if link_name is None or link_name not in fk_dict:
            raise KeyError(f"link {li} not found in chain (URDF link name missing)")
        T = fk_dict[link_name]
        m = T.get_matrix()  # (B, 4, 4)
        pos_list.append(m[:, :3, 3])
        R_list.append(m[:, :3, :3])

    pos_batch = torch.stack(pos_list, dim=1)  # (B, L, 3)
    R_batch = torch.stack(R_list, dim=1)      # (B, L, 3, 3)
    return pos_batch, R_batch
