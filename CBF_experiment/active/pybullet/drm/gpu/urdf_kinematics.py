"""从 PyBullet 提取运动学链信息，构建可在 GPU 上批量 FK 的数据结构。

不依赖 urdfpy/yourdfpy，仅通过 p.getJointInfo 读取：
  - parent link 索引
  - joint type (REV / PRIS / FIXED)
  - parent frame pos & quat (link 在 parent 系下的固定 offset)
  - joint axis (在 link 系下的运动轴)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p
import torch


@dataclass
class KinematicChain:
    """机器人运动学链描述。所有 tensor 在指定 device 上。"""
    n_links: int
    parent: list[int]                       # parent[i] = parent link 索引 (-1 表示 base)
    joint_type: list[str]                   # 'rev' / 'pris' / 'fixed'
    active_joint_idx: list[int]             # link i 对应的 active joint 在 q 向量中的索引 (-1 表示 fixed)
    parent_pos: torch.Tensor                # (n_links, 3)  link 原点在 parent 系下的位置
    parent_R: torch.Tensor                  # (n_links, 3, 3) link 原点在 parent 系下的旋转
    axis: torch.Tensor                      # (n_links, 3)  joint 轴在 link 系下的方向（fixed 时为 0）
    link_names: list[str]


def _quat_to_R(quat) -> np.ndarray:
    """[x,y,z,w] → (3,3) numpy."""
    R = p.getMatrixFromQuaternion(list(quat))
    return np.array(R, dtype=np.float64).reshape(3, 3)


def extract_chain_from_pybullet(
    body_id: int,
    active_joints: list[int],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> KinematicChain:
    """从 PyBullet 提取整个机器人的运动学链。

    Parameters
    ----------
    body_id        : PyBullet body id
    active_joints  : Robot.active_joints 中给出的索引顺序（决定 q 向量布局）
    """
    n = p.getNumJoints(body_id)
    parent: list[int] = []
    joint_type: list[str] = []
    active_joint_idx: list[int] = []
    parent_pos = np.zeros((n, 3), dtype=np.float64)
    parent_R = np.zeros((n, 3, 3), dtype=np.float64)
    axis = np.zeros((n, 3), dtype=np.float64)
    link_names: list[str] = []

    active_set = {int(j): i for i, j in enumerate(active_joints)}

    for i in range(n):
        info = p.getJointInfo(body_id, i)
        link_names.append(info[12].decode())
        jtype = int(info[2])
        ax = info[13]
        ppos = info[14]
        porn = info[15]
        par = int(info[16])

        parent.append(par)
        if jtype == p.JOINT_REVOLUTE:
            joint_type.append("rev")
        elif jtype == p.JOINT_PRISMATIC:
            joint_type.append("pris")
        else:
            joint_type.append("fixed")

        if i in active_set:
            active_joint_idx.append(active_set[i])
        else:
            active_joint_idx.append(-1)

        parent_pos[i] = np.array(ppos, dtype=np.float64)
        parent_R[i] = _quat_to_R(porn)
        axis[i] = np.array(ax, dtype=np.float64)

    return KinematicChain(
        n_links=n,
        parent=parent,
        joint_type=joint_type,
        active_joint_idx=active_joint_idx,
        parent_pos=torch.tensor(parent_pos, dtype=dtype, device=device),
        parent_R=torch.tensor(parent_R, dtype=dtype, device=device),
        axis=torch.tensor(axis, dtype=dtype, device=device),
        link_names=link_names,
    )
