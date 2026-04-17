"""GPU 加速：批量 FK + capsule 自碰撞粗筛 + 批量 COLRM。

注意 import 顺序：torch 必须在 PyBullet/COAL 等 native 库之前 import，
否则 Windows 上会出现 fbgemm.dll 加载失败（libomp 冲突）。
"""
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch as _torch  # noqa: F401  必须最先导入
