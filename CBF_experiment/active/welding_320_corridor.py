from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class JointSpaceCorridor:
    A: np.ndarray
    b: np.ndarray
    source: str

    def contains(self, q: np.ndarray, atol: float = 1e-9) -> bool:
        q = np.array(q, dtype=float)
        return bool(np.all(self.A @ q <= self.b + atol))


@dataclass(frozen=True)
class PyBulletModelSnapshot:
    robot_urdf: str
    workpiece_urdf: str
    active_joint_names: tuple[str, ...]
    active_joint_limits: tuple[tuple[float, float], ...]

    def signature(self) -> str:
        payload = "|".join([
            self.robot_urdf,
            self.workpiece_urdf,
            ",".join(self.active_joint_names),
            ",".join(f"{lo:.6f}:{hi:.6f}" for lo, hi in self.active_joint_limits),
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
