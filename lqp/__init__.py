"""Learned QP modules for 2D nominal control.

This package adds a lightweight implementation of the paper-inspired
"MPC-like learned QP controller" and the utilities required to train and
evaluate it in the current 2D obstacle-avoidance project.
"""

from .qp_policy import LearnedQPPolicy, LearnedQPConfig
from .two_d_core import (
    TwoDConfig,
    build_default_problem,
    build_track_reference,
    get_min_h,
    nominal_pd_control,
    step_dynamics,
)

__all__ = [
    "LearnedQPConfig",
    "LearnedQPPolicy",
    "TwoDConfig",
    "build_default_problem",
    "build_track_reference",
    "get_min_h",
    "nominal_pd_control",
    "step_dynamics",
]
