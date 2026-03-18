# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **CBF-QP (Control Barrier Function - Quadratic Programming) based safe robot navigation and obstacle avoidance**, forked from Dai et al. (IEEE RA-L 2023). Extended with custom experiments for a JAKA 6-DOF arm on a 2-DOF mobile base (8 DOF total), targeting welding trajectory planning (免示教焊接轨迹规划).

## Install & Run

```bash
# Install the original library package (editable)
pip install -e .

# Install lqp module dependencies
pip install -r requirements-lqp.txt
```

### Entry Points

```bash
# JAKA 3D CBF-QP experiments (main active work)
python CBF_experiment/3_11_jaka_3d_cbf_qp_experiment.py   # single obstacle
python CBF_experiment/3_12_jaka_3d_cbf_qp_experiment.py   # multi-obstacle + URDF workpiece
python CBF_experiment/3_13_jaka_3d_cbf_qp_experiment.py   # multi-obstacle + MPC-DCBF

# 2D benchmarks
python CBF_experiment/3_11_2D_CFB_QP.py
python CBF_experiment/3_11_2D_CFB_MCP_QP.py
python CBF_experiment/3_11_2D_CFB_MCP_QP_deg2.py

# Original FR3 experiments (require Julia + DifferentiableCollisions.jl)
python -m DifferentiableOptimizationCBF.unicycle_exp
python -m DifferentiableOptimizationCBF.three_blocks_exp
python -m DifferentiableOptimizationCBF.two_walls_exp

# Neural SDF training
python CBF_Net/3_13_CBF_Net.py

# Learned QP policy training (PPO via Stable-Baselines3)
python -m lqp.train_lqp
```

No test suite, linter, or CI/CD pipeline exists.

## Architecture

### `CBF_experiment/` — Custom JAKA Experiments (Primary Development)

All JAKA experiments use PyBullet simulation. File naming uses `3_XX_` date-based versioning (March 11, 12, 13).

Key abstractions (in 3_12 and 3_13 versions):
- `ExperimentConfig` — dataclass holding all experiment parameters
- `SimulationScene` — PyBullet world setup
- `JakaRobot` — robot wrapper (FK, Jacobian via PyBullet)
- `Obstacle` ABC → `SphereObstacle`, `PlateObstacle`, `URDFObstacle` — factory via `create_obstacle()`
- `Controller` ABC → `CBFQPController`, `MPCDCBFController` — factory via `create_controller()`
- `LineSlerpTrajectory` — reference trajectory generation
- `AvoidanceExperiment` — orchestrates the full experiment loop

QP solving: `scipy.optimize.minimize` with SLSQP method (not ProxSuite).

### `DifferentiableOptimizationCBF/` — Original Library (FR3 Robot)

- `BaseController` → `ThreeBlocksController` / `TwoWallsController` — FR3 robot controllers using Pinocchio for kinematics and Julia for collision distance
- `CBFQPSolver` — ProxSuite dense QP solver wrapper
- `envs/` — PyBullet environments (`FR3BaseEnv`, `ThreeBlocksEnv`, `TwoWallsEnv`, `UnicycleEnv`)
- `dc_utils/*.jl` — Julia scripts for DifferentiableCollisions CBF gradient computation
- `robots/` — FR3 URDF and mesh files

### `lqp/` — Learned QP Policy

RL-trained structured QP controller used as nominal controller with CBF-QP safety filter.

- `pdhg.py` — differentiable PDHG unrolled QP solver (PyTorch)
- `qp_policy.py` — `LearnedQPPolicy(nn.Module)` with learnable P, H, q(state), b(state)
- `cbf_filter.py` — CVXPY-based CBF-QP safety filter for 2D
- `two_d_core.py` — 2D dynamics, RRT* planner, PD controller
- `envs/two_d_tracking_env.py` — Gymnasium environment
- `train_lqp.py` — PPO training via Stable-Baselines3

### `CBF_Net/` — Neural SDF

`NeuralSDF` network learning signed distance fields from point clouds (surface loss + Eikonal constraint). Intended as CBF h(x) for arbitrary obstacle shapes.

## Key Dependencies

- **3D simulation**: pybullet, pinocchio (FR3 only)
- **QP solvers**: scipy (JAKA), cvxpy+OSQP (2D), proxsuite (FR3)
- **ML**: torch, gymnasium, stable-baselines3
- **Julia**: pyjulia + DifferentiableCollisions.jl (FR3 only, not needed for JAKA)
- **Visualization**: matplotlib, imageio

## Important Notes

- The JAKA robot URDF path is hardcoded to a local Windows path — must be updated per machine.
- The original FR3 experiments require a Julia installation; JAKA experiments are pure Python.
- Code comments and UI strings are primarily in Chinese.
- The 2D experiments use CVXPY; the 3D JAKA experiments use scipy SLSQP — these are different solver stacks.
