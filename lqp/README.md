# Learned QP For 2D CBF Project

## What is implemented

This folder implements the paper idea from `MPC-Inspired Reinforcement Learning for Verifiable Model-Free Control`
in a form that fits the current repository:

- the controller keeps a **QP structure**
- QP parameters are **partly learned**
- the online forward pass uses a **fixed number of PDHG iterations**
- the learned controller is used as a **nominal controller**
- the existing `CBF-QP` remains the **safety filter**

## Mapping from paper to this project

Paper architecture:

1. Build a QP policy in standard form
2. Unroll a small number of solver iterations
3. Learn the QP parameters with RL
4. Use the first control block as the action

Current project mapping:

1. `lqp/qp_policy.py`
   - defines the learnable QP controller
   - keeps `P` and `H` state-independent
   - generates `q` and `b` from state and reference by affine maps
2. `lqp/pdhg.py`
   - implements the fixed-step PDHG-style unrolled solver
3. `lqp/envs/two_d_tracking_env.py`
   - wraps the existing 2D mass-point task as a `gymnasium` environment
4. existing `CBF_experiment/history/3_11_2D_CFB_MCP_QP_deg2.py`
   - can switch its nominal controller backend to `learned_qp`
   - still applies `CBF-QP` as the online safety layer

## Why nominal + CBF

The paper focuses on learning a structured QP controller, but this repo is centered on safe navigation.
For the first integration stage, it is safer to:

- let RL learn a performant nominal policy
- keep collision avoidance in the existing `CBF-QP`
- compare against `pd`, `rrt`, and `mpc` baselines

This gives a cleaner diagnosis:

- if the nominal controller is poor, the RL side is the bottleneck
- if the nominal controller is reasonable but collisions remain, the CBF filter or linearization is the bottleneck

## Training flow

1. create the 2D environment from `lqp/two_d_core.py`
2. train `LearnedQPPolicy` with PPO
3. save a checkpoint
4. evaluate the policy with and without CBF filtering
5. optionally plug it into the benchmark script as a new nominal backend

## Important limitation

This integration is **paper-inspired**, not a full reproduction of the paper.

- the paper gives guarantees for linear constrained systems
- this project contains obstacle avoidance and a nonlinear safety layer
- therefore the learned QP here should be treated as a structured nominal controller, not as a drop-in replacement for the full safe controller
