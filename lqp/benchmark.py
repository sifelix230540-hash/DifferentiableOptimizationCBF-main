from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from lqp.cbf_filter import CBFSafetyFilter2D
from lqp.envs import TwoDTrackingEnv
from lqp.qp_policy import LearnedQPPolicy
from lqp.train_lqp import LearnedQPActorCriticPolicy
from lqp.two_d_core import (
    build_default_problem,
    get_obstacles_at,
    get_path_target,
    nominal_pd_control,
    plan_static_rrt_star,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark nominal controllers on the 2D task.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--dynamic-obstacles", action="store_true")
    parser.add_argument("--use-cbf", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_policy(checkpoint: str, device: str):
    path = Path(checkpoint)
    if path.suffix == ".zip":
        model = PPO.load(
            str(path),
            device=device,
            custom_objects={"policy_class": LearnedQPActorCriticPolicy},
        )
        return "sb3", model
    return "pt", LearnedQPPolicy.load_checkpoint(path, map_location=device)


def benchmark_backend(
    backend: str,
    env: TwoDTrackingEnv,
    episodes: int,
    checkpoint: str = "",
    use_cbf: bool = False,
    device: str = "cpu",
) -> dict[str, float]:
    policy_kind = None
    policy = None
    if backend == "learned_qp" and checkpoint:
        policy_kind, policy = load_policy(checkpoint, device)
    cbf = CBFSafetyFilter2D(env.cfg, n_obs=len(env.static_obs) + len(env.dyn_defs)) if use_cbf else None
    _, static_obs, _ = build_default_problem()
    rrt_path = plan_static_rrt_star(env.cfg.start[:2], env.cfg.goal, static_obs, env.cfg)

    returns = []
    min_hs = []
    reached = 0
    collisions = 0
    latencies = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_return = 0.0
        ep_min_h = float("inf")

        while not done:
            t0 = time.perf_counter()
            if backend == "pd":
                u_nom = nominal_pd_control(env.state, env.cfg)
            elif backend == "rrt":
                target = get_path_target(rrt_path, env.state[:2], env.cfg.path_lookahead)
                u_nom = nominal_pd_control(env.state, env.cfg, target=target)
            elif backend == "learned_qp":
                if policy is None:
                    raise ValueError("learned_qp benchmark requires --checkpoint")
                if policy_kind == "sb3":
                    u_nom, _ = policy.predict(obs, deterministic=True)
                    u_nom = np.asarray(u_nom, dtype=float)
                else:
                    u_nom = policy.act_numpy(env.state, env.cfg.goal, device=device)
            else:
                raise ValueError(f"unknown backend: {backend}")

            if cbf is not None:
                obs_list = get_obstacles_at(env.steps * env.cfg.dt, env.static_obs, env.dyn_defs)
                u_cmd, _ = cbf.filter(env.state, u_nom, obs_list)
            else:
                u_cmd = u_nom
            latencies.append((time.perf_counter() - t0) * 1000.0)

            obs, reward, terminated, truncated, info = env.step(u_cmd)
            done = terminated or truncated
            ep_return += reward
            ep_min_h = min(ep_min_h, info["min_h"])
            if terminated and info["reached"]:
                reached += 1
            if terminated and info["collision"]:
                collisions += 1

        returns.append(ep_return)
        min_hs.append(ep_min_h)

    return {
        "mean_return": float(np.mean(returns)),
        "reach_rate": reached / float(episodes),
        "collision_rate": collisions / float(episodes),
        "mean_min_h": float(np.mean(min_hs)),
        "mean_latency_ms": float(np.mean(latencies)),
    }


def main() -> None:
    args = parse_args()
    env = TwoDTrackingEnv(include_dynamic_obs=args.dynamic_obstacles)
    results = {}
    for backend in ["pd", "rrt"] + (["learned_qp"] if args.checkpoint else []):
        results[backend] = benchmark_backend(
            backend=backend,
            env=env,
            episodes=args.episodes,
            checkpoint=args.checkpoint,
            use_cbf=args.use_cbf,
            device=args.device,
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
