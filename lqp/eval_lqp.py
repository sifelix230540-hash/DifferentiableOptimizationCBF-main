"""
评估 / 可视化训练好的策略。

把所有参数集中在下方 ========== 配置区 ========== 中修改，
然后直接运行本文件即可（无需命令行参数）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "KaiTi"]
matplotlib.rcParams["axes.unicode_minus"] = False

from lqp.cbf_filter import CBFSafetyFilter2D
from lqp.envs import TwoDTrackingEnv
from lqp.two_d_core import get_obstacles_at, nominal_pd_control

# ========== 配置区（修改这里的参数即可） ==========

CHECKPOINT = "lqp/checkpoints/ppo_simple.zip"
VECNORM_PATH = "lqp/checkpoints/ppo_simple_vecnormalize.pkl"  # 留空则不加载
EPISODES = 10
USE_CBF = False
DYNAMIC_OBSTACLES = False
BACKEND = "learned"                # "learned" 或 "pd"
DEVICE = "cpu"
GOAL_TOL = 0.3
FIXED_START = True
MAX_STEPS = 1500
PLOT_TRAJ = True
PLOT_EPISODES = 5

# =================================================


def load_model(checkpoint: str, device: str):
    return PPO.load(str(checkpoint), device=device)


def run_episode(env: TwoDTrackingEnv, model, cbf, seed: int, normalize_obs_fn=None):
    obs, _ = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    ep_min_h = float("inf")
    trajectory = [env.state[:2].copy()]
    start_pos = env.state[:2].copy()
    reached = False
    collision = False
    latencies: list[float] = []

    while not done:
        t0 = time.perf_counter()
        if BACKEND == "learned":
            obs_input = normalize_obs_fn(obs) if normalize_obs_fn else obs
            u_nom, _ = model.predict(obs_input, deterministic=True)
            u_nom = np.asarray(u_nom, dtype=float)
        else:
            u_nom = nominal_pd_control(env.state, env.cfg)

        if cbf is not None:
            obs_list = get_obstacles_at(
                env.steps * env.cfg.dt, env.static_obs, env.dyn_defs,
            )
            u_cmd, _ = cbf.filter(env.state, u_nom, obs_list)
        else:
            u_cmd = u_nom
        latencies.append((time.perf_counter() - t0) * 1000.0)

        obs, reward, terminated, truncated, info = env.step(u_cmd)
        done = terminated or truncated
        ep_return += reward
        ep_min_h = min(ep_min_h, info["min_h"])
        trajectory.append(env.state[:2].copy())
        if terminated and info["reached"]:
            reached = True
        if terminated and info["collision"]:
            collision = True

    final_dist = float(np.linalg.norm(env.state[:2] - env.cfg.goal))
    return {
        "trajectory": np.array(trajectory),
        "start_pos": start_pos,
        "return": ep_return,
        "min_h": ep_min_h,
        "reached": reached,
        "collision": collision,
        "steps": len(trajectory) - 1,
        "final_dist": final_dist,
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def plot_results(episodes_data: list[dict], env: TwoDTrackingEnv):
    fig, ax = plt.subplots(figsize=(10, 6))

    for o in env.static_obs:
        circle = plt.Circle(o["center"], o["radius"], color="gray", alpha=0.5)
        ax.add_patch(circle)
        safe_circle = plt.Circle(
            o["center"], o["radius"] + env.cfg.safety_margin,
            color="gray", alpha=0.15, linestyle="--", fill=False,
        )
        ax.add_patch(safe_circle)

    goal_circle = plt.Circle(env.cfg.goal, GOAL_TOL, color="red", alpha=0.1)
    ax.add_patch(goal_circle)

    cmap = plt.cm.tab10
    n_plot = min(PLOT_EPISODES, len(episodes_data))
    for i in range(n_plot):
        ep = episodes_data[i]
        traj = ep["trajectory"]
        color = cmap(i % 10)
        label = f"ep{i} ({'到达' if ep['reached'] else '碰撞' if ep['collision'] else '超时'})"
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, label=label)
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=7)
        ax.plot(traj[-1, 0], traj[-1, 1], "s", color=color, markersize=7)

    ax.plot(*env.cfg.goal, "r*", markersize=16, label="目标")

    if FIXED_START:
        ax.plot(*env.cfg.start[:2], "g^", markersize=12, label="固定起点")
    else:
        starts = np.array([ep["start_pos"] for ep in episodes_data[:n_plot]])
        ax.scatter(
            starts[:, 0], starts[:, 1],
            marker="^", s=90, c="green", label="各轨迹起点",
        )

    xmin, xmax, ymin, ymax = env.cfg.plan_bounds
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    reaches = sum(e["reached"] for e in episodes_data)
    mode = "固定起点" if FIXED_START else "随机起点"
    ax.set_title(f"PPO 评估 ({mode})  |  到达率={reaches}/{len(episodes_data)}")
    plt.tight_layout()
    plt.show()


def main() -> None:
    env = TwoDTrackingEnv(
        include_dynamic_obs=DYNAMIC_OBSTACLES,
        goal_tol=GOAL_TOL,
        fixed_start=FIXED_START,
        max_steps=MAX_STEPS,
    )

    model = load_model(CHECKPOINT, DEVICE)

    normalize_obs_fn = None
    if VECNORM_PATH and Path(VECNORM_PATH).exists():
        dummy_env = DummyVecEnv([lambda: TwoDTrackingEnv(
            fixed_start=FIXED_START, goal_tol=GOAL_TOL, max_steps=MAX_STEPS,
        )])
        vec_norm = VecNormalize.load(VECNORM_PATH, dummy_env)
        vec_norm.training = False
        vec_norm.norm_reward = False

        def _normalize(obs: np.ndarray) -> np.ndarray:
            return vec_norm.normalize_obs(obs)

        normalize_obs_fn = _normalize
        print(f"已加载 VecNormalize: {VECNORM_PATH}")

    cbf = (
        CBFSafetyFilter2D(env.cfg, n_obs=len(env.static_obs) + len(env.dyn_defs))
        if USE_CBF else None
    )

    episodes_data: list[dict] = []
    for ep in range(EPISODES):
        result = run_episode(env, model, cbf, seed=ep, normalize_obs_fn=normalize_obs_fn)
        status = "到达" if result["reached"] else ("碰撞" if result["collision"] else "超时")
        print(
            f"  ep {ep:3d} | {status} | steps={result['steps']:4d} | "
            f"start=({result['start_pos'][0]:.2f},{result['start_pos'][1]:.2f}) | "
            f"return={result['return']:8.2f} | min_h={result['min_h']:.3f} | "
            f"final_dist={result['final_dist']:.3f}"
        )
        episodes_data.append(result)

    reaches = sum(e["reached"] for e in episodes_data)
    collisions = sum(e["collision"] for e in episodes_data)
    metrics = {
        "reach_rate": reaches / EPISODES,
        "collision_rate": collisions / EPISODES,
        "mean_return": float(np.mean([e["return"] for e in episodes_data])),
        "mean_min_h": float(np.mean([e["min_h"] for e in episodes_data])),
        "mean_final_dist": float(np.mean([e["final_dist"] for e in episodes_data])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes_data])),
        "mean_action_latency_ms": float(np.mean([e["latency_ms"] for e in episodes_data])),
    }
    print("\n========== 汇总指标 ==========")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if PLOT_TRAJ:
        plot_results(episodes_data, env)


if __name__ == "__main__":
    main()
