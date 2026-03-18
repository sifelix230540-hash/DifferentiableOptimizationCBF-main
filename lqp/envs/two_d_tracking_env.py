from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..two_d_core import (
    TwoDConfig,
    build_default_problem,
    get_min_h,
    get_obstacles_at,
    reward_terms,
    step_dynamics,
)


class TwoDTrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: TwoDConfig | None = None,
        include_dynamic_obs: bool = False,
        max_steps: int | None = None,
        goal_tol: float = 0.3,
        fixed_start: bool = True,
        randomize_start_radius: float = 0.0,
    ):
        super().__init__()
        default_cfg, static_obs, dyn_defs = build_default_problem()
        self.cfg = replace(default_cfg, **(config.__dict__ if config is not None else {}))
        self.static_obs = static_obs
        self.dyn_defs = dyn_defs if include_dynamic_obs else []
        self.episode_steps = max_steps or self.cfg.max_steps

        self.goal_tol = goal_tol
        self.fixed_start = fixed_start
        self.randomize_start_radius = randomize_start_radius

        obs_dim = 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-self.cfg.u_max,
            high=self.cfg.u_max,
            shape=(2,),
            dtype=np.float32,
        )
        self.state = self.cfg.start.copy()
        self.steps = 0

    def _make_obs(self) -> np.ndarray:
        pos = self.state[:2]
        vel = self.state[2:]
        goal_delta = self.cfg.goal - pos
        obs_list = get_obstacles_at(
            self.steps * self.cfg.dt, self.static_obs, self.dyn_defs,
        )
        nearest = min(
            obs_list,
            key=lambda o: np.linalg.norm(pos - o["center"]) - o["radius"],
        )
        nearest_delta = nearest["center"] - pos
        return np.concatenate([vel, goal_delta, nearest_delta]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.fixed_start:
            pos = self.cfg.start[:2].copy()
        else:
            goal = self.cfg.goal
            xmin, xmax, ymin, ymax = self.cfg.plan_bounds
            r = self.randomize_start_radius if self.randomize_start_radius > 0 else 5.0
            pos = self.cfg.start[:2].copy()
            for _ in range(200):
                angle = self.np_random.uniform(-np.pi, np.pi)
                dist = self.np_random.uniform(0.3, r)
                px = self.cfg.start[0] + dist * np.cos(angle)
                py = self.cfg.start[1] + dist * np.sin(angle)
                px = float(np.clip(px, xmin + 0.2, xmax - 0.2))
                py = float(np.clip(py, ymin + 0.2, ymax - 0.2))
                cand = np.array([px, py], dtype=float)
                if np.linalg.norm(cand - goal) < 0.5:
                    continue
                if all(
                    np.linalg.norm(cand - o["center"])
                    > o["radius"] + self.cfg.safety_margin + 0.1
                    for o in self.static_obs
                ):
                    pos = cand
                    break

        vel = np.zeros(2, dtype=float)
        self.state = np.concatenate([pos, vel])
        self.steps = 0
        return self._make_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -self.cfg.u_max, self.cfg.u_max)
        prev_state = self.state.copy()
        next_state = step_dynamics(self.state, action, self.cfg)
        self.steps += 1

        obs_list = get_obstacles_at(
            self.steps * self.cfg.dt, self.static_obs, self.dyn_defs,
        )
        terms = reward_terms(
            next_state, action, self.cfg.goal, obs_list, self.cfg, x_prev=prev_state,
        )

        dist_to_goal = float(np.linalg.norm(next_state[:2] - self.cfg.goal))
        reached = dist_to_goal < self.goal_tol
        collision = terms["collision"] > 0.5
        out_of_bounds = terms["out_of_bounds"] > 0.5
        timeout = self.steps >= self.episode_steps
        terminated = bool(reached or collision or out_of_bounds)
        truncated = bool(timeout and not terminated)
        self.state = next_state

        if reached:
            terms["reward"] += 100.0

        info = {
            "reached": reached,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "min_h": get_min_h(next_state, obs_list, self.cfg),
            "pos_err": terms["pos_err"],
        }
        return self._make_obs(), float(terms["reward"]), terminated, truncated, info
