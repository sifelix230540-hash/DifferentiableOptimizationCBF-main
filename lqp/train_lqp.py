from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from lqp.envs import TwoDTrackingEnv
from lqp.qp_policy import LearnedQPConfig, LearnedQPPolicy


class PassThroughExtractor(nn.Module):
    def __init__(self, features_dim: int):
        super().__init__()
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return features, features

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return features


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class LearnedQPActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        self.qp_config = kwargs.pop("qp_config", LearnedQPConfig())
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PassThroughExtractor(self.features_dim)

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if not hasattr(self.action_dist, "proba_distribution_net"):
            raise NotImplementedError("Current action distribution does not support Gaussian PPO.")

        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi,
            log_std_init=self.log_std_init,
        )
        self.qp_actor = LearnedQPPolicy(self.qp_config)
        self.value_net = ValueNetwork(self.features_dim)

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    @staticmethod
    def _split_obs(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = obs[..., :4]
        ref = obs[..., :2] + obs[..., 4:6]
        return state, ref

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        state, ref = self._split_obs(latent_pi)
        mean_actions = self.qp_actor(state, ref)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learned QP nominal controller.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--episodes-per-epoch", type=int, default=12)
    parser.add_argument("--total-timesteps", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--exploration-std", type=float, default=0.35)
    parser.add_argument("--output", type=str, default="lqp/checkpoints/learned_qp_nominal.pt")
    parser.add_argument("--sb3-output", type=str, default="")
    parser.add_argument("--logdir", type=str, default="runs/learned_qp_nominal")
    parser.add_argument("--dynamic-obstacles", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments (use SubprocVecEnv when > 1)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_env(dynamic_obstacles: bool, seed: int, rank: int = 0):
    def _init():
        env = TwoDTrackingEnv(include_dynamic_obs=dynamic_obstacles)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    probe_env = TwoDTrackingEnv(include_dynamic_obs=args.dynamic_obstacles)
    cfg = LearnedQPConfig(u_max=probe_env.cfg.u_max)
    total_timesteps = args.total_timesteps
    if total_timesteps <= 0:
        total_timesteps = args.epochs * args.episodes_per_epoch * probe_env.episode_steps

    n_envs = max(1, args.n_envs)
    env_fns = [make_env(args.dynamic_obstacles, args.seed, rank=i) for i in range(n_envs)]
    if n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
        print(f"Using SubprocVecEnv with {n_envs} parallel environments")
    else:
        vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                           clip_reward=10.0, gamma=args.gamma)
    model = PPO(
        policy=LearnedQPActorCriticPolicy,
        env=vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.entropy_coef,
        vf_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        tensorboard_log=args.logdir,
        device=args.device,
        seed=args.seed,
        verbose=1,
        policy_kwargs={
            "qp_config": cfg,
            "log_std_init": float(np.log(args.exploration_std)),
            "ortho_init": False,
        },
    )

    model.learn(total_timesteps=total_timesteps, tb_log_name="learned_qp_ppo")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.policy.qp_actor.save_checkpoint(out_path)

    sb3_path = Path(args.sb3_output) if args.sb3_output else out_path.with_suffix(".zip")
    sb3_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(sb3_path))

    norm_path = out_path.with_name("vecnormalize.pkl")
    vec_env.save(str(norm_path))

    vec_env.close()
    print(f"saved actor checkpoint to {out_path}")
    print(f"saved sb3 PPO model to {sb3_path}")
    print(f"saved VecNormalize stats to {norm_path}")


if __name__ == "__main__":
    main()
