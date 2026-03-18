"""
极简 PPO 训练：标准 MLP 策略 + 固定起点 [0,0] -> [5,0]

把所有参数集中在下方 ========== 配置区 ========== 中修改，
然后直接运行本文件即可。
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from lqp.envs import TwoDTrackingEnv

# ========== 配置区 ==========

TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 8
FIXED_START = True
GOAL_TOL = 0.3
MAX_STEPS = 1500
SEED = 42
OUTPUT_DIR = "lqp/checkpoints"
MODEL_NAME = "ppo_simple"
LOGDIR = "runs/ppo_simple"

# PPO 超参数
LR = 3e-4
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# MLP 策略网络结构
NET_ARCH = [128, 128]

# =================================================


def make_env(rank: int):
    def _init():
        env = TwoDTrackingEnv(
            fixed_start=FIXED_START,
            goal_tol=GOAL_TOL,
            max_steps=MAX_STEPS,
        )
        env.reset(seed=SEED + rank)
        return Monitor(env)
    return _init


def main() -> None:
    env_fns = [make_env(i) for i in range(N_ENVS)]
    if N_ENVS > 1:
        vec_env = SubprocVecEnv(env_fns)
        print(f"SubprocVecEnv x {N_ENVS}")
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=GAMMA,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        tensorboard_log=LOGDIR,
        device="cpu",
        seed=SEED,
        verbose=1,
        policy_kwargs={"net_arch": NET_ARCH},
    )

    print(f"开始训练: {TOTAL_TIMESTEPS} 步, 固定起点={FIXED_START}, "
          f"goal_tol={GOAL_TOL}, max_steps={MAX_STEPS}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="ppo_simple")

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"{MODEL_NAME}.zip"
    norm_path = out / f"{MODEL_NAME}_vecnormalize.pkl"

    model.save(str(model_path))
    vec_env.save(str(norm_path))
    vec_env.close()

    print(f"\n模型已保存: {model_path}")
    print(f"归一化统计: {norm_path}")


if __name__ == "__main__":
    main()
