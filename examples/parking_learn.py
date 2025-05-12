import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import f1tenth_gym.envs

import gymnasium as gym
from stable_baselines3 import PPO

from wandb.integration.sb3 import WandbCallback
import wandb
import argparse
import datetime

class CustomWandCallback(WandbCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _on_step(self) -> bool:
        # Log custom metric if present
        for key in self.locals['infos'][0]:
            if 'custom' in key:
                wandb.log({key : self.locals['infos'][0][key]}, step=self.num_timesteps)

        return super()._on_step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_freq', type=int, default=1_000_000)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    args = parser.parse_args()
    run = wandb.init(
        project="parking_ppo",
        sync_tensorboard=True,
        save_code=True,
    )

    env = gym.make(
        "parking-v0",
        config={
            "map": "levine_parking",
            "num_agents": 1,
            "timestep": 0.01,
            "num_beams": 36,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "rl_parking"},
            "reset_config": {"type": "rl_random_static"},
        },
    )

    # will be faster on cpu
    name = datetime.date.today("%I:%M%p_%B-%d-%Y")
    model = PPO(
        "MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{name}", device="cpu", seed=42
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=CustomWandCallback(
            gradient_save_freq=0, model_save_path=f"models/{name}", verbose=2, model_save_freq=args.save_freq
        ),
    )
    run.finish()

if __name__ == '__main__':
    main()
