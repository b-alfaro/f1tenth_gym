# import sys
# import os
# import argparse
# import datetime

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import f1tenth_gym.envs
# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback
# from wandb.integration.sb3 import WandbCallback
# import wandb



# class CustomWandbCallback(WandbCallback):
#     def _on_step(self) -> bool:
#         # Log custom metrics if present in infos
#         if "infos" in self.locals and self.locals["infos"]:
#             info = self.locals["infos"][0]
#             for key, value in info.items():
#                 if 'custom' in key:
#                     wandb.log({key: value}, step=self.num_timesteps)
#         return super()._on_step()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--save_freq', type=int, default=1_000_000)
#     parser.add_argument('--timesteps', type=int, default=20_000_000)
#     args = parser.parse_args()

#     run = wandb.init(
#         project="parking_ppo",
#         sync_tensorboard=True,
#         save_code=False,
#     )

#     # Environment setup
#     env = gym.make(
#         "parking-v0",
#         config={
#             "map": "levine_parking",
#             "num_agents": 1,
#             "timestep": 0.01,
#             "num_beams": 36,
#             "integrator": "rk4",
#             "control_input": ["speed", "steering_angle"],
#             "observation_config": {"type": "rl_parking"},
#             "reset_config": {"type": "rl_random_static"},
#         },
#     )

#     # Timestamped run name
#     name = datetime.datetime.now().strftime("%I:%M%p_%B-%d-%Y")

#     # Create model
#     from stable_baselines3.common.schedules import LinearSchedule


    # model = PPO(
    #     "MultiInputPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log=f"runs/{name}",
    #     device="cpu",
    #     seed=42,
    #     learning_rate=linear_schedule(3e-4),
    #     ent_coef=linear_schedule(0.01)
    # )


#     # Callbacks
#     checkpoint_callback = CheckpointCallback(
#         save_freq=args.save_freq,
#         save_path=f"models/{name}",
#         name_prefix="rl_model",
#         verbose=1,
#     )

#     custom_wandb_callback = CustomWandbCallback(
#         gradient_save_freq=0,
#         verbose=2
#     )

#     # Learn with progress bar and both callbacks
#     model.learn(
#         total_timesteps=args.timesteps,
#         callback=[checkpoint_callback, custom_wandb_callback],
#         progress_bar=True
#     )

#     run.finish()

# if __name__ == '__main__':
#     main()

import sys
import os
import argparse
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import f1tenth_gym.envs
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from wandb.integration.sb3 import WandbCallback
import wandb


def linear_schedule(initial_value):
    def schedule(progress):
        return progress * initial_value
    return schedule
class CustomWandbCallback(WandbCallback):
    def _on_step(self) -> bool:
        if "infos" in self.locals and self.locals["infos"]:
            info = self.locals["infos"][0]
            for key, value in info.items():
                if 'custom' in key:
                    wandb.log({key: value}, step=self.num_timesteps)
        return super()._on_step()

class EntropySchedulerCallback(BaseCallback):
    def __init__(self, initial_ent_coef: float, final_ent_coef: float, target_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.target_timesteps = target_timesteps

    def _on_step(self) -> bool:
        progress = min(1.0, self.model.num_timesteps / float(self.target_timesteps))
        new_ent_coef = self.initial_ent_coef + progress * (self.final_ent_coef - self.initial_ent_coef)
        if hasattr(self.model, "ent_coef"):
            self.model.ent_coef = new_ent_coef
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_freq', type=int, default=1_000_000)
    parser.add_argument('--timesteps', type=int, default=20_000_000)
    args = parser.parse_args()

    run = wandb.init(
        project="parking_ppo",
        sync_tensorboard=True,
        save_code=False,
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

    name = datetime.datetime.now().strftime("%I:%M%p_%B-%d-%Y")

    initial_ent = 0.01
    final_ent = 0.0



    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{name}",
        device="cpu",
        seed=42,
        learning_rate=3e-4,
        # learning_rate=linear_schedule(3e-4),
        ent_coef=linear_schedule(0.01)
    )


    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=f"models/{name}",
        name_prefix="rl_model",
        verbose=1,
    )

    custom_wandb_callback = CustomWandbCallback(
        gradient_save_freq=0,
        verbose=2
    )

    entropy_scheduler = EntropySchedulerCallback(
        initial_ent_coef=initial_ent,
        final_ent_coef=final_ent,
        target_timesteps=args.timesteps
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, custom_wandb_callback, entropy_scheduler],
        progress_bar=True
    )

    run.finish()

if __name__ == '__main__':
    main()