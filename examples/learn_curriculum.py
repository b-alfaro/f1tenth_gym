import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import f1tenth_gym.envs
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from wandb.integration.sb3 import WandbCallback
import wandb
import argparse
import datetime
import numpy as np
import torch

class CustomWandCallback(WandbCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _on_step(self) -> bool:
        # Log custom metric if present
        for key in self.locals['infos'][0]:
            if 'custom' in key:
                wandb.log({key : self.locals['infos'][0][key]}, step=self.num_timesteps)
        return super()._on_step()

def evaluate_model(env, model, n_episodes=10):
    """Evaluate model performance over n episodes"""
    success_count = 0
    total_rewards = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        trunc = False
        
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            episode_reward += reward
            
        if done and not trunc:  # Success if done without timeout
            success_count += 1
        total_rewards.append(episode_reward)
    
    success_rate = success_count / n_episodes
    avg_reward = np.mean(total_rewards)
    return success_rate, avg_reward

def train_stage(stage, timesteps, save_freq, load_path=None):
    """Train model for a specific stage"""
    # Create environment with specific stage
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
        stage=stage
    )
    
    # Set device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize or load model
    if load_path:
        model = PPO.load(load_path, env=env, device=device)
        print(f"Loaded model from {load_path}")
    else:
        name = datetime.datetime.now().strftime("%I:%M%p_%B-%d-%Y")
        model = PPO(
            "MultiInputPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=f"runs/stage_{stage}_{name}", 
            device=device, 
            seed=42
        )
    
    # Train model
    model.learn(
        total_timesteps=timesteps,
        callback=CustomWandCallback(
            gradient_save_freq=0,
            model_save_path=f"models/stage_{stage}",
            verbose=2,
            model_save_freq=save_freq
        ),
    )
    
    # Save final model
    model.save(f"models/stage_{stage}_final")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_freq', type=int, default=700_000)
    parser.add_argument('--timesteps', type=int, default=700_000)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--success_threshold', type=float, default=0.8)
    parser.add_argument('--stage', type=int, default=1, help='Stage to train (1, 2, or 3)')
    args = parser.parse_args()
    
    # Validate stage number
    if args.stage not in [1, 2, 3]:
        raise ValueError("Stage must be 1, 2, or 3")
    
    # Initialize wandb
    run = wandb.init(
        project="parking_curriculum",
        sync_tensorboard=True,
        save_code=False,
    )
    
    print(f"\nStarting training for stage {args.stage}")
    
    # Load previous stage's model if not first stage
    load_path = None if args.stage == 1 else f"models/stage_{args.stage-1}_final.zip"
    
    # Train current stage
    model = train_stage(
        stage=args.stage,
        timesteps=args.timesteps,
        save_freq=args.save_freq,
        load_path=load_path
    )
    
    # Evaluate performance
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
        stage=args.stage
    )
    
    success_rate, avg_reward = evaluate_model(env, model, args.eval_episodes)
    print(f"Stage {args.stage} evaluation:")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    
    wandb.log({
        f"stage_{args.stage}_success_rate": success_rate,
        f"stage_{args.stage}_avg_reward": avg_reward
    })
    
    # If performance is not satisfactory, you might want to retrain
    if success_rate < args.success_threshold:
        print(f"Stage {args.stage} performance below threshold. Consider retraining with different parameters.")
    
    run.finish()

if __name__ == '__main__':
    main() 