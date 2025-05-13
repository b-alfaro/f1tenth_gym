import gymnasium as gym
from stable_baselines3 import PPO
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import f1tenth_gym.envs
import argparse
import numpy as np

def main(n=10, timeout=10):
    '''
    perform eval on n environments with a limit of timeout seconds
    '''
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO model in F1TENTH environment')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    args = parser.parse_args()
    model_path = args.model
    model = PPO.load(model_path, print_system_info=True, device="cpu")
    eval_env = gym.make(
        "f1tenth_gym:parking-v0",
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
        render_mode="human",
    )
    for _ in range(n):
        obs, info = eval_env.reset()
        done = False
        trunc = False
        steps = 0
        while not done and not trunc:
            action, _states = model.predict(obs, deterministic=True)
            # action = np.zeros((1,2))
            obs, reward, done, trunc, info = eval_env.step(action)
            print(obs['pose'], reward, eval_env.unwrapped.waypoint_idx)
            # print(done)
            # print(obs['waypoint_idx'])
            steps += 1
            eval_env.render()

            # VecEnv resets automatically
            # if done:
            #   obs = env.reset()
        eval_env.close()

if __name__ == '__main__':
    main()