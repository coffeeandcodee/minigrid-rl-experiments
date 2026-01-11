import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import numpy as np

# 1. Create environment
# ImgObsWrapper converts the dict observation to just the image (required for SB3)
env = gym.make("MiniGrid-Empty-5x5-v0")
env = ImgObsWrapper(env)
env = Monitor(env)

# 2. Create and train agent
model = PPO("MlpPolicy", env, verbose=1, seed=42)
model.learn(total_timesteps=50_000)

# 3. Evaluate
obs, _ = env.reset()
total_reward = 0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Total reward: {total_reward}")