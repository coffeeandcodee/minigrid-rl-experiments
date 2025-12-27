import gymnasium as gym
import minigrid

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")



env.reset()

import time
time.sleep(5)

env.close()