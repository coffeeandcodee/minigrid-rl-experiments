"""
Simple interactive MiniGrid player.
Use keyboard to control the agent and understand the environment.
"""

import gymnasium as gym
import minigrid

# Create a simple empty 5x5 environment
env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")

# MiniGrid actions:
# 0 = turn left
# 1 = turn right
# 2 = move forward
# 3 = pick up
# 4 = drop
# 5 = toggle (open doors, etc.)
# 6 = done

action_map = {
    'a': 0,  # turn left
    'd': 1,  # turn right
    'w': 2,  # move forward
    'p': 3,  # pick up
    'o': 4,  # drop
    't': 5,  # toggle
    'q': None  # quit
}

print("=== MiniGrid Interactive Player ===")
print("\nControls:")
print("  w = move forward")
print("  a = turn left")
print("  d = turn right")
print("  p = pick up object")
print("  o = drop object")
print("  t = toggle (open doors, etc.)")
print("  q = quit")
print("\nGoal: Reach the green square!")
print("=" * 35)

obs, info = env.reset()
total_reward = 0
steps = 0

while True:
    action_input = input(f"\nStep {steps} | Total reward: {total_reward} | Action: ").strip().lower()
    
    if action_input == 'q':
        print("Quitting...")
        break
    
    if action_input not in action_map:
        print(f"Invalid action '{action_input}'. Use w/a/d/p/o/t or q to quit.")
        continue
    
    action = action_map[action_input]
    obs, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    steps += 1
    
    if reward > 0:
        print(f"  -> Got reward: {reward}")
    
    if terminated:
        print(f"\n*** Episode finished! ***")
        print(f"Total steps: {steps}")
        print(f"Total reward: {total_reward}")
        
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again == 'y':
            obs, info = env.reset()
            total_reward = 0
            steps = 0
        else:
            break
    
    if truncated:
        print(f"\n*** Episode truncated (max steps reached) ***")
        obs, info = env.reset()
        total_reward = 0
        steps = 0

env.close()
print("Done!")