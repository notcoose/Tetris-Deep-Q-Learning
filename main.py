import cv2
import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
env.reset(seed=42)

terminated = False
while not terminated:
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    key = cv2.waitKey(150) # timeout to see the movement
print("Game Over!")
