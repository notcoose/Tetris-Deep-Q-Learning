import gymnasium as gym
import cv2

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi") # ansi --> uses terminal based graphics to display the game moves list, "human" --> uses a GUI interface to visualize the game
    env.reset(seed=42)

    terminated = False
    while not terminated:
        print(env.render() + "\n")
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
    cv2.waitKey(300) # timeout to see the movement
    print("Game Over!")