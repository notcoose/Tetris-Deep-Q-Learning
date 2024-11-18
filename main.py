import gymnasium as gym
import cv2
import torch
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
from deepqnetwork import deepqnetwork
from experience_replay import ExperienceReplay

def preprocess_state(observation):
    # Convert observation dictionary to flat array
    state = []
    for value in observation.values():
        if isinstance(value, np.ndarray):
            state.extend(value.flatten())
    return np.array(state, dtype=np.float32)

if __name__ == "__main__":
    # Initialize environment
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    state, _ = env.reset(seed=42)
    
    # Initialize DQN and Experience Replay
    state_dim = 56  # Update this based on your actual state dimension
    action_dim = env.action_space.n
    dqn = deepqnetwork(state_dim, action_dim)
    replay_buffer = ExperienceReplay(capacity=10000, state_dim=state_dim)

    # Game loop
    terminated = False
    while not terminated:
        print(env.render() + "\n")
        
        # Get current state
        current_state = preprocess_state(state)
        
        # Get action (random for now, you'll implement epsilon-greedy later)
        action = env.action_space.sample()
        
        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Store transition in replay buffer
        replay_buffer.store(
            current_state, 
            action, 
            reward, 
            preprocess_state(next_state), 
            terminated
        )
        
        # Update current state
        state = next_state
        
        cv2.waitKey(300)  # timeout to see the movement
    
    print("Game Over!")
    print(f"Final Buffer Size: {len(replay_buffer)}")
