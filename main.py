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

def epsilon_greedy_action(dqn, state, epsilon, action_dim):
    #random number is used to decide action
    if np.random.rand() < epsilon:
        #explore- choose a random action
        return np.random.randint(action_dim)
    else:
        #exploit- choose the action with the highest q-val
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = dqn(state_tensor)
        return torch.argmax(q_values).item()

if __name__ == "__main__":
    # Initialize environment
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    state, _ = env.reset(seed=42)
    
    # Initialize DQN and Experience Replay
    state_dim = preprocess_state(state).shape[0] 
    action_dim = env.action_space.n
    dqn = deepqnetwork(state_dim, action_dim)
    replay_buffer = ExperienceReplay(capacity=10000, state_dim=state_dim)

    #initialize greedy params
    epsilon = 1.0  #initial exploration rate
    epsilon_min = 0.1  #min exploration rate
    epsilon_decay = 0.995  #decay factor
    gamma = 0.99

    # Game loop
    terminated = False
    while not terminated:
        print(env.render() + "\n")
        
        # Get current state
        current_state = preprocess_state(state)
        
        # Get action using epsilon-greedy algorithm
        action = epsilon_greedy_action(dqn, current_state, epsilon, action_dim)
        
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

        #update epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # Update current state
        state = next_state
        
        cv2.waitKey(300)  # timeout to see the movement
    
    print("Game Over!")
    print(f"Final Buffer Size: {len(replay_buffer)}")
