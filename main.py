import gymnasium as gym
import cv2
import torch
import numpy as np
import datetime as dt
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

class TetrisAgent:
    def run(self, is_training = True, render_mode = "ansi"):
        #creating and writing to log file
        if(is_training):
            traininglog = open("traininglog.txt", "w")

            start = dt.datetime.now()
            traininglog.write(f"Start time: {start}\n")

        # Initialize environment
        env = gym.make("tetris_gymnasium/Tetris", render_mode = render_mode)
        rewards = []
    
        # get action and state space dims
        state_dim = preprocess_state(state).shape[0]  #temp
        action_dim = env.action_space.n #number of actions

        #"policy" network
        dqn = deepqnetwork(state_dim, action_dim)

        if is_training:
            replay_buffer = ExperienceReplay(capacity=10000, state_dim=state_dim)

            #initialize greedy params
            epsilon = 1.0  #initial exploration rate
            epsilon_min = 0.1  #min exploration rate
            epsilon_decay = 0.995  #decay factor
            epsilon_hist = []  #store epsilon values for plotting
            gamma = 0.99

            #"target" network
            target_dqn = deepqnetwork(state_dim, action_dim)

            #copies weights and biases from policy network to target network
            target_dqn.load_state_dict(dqn.state_dict())

            #syncs target network with policy network
            count = 0

        #else if not training
            # need to add loading of model
            #target_dqn.load_state_dict(torch.load())
            #target_dqn.eval()

        #arbitrary number of episodes, change as you wish
        for episode in range(100000):
            terminated = False
            episode_reward = 0.0
            state, _ = env.reset(seed=42)

            #1000 is arbitrary, change as you wish for early stopping
            while not terminated and episode_reward < 1000:
                print(env.render() + "\n")
        
                # Get current state
                current_state = preprocess_state(state)
        
                # Get action using epsilon-greedy algorithm
                action = epsilon_greedy_action(dqn, current_state, epsilon, action_dim)
        
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if is_training:
                    # Store transition in replay buffer
                    replay_buffer.store(
                        current_state, 
                        action, 
                        reward, 
                        preprocess_state(next_state), 
                        terminated
                    )
                    count += 1    

            traininglog.write(f"Episode: {episode}, Reward: {episode_reward}\n")
            rewards.append(episode_reward)
            
            #update epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            
            # Update current state
            state = next_state
        
            cv2.waitKey(300)  # timeout to see the movement
        
        print("Game Over!")
        print(f"Final Buffer Size: {len(replay_buffer)}")
        traininglog.close()

if __name__ == "__main__":
    agent = TetrisAgent()
    agent.run()