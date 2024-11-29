import gymnasium as gym
import cv2
import torch
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from seaborn import set_style
from tetris_gymnasium.envs.tetris import Tetris
from deepqnetwork import deepqnetwork
from experience_replay import ExperienceReplay

def plot_reward(rewards, i):
    set_style("whitegrid")
    plt.figure(figsize=(10,6))
    plt.plot(rewards)
    plt.title(f"Episode Rewards Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.savefig(f"plots/Reward/reward_plot_{i}.png")
    plt.close()

def plot_epsilon(epsilons, i):
    set_style("whitegrid")
    plt.figure(figsize=(10,6))
    plt.plot(epsilons)
    plt.title("Episode Epsilon Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.savefig(f"plots/Epsilon/epsilon_plot_{i}.png")
    plt.close()

def plot_cum_rewards(cum_rewards, i):
    set_style("whitegrid")
    plt.figure(figsize=(10,6))
    plt.plot(cum_rewards)
    plt.title("Cumulative Rewards Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.savefig(f"plots/Cumulative_Rewards/cumulative_plot{i}.png")
    plt.close()

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
        cumReward = 0
        cum_rewards = [0]

        state,_ = env.reset(seed=42)
    
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
            epsilon_hist = [epsilon]  #store epsilon values for plotting
            gamma = 0.99

            #"target" network
            target_dqn = deepqnetwork(state_dim, action_dim)

            #copies weights and biases from policy network to target network
            target_dqn.load_state_dict(dqn.state_dict())

            #Adam optimizer initialization, learning rate set to 0.001
            self.optimizer = torch.optim.Adam(dqn.parameters(), lr=0.001)            

            #needs syncs target network with policy network
            count = 0


        #else if not training
            # need to add loading of model
            #target_dqn.load_state_dict(torch.load())
            #target_dqn.eval()

        iteration = 1
        #arbitrary number of episodes, change as you wish
        for episode in range(1000):
            episode += 1 #to graph all graphs properly
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
            cumReward += episode_reward
            cum_rewards.append(cumReward)
            
            #update epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            
            epsilon_hist.append(epsilon)
            
            # Update current state
            state = next_state

            if episode % 100 == 0:
                plot_reward(rewards, iteration)
                plot_epsilon(epsilon_hist, iteration)
                plot_cum_rewards(cum_rewards, iteration)
                iteration+=1

        
            cv2.waitKey(300)  # timeout to see the movement
        
        print("Game Over!")
        print(f"Final Buffer Size: {len(replay_buffer)}")
        traininglog.close()

    #optimizer for dqn/policy network
    def optimize(self, small_batch, dqn, target_dqn):
        #for state, action, next_state, reward, terminated in small_batch:
        #    if terminated:
        #        target = reward
        #    else:
        #        with torch.no_grad():
        #            target_qval = reward + self.discount_factor * torch.max(target_dqn(next_state))
        #        
        #    current_qval = dqn(state)
        #


        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*small_batch)

        #stacking tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float()

        with torch.no_grad():
            #calculating target q values (expected returns), using .99 as discount factor
            target = rewards + (1 - terminations) * .99 * target_dqn(new_states).max(dim=1)[0]

            #calcuate current policy q values
            current_qval = dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        #using MSE, change to whatever loss function you want
        loss = torch.nn.MSELoss(current_qval, target)

        #zeroing gradients, backpropagating, and updating weights and biases
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    agent = TetrisAgent()
    agent.run()
