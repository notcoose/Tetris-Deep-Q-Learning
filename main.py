import gymnasium as gym
import cv2
import torch
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os

from seaborn import set_style
from tetris_gymnasium.envs.tetris import Tetris
from deepqnetwork import deepqnetwork
from experience_replay import ExperienceReplay

model_dir_name = "models"
os.makedirs(model_dir_name, exist_ok=True)

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

def compute_line_clear_reward(lines_cleared):
    """Assign reward based on the number of lines cleared."""
    if lines_cleared == 1:
        return 10
    elif lines_cleared == 2:
        return 30
    elif lines_cleared == 3:
        return 60
    elif lines_cleared == 4:
        return 100
    else:
        return 0

def compute_stack_height(grid):
    for i, row in enumerate(grid):
        if np.any(row):  # Check if the row contains any filled cells
            return len(grid) - i  # Stack height is from the top
    return 0  # No filled rows

def compute_gaps_in_rows(grid):
    gaps = 0
    for col in range(grid.shape[1]):  # Iterate over columns
        filled = False
        for row in range(grid.shape[0]):  # Iterate over rows top to bottom
            if grid[row, col] == 1:  # Encounter filled cell
                filled = True
            elif filled and grid[row, col] == 0:  # Empty below filled cell
                gaps += 1
    return gaps


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
        self.savedmodel = os.path.join(model_dir_name, "tetris_model.pt")

        self.recent_rewards = []
        self.recent_rewards_window = 100

        #creating and writing to log file
        if(is_training):
            traininglog = open("traininglog.txt", "w")

            start = dt.datetime.now()
            traininglog.write(f"Start time: {start}\n")

        # Initialize environment
        env = gym.make("tetris_gymnasium/Tetris", render_mode = render_mode)
        rewards = []
        best_reward = 0
        cumReward = 0
        cum_rewards = [0]

        state,_ = env.reset(seed=42)
    
        # get action and state space dims
        state_dim = preprocess_state(state).shape[0]  #temp
        action_dim = env.action_space.n #number of actions

        #"policy" network
        dqn = deepqnetwork(state_dim, action_dim)

        if is_training:
            replay_buffer = ExperienceReplay(capacity=50000, state_dim=state_dim)

            #initialize greedy params
            epsilon = 1.0  #initial exploration rate
            epsilon_min = 0.1  #min exploration rate
            epsilon_decay = 0.999    #decay factor
            epsilon_hist = [epsilon]  #store epsilon values for plotting
            gamma = 0.99

            #"target" network
            target_dqn = deepqnetwork(state_dim, action_dim)

            #copies weights and biases from policy network to target network
            target_dqn.load_state_dict(dqn.state_dict())

            #Adam optimizer initialization, learning rate set to 0.001
            self.optimizer = torch.optim.Adam(dqn.parameters(), lr=0.0005)            

        else:
            dqn.load_state_dict(torch.load(self.savedmodel))
            dqn.eval()

        iteration = 1
        count = 0
        #arbitrary number of episodes, change as you wish
        for episode in range(50000):
            episode += 1 #to graph all graphs properly
            terminated = False
            episode_reward = 0.0
            state, _ = env.reset(seed=42)
            self.recent_rewards.append(episode_reward)

            if len(self.recent_rewards) > self.recent_rewards_window:
                self.recent_rewards.pop(0)

            if len(self.recent_rewards) > 0:
                previous_avg_reward = np.mean(self.recent_rewards)
            else:
                previous_avg_reward = 0

            #1000 is arbitrary, change as you wish for early stopping
            while not terminated and episode_reward < 10000:
                print(env.render() + "\n")
        
                # Get current state
                current_state = preprocess_state(state)
        
                # Get action using epsilon-greedy algorithm
                action = epsilon_greedy_action(dqn, current_state, epsilon, action_dim)
        
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                lines_cleared = info.get("lines_cleared", 0)  # Default to 0 if not provided
                reward += compute_line_clear_reward(lines_cleared)


                grid = next_state['board']
                stack_height = compute_stack_height(grid)
                gaps_in_rows = compute_gaps_in_rows(grid)

                reward += -10 * stack_height  # Penalize based on stack height
                reward += -50 * gaps_in_rows  # Penalize gaps more
                reward += 50 * lines_cleared

                if terminated:
                    reward -= 1000

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

            if is_training and episode_reward > best_reward:
                traininglog.write(f"New best reward: {episode_reward}\n")
                best_reward = episode_reward
                torch.save(dqn.state_dict(), self.savedmodel)
                traininglog.write(f"Model saved\n")

                #update target network every 1000 steps (arbitrary)
                if(len(replay_buffer) > 1000):
                    #small batch size is 32, change as you wish
                    small_batch = replay_buffer.sample(64)
                    self.optimize(small_batch, dqn, target_dqn)

                    #update epsilon
                    #if epsilon > epsilon_min:
                    #    epsilon *= epsilon_decay
                    if episode_reward > previous_avg_reward:
                        epsilon *= 0.99
                    else:
                        epsilon *= epsilon_decay

                    
                    epsilon_hist.append(epsilon)


                    #copies policy to target network eveyr 10 steps, change as you wish
                    # Soft update example
                    tau = 0.005  # Small update factor
                    for target_param, policy_param in zip(target_dqn.parameters(), dqn.parameters()):
                        target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
                    count = 0  # Reset count


            rewards.append(episode_reward)
            cumReward += episode_reward
            cum_rewards.append(cumReward)
        
            # Update current state
            state = next_state

            if episode % 100 == 0:
                os.makedirs("plots/Reward", exist_ok=True)
                os.makedirs("plots/Epsilon", exist_ok=True)
                os.makedirs("plots/Cumulative_Rewards", exist_ok=True)
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
        states, actions, rewards, next_states, terminations = small_batch

        # Ensure all tensors are the correct shape
        actions = actions.view(-1, 1)  # Convert actions to the right shape for indexing

        with torch.no_grad():
            # Calculate target Q-values (expected returns)
            target_q_values = rewards + (1 - terminations) * 0.99 * target_dqn(next_states).max(dim=1)[0]

        # Calculate current Q-values for the taken actions
        current_q_values = dqn(states).gather(1, actions).squeeze()

        # Compute loss using MSE
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)

        # Zero gradients, backpropagate, and update weights
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

if __name__ == "__main__":
    agent = TetrisAgent()
    agent.run()
