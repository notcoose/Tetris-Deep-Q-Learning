import gymnasium as gym
import cv2
import torch
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os

from seaborn import set_style
from tetris_gymnasium.envs.tetris import Tetris
from deepqnetwork import DeepQNetwork
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

#our reward function
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
    grid = observation['board']  # Assuming 'board' is the Tetris grid
    if len(grid.shape) == 2:  # If the grid is 2D, add only a channel dimension
        grid = grid[np.newaxis, :, :]
    return grid.astype(np.float32)

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
    def run(self, is_training = True, render_mode = "ansii"):
        self.savedmodel = os.path.join(model_dir_name, "tetris_model.pt")

        self.recent_rewards = []
        self.recent_rewards_window = 100

        #creating and writing to log file
        if(is_training):
            traininglog = open("traininglog.txt", "w")

            start = dt.datetime.now()
            traininglog.write(f"Start time: {start}\n")

        # Initialize environment
        # Initialize environment
        env = gym.make("tetris_gymnasium/Tetris", render_mode=render_mode)
        rewards = []
        best_reward = 0
        cumReward = 0
        cum_rewards = [0]

        state, _ = env.reset(seed=42)

        # Get action and state space dimensions
        state_dim = (24, 18)
        action_dim = env.action_space.n  # Number of possible actions

        # Instantiate the DeepQNetwork
        dqn = DeepQNetwork(grid_shape=state_dim, action_dim=action_dim)

        if is_training:
            replay_buffer = ExperienceReplay(capacity=50000, state_dim=state_dim)

            # Warm up the replay buffer with random actions
            while len(replay_buffer) < 1000:  # Choose a batch size of 1000 for warm-up
                random_action = np.random.randint(action_dim)
                next_state, reward, terminated, truncated, info = env.step(random_action)
                replay_buffer.store(preprocess_state(state), random_action, reward, preprocess_state(next_state), terminated)
                state = next_state
                if terminated:
                    state, _ = env.reset()

            # Initialize greedy parameters
            epsilon_start = 1.0  # Initial exploration rate
            epsilon_min = 0.1  # Min exploration rate
            epsilon_decay = 0.999  # Decay factor
            epsilon = epsilon_start  # Initialize epsilon to its starting value
            epsilon_hist = [epsilon]  # Store epsilon values for plotting
            epsilon_decay_steps = 10000

            # "Target" network
            target_dqn = DeepQNetwork(grid_shape=state_dim, action_dim=action_dim)
            target_dqn.load_state_dict(dqn.state_dict())  # Copy weights from policy network to target network

            # Adam optimizer initialization
            self.optimizer = torch.optim.Adam(dqn.parameters(), lr=0.0001)

        iteration = 1
        count = 0

        #arbitrary number of episodes, change as you wish
        for episode in range(50000):
            episode += 1 #to graph all graphs properly
            terminated = False
            episode_reward = 0.0
            state, _ = env.reset(seed=42)
            self.recent_rewards.append(episode_reward)
            epsilon = max(epsilon_min, epsilon_start - (episode / epsilon_decay_steps))
            epsilon_hist.append(epsilon)

            if len(self.recent_rewards) > self.recent_rewards_window:
                self.recent_rewards.pop(0)

            #1000 is arbitrary, change as you wish for early stopping
            while not terminated and episode_reward < 10000:
                if render_mode == "human":
                    env.render()
                elif render_mode == "ansii":
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

                reward += 20 * lines_cleared  # Encourage clearing lines
                reward -= stack_height  # Penalize high stacks moderately
                reward -= 5 * gaps_in_rows  # Penalize gaps

                reward = np.clip(reward, -1, 1)

                #if terminated:
                    #reward -= 1000

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
                    #small batch size is 128, change as you wish
                    small_batch = replay_buffer.sample(128)
                    self.optimize(small_batch, dqn, target_dqn)

                    #update epsilon
                    if epsilon > epsilon_min:
                        epsilon *= epsilon_decay

                    #copies policy to target network
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
        # Transpose the list of experiences and separate each element
        states, actions, rewards, next_states, terminations = small_batch

        # Ensure `states` and `next_states` have the correct shape
        states = states.squeeze()  # Remove unnecessary dimensions if any
        next_states = next_states.squeeze()

        if len(states.shape) == 3:  # If shape is [batch_size, height, width], add channel
            states = states.unsqueeze(1)  # Shape -> [batch_size, 1, height, width]
        if len(next_states.shape) == 3:
            next_states = next_states.unsqueeze(1)

        print(f"Shape of states after preprocessing: {states.shape}")
        print(f"Shape of next_states after preprocessing: {next_states.shape}")

        # Ensure actions tensor has the correct shape
        actions = actions.view(-1, 1)

        with torch.no_grad():
            # Calculate target Q-values
            target_q_values = rewards + (1 - terminations) * 0.99 * target_dqn(next_states).max(dim=1)[0]

        # Calculate current Q-values for the taken actions
        current_q_values = dqn(states).gather(1, actions).squeeze()

        # Compute loss
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)

        # Zero gradients, backpropagate, and update weights
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

if __name__ == "__main__":
    agent = TetrisAgent()

    # insert render_mode="human" for visual rendering like traditional tetris,
    # defeult is ansii rendering in terminal
    agent.run()

