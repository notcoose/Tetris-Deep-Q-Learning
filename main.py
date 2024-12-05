import os
import yaml
from random import random, sample
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from deepqnetwork import DQN
from tetris import Tetris
from collections import deque
from seaborn import set_style
import matplotlib.pyplot as plt

# function to plot rewards over episodes
def plot_reward(rewards, i):
    sns.set_style("whitegrid")
    plot_dir = "plots/Reward"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rewards)), rewards, marker='o', label="Rewards")
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Index")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig(f"{plot_dir}/reward_plot_{i}.png")
    plt.close()

# function to plot scores over episodes
def plot_scores(scores, i):
    plot_dir = "plots/Score"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(scores)), scores, linestyle='-', linewidth=2, label="Scores")
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Scores")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(f"{plot_dir}/score_plot_{i}.png")
    plt.close()

# function to plot epsilon decay over episodes
def plot_epsilon(epsilons, i):
    set_style("whitegrid")
    plot_dir = "plots/Epsilon"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons)
    plt.title("Episode Epsilon Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.savefig(f"{plot_dir}/epsilon_plot_{i}.png")
    plt.close()

# function to plot cumulative rewards over episodes
def plot_cum_rewards(cum_rewards, i):
    sns.set_style("whitegrid")
    plot_dir = "plots/Reward"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cum_rewards)), cum_rewards, marker='o', label="Rewards")
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Index")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig(f"{plot_dir}/reward_plot_{i}.png")
    plt.close()

# function to plot running average of scores
def plot_running_average(scores, i, window_size=10):
    plot_dir = "plots/Score_avg"
    os.makedirs(plot_dir, exist_ok=True)
    running_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(running_avg)), running_avg, linestyle='-', linewidth=2, label="Running Average")
    plt.title("Running Average of Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Running Average Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(f"{plot_dir}/running_avg_plot_{i}.png")
    plt.close()

# class representing the tetris agent
class TetrisAgent:

    def __init__(self, parameter_set):
        # load parameters from yaml file
        with open('parameters.yaml', 'r') as f:
            all_parameters = yaml.safe_load(f)
            parameters = all_parameters[parameter_set]
            self.parameter_set = parameter_set
            self.learning_rate = parameters['learning_rate']
            self.batch_size = parameters['batch_size']
            self.episodes = parameters['episodes']
            self.gamma = parameters['gamma']
            self.epsilon = parameters['epsilon']
            self.epsilon_decay = parameters['epsilon_decay']
            self.epsilon_min = parameters['epsilon_min']
            self.replay_memory_size = parameters['replay_memory_size']
            self.target_update = parameters['target_update']
            self.width = parameters['width']
            self.height = parameters['height']
            self.block_size = parameters['block_size']

    # function to prepare and calculate target q-values
    def compute_targets(reward_batch, done_batch, next_pred_batch, gamma):
        return torch.cat(
            tuple(reward if done else reward + (prediction * gamma)
                for reward, done, prediction in zip(reward_batch, done_batch, next_pred_batch))
        )[:, None]

    # function to prepare a batch from replay memory
    def prepare_batch(self, memory):
        batch = sample(memory, min(len(memory), self.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))
        return state_batch, reward_batch, next_state_batch, done_batch

    # main training loop for the agent
    def train(self):
        buffer_cap = self.replay_memory_size / 15
        episode = 0
        best_reward = 0
        cum_reward = 0
        cum_rewards = []
        scores = []
        rewards = []
        epsilons = []
        env = Tetris(width=self.width, height=self.height, block_size=self.block_size)
        model = DQN()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        state = env.reset()
        memory = deque(maxlen=self.replay_memory_size)

        while episode < self.episodes:
            next_moves = env.get_next_states()
            epsilon = max(self.epsilon_min, self.epsilon - (self.epsilon_decay * episode))
            epsilons.append(epsilon)
            next_actions, next_states = zip(*next_moves.items())
            next_states = torch.stack(next_states)
            model.eval()
            with torch.no_grad():
                predictions = model(next_states)[:, 0]

            # choose action based on epsilon-greedy policy
            if random() < epsilon:
                index = next_actions.index(sample(next_actions, 1)[0])
            else:
                index = torch.argmax(predictions).item()

            # take action
            action = next_actions[index]
            next_state = next_states[index, :]
            reward, done = env.step(action, render=True)

            # update metrics
            rewards.append(reward)
            cum_reward += reward
            cum_rewards.append(cum_reward)
            memory.append([state, reward, next_state, done])

            # check if buffer is passed
            if done:
                final_cleared = env.cleared_lines
                final_score = env.score
                state = env.reset()
            else:
                state = next_state
                continue
            if len(memory) < buffer_cap:
                continue

            # prepare training batch
            state_batch, reward_batch, next_state_batch, done_batch = self.prepare_batch(memory)
            q_vals = model(state_batch)
            model.eval()
            with torch.no_grad():
                next_pred_batch = model(next_state_batch)
            model.train()

            # compute target values
            y_batch = self.compute_targets(reward_batch, done_batch, next_pred_batch, self.gamma)

            episode += 1
            scores.append(final_score)
            optimizer.zero_grad()
            loss_eq = criterion(q_vals, y_batch)
            loss_eq.backward()
            optimizer.step()

            # print episode summary
            print("Episode: {}/{}, Score: {}, Cleared lines: {}".format(
                episode,
                self.episodes,
                final_score,
                final_cleared))

            # periodically plot and save model
            if episode % 50 == 0:
                plot_reward(rewards, episode)
                plot_scores(scores, episode)
                plot_epsilon(epsilons, episode)
                plot_cum_rewards(cum_rewards, episode)
                plot_running_average(scores, episode)

            if episode % 1000 == 0:
                torch.save(model, "{}/tetris_{}".format(os.path.dirname(os.path.realpath(__file__)), episode))

# entry point for training the agent
if __name__ == "__main__":
    agent = TetrisAgent(parameter_set="tetris")
    agent.train()
