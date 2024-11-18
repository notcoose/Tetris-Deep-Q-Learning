import numpy as np
from collections import deque
import torch
import random

class ExperienceReplay:
    def __init__(self, capacity, state_dim):
        """
        Initialize Experience Replay Buffer
        
        Args:
            capacity (int): Maximum size of the replay buffer
            state_dim (int): Dimension of the state space
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.state_dim = state_dim
        
    def store(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        # Convert numpy arrays to torch tensors if they aren't already
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)
            
        # Ensure state dimensions are correct
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
            
        # Store the transition
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer
        
        Args:
            batch_size (int): Size of the batch to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
            
        # Randomly sample transitions
        transitions = random.sample(self.memory, batch_size)
        
        # Unzip the transitions into separate batches
        batch = list(zip(*transitions))
        
        # Convert to appropriate torch tensors
        states = torch.cat(batch[0])
        actions = torch.tensor(batch[1])
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.cat(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Returns current size of the replay buffer"""
        return len(self.memory)

if __name__ == "__main__":
    # Example usage and testing
    state_dim = 56  # Same as in your DQN
    buffer = ExperienceReplay(capacity=10000, state_dim=state_dim)
    
    # Example of storing a transition
    state = torch.rand(1, state_dim)
    action = 0
    reward = 1.0
    next_state = torch.rand(1, state_dim)
    done = False
    
    buffer.store(state, action, reward, next_state, done)
    
    # Example of sampling from buffer (once it has enough samples)
    if len(buffer) > 32:
        states, actions, rewards, next_states, dones = buffer.sample(32)
        print(f"Sampled batch shapes:")
        print(f"States: {states.shape}")
        print(f"Actions: {actions.shape}")
        print(f"Rewards: {rewards.shape}")
        print(f"Next states: {next_states.shape}")
        print(f"Dones: {dones.shape}")
