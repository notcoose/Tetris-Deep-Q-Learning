import torch
import torch.nn as nn
state_dim= 944
action_dim = 8
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            #input
            nn.Linear(state_dim, state_dim//2),
            nn.ReLU(),

            #hidden layer 1
            nn.Linear(state_dim//2, state_dim//4),
            nn.ReLU(),

            #hidden layer 2
            nn.Linear(state_dim//4, state_dim//8),
            nn.ReLU(),

            #output layer
            nn.Linear(state_dim//8, action_dim)
        )

    def forward(self, x):
        return self.model(x)

"""
#input side
state_dim = 944
action_dim = 8

#start network
dqn = DQN(state_dim, action_dim)

# Example forward pass
state = torch.randn(1, state_dim)
q_values = dqn(state)
print(q_values)
"""
