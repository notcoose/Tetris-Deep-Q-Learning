import torch
from torch import nn
import torch.nn.functional as F
import numpy as np #will prob need to flatten observation dictionary of np arrays

class deepqnetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super(deepqnetwork, self).__init__()
        
        #basic network structure, tbd
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x)) #relu activation function
        return self.fc2(x)

if __name__ == "__main__":
    state_dim = 56 #cumulative length of arrays of observation dictionary
    action_dim = 8 #number of possible actions

    dqn = deepqnetwork(state_dim, action_dim)

    state = torch.rand(1, state_dim) #random state to pass through network

    output = dqn(state)
    print(output)