import torch
from torch import nn
import torch.nn.functional as F

class deepqnetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(deepqnetwork, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln3 = nn.LayerNorm(hidden_dim // 4)
        
        self.fc4 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.ln4 = nn.LayerNorm(hidden_dim // 8)

        self.fc5 = nn.Linear(hidden_dim // 8, action_dim)  # Output layer

    def forward(self, x):
        # Pass through all hidden layers
        x = F.relu(self.ln1(self.fc1(x)))  # Layer 1
        x = F.relu(self.ln2(self.fc2(x)))  # Layer 2
        x = F.relu(self.ln3(self.fc3(x)))  # Layer 3
        x = F.relu(self.ln4(self.fc4(x)))  # Layer 4

        # Output layer (no activation, raw Q-values)
        x = self.fc5(x)
        return x

if __name__ == "__main__":
    state_dim = 56  # Cumulative length of arrays of observation dictionary
    action_dim = 8  # Number of possible actions

    dqn = deepqnetwork(state_dim, action_dim)

    # Random state to pass through network (batch size of 1)
    state = torch.rand(1, state_dim)

    output = dqn(state)
    print(output)
