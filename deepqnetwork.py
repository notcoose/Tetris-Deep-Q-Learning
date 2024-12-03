import torch
from torch import nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, grid_shape, action_dim, hidden_dim=256):
        """
        Deep Q-Network with convolutional layers.

        Args:
            grid_shape (tuple): Shape of the grid (e.g., (24, 18) for Tetris).
            action_dim (int): Number of possible actions.
            hidden_dim (int): Number of hidden units in fully connected layers.
        """
        super(DeepQNetwork, self).__init__()
        
        self.grid_shape = grid_shape
        self.action_dim = action_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # Retain grid size
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Dynamically calculate flattened size
        dummy_input = torch.zeros(1, 1, grid_shape[0], grid_shape[1])  # [Batch, Channel, Height, Width]
        flattened_size = self._get_flattened_size(dummy_input)

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, action_dim)  # Output layer for Q-values

    def _get_flattened_size(self, x):
        """Pass a dummy tensor through conv layers to compute the flattened size."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(torch.prod(torch.tensor(x.shape[1:])))  # Compute product of all dimensions except batch

    def forward(self, x):
        """
        Forward pass for the Deep Q-Network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 1, grid_height, grid_width).
        
        Returns:
            torch.Tensor: Q-values for each action.
        """
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, flattened_size)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output Q-values for each action

        return x
