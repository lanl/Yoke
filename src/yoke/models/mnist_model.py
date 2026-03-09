"""Defines neural network architecture for MNIST digit classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class mnist_CNN(nn.Module):
    """CNN for classifying MNIST digits.

    Maps a 1x28x28 MNIST image to a log-probability distribution over digits 0-9.

    Convolutional Neural Network module that takes a single-channel 28x28 input
    image and produces a 10-dimensional log-probability vector via a series of
    four convolutional blocks (conv → ReLU → optional pooling), a dropout, and
    two fully connected layers.

    Args:
        conv1_size (int):   Number of output channels for the first conv layer.
        conv2_size (int):   Number of output channels for the second conv layer.
        conv3_size (int):   Number of output channels for the third conv layer.
        conv4_size (int):   Number of output channels for the fourth conv layer.

    Input:
        x (torch.Tensor):   Float tensor of shape (N, 1, 28, 28), where N is batch size.

    Output:
        torch.Tensor:       Float tensor of shape (N, 10) containing log-probabilities
                            for each digit class.

    """

    # Net class inherits from nn.Module, which is the base class for all
    # neural network modules in PyTorch. Inheriting from nn.Module allows you to
    # define your own custom neural network layers and operations.
    def __init__(
        self,
        conv1_size: int = 32,
        conv2_size: int = 64,
        conv3_size: int = 128,
        conv4_size: int = 128,
    ) -> None:
        """__init__ initializes the layers."""
        super().__init__()

        # First convolutional layer with 1 input channel,
        # 32 output channels, kernel size 3, and stride 1
        self.conv1 = nn.Conv2d(1, conv1_size, 3, 1, padding=1)

        # Second convolutional layer with 32 input channels,
        # 64 output channels, kernel size 3, and stride 1
        self.conv2 = nn.Conv2d(conv1_size, conv2_size, 3, 1, padding=1)

        # Second convolutional layer with 64 input channels,
        # 128 output channels, kernel size 3, and stride 1
        self.conv3 = nn.Conv2d(conv2_size, conv3_size, 3, 1, padding=1)

        # Second convolutional layer with 128 input channels,
        # 128 output channels, kernel size 3, and stride 1
        self.conv4 = nn.Conv2d(conv3_size, conv4_size, 3, 1, padding=1)

        # Dropout layer with 0.5 dropout rate
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer with 6272 input features and 512 output features
        self.fc1 = nn.Linear(conv4_size * 7 * 7, 512)

        # Fully connected layer with 512 input features and 10 output features
        # (for the 10 classes in MNIST)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the network."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # takes a vector of raw data as input and returns a vector of probabilities,
        # where each element represents the probability of the corresponding class.
        output = F.log_softmax(x, dim=1)
        return output
