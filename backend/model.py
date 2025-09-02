import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    A "Lite" Convolutional Neural Network for CIFAR-10.
    This version is smaller to easily fit within WebSocket message limits.
    """
    def __init__(self):
        super(CNNModel, self).__init__()
        # Reduced convolutional layers
        # Input: 3 channels, Output: 8 channels
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2)
        # Input: 8 channels, Output: 16 channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Reduced fully connected layers
        # New flattened size is 16 (from conv2) * 8 * 8 = 1024
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10) # 10 classes for CIFAR-10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the feature maps
        x = x.view(-1, 16 * 8 * 8)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x