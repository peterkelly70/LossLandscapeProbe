"""
Simple CNN Model for CIFAR-10/100
================================

A simple CNN architecture used for CIFAR-10 and CIFAR-100 classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple CNN architecture for image classification.
    
    This model consists of:
    - 3 convolutional layers with batch normalization and max pooling
    - 2 fully connected layers with dropout
    - Configurable number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
    """
    
    def __init__(self, num_channels=32, dropout_rate=0.2, num_classes=10):
        """
        Initialize the SimpleCNN model.
        
        Args:
            num_channels (int): Number of channels in the first conv layer
            dropout_rate (float): Dropout probability
            num_classes (int): Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
        """
        super(SimpleCNN, self).__init__()
        
        # Ensure num_channels is an integer and divisible by the default groups=1
        num_channels = int(num_channels)
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(num_channels, num_channels*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels*2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(num_channels*2, num_channels*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels*4)
        
        # Calculate size after convolutions and pooling
        # Input: 32x32x3 -> After 3 max pooling layers: 4x4x(num_channels*4)
        fc_input_size = 4 * 4 * num_channels * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """Forward pass through the network"""
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
