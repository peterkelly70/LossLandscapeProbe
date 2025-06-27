"""Model definitions for CIFAR training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_model(dataset='cifar10'):
    """Get a model for the specified dataset.
    
    Args:
        dataset (str): The dataset to get the model for. Either 'cifar10' or 'cifar100'.
        
    Returns:
        torch.nn.Module: A PyTorch model for the specified dataset.
    """
    num_classes = 10 if dataset == 'cifar10' else 100
    
    # Use a ResNet-18 model
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    
    # Adjust the first convolutional layer for CIFAR's 32x32 input size
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove initial max pooling
    
    return model


class SimpleCNN(nn.Module):
    """A simple CNN model for CIFAR classification."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CIFARModel(nn.Module):
    """A model for CIFAR classification with configurable architecture."""
    
    def __init__(self, num_classes=10, arch='resnet18'):
        super(CIFARModel, self).__init__()
        self.num_classes = num_classes
        self.arch = arch
        
        if arch == 'resnet18':
            self.model = models.resnet18(pretrained=False, num_classes=num_classes)
            # Adjust for CIFAR input size
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()
        elif arch == 'simple_cnn':
            self.model = SimpleCNN(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
    
    def forward(self, x):
        return self.model(x)
    
    def get_optimizer(self, lr=0.1, momentum=0.9, weight_decay=5e-4):
        """Get an optimizer for this model."""
        return torch.optim.SGD(
            self.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )


# Alias for backward compatibility
get_model = CIFARModel
