#!/usr/bin/env python3
"""
CIFAR Core Module
================

Core training logic for CIFAR datasets, including:
- Dataset loading
- Model creation
- Meta-model optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Define a simple CNN model for CIFAR datasets
class SimpleCNN(nn.Module):
    def __init__(self, num_channels=32, dropout_rate=0.2, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_channels * 4 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, self.fc1.in_features)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_cifar_loaders(dataset='cifar10', data_fraction=1.0, batch_size=128, num_workers=2, 
                     train_shuffle=True, val_shuffle=False, pin_memory=True):
    """
    Get CIFAR data loaders.
    
    Args:
        dataset: Dataset name ('cifar10' or 'cifar100')
        data_fraction: Fraction of training data to use (0.0 to 1.0)
        batch_size: Batch size for data loaders
        num_workers: Number of subprocesses to use for data loading
        train_shuffle: Whether to shuffle training data
        val_shuffle: Whether to shuffle validation data
        pin_memory: If True, the data loader will copy tensors into CUDA pinned memory
                   before returning them (only use with GPU training)
        
    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    # Use the root data directory
    data_dir = Path(__file__).parent / 'data'
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Select the appropriate dataset
    if dataset.lower() == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=str(data_dir), train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=str(data_dir), train=False, download=True, transform=transform_test
        )
        num_classes = 10
    elif dataset.lower() == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=str(data_dir), train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root=str(data_dir), train=False, download=True, transform=transform_test
        )
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Use a subset of training data if data_fraction < 1.0
    if data_fraction < 1.0:
        num_train = len(trainset)
        rng = np.random.default_rng(42)  # Create a new random number generator with fixed seed
        indices = rng.permutation(num_train)  # Shuffle indices using the new API
        split = int(np.floor(data_fraction * num_train))
        train_idx = indices[:split]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=val_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, num_classes


def create_model(config, num_classes=10):
    """
    Create a model from a configuration.
    
    Args:
        config: Dictionary with hyperparameters
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    return SimpleCNN(
        num_channels=config['num_channels'],
        dropout_rate=config['dropout_rate'],
        num_classes=num_classes
    )


def create_optimizer(model_or_params, config):
    """
    Create an optimizer from a configuration.
    
    Args:
        model_or_params: PyTorch model or list of parameters
        config: Dictionary with hyperparameters
        
    Returns:
        PyTorch optimizer
    """
    # Handle both model and parameter list inputs
    if isinstance(model_or_params, torch.nn.Module):
        params = model_or_params.parameters()
    else:
        params = model_or_params
    
    # Create optimizer
    optimizer_type = config['optimizer'].lower()
    if optimizer_type == 'sgd':
        return optim.SGD(
            params,
            lr=config['learning_rate'],
            momentum=config.get('momentum', 0.9),  # Default momentum for SGD
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif optimizer_type == 'adam':
        return optim.Adam(
            params,
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif optimizer_type == 'adamw':
        return optim.AdamW(
            params,
            lr=config['learning_rate'],
            betas=(
                config.get('beta1', 0.9),
                config.get('beta2', 0.999)
            ),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


def get_default_hyperparameter_configs():
    """
    Get default hyperparameter configurations for meta-model training.
    
    Returns:
        List of hyperparameter configurations
    """
    return [
        {
            'num_channels': 32,
            'dropout_rate': 0.2,
            'optimizer': 'sgd',
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-4
        },
        {
            'num_channels': 32,
            'dropout_rate': 0.2,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'momentum': 0.0,
            'weight_decay': 5e-4
        },
        {
            'num_channels': 64,
            'dropout_rate': 0.3,
            'optimizer': 'sgd',
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-4
        },
        {
            'num_channels': 64,
            'dropout_rate': 0.3,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'momentum': 0.0,
            'weight_decay': 5e-4
        },
        {
            'num_channels': 64,
            'dropout_rate': 0.5,
            'optimizer': 'sgd',
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4
        }
    ]


def get_cifar100_transfer_config():
    """
    Get the default configuration for CIFAR-100 transfer learning.
    
    Returns:
        Dictionary with hyperparameters
    """
    return {
        'num_channels': 64,
        'dropout_rate': 0.3,
        'optimizer': 'adam',
        'learning_rate': 0.0005,
        'momentum': 0.0,
        'weight_decay': 1e-4
    }


def get_best_known_config(dataset='cifar10'):
    """
    Get the best known hyperparameters for the given dataset.
    Used when skipping meta-model optimization.
    
    Args:
        dataset: Dataset name ('cifar10' or 'cifar100')
        
    Returns:
        Dictionary with hyperparameters
    """
    if dataset.lower() == 'cifar10':
        return {
            'num_channels': 32,
            'dropout_rate': 0.2,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'momentum': 0.0,
            'weight_decay': 0.0005
        }
    elif dataset.lower() == 'cifar100':
        return {
            'num_channels': 64,
            'dropout_rate': 0.3,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'momentum': 0.0,
            'weight_decay': 0.0005
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
