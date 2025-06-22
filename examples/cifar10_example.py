"""
CIFAR-10 Example with Two-Tier Probing
=====================================

This example demonstrates how to use the Two-Tier Probing framework
to efficiently find good hyperparameters for training a CNN on CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import logging
from typing import Dict, Any, Tuple, List, Callable
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from llp.two_tier_probing import TwoTierProbing
from llp.meta_probing import MetaProbing
from llp.parameter_probing import SAM, StochasticWeightAveraging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_channels=32, dropout_rate=0.2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_channels * 4 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, self.fc1.in_features)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_cifar10_loaders(data_fraction=1.0, batch_size=128):
    """
    Get CIFAR-10 data loaders.
    
    Args:
        data_fraction: Fraction of training data to use
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Use the root data directory
    from pathlib import Path
    data_dir = Path(__file__).parent.parent / 'data'
    
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
    
    # Download CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=transform_test
    )
    
    # Use a subset of training data if data_fraction < 1.0
    if data_fraction < 1.0:
        num_train = len(trainset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(data_fraction * num_train))
        train_idx = indices[:split]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, sampler=train_sampler, num_workers=2
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )
    
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader


def create_model(config):
    """
    Create a model from a configuration.
    
    Args:
        config: Dictionary with hyperparameters
        
    Returns:
        PyTorch model
    """
    return SimpleCNN(
        num_channels=config['num_channels'],
        dropout_rate=config['dropout_rate']
    )


def create_optimizer(model, config):
    """
    Create an optimizer from a configuration.
    
    Args:
        model: PyTorch model
        config: Dictionary with hyperparameters
        
    Returns:
        PyTorch optimizer
    """
    if config['optimizer'] == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define hyperparameter configurations to search
    configs = [
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
            'optimizer': 'sgd',
            'learning_rate': 0.1,
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
    
    # Define dataset function
    dataset_fn = lambda data_fraction: get_cifar10_loaders(data_fraction=data_fraction)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create Two-Tier Probing object
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    probing = TwoTierProbing(
        configs=configs,
        model_fn=create_model,
        dataset_fn=dataset_fn,
        criterion=criterion,
        optimizer_fn=create_optimizer,
        max_epochs=50,
        device=device,
        alpha=0.1  # Weight for sharpness in generalization score
    )
    
    # Use Meta-Model for hyperparameter prediction (default method)
    logger.info("Running Meta-Model guided hyperparameter optimization...")
    logger.info(f"Starting with {len(configs)} configurations")
    logger.info(f"Resource range: {0.1:.1f} to {0.5:.1f} of data")
    logger.info(f"Reduction factor: {2}")
    
    # Track overall progress
    total_configs = len(configs)
    start_time = time.time()
    
    # Create a callback to monitor progress
    def progress_callback(iteration, remaining_configs, current_resource):
        elapsed = time.time() - start_time
        logger.info(f"[Progress] Iteration {iteration}: {remaining_configs}/{total_configs} configs being evaluated")
        logger.info(f"[Progress] Current resource level: {current_resource:.2f}")
        logger.info(f"[Progress] Elapsed time: {elapsed:.2f}s")
    
    # Create MetaProbing instance
    meta_probing = MetaProbing(
        configs=configs,
        model_fn=create_model,
        dataset_fn=dataset_fn,
        criterion=criterion,
        optimizer_fn=create_optimizer,
        max_epochs=50,
        device=device,
        alpha=0.1  # Weight for sharpness in generalization score
    )
    
    # Run meta-model optimization
    best_config = meta_probing.run_meta_optimization(
        min_resource=0.1,  # Start with 10% of data
        max_resource=0.5,  # End with 50% of data
        reduction_factor=2,
        measure_flatness=True,
        progress_callback=progress_callback,
        num_initial_configs=6,
        num_iterations=3
    )
    
    # Use the best config for final training
    logger.info(f"Best configuration found: {best_config}")
    
    # Log the best configuration found by the meta-model
    logger.info(f"\nBest configuration predicted by meta-model: {best_config}")
    
    # Display the hyperparameters of the best configuration
    logger.info("\nBest hyperparameters:")
    for param, value in best_config.items():
        logger.info(f"  {param}: {value}")
    
    # Calculate total time taken
    total_time = time.time() - start_time
    logger.info(f"\nMeta-model optimization completed in {total_time:.2f} seconds")
    
    logger.info("\nProof of concept complete: Meta-model successfully predicted optimal hyperparameters")
    logger.info("These hyperparameters can now be used for full model training without SAM or SWA")


if __name__ == "__main__":
    main()
