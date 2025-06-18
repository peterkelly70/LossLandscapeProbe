"""
CIFAR-100 Training with Two-Tier Probing
=======================================

This example demonstrates how to use the Two-Tier Probing framework
to efficiently find good hyperparameters for training a CNN on CIFAR-100.
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
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from llp.two_tier_probing import TwoTierProbing
from llp.meta_probing import MetaProbing
from llp.parameter_probing import SAM, StochasticWeightAveraging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cifar100_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Define a CNN model for CIFAR-100
class CIFAR100CNN(nn.Module):
    def __init__(self, num_channels=32, dropout_rate=0.2):
        super(CIFAR100CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_channels * 4, num_channels * 8, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_channels * 8 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 100)  # 100 classes for CIFAR-100
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(num_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_channels * 2)
        self.batch_norm3 = nn.BatchNorm2d(num_channels * 4)
        self.batch_norm4 = nn.BatchNorm2d(num_channels * 8)
    
    def forward(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.relu(self.batch_norm4(self.conv4(x)))
        x = x.view(-1, self.fc1.in_features)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def get_cifar100_loaders(data_fraction=1.0, batch_size=128):
    """
    Get CIFAR-100 data loaders.
    
    Args:
        data_fraction: Fraction of training data to use
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Download CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
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
    return CIFAR100CNN(
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
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, device, max_epochs=100):
    """
    Train and evaluate a model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to use for training
        max_epochs: Maximum number of epochs
        
    Returns:
        Dictionary with training results
    """
    model.to(device)
    
    train_losses = []
    val_accs = []
    epoch_times = []
    
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100.0 * correct / total
        val_accs.append(val_acc)
        
        # Track best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
        
        # Record epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{max_epochs} | "
                   f"Train Loss: {train_loss:.4f} | "
                   f"Val Acc: {val_acc:.2f}% | "
                   f"Time: {epoch_time:.2f}s")
    
    # Save final model
    torch.save(model.state_dict(), 'cifar100_multisamplesize_trained.pth')
    
    # Return results
    results = {
        'train_losses': train_losses,
        'test_accs': [acc / 100.0 for acc in val_accs],  # Convert to 0-1 range
        'best_acc': best_acc / 100.0,  # Convert to 0-1 range
        'best_epoch': best_epoch,
        'epochs': list(range(1, max_epochs + 1)),
        'epoch_times': epoch_times
    }
    
    # Save results to .pth file
    torch.save(results, 'cifar100_multisamplesize_results.pth')
    # Also save with the standard name for backward compatibility
    torch.save(results, 'cifar100_results.pth')
    
    return results


def main():
    # Define configurations to try
    configs = [
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
        },
        {
            'num_channels': 64,
            'dropout_rate': 0.4,
            'optimizer': 'adam',
            'learning_rate': 0.0005,
            'momentum': 0.0,
            'weight_decay': 1e-4
        }
    ]
    
    # Define dataset function
    dataset_fn = lambda data_fraction: get_cifar100_loaders(data_fraction=data_fraction)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create Two-Tier Probing object
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
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
    logger.info("Running Meta-Model guided hyperparameter optimization...")
    logger.info(f"Starting with {len(configs)} configurations")
    logger.info(f"Resource range: {0.1:.1f} to {0.5:.1f} of data")
    logger.info(f"Reduction factor: {2}")
    
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
    
    # Calculate total time taken for meta-model optimization
    meta_time = time.time() - start_time
    logger.info(f"\nMeta-model optimization completed in {meta_time:.2f} seconds")
    
    # Train the final model with the best configuration
    logger.info("\nTraining final model with best configuration...")
    
    # Get full dataset
    train_loader, val_loader = dataset_fn(data_fraction=1.0)
    
    # Create model with best configuration
    model = create_model(best_config)
    
    # Create optimizer with best configuration
    optimizer = create_optimizer(model, best_config)
    
    # Train and evaluate the model
    results = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_epochs=100
    )
    
    # Log final results
    logger.info(f"\nFinal results:")
    logger.info(f"Best validation accuracy: {results['best_acc']*100:.2f}%")
    logger.info(f"Best epoch: {results['best_epoch']}")
    
    # Calculate total time taken
    total_time = time.time() - start_time
    logger.info(f"\nTotal training time: {total_time:.2f} seconds")
    
    # Save configuration to results
    results['best_config'] = best_config
    torch.save(results, 'cifar100_multisamplesize_results.pth')
    # Also save with the standard name for backward compatibility
    torch.save(results, 'cifar100_results.pth')
    
    logger.info("\nCIFAR-100 training complete!")
    logger.info("Results saved to cifar100_results.pth")


if __name__ == "__main__":
    main()
