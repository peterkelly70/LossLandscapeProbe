#!/usr/bin/env python3
"""
CIFAR Core Training Module
=========================

This module contains the core functionality for training models on CIFAR-10 and CIFAR-100 datasets
using the meta-model approach for hyperparameter optimization.

It provides functions for:
- Loading and preprocessing datasets
- Creating models and optimizers
- Running meta-model optimization
- Training and evaluating models

This module focuses on the training logic and logs all necessary information for later reporting.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

# Add the project root directory to the path so we can import the LLP package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Also add the src directory to the path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llp.models.simple_cnn import SimpleCNN
from llp.two_tier_probing import TwoTierProbing
from llp.meta_probing import MetaProbing

# Define setup_logger function locally since llp.utils.logging_utils doesn't exist
def setup_logger():
    """Set up a basic logger for the script"""
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to prevent duplicate messages
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RESOURCE_LEVELS = [0.1, 0.2, 0.3, 0.4]
DEFAULT_NUM_ITERATIONS = 3  # Number of meta-model iterations
DEFAULT_NUM_CONFIGS = 10    # Number of configurations to try per iteration
DEFAULT_EPOCHS = 100        # Number of epochs for final training
DEFAULT_BATCH_SIZE = 128

# Create results directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Date-time string for unique filenames
DATE_STR = datetime.now().strftime("%Y%m%d_%H%M%S")


def get_cifar10_loaders(subset_fraction=1.0, batch_size=DEFAULT_BATCH_SIZE):
    """Get CIFAR-10 data loaders with optional subset sampling."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Download and load the training data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    
    # If using a subset
    if subset_fraction < 1.0:
        num_train = len(trainset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(subset_fraction * num_train))
        train_idx = indices[:split]
        trainset = Subset(trainset, train_idx)
    
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    
    # Download and load the test data
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    
    return trainloader, testloader


def get_cifar100_loaders(subset_fraction=1.0, batch_size=DEFAULT_BATCH_SIZE):
    """Get CIFAR-100 data loaders with optional subset sampling."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Download and load the training data
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    
    # If using a subset
    if subset_fraction < 1.0:
        num_train = len(trainset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(subset_fraction * num_train))
        train_idx = indices[:split]
        trainset = Subset(trainset, train_idx)
    
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    
    # Download and load the test data
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    
    return trainloader, testloader


def create_model(config, num_classes=10):
    """Create a SimpleCNN model with the given configuration.
    
    Args:
        config: Dictionary with hyperparameters
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
        
    Returns:
        PyTorch model
    """
    num_channels = config.get('num_channels', 32)
    dropout_rate = config.get('dropout_rate', 0.2)
    
    # Create a SimpleCNN model with the specified number of classes
    model = SimpleCNN(num_channels=num_channels, dropout_rate=dropout_rate, num_classes=num_classes)
    
    return model


def create_optimizer(model, config):
    """Create an optimizer based on the configuration.
    
    Args:
        model: PyTorch model
        config: Dictionary with hyperparameters
        
    Returns:
        PyTorch optimizer
    """
    optimizer_name = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    momentum = config.get('momentum', 0.9)
    weight_decay = config.get('weight_decay', 0.0005)
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        logger.warning(f"Unknown optimizer: {optimizer_name}, using Adam")
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def run_meta_optimization(dataset_name, resource_level, num_iterations=DEFAULT_NUM_ITERATIONS, num_configs=DEFAULT_NUM_CONFIGS):
    """Run meta-model optimization to find the best hyperparameters.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        resource_level: Fraction of dataset to use (e.g., 0.1 for 10%)
        num_iterations: Number of meta-model iterations
        num_configs: Number of configurations to try per iteration
        
    Returns:
        Dictionary with best hyperparameter configuration
    """
    logger.info(f"Running meta-optimization for {dataset_name} with resource level {resource_level}")
    
    # Get data loaders for the specified dataset
    if dataset_name.lower() == 'cifar10':
        trainloader, testloader = get_cifar10_loaders(subset_fraction=resource_level)
        num_classes = 10
    else:  # cifar100
        trainloader, testloader = get_cifar100_loaders(subset_fraction=resource_level)
        num_classes = 100
    
    # Define the hyperparameter search space
    search_space = {
        'num_channels': [16, 32, 64],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'optimizer': ['adam', 'sgd'],
        'learning_rate': [0.1, 0.01, 0.001, 0.0001],
        'momentum': [0.0, 0.9, 0.95],
        'weight_decay': [0.0, 0.0001, 0.0005, 0.001]
    }
    
    # Create the meta-model probing object
    # Create a list of configurations from the search space
    configs = []
    for _ in range(num_configs):
        config = {}
        for param, values in search_space.items():
            config[param] = np.random.choice(values)
        configs.append(config)
    
    # Define model and dataset functions
    def model_fn(config):
        return create_model(config, num_classes=num_classes)
        
    def dataset_fn(fraction):
        if dataset_name.lower() == 'cifar10':
            return get_cifar10_loaders(subset_fraction=fraction)
        else:  # cifar100
            return get_cifar100_loaders(subset_fraction=fraction)
    
    def optimizer_fn(model, config):
        return create_optimizer(model, config)
    
    meta_probing = MetaProbing(
        configs=configs,
        model_fn=model_fn,
        dataset_fn=dataset_fn,
        criterion=nn.CrossEntropyLoss(),
        optimizer_fn=optimizer_fn
    )
    
    # Define the evaluation function
    def evaluate_config(config):
        # Create model with the given configuration
        model = create_model(config, num_classes=num_classes)
        
        # Create optimizer
        optimizer = create_optimizer(model, config)
        
        # Create two-tier probing object
        probing = TwoTierProbing(model, optimizer, nn.CrossEntropyLoss())
        
        # Train for a small number of epochs to evaluate the configuration
        eval_epochs = 5
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Train and evaluate
        train_loss, train_acc, val_loss, val_acc = probing.train_and_evaluate(
            trainloader, testloader, epochs=eval_epochs, device=device
        )
        
        # Return metrics for the meta-model
        return {
            'val_accuracy': val_acc,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'train_loss': train_loss
        }
    
    # Run the meta-optimization
    best_config = meta_probing.run_meta_optimization(
        max_resource=1.0,  # Full dataset for final evaluation
        min_resource=resource_level,  # Start with the specified resource level
        num_iterations=num_iterations,
        measure_flatness=True  # Measure loss landscape flatness
    )
    
    logger.info(f"Best configuration found: {best_config}")
    return best_config


def train_and_evaluate(config, dataset_name, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, resource_level=1.0):
    """Train and evaluate a model with the given configuration.
    
    Args:
        config: Dictionary with hyperparameters
        dataset_name: 'cifar10' or 'cifar100'
        epochs: Number of epochs to train
        batch_size: Batch size for training
        resource_level: Fraction of dataset to use (default: full dataset)
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Training model with config: {config}")
    
    # Get data loaders for the specified dataset
    if dataset_name.lower() == 'cifar10':
        trainloader, testloader = get_cifar10_loaders(subset_fraction=resource_level, batch_size=batch_size)
        num_classes = 10
    else:  # cifar100
        trainloader, testloader = get_cifar100_loaders(subset_fraction=resource_level, batch_size=batch_size)
        num_classes = 100
    
    # Create model with the given configuration
    model = create_model(config, num_classes=num_classes)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create two-tier probing object
    probing = TwoTierProbing(model, optimizer, nn.CrossEntropyLoss())
    
    # Train for the specified number of epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Determine the model type for directory structure
    model_type = f"cifa{num_classes}" if resource_level == 1.0 else f"cifa{num_classes}_{int(resource_level*100)}"
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "reports", model_type)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create the training log file path
    training_log_path = os.path.join(reports_dir, f"{model_type}_training_log.txt")
    
    # Open the training log file in append mode
    with open(training_log_path, 'w') as log_file:
        log_file.write(f"Training log for {dataset_name} model with {int(resource_level*100)}% resource level\n")
        log_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.write(f"Configuration: {json.dumps(config, indent=2)}\n\n")
        log_file.write("Epoch,TrainingLoss,TrainingAccuracy,ValidationLoss,ValidationAccuracy,ElapsedTime\n")
    
    # Define progress callback for detailed logging that also writes to the log file
    def progress_callback(epoch, loss, accuracy, elapsed_time):
        logger.info(f"Epoch {epoch}/{epochs}: loss={loss:.4f}, accuracy={accuracy:.4f}, time={elapsed_time:.2f}s")
        
        # Write to the log file in real-time
        with open(training_log_path, 'a') as log_file:
            # Write the epoch data in CSV format
            log_file.write(f"{epoch},{loss:.4f},{accuracy:.4f},,,{elapsed_time:.2f}\n")
            log_file.flush()  # Ensure the data is written immediately
    
    # Train and evaluate
    train_loss, train_acc, val_loss, val_acc = probing.train_and_evaluate(
        trainloader, testloader, epochs=epochs, device=device, progress_callback=progress_callback
    )
    
    # Save the trained model to both results and reports directories
    # 1. Save to results directory with timestamp (for archival)
    model_dir = os.path.join(RESULTS_DIR, f"{dataset_name}_{int(resource_level*100)}pct_{DATE_STR}")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    # 2. Save to reports directory with fixed name (for website)
    # Create model-specific directory in reports
    model_type = f"cifa{num_classes}" if resource_level == 1.0 else f"cifa{num_classes}_{int(resource_level*100)}"
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "reports", model_type)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save the model with a fixed name
    fixed_model_path = os.path.join(reports_dir, "latest_model.pth")
    torch.save(model.state_dict(), fixed_model_path)
    
    # Save training log to the reports directory with a fixed name
    training_log_path = os.path.join(reports_dir, f"{model_type}_training_log.txt")
    
    # Create a training log with all the information
    with open(training_log_path, 'w') as log_file:
        log_file.write(f"Training log for {dataset_name} model with {int(resource_level*100)}% resource level\n")
        log_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.write(f"Configuration: {json.dumps(config, indent=2)}\n\n")
        log_file.write(f"Final validation accuracy: {val_acc:.4f}\n")
        log_file.write(f"Final training accuracy: {train_acc:.4f}\n")
        log_file.write(f"Final validation loss: {val_loss:.4f}\n")
        log_file.write(f"Final training loss: {train_loss:.4f}\n\n")
        log_file.write("Epoch,TrainingLoss,TrainingAccuracy,ValidationLoss,ValidationAccuracy\n")
        
        # Add epoch data if available from the probing object
        if hasattr(probing, 'epoch_metrics') and probing.epoch_metrics:
            for epoch_data in probing.epoch_metrics:
                log_file.write(f"{epoch_data['epoch']},{epoch_data['train_loss']:.4f},{epoch_data['train_acc']:.4f},{epoch_data['val_loss']:.4f},{epoch_data['val_acc']:.4f}\n")
    
    logger.info(f"Training log saved to {training_log_path}")
    
    # Create symbolic link in project root for easy access
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    symlink_path = os.path.join(project_root, f"{model_type}_training_log.txt")
    
    # Remove existing symlink if it exists
    if os.path.exists(symlink_path) and os.path.islink(symlink_path):
        os.unlink(symlink_path)
    
    # Create relative symlink
    try:
        os.symlink(os.path.relpath(training_log_path, project_root), symlink_path)
        logger.info(f"Created symbolic link at {symlink_path}")
    except Exception as e:
        logger.warning(f"Could not create symbolic link: {e}")
    
    # Save the configuration and results
    results = {
        'config': config,
        'val_accuracy': val_acc,
        'train_accuracy': train_acc,
        'val_loss': val_loss,
        'train_loss': train_loss,
        'epochs': epochs,
        'resource_level': resource_level,
        'dataset': dataset_name,
        'model_path': model_path
    }
    
    results_path = os.path.join(model_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Training completed. Results saved to {results_path}")
    return results
