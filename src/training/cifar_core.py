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
from typing import Dict, List, Tuple, Any, Optional

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
DEFAULT_SAMPLE_SIZES = [0.1, 0.2, 0.3, 0.4]
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


def run_meta_optimization(dataset_name, sample_size, num_iterations=DEFAULT_NUM_ITERATIONS, num_configs=DEFAULT_NUM_CONFIGS):
    """Run meta-model optimization to find the best hyperparameters.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        sample_size: Fraction of dataset to use (e.g., 0.1 for 10%)
        num_iterations: Number of meta-model iterations
        num_configs: Number of configurations to try per iteration
        
    Returns:
        Dictionary with best hyperparameter configuration
    """
    logger.info(f"Running meta-optimization for {dataset_name} with sample size {sample_size}")
    
    # Get data loaders for the specified dataset
    if dataset_name.lower() == 'cifar10':
        trainloader, testloader = get_cifar10_loaders(subset_fraction=sample_size)
        num_classes = 10
    else:  # cifar100
        trainloader, testloader = get_cifar100_loaders(subset_fraction=sample_size)
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
    
    # Setup logging to the appropriate report directory with correct naming (cifar not cifa)
    model_type = f"cifar{num_classes}" if sample_size == 1.0 else f"cifar{num_classes}_{int(sample_size*100)}"
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    reports_dir = os.path.join(project_root, "reports", model_type)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Define log file path in the reports directory using the new sample percentage naming convention
    log_file_path = os.path.join(reports_dir, f"{dataset_name}_meta_model_sample{int(sample_size*100)}pct.log")
    
    # Also create a logs directory at the project root for easier discovery
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logs_file_path = os.path.join(logs_dir, f"{dataset_name}_meta_model_sample{int(sample_size*100)}pct.log")
    
    # Add file handler to logger for project reports directory
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    # Add file handler to logger for project logs directory
    logs_file_handler = logging.FileHandler(logs_file_path, mode='w')
    logs_file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(logs_file_handler)
    
    # Also log to web server directory if it exists
    web_server_dir = "/var/www/html/loss.computer-wizard.com.au/reports"
    web_model_dir = os.path.join(web_server_dir, f"{model_type}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(web_model_dir, exist_ok=True)
        web_log_path = os.path.join(web_model_dir, f"{dataset_name}_meta_model_sample{int(sample_size*100)}pct.log")
        
        # Add another file handler for web server directory
        web_file_handler = logging.FileHandler(web_log_path, mode='w')
        web_file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(web_file_handler)
        logger.info(f"Also logging to web server at {web_log_path}")
    except Exception as e:
        logger.warning(f"Could not set up logging to web server: {e}")
    
    logger.info(f"Meta-model logs will be saved to {log_file_path} and {logs_file_path}")
    
    # Run the meta-optimization
    best_config = meta_probing.run_meta_optimization(
        max_resource=1.0,  # Full dataset for final evaluation
        min_resource=sample_size,  # Start with the specified sample size
        num_iterations=num_iterations,
        measure_flatness=True  # Measure loss landscape flatness
    )
    
    # Remove the file handlers after meta-optimization
    logger.removeHandler(file_handler)
    
    # Also remove web server file handler if it was added
    try:
        if 'web_file_handler' in locals():
            logger.removeHandler(web_file_handler)
    except Exception as e:
        logger.warning(f"Error removing web server file handler: {e}")
    
    logger.info(f"Best configuration found: {best_config}")
    return best_config


def train_and_evaluate(config, dataset_name, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, sample_size=1.0, max_training_time=3600*6):
    """Train and evaluate a model with the given configuration.
    
    Args:
        config: Dictionary with hyperparameters
        dataset_name: 'cifar10' or 'cifar100'
        epochs: Number of epochs to train
        batch_size: Batch size for training
        sample_size: Fraction of dataset to use (default: full dataset)
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Training model with config: {config}")
    
    # Get data loaders for the specified dataset
    if dataset_name.lower() == 'cifar10':
        trainloader, testloader = get_cifar10_loaders(subset_fraction=sample_size, batch_size=batch_size)
        num_classes = 10
    else:  # cifar100
        trainloader, testloader = get_cifar100_loaders(subset_fraction=sample_size, batch_size=batch_size)
        num_classes = 100
    
    # Create model with the given configuration
    model = create_model(config, num_classes=num_classes)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create two-tier probing object with the correct constructor parameters
    # Clear CUDA cache and garbage collect before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Track start time for timeout
    training_start_time = time.time()
    
    # Define helper function to log memory usage
    def log_memory_usage():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    
    # Log initial memory usage
    logger.info("Starting training with memory usage:")
    log_memory_usage()
    
    # Define model_fn and optimizer_fn for TwoTierProbing
    def model_fn(cfg):
        return create_model(cfg, num_classes=num_classes)
        
    def optimizer_fn(mdl, cfg):
        return create_optimizer(mdl, cfg)
        
    def dataset_fn(fraction):
        if dataset_name.lower() == 'cifar10':
            return get_cifar10_loaders(subset_fraction=fraction, batch_size=batch_size)
        else:  # cifar100
            return get_cifar100_loaders(subset_fraction=fraction, batch_size=batch_size)
    
    # Initialize with a single configuration (the one we're training)
    probing = TwoTierProbing(
        configs=[config],
        model_fn=model_fn,
        dataset_fn=dataset_fn,
        criterion=nn.CrossEntropyLoss(),
        optimizer_fn=optimizer_fn,
        max_epochs=epochs,
        device=device
    )
    
    # Log memory usage after model initialization
    logger.info("After model initialization:")
    log_memory_usage()
    
    # Determine the model type for directory structure
    # Use sample_pct in the name to clearly indicate percentage of dataset used
    model_type = f"cifar{num_classes}" if sample_size == 1.0 else f"cifar{num_classes}_sample{int(sample_size*100)}pct"
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create reports directory path
    reports_dir = os.path.join(project_root, "reports", model_type)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Also create logs directory at project root
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create the training log file paths - one in reports directory and one in logs directory
    reports_log_path = os.path.join(reports_dir, f"{model_type}_training_log.txt")
    logs_log_path = os.path.join(logs_dir, f"{model_type}_training_log.txt")
    
    # Store all log paths in a list for easier handling
    training_log_paths = [reports_log_path, logs_log_path]
    
    # Also prepare web server path if available
    web_server_dir = "/var/www/html/loss.computer-wizard.com.au/reports"
    web_model_dir = os.path.join(web_server_dir, model_type)
    web_log_path = None
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(web_model_dir, exist_ok=True)
        web_log_path = os.path.join(web_model_dir, f"{model_type}_training_log.txt")
        training_log_paths.append(web_log_path)
        logger.info(f"Will also log training progress to web server at {web_log_path}")
    except Exception as e:
        logger.warning(f"Could not set up logging to web server: {e}")
    
    # Define a custom JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.str_):
                return str(obj)
            return super(NumpyEncoder, self).default(obj)
    
    # Write initial log content to all log files
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Log the start of training
    logger.info(f"Training {dataset_name} model with {int(sample_size*100)}% sample size")
    logger.info(f"Configuration: {config}")
    
    # Create results directory with timestamp and sample size
    model_dir = os.path.join(RESULTS_DIR, f"{dataset_name}_{int(sample_size*100)}pct_{DATE_STR}")
    os.makedirs(model_dir, exist_ok=True)
    
    header_content = f"Training log for {dataset_name} model with {int(sample_size*100)}% sample size\n"
    header_content += f"Generated on: {timestamp}\n\n"
    header_content += f"Configuration: {json.dumps(config, indent=2, cls=NumpyEncoder)}\n\n"
    header_content += "Epoch,TrainingLoss,TrainingAccuracy,ValidationLoss,ValidationAccuracy,ElapsedTime\n"
    
    # Write to all log paths
    for log_path in training_log_paths:
        try:
            with open(log_path, 'w') as log_file:
                log_file.write(header_content)
        except Exception as e:
            logger.warning(f"Could not write to log file {log_path}: {e}")
    
    # Define progress callback for detailed logging that also writes to the log file
    def progress_callback(epoch, loss, accuracy, elapsed_time):
        logger.info(f"Epoch {epoch}/{epochs}: loss={loss:.4f}, accuracy={accuracy:.4f}, time={elapsed_time:.2f}s")
        
        # Write to all log files in real-time
        log_line = f"{epoch},{loss:.4f},{accuracy:.4f},,,{elapsed_time:.2f}\n"
        
        for log_path in training_log_paths:
            try:
                with open(log_path, 'a') as log_file:
                    log_file.write(log_line)
                    log_file.flush()  # Ensure the data is written immediately
            except Exception as e:
                logger.warning(f"Could not write to log file {log_path}: {e}")
    
    # Define progress callback for detailed logging that also writes to the log file
    def progress_callback_wrapper(epoch_data):
        epoch = epoch_data['epoch']
        loss = epoch_data['train_loss']
        accuracy = epoch_data['train_acc']
        val_loss = epoch_data['val_loss']
        val_acc = epoch_data['val_acc']
        elapsed_time = 0  # Not available in this context
        
        logger.info(f"Epoch {epoch}/{epochs}: loss={loss:.4f}, accuracy={accuracy:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Write to all log files in real-time
        log_line = f"{epoch},{loss:.4f},{accuracy:.4f},{val_loss:.4f},{val_acc:.4f},{elapsed_time:.2f}\n"
        
        for log_path in training_log_paths:
            try:
                with open(log_path, 'a') as log_file:
                    log_file.write(log_line)
                    log_file.flush()  # Ensure the data is written immediately
            except Exception as e:
                logger.warning(f"Could not write to log file {log_path}: {e}")
    
    # Register the callback with the probing object
    probing.epoch_callback = progress_callback_wrapper
    
    # Train and evaluate using the new API with timeout
    remaining_time = max_training_time - (time.time() - training_start_time)
    if remaining_time <= 0:
        raise TimeoutError("Training aborted: Maximum training time reached before starting training loop")
        
    logger.info(f"Starting training with max time: {max_training_time/3600:.1f} hours")
    
    try:
        result = probing.train_and_evaluate(
            config=config,
            sample_size=sample_size,
            measure_flatness=True,
            max_training_time=remaining_time
        )
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        # Log memory usage on error
        logger.error("Memory usage at time of error:")
        log_memory_usage()
        raise  # Re-raise the exception to be handled by the caller
    
    # Extract results
    val_loss = result.val_loss
    val_acc = result.val_metric
    
    # Get the last epoch metrics for training loss and accuracy
    if probing.epoch_metrics and len(probing.epoch_metrics) > 0:
        last_epoch = probing.epoch_metrics[-1]
        train_loss = last_epoch['train_loss']
        train_acc = last_epoch['train_acc']
    else:
        train_loss = 0.0
        train_acc = 0.0
    
    # Save the trained model to both results and reports directories
    # 1. Save to results directory with timestamp (for archival)
    model_dir = os.path.join(RESULTS_DIR, f"{dataset_name}_{int(sample_size*100)}pct_{DATE_STR}")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    # 2. Save to reports directory with fixed name (for website)
    # Create model-specific directory in reports using consistent naming
    model_type = f"cifar{num_classes}" if sample_size == 1.0 else f"cifar{num_classes}_{int(sample_size*100)}"
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "reports", model_type)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save the model with a fixed name
    fixed_model_path = os.path.join(reports_dir, "latest_model.pth")
    torch.save(model.state_dict(), fixed_model_path)
    
    # Prepare final training log content with summary information
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary_content = f"Training log for {dataset_name} model with {int(sample_size*100)}% sample percentage\n"
    summary_content += f"Generated on: {timestamp}\n\n"
    summary_content += f"Configuration: {json.dumps(config, indent=2, cls=NumpyEncoder)}\n\n"
    summary_content += f"Final validation accuracy: {val_acc:.4f}\n"
    summary_content += f"Final training accuracy: {train_acc:.4f}\n"
    summary_content += f"Final validation loss: {val_loss:.4f}\n"
    summary_content += f"Final training loss: {train_loss:.4f}\n\n"
    summary_content += "Epoch,TrainingLoss,TrainingAccuracy,ValidationLoss,ValidationAccuracy\n"
    
    # Add epoch data if available
    if hasattr(probing, 'epoch_metrics') and probing.epoch_metrics:
        for epoch_data in probing.epoch_metrics:
            summary_content += f"{epoch_data['epoch']},{epoch_data['train_loss']:.4f},{epoch_data['train_acc']:.4f},{epoch_data['val_loss']:.4f},{epoch_data['val_acc']:.4f}\n"
    
    # Define log paths for final logs
    reports_log_path = os.path.join(reports_dir, f"{model_type}_training_log.txt")
    logs_log_path = os.path.join(logs_dir, f"{model_type}_training_log.txt")
    project_root_log_path = os.path.join(project_root, f"{model_type}_training_log.txt")
    
    # List of all paths to write the final log to
    final_log_paths = [reports_log_path, logs_log_path, project_root_log_path]
    
    # Add web server path if available
    web_log_path = os.path.join(web_model_dir, f"{model_type}_training_log.txt") if web_model_dir else None
    if web_log_path:
        final_log_paths.append(web_log_path)
    
    # Write final logs to all locations
    for log_path in final_log_paths:
        try:
            with open(log_path, 'w') as log_file:
                log_file.write(summary_content)
            logger.info(f"Training log saved to {log_path}")
        except Exception as e:
            logger.warning(f"Could not write final log to {log_path}: {e}")
    
    # No need for symbolic links as we're writing directly to all locations
    
    # Save the configuration and results
    results = {
        'config': config,
        'val_accuracy': val_acc,
        'train_accuracy': train_acc,
        'val_loss': val_loss,
        'train_loss': train_loss,
        'epochs': epochs,
        'sample_size': sample_size,
        'dataset': dataset_name,
        'model_path': model_path
    }
    
    results_path = os.path.join(model_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    logger.info(f"Training completed. Results saved to {results_path}")
    return results
