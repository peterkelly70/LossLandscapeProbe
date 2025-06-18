#!/usr/bin/env python3
"""
Resource Level Comparison for CIFAR-100

This script compares the effectiveness of meta-models trained at different resource levels
(0.1, 0.2, 0.3, 0.4) on the CIFAR-100 dataset. For each resource level, it:
1. Trains a meta-model
2. Predicts optimal hyperparameters
3. Trains a full model with those hyperparameters
4. Records training loss and test accuracy

Results are saved to a CSV file for analysis.
"""

import os
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import subprocess
import sys

# Add the parent directory to the path so we can import the LLP package
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llp.models.simple_cnn import SimpleCNN
from llp.two_tier_probing import TwoTierProbing
from llp.meta_probing import MetaModelProbing
from llp.utils.logging_utils import setup_logger

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

# Constants
RESOURCE_LEVELS = [0.1, 0.2, 0.3, 0.4]
NUM_ITERATIONS = 3  # Number of meta-model iterations
NUM_CONFIGS = 10    # Number of configurations to try per iteration
EPOCHS = 100        # Number of epochs for final training
BATCH_SIZE = 128
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Date-time string for unique filenames
DATE_STR = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_cifar100_loaders(subset_fraction=1.0):
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
    
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)
    
    # Download and load the test data
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)
    
    return trainloader, testloader

def create_model(config):
    """Create a SimpleCNN model with the given configuration."""
    # For CIFAR-100, we need 100 output classes instead of 10
    num_channels = config.get('num_channels', 32)
    dropout_rate = config.get('dropout_rate', 0.2)
    return SimpleCNN(num_channels=num_channels, dropout_rate=dropout_rate, num_classes=100)

def create_optimizer(model, config):
    """Create an optimizer based on the configuration."""
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

def train_and_evaluate(config, epochs=100):
    """Train a model with the given configuration and evaluate on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get data loaders
    train_loader, test_loader = get_cifar100_loaders()
    
    # Create model and optimizer
    model = create_model(config)
    model = model.to(device)
    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Training
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{epochs} | '
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                       f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    training_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    final_test_acc = 100. * correct / total
    
    return {
        'final_test_acc': final_test_acc,
        'training_time': training_time,
        'history': history
    }

def generate_test_report(config, resource_level):
    """Generate a test report for a model with the given configuration."""
    logger.info("Generating test report...")
    
    # Convert config values to strings for command line
    config_str = {k: str(v) if not isinstance(v, str) else v for k, v in config.items()}
    
    # Build the command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate_test_report.py'),
        '--dataset', 'cifar100',  # Specify CIFAR-100 dataset
        '--num_channels', str(config_str.get('num_channels', '32')),
        '--dropout_rate', str(config_str.get('dropout_rate', '0.2')),
        '--optimizer', config_str.get('optimizer', 'adam'),
        '--learning_rate', str(config_str.get('learning_rate', '0.001')),
        '--momentum', str(config_str.get('momentum', '0.9')),
        '--weight_decay', str(config_str.get('weight_decay', '0.0005')),
        '--resource_level', str(resource_level),
        '--num_classes', '100'  # CIFAR-100 has 100 classes
    ]
    
    # Run the command
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Test report generated successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating test report: {e}")
        return False

def update_website():
    """Update the website with the latest reports and visualizations."""
    logger.info("Updating website...")
    
    # Build the command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'setup_website.py')
    ]
    
    # Run the command
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Website updated successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error updating website: {e}")
        return False

def run_experiment():
    """Run the resource level comparison experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    results = []
    
    for resource_level in RESOURCE_LEVELS:
        logger.info(f"\n{'='*80}\nTesting resource level: {resource_level}\n{'='*80}")
        
        # Define the dataset function with the current resource level
        def dataset_fn(fraction=1.0):
            return get_cifar100_loaders(subset_fraction=fraction * resource_level)
        
        # Create the TwoTierProbing object
        probing = TwoTierProbing(
            model_fn=create_model,
            optimizer_fn=create_optimizer,
            dataset_fn=dataset_fn,
            device=device
        )
        
        # Create the MetaModelProbing object
        meta_probing = MetaModelProbing(probing)
        
        # Define the hyperparameter space
        hyperparameter_space = {
            'num_channels': {'type': 'int', 'min': 16, 'max': 64},
            'dropout_rate': {'type': 'float', 'min': 0.0, 'max': 0.5},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd']},
            'learning_rate': {'type': 'float', 'min': 0.0001, 'max': 0.01, 'log': True},
            'momentum': {'type': 'float', 'min': 0.0, 'max': 0.99},
            'weight_decay': {'type': 'float', 'min': 0.0001, 'max': 0.001, 'log': True}
        }
        
        # Run meta-model optimization
        start_time = time.time()
        best_config = meta_probing.run_meta_optimization(
            hyperparameter_space=hyperparameter_space,
            num_iterations=NUM_ITERATIONS,
            num_configs=NUM_CONFIGS,
            resource_level=resource_level
        )
        meta_time = time.time() - start_time
        
        logger.info(f"Meta-model optimization completed in {meta_time:.2f}s")
        logger.info(f"Best configuration: {best_config}")
        
        # Train a full model with the best configuration
        logger.info("Training full model with best configuration...")
        eval_result = train_and_evaluate(best_config, epochs=EPOCHS)
        
        # Generate test report for this configuration
        generate_test_report(best_config, resource_level)
        
        # Save results
        result = {
            'resource_level': resource_level,
            'best_config': best_config,
            'meta_time': meta_time,
            'training_time': eval_result['training_time'],
            'final_test_acc': eval_result['final_test_acc'],
            'history': eval_result['history']
        }
        
        results.append(result)
        
        # Save intermediate results
        with open(os.path.join(RESULTS_DIR, f'cifar100_resource_level_{resource_level}_{DATE_STR}.json'), 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            result_copy = result.copy()
            result_copy['best_config'] = {k: v.item() if hasattr(v, 'item') else v 
                                         for k, v in result_copy['best_config'].items()}
            json.dump(result_copy, f, indent=2)
    
    # Create summary DataFrame
    summary = []
    for result in results:
        summary.append({
            'resource_level': result['resource_level'],
            'meta_time': result['meta_time'],
            'training_time': result['training_time'],
            'final_test_acc': result['final_test_acc'],
            'total_time': result['meta_time'] + result['training_time'],
            'num_channels': result['best_config'].get('num_channels'),
            'dropout_rate': result['best_config'].get('dropout_rate'),
            'optimizer': result['best_config'].get('optimizer'),
            'learning_rate': result['best_config'].get('learning_rate'),
            'momentum': result['best_config'].get('momentum'),
            'weight_decay': result['best_config'].get('weight_decay')
        })
    
    df = pd.DataFrame(summary)
    
    # Save summary to CSV
    csv_path = os.path.join(RESULTS_DIR, f'cifar100_resource_level_comparison_{DATE_STR}.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    
    # Print summary
    logger.info("\nSummary of Results:")
    logger.info(f"\n{df.to_string()}")
    
    # Generate test report for the best model
    best_result = max(results, key=lambda x: x['final_test_acc'])
    logger.info(f"\nBest result achieved with resource level {best_result['resource_level']}")
    logger.info(f"Test accuracy: {best_result['final_test_acc']:.2f}%")
    logger.info(f"Configuration: {best_result['best_config']}")
    
    # Update the website with all the reports
    update_website()

if __name__ == "__main__":
    run_experiment()
