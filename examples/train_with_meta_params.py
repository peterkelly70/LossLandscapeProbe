"""
Train CIFAR-10 with Meta-Model Predicted Hyperparameters
=======================================================

This script trains a CNN on CIFAR-10 using the hyperparameters predicted by the meta-model
and compares the results against state-of-the-art accuracy.
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
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from llp.meta_probing import MetaProbing
from examples.cifar10_example import SimpleCNN, get_cifar10_loaders, create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# State-of-the-art accuracy for simple CNN on CIFAR-10 (approximate)
SOTA_ACCURACY = 0.93  # ~93% for advanced CNN architectures

def train_model(config, epochs=100, device=None):
    """
    Train a model with the given configuration for the specified number of epochs.
    
    Args:
        config: Hyperparameter configuration
        epochs: Number of epochs to train
        device: Device to train on
        
    Returns:
        Tuple of (trained model, training history, test accuracy)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Training on device: {device}")
    
    # Create model with the given configuration
    model = create_model(config)
    model = model.to(device)
    
    # Create optimizer
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Get data loaders (full dataset)
    train_loader, test_loader = get_cifar10_loaders(data_fraction=1.0, batch_size=128)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Training loop
    start_time = time.time()
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
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
        train_acc = correct / total
        
        # Testing phase
        model.eval()
        test_loss = 0.0
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
        test_acc = correct / total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - start_time
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                   f"Time: {epoch_time:.2f}s, Total: {total_elapsed:.2f}s")
    
    # Final evaluation
    model.eval()
    test_loss = 0.0
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
    test_acc = correct / total
    
    logger.info(f"Final Test Accuracy: {test_acc:.4f}")
    logger.info(f"Total training time: {time.time() - start_time:.2f}s")
    
    return model, history, test_acc

def plot_training_history(history, title="Training History"):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        title: Plot title
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('training_history.png')
    logger.info("Training history plot saved as 'training_history.png'")
    
    # Show the plot
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Meta-model predicted hyperparameters
    meta_config = {
        'num_channels': 32,
        'dropout_rate': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'momentum': 0.0,
        'weight_decay': 0.0005
    }
    
    # Number of epochs for full training
    epochs = 100
    
    logger.info("Starting training with meta-model predicted hyperparameters")
    logger.info(f"Configuration: {meta_config}")
    logger.info(f"Training for {epochs} epochs")
    
    # Train the model
    model, history, test_acc = train_model(meta_config, epochs=epochs)
    
    # Compare with state-of-the-art
    logger.info("\nResults Comparison:")
    logger.info(f"{'Model':<30} {'Accuracy':<10}")
    logger.info(f"{'-'*40}")
    logger.info(f"{'Meta-model predicted params':<30} {test_acc:.4f}")
    logger.info(f"{'State-of-the-art (approx)':<30} {SOTA_ACCURACY:.4f}")
    logger.info(f"{'Difference':<30} {test_acc - SOTA_ACCURACY:.4f}")
    
    # Calculate percentage of SOTA achieved
    sota_percentage = (test_acc / SOTA_ACCURACY) * 100
    logger.info(f"Meta-model achieved {sota_percentage:.2f}% of state-of-the-art accuracy")
    
    # Plot training history
    plot_training_history(history, title="Training with Meta-Model Predicted Hyperparameters")
    
    # Save the model
    torch.save(model.state_dict(), 'meta_model_trained.pth')
    logger.info("Model saved as 'meta_model_trained.pth'")

if __name__ == "__main__":
    main()
