#!/usr/bin/env python3
"""
CIFAR-100 Transfer Example
=========================

This script tests the generalization capability of the meta-model trained on CIFAR-10
by applying it to the CIFAR-100 dataset without retraining.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import logging
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llp.meta_probing import MetaProbing

# Define SimpleCNN model for CIFAR-100 (adapted from cifar10_example.py)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100, num_channels=32, dropout_rate=0.2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_channels * 4 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)  # Changed to num_classes for CIFAR-100
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, self.fc1.in_features)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cifar100_transfer.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Define transforms
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
    
    # Load CIFAR-100
    logger.info("Loading CIFAR-100 dataset...")
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    
    # Define dataset function
    def get_cifar100_loaders(data_fraction=1.0):
        if data_fraction < 1.0:
            # Use a subset of training data
            indices = torch.randperm(len(trainset))[:int(len(trainset) * data_fraction)]
            subset = torch.utils.data.Subset(trainset, indices)
            train_loader = torch.utils.data.DataLoader(
                subset, batch_size=128, shuffle=True, num_workers=2)
        else:
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=True, num_workers=2)
            
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    # Define model function (adapted for 100 classes)
    def create_model(config):
        return SimpleCNN(
            num_classes=100,
            num_channels=config.get('num_channels', 32),
            dropout_rate=config.get('dropout_rate', 0.2)
        ).to(device)
    
    # Define optimizer function
    def create_optimizer(model, config):
        optimizer_name = config.get('optimizer', 'adam').lower()
        lr = config.get('learning_rate', 0.001)
        momentum = config.get('momentum', 0.9)
        weight_decay = config.get('weight_decay', 0.0005)
        
        if optimizer_name == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define hyperparameter configurations to explore
    configs = [
        {'num_channels': 32, 'dropout_rate': 0.2, 'optimizer': 'adam', 'learning_rate': 0.001, 'momentum': 0.0, 'weight_decay': 0.0005},
        {'num_channels': 64, 'dropout_rate': 0.3, 'optimizer': 'sgd', 'learning_rate': 0.01, 'momentum': 0.9, 'weight_decay': 0.0001},
        {'num_channels': 32, 'dropout_rate': 0.1, 'optimizer': 'adam', 'learning_rate': 0.0005, 'momentum': 0.0, 'weight_decay': 0.001},
        {'num_channels': 48, 'dropout_rate': 0.25, 'optimizer': 'sgd', 'learning_rate': 0.005, 'momentum': 0.9, 'weight_decay': 0.0005},
        {'num_channels': 64, 'dropout_rate': 0.2, 'optimizer': 'adam', 'learning_rate': 0.001, 'momentum': 0.0, 'weight_decay': 0.0001},
        {'num_channels': 32, 'dropout_rate': 0.3, 'optimizer': 'sgd', 'learning_rate': 0.01, 'momentum': 0.9, 'weight_decay': 0.001},
    ]
    
    # Initialize meta-probing with CIFAR-10 trained meta-model
    meta_model_dir = os.path.join(os.path.dirname(__file__), '../models/meta')
    probing = MetaProbing(
        configs=configs,
        model_fn=create_model,
        dataset_fn=get_cifar100_loaders,
        criterion=criterion,
        optimizer_fn=create_optimizer,
        max_epochs=50,
        device=device,
        meta_model_dir=meta_model_dir
    )
    
    # Run meta-optimization
    logger.info("Running meta-optimization with CIFAR-10 trained meta-model on CIFAR-100...")
    best_config = probing.run_meta_optimization(
        min_resource=0.1,
        max_resource=0.5,
        reduction_factor=2,
        num_initial_configs=6,
        num_iterations=3
    )
    
    logger.info(f"Best configuration predicted by meta-model: {best_config}")
    
    # Train model with best configuration
    logger.info("Training final model with predicted best configuration...")
    train_loader, test_loader = get_cifar100_loaders(data_fraction=1.0)
    model = create_model(best_config)
    optimizer = create_optimizer(model, best_config)
    
    # Training loop
    epochs = 50
    best_acc = 0.0
    train_losses = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = correct / total
        test_accs.append(test_acc)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Save best model
            torch.save(model.state_dict(), 'cifar100_multisamplesize_trained.pth')
    
    # Final evaluation
    model.load_state_dict(torch.load('cifar100_multisamplesize_trained.pth'))
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    final_acc = correct / total
    
    # Compare to CIFAR-10 results
    cifar10_acc = 0.847  # Our CIFAR-10 accuracy
    relative_performance = final_acc / cifar10_acc
    
    logger.info(f"Final Test Accuracy on CIFAR-100: {final_acc:.4f}")
    logger.info(f"Best Test Accuracy on CIFAR-100: {best_acc:.4f}")
    logger.info(f"CIFAR-10 Accuracy (for reference): {cifar10_acc:.4f}")
    logger.info(f"Relative Performance (CIFAR-100/CIFAR-10): {relative_performance:.4f}")
    logger.info(f"This represents how well the meta-model transfers from CIFAR-10 to CIFAR-100")
    
    # Save results
    results = {
        'best_config': best_config,
        'final_acc': final_acc,
        'best_acc': best_acc,
        'train_losses': train_losses,
        'test_accs': test_accs,
        'cifar10_acc': cifar10_acc,
        'relative_performance': relative_performance
    }
    
    torch.save(results, 'cifar100_multisamplesize_results.pth')
    # Also save with the standard name for backward compatibility
    torch.save(results, 'cifar100_transfer_results.pth')
    logger.info("Results saved to cifar100_transfer_results.pth")

if __name__ == "__main__":
    main()
