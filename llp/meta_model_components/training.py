#!/usr/bin/env python3
"""
Meta-Model Training Module
=========================

Training and evaluation functions for the meta-model.
"""

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def train_one_epoch(model, loader, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model to train
        loader: DataLoader with training data
        optimizer: PyTorch optimizer
        device: Device to train on (cuda or cpu)
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    if loader is None:
        logger.error("train_loader is None! Cannot train model.")
        return float('inf')

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # MSE loss for regression
        loss = torch.nn.functional.mse_loss(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
    
    return running_loss / total_samples if total_samples > 0 else float('inf')

def evaluate_model(model, loader, device):
    """
    Evaluate model on validation data.
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader with validation data
        device: Device to evaluate on (cuda or cpu)
        
    Returns:
        Negative MSE loss (higher is better, for compatibility with accuracy metrics)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    if loader is None:
        logger.error("val_loader is None! Cannot evaluate model.")
        return -float('inf')
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # MSE loss for regression
            loss = torch.nn.functional.mse_loss(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    # Return negative MSE as accuracy (higher is better)
    return -running_loss / total_samples if total_samples > 0 else -float('inf')

def create_optimizer(model, optimizer_config):
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        optimizer_config: Dictionary with optimizer configuration
        
    Returns:
        PyTorch optimizer
    """
    lr = optimizer_config.get('learning_rate', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 0.0001)
    optimizer_type = optimizer_config.get('optimizer', 'adam').lower()
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(optimizer_config.get('beta1', 0.9), optimizer_config.get('beta2', 0.999)),
            eps=optimizer_config.get('eps', 1e-8)
        )
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(optimizer_config.get('beta1', 0.9), optimizer_config.get('beta2', 0.999)),
            eps=optimizer_config.get('eps', 1e-8)
        )
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=optimizer_config.get('nesterov', False)
        )
    else:
        # Default to Adam
        logger.warning(f"Unknown optimizer type: {optimizer_type}, defaulting to Adam")
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
