#!/usr/bin/env python3
"""
Hyperparameter Optimization Module
================================

Functions for hyperparameter optimization and configuration evaluation.
"""

import torch
import logging
import numpy as np
import copy
import random
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def get_hyperparameter_space():
    """
    Define the hyperparameter search space.
    
    Returns:
        Dictionary with hyperparameter ranges and types
    """
    return {
        'learning_rate': {
            'type': 'float',
            'range': [1e-5, 1e-1],
            'log_scale': True
        },
        'weight_decay': {
            'type': 'float',
            'range': [1e-6, 1e-3],
            'log_scale': True
        },
        'batch_size': {
            'type': 'int',
            'range': [16, 256],
            'log_scale': True
        },
        'optimizer': {
            'type': 'categorical',
            'choices': ['adam', 'adamw', 'sgd']
        },
        'momentum': {
            'type': 'float',
            'range': [0.8, 0.99],
            'condition': {'optimizer': 'sgd'}
        },
        'nesterov': {
            'type': 'bool',
            'condition': {'optimizer': 'sgd'}
        }
    }

def sample_hyperparameter_configs(num_configs, base_config=None, perturbation_scale=None, seed=None):
    """
    Sample hyperparameter configurations from the search space.
    
    Args:
        num_configs: Number of configurations to sample
        base_config: Optional base configuration to perturb
        perturbation_scale: Scale of perturbations (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        List of hyperparameter configurations
    """
    from ..hyperparameter_utils import sample_hyperparameter_configs, generate_perturbed_configs
    
    if base_config is not None:
        # Generate perturbations around the base configuration
        return generate_perturbed_configs(
            base_config=base_config,
            num_perturbations=num_configs,
            perturbation_scale=perturbation_scale,
            seed=seed
        )
    else:
        # Sample new configurations
        return sample_hyperparameter_configs(
            num_configs=num_configs,
            seed=seed
        )

def evaluate_configuration(config, subset_idx, resource_level, model_creator, data_loader_creator, device):
    """
    Evaluate a hyperparameter configuration on a specific data subset.
    
    Args:
        config: Hyperparameter configuration to evaluate
        subset_idx: Index of the data subset to use
        resource_level: Fraction of the data to use (0.0 to 1.0)
        model_creator: Function to create a model
        data_loader_creator: Function to create a data loader
        device: Device to use for training and evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    from ..meta_model_components.training import create_optimizer
    
    try:
        # Create model
        model = model_creator(config)
        model.to(device)
        
        # Create optimizer
        optimizer = create_optimizer(model, config)
        
        # Create data loader
        train_loader, val_loader = data_loader_creator(subset_idx, config.get('batch_size', 128))
        
        # Train for a few epochs
        epochs = max(1, int(10 * resource_level))
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Evaluate
            val_accuracy = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_accuracy += (predicted == targets).sum().item()
            
            val_accuracy /= len(val_loader.dataset)
            val_accuracies.append(val_accuracy)
        
        # Return results
        return {
            'config': config,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'final_train_loss': train_losses[-1],
            'final_val_accuracy': val_accuracies[-1],
            'avg_val_accuracy': sum(val_accuracies) / len(val_accuracies),
            'subset_idx': subset_idx,
            'resource_level': resource_level,
            'status': 'success'
        }
    
    except Exception as e:
        # Log the error
        logger.error(f"Error evaluating configuration: {e}")
        
        # Return failure result
        return {
            'config': config,
            'train_losses': [],
            'val_accuracies': [],
            'final_train_loss': float('inf'),
            'final_val_accuracy': 0.0,
            'avg_val_accuracy': 0.0,
            'subset_idx': subset_idx,
            'resource_level': resource_level,
            'status': 'failure',
            'error': str(e)
        }
