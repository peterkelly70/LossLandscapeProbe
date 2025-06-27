#!/usr/bin/env python3
"""
Meta-Model Implementation Module
==============================

Implementation of the meta-model for hyperparameter prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class MetaModel(nn.Module):
    """
    Meta-model for hyperparameter prediction.
    
    This model takes meta-features as input and predicts optimal hyperparameters.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        """
        Initialize the meta-model.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Build the layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

def extract_meta_features(evaluation_results, config):
    """
    Extract meta-features from evaluation results.
    
    Args:
        evaluation_results: List of evaluation result dictionaries
        config: Meta-model configuration
        
    Returns:
        Dictionary of meta-features
    """
    # Initialize meta-features
    meta_features = {
        'num_configs': len(evaluation_results),
        'best_accuracy': 0.0,
        'worst_accuracy': 1.0,
        'mean_accuracy': 0.0,
        'accuracy_std': 0.0,
        'learning_rate_range': [float('inf'), float('-inf')],
        'batch_size_range': [float('inf'), float('-inf')],
        'weight_decay_range': [float('inf'), float('-inf')],
        'optimizer_counts': {},
        'configs_by_performance': [],
        'num_classes': 0,
        'input_channels': 0,
        'input_size': 0
    }
    
    # Extract features from evaluation results
    accuracies = []
    
    for result in evaluation_results:
        config_data = result['config']
        avg_accuracy = result.get('avg_val_accuracy', 0.0)
        accuracies.append(avg_accuracy)
        
        # Track best and worst accuracy
        meta_features['best_accuracy'] = max(meta_features['best_accuracy'], avg_accuracy)
        meta_features['worst_accuracy'] = min(meta_features['worst_accuracy'], avg_accuracy)
        
        # Track hyperparameter ranges
        if 'learning_rate' in config_data:
            lr = config_data['learning_rate']
            meta_features['learning_rate_range'][0] = min(meta_features['learning_rate_range'][0], lr)
            meta_features['learning_rate_range'][1] = max(meta_features['learning_rate_range'][1], lr)
            
        if 'batch_size' in config_data:
            bs = config_data['batch_size']
            meta_features['batch_size_range'][0] = min(meta_features['batch_size_range'][0], bs)
            meta_features['batch_size_range'][1] = max(meta_features['batch_size_range'][1], bs)
            
        if 'weight_decay' in config_data:
            wd = config_data['weight_decay']
            meta_features['weight_decay_range'][0] = min(meta_features['weight_decay_range'][0], wd)
            meta_features['weight_decay_range'][1] = max(meta_features['weight_decay_range'][1], wd)
        
        # Track optimizer types
        optimizer_type = config_data.get('optimizer', 'unknown')
        if optimizer_type not in meta_features['optimizer_counts']:
            meta_features['optimizer_counts'][optimizer_type] = 0
        meta_features['optimizer_counts'][optimizer_type] += 1
        
        # Store config with its performance for ranking
        meta_features['configs_by_performance'].append({
            'config': config_data,
            'accuracy': avg_accuracy
        })
    
    # Calculate statistics
    if accuracies:
        meta_features['mean_accuracy'] = sum(accuracies) / len(accuracies)
        meta_features['accuracy_std'] = (sum((x - meta_features['mean_accuracy'])**2 for x in accuracies) / len(accuracies))**0.5 if len(accuracies) > 1 else 0.0
    
    # Sort configs by performance (descending)
    meta_features['configs_by_performance'].sort(key=lambda x: x['accuracy'], reverse=True)
    
    return meta_features

def create_meta_model_datasets(meta_features, best_config):
    """
    Create datasets for meta-model training.
    
    Args:
        meta_features: Dictionary of meta-features
        best_config: Best hyperparameter configuration
        
    Returns:
        Tuple of (features, targets) for meta-model training
    """
    # Extract features
    features = []
    
    # Numerical features
    features.append(meta_features['best_accuracy'])
    features.append(meta_features['worst_accuracy'])
    features.append(meta_features['mean_accuracy'])
    features.append(meta_features['accuracy_std'])
    
    # Learning rate range
    features.append(meta_features['learning_rate_range'][0])
    features.append(meta_features['learning_rate_range'][1])
    
    # Batch size range
    features.append(meta_features['batch_size_range'][0])
    features.append(meta_features['batch_size_range'][1])
    
    # Weight decay range
    features.append(meta_features['weight_decay_range'][0])
    features.append(meta_features['weight_decay_range'][1])
    
    # Optimizer counts
    for opt in ['adam', 'adamw', 'sgd']:
        features.append(meta_features['optimizer_counts'].get(opt, 0) / meta_features['num_configs'])
    
    # Dataset properties
    features.append(meta_features['num_classes'])
    features.append(meta_features['input_channels'])
    features.append(meta_features['input_size'])
    
    # Convert to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    # Extract targets (best hyperparameters)
    targets = []
    
    # Learning rate
    targets.append(best_config.get('learning_rate', 0.001))
    
    # Weight decay
    targets.append(best_config.get('weight_decay', 0.0001))
    
    # Batch size
    targets.append(best_config.get('batch_size', 128))
    
    # Optimizer (one-hot encoded)
    optimizer_type = best_config.get('optimizer', 'adam')
    for opt in ['adam', 'adamw', 'sgd']:
        targets.append(1.0 if optimizer_type == opt else 0.0)
    
    # Convert to tensor
    targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(0)
    
    return features_tensor, targets_tensor
