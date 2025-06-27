#!/usr/bin/env python3
"""
CIFAR Meta-Model Optimizer Module
===============================

CIFAR-specific implementation of the meta-model optimizer.
"""

import torch
import logging
import numpy as np
import copy
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from .optimizer import MetaModelOptimizer
from .config import MetaModelConfig
from .training import train_one_epoch, evaluate_model, create_optimizer
from .hyperparameter_optimization import sample_hyperparameter_configs, evaluate_configuration

logger = logging.getLogger(__name__)

class CIFARMetaModelOptimizer(MetaModelOptimizer):
    """
    CIFAR-specific meta-model optimizer for hyperparameter optimization.
    
    This class handles the meta-model training and optimization process for
    CIFAR datasets, including evaluation on multiple data subsets.
    """
    
    def __init__(self, config: Optional[MetaModelConfig] = None, **kwargs):
        """
        Initialize the CIFAR meta-model optimizer.
        
        Args:
            config: Configuration object with all parameters. If None, uses defaults.
            **kwargs: Alternative way to provide parameters that will override the config.
        """
        super().__init__(config, **kwargs)
        
        # Initialize CIFAR-specific components
        self._initialize_cifar_components()
    
    def _initialize_cifar_components(self):
        """Initialize CIFAR-specific components."""
        from ..cifar_core import get_cifar_loaders
        from ..data_sampling import get_multiple_subsets
        
        # Get data subsets
        self.data_subsets = get_multiple_subsets(
            dataset_name=self.config.dataset,
            num_subsets=self.config.num_data_subsets,
            subset_size=self.config.subset_size,
            ensure_disjoint=self.config.ensure_disjoint_subsets,
            seed=self.config.random_seed
        )
        
        # Get main data loaders
        self.train_loader, self.val_loader = get_cifar_loaders(
            dataset_name=self.config.dataset,
            batch_size=self.config.batch_size
        )
        
        logger.info(f"Initialized {len(self.data_subsets)} data subsets for {self.config.dataset}")
    
    def optimize_hyperparameters(self, num_configs=10):
        """
        Train a meta-model to predict optimal hyperparameters for CIFAR datasets.
        
        Process:
        1. Sample hyperparameter configurations
        2. Evaluate these configurations on multiple data subsets
        3. Extract meta-features from the evaluation results
        4. Train a meta-model to predict the best hyperparameters
        5. Return the best hyperparameters for actual model training
        
        Args:
            num_configs: Number of configurations to evaluate
            
        Returns:
            Dictionary with best hyperparameters as basic Python types
        """
        logger.info(f"Starting meta-model training with {num_configs} configurations")
        
        # Sample hyperparameter configurations
        from ..hyperparameter_utils import sample_hyperparameter_configs
        configs = sample_hyperparameter_configs(num_configs=num_configs, seed=self.config.random_seed)
        logger.info(f"Sampled {len(configs)} hyperparameter configurations")
        
        # Initialize containers for meta-model training data
        meta_features = []
        meta_targets = []
        best_config_found = None
        best_accuracy = -float('inf')
        
        # Evaluate configurations on multiple data subsets
        total_subsets = len(self.data_subsets)
        total_evaluations = len(configs) * total_subsets
        evaluation_counter = 0
        
        # Evaluate each configuration on each subset
        for config_idx, config in enumerate(configs):
            logger.info(f"Evaluating configuration {config_idx+1}/{len(configs)}: {config}")
            subset_results = []
            
            for subset_idx in range(total_subsets):
                evaluation_counter += 1
                overall_progress = (evaluation_counter / total_evaluations) * 100
                logger.info(f"Evaluating dataset {subset_idx+1}/{total_subsets} [Overall Progress: {evaluation_counter}/{total_evaluations} ({overall_progress:.1f}%)]")
                
                # Evaluate the configuration on this subset - fail fast
                result = self._evaluate_configuration(config, subset_idx, resource_level=1)
                
                # Track best configuration based on validation accuracy
                if result['val_acc'] > best_accuracy:
                    best_accuracy = result['val_acc']
                    best_config_found = copy.deepcopy(config)
                    logger.info(f"New best configuration found with accuracy: {best_accuracy:.4f}")
                
                subset_results.append(result)
            
            # Extract meta-features from subset results
            config_meta_features = self._extract_meta_features(subset_results)
            meta_features.append(config_meta_features)
            
            # Use best accuracy as target
            best_subset_accuracy = max([r['val_acc'] for r in subset_results])
            meta_targets.append(best_subset_accuracy)
        
        # Train meta-model on collected data
        self._train_meta_model_on_data(meta_features, meta_targets)
        
        # Create a new dictionary with ONLY the hyperparameters we need
        # Don't even try to clean the original config - start fresh
        training_hyperparams = {}
        
        if best_config_found:
            # Extract only these specific keys we need for training
            keys_to_extract = [
                'learning_rate', 'weight_decay', 'optimizer', 'momentum', 'batch_size',
                'scheduler', 'step_size', 'gamma', 'model_type', 'dropout',
                'beta1', 'beta2', 'eps'
            ]
            
            # Extract only the keys we need and convert to basic Python types
            for key in keys_to_extract:
                if key in best_config_found:
                    value = best_config_found[key]
                    
                    # Convert to basic Python types
                    if isinstance(value, torch.Tensor):
                        training_hyperparams[key] = value.item() if value.numel() == 1 else value.tolist()
                    elif isinstance(value, np.ndarray):
                        training_hyperparams[key] = value.tolist()
                    elif isinstance(value, (np.int_, np.float_)):
                        training_hyperparams[key] = value.item()
                    elif isinstance(value, (int, float, str, bool)) or value is None:
                        training_hyperparams[key] = value
        
        # Add the best accuracy directly to the hyperparameters
        training_hyperparams['best_accuracy'] = float(best_accuracy)
        
        # Log the best hyperparameters and accuracy
        logger.info("Meta-model training complete. Best hyperparameters for final training:")
        logger.info(f"Best accuracy during evaluation: {best_accuracy:.4f}")
        for key, value in training_hyperparams.items():
            logger.info(f"  {key}: {value}")
        
        # Return the hyperparameters for final training
        return training_hyperparams
    
    def _evaluate_configuration(self, config, subset_idx, resource_level=1):
        """
        Evaluate a single hyperparameter configuration on a specific data subset.
        
        Args:
            config: Hyperparameter configuration to evaluate
            subset_idx: Index of the data subset to use
            resource_level: Resource level for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        # Get data subset - fail fast if index is invalid
        subset = self.data_subsets[subset_idx]  # Will raise IndexError if invalid
        
        # Create data loaders for this subset
        from ..cifar_core import create_sampled_loader
        train_loader = create_sampled_loader(
            dataset_name=self.config.dataset,
            subset=subset,
            batch_size=config.get('batch_size', self.config.batch_size)
        )
        
        # Create model and optimizer
        from ..cifar_core import create_model
        model = create_model(config=config)
        model.to(self.device)
        
        # Create optimizer
        from ..cifar_core import create_optimizer
        optimizer = create_optimizer(model, config)
        
        # Train for a few epochs
        epochs = max(1, int(10 * resource_level))
        train_loss = 0.0
        val_acc = 0.0
        
        for epoch in range(epochs):
            # Train for one epoch
            from ..cifar_core import train_one_epoch, evaluate_model
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=self.device
            )
            
            # Evaluate
            val_acc = evaluate_model(
                model=model,
                loader=self.val_loader,
                device=self.device
            )
        
        # Return results as simple Python types
        return {
            'config': config,
            'subset_idx': subset_idx,
            'train_loss': float(train_loss),
            'val_acc': float(val_acc),
            'resource_level': resource_level
        }
        
    def _extract_meta_features(self, subset_results):
        """
        Extract meta-features from subset evaluation results.
        
        Args:
            subset_results: List of evaluation results for each subset
            
        Returns:
            Dictionary of meta-features
        """
        # Basic statistics across subsets
        accuracies = [r['val_acc'] for r in subset_results]
        losses = [r['train_loss'] for r in subset_results]
        
        meta_features = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies) if len(accuracies) > 1 else 0.0,
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies),
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses) if len(losses) > 1 else 0.0,
        }
        
        # Add configuration parameters as features
        if subset_results:
            config = subset_results[0]['config']
            for key, value in config.items():
                if isinstance(value, (int, float)):
                    meta_features[f'config_{key}'] = value
        
        return meta_features
    
    def _train_meta_model_on_data(self, meta_features, meta_targets):
        """
        Train a meta-model (random forest or MLP) to predict hyperparameter performance.
        This is the core of our approach - we're training a model that can predict
        which hyperparameters will maximize accuracy on the target model.
        
        Args:
            meta_features: List of meta-feature dictionaries
            meta_targets: List of target values (accuracies)
        """
        # Fail fast - if we don't have data, fail loudly
        assert meta_features, "No meta-features available for meta-model training"
        assert meta_targets, "No meta-targets available for meta-model training"
        assert len(meta_features) == len(meta_targets), "Meta-features and targets must have the same length"
        
        # Log what we're doing
        logger.info(f"Training meta-model on {len(meta_features)} hyperparameter configurations")
        logger.info(f"Target accuracies range: {min(meta_targets):.4f} to {max(meta_targets):.4f}")
        
        # Convert meta-features to tensor
        feature_keys = sorted(meta_features[0].keys())
        features = torch.tensor(
            [[mf[k] for k in feature_keys] for mf in meta_features],
            dtype=torch.float32
        )
        
        # Convert targets to tensor
        targets = torch.tensor(meta_targets, dtype=torch.float32).unsqueeze(1)
        
        # Create and train meta-model
        meta_model = self._create_meta_model(
            input_dim=len(feature_keys),
            output_dim=1
        )
        
        # Train meta-model
        self._train_meta_model(
            meta_model=meta_model,
            features=features,
            targets=targets,
            epochs=50
        )
        
        logger.info(f"Meta-model trained successfully on {len(meta_features)} samples")
        logger.info("Meta-model can now predict hyperparameter performance")
    
    def _create_meta_model(self, input_dim, output_dim):
        """
        Create a simple MLP meta-model.
        
        Args:
            input_dim: Input dimension (number of features)
            output_dim: Output dimension (number of targets)
            
        Returns:
            PyTorch meta-model
        """
        # Create a simple MLP meta-model
        meta_model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim)
        )
        
        # Move to device
        meta_model.to(self.device)
        
        return meta_model
        
    def _train_meta_model(self, meta_model, features, targets, epochs=50):
        """
        Train the meta-model on features and targets.
        
        Args:
            meta_model: PyTorch meta-model
            features: Feature tensor
            targets: Target tensor
            epochs: Number of training epochs
        """
        # Move data to device
        features = features.to(self.device)
        targets = targets.to(self.device)
        
        # Create optimizer with weight decay for regularization
        optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            outputs = meta_model(features)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f"Meta-model epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        logger.info(f"Meta-model training completed after {epochs} epochs")
