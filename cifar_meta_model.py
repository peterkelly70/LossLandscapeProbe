#!/usr/bin/env python3
"""
CIFAR Meta-Model Module
======================

Hyperparameter prediction and meta-model optimization for CIFAR datasets.
This module handles:
- Meta-model training
- Hyperparameter prediction
- Configuration evaluation
"""

import torch
import numpy as np
import logging
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import random

# Import LLP modules
from llp.meta_probing import MetaProbing
from llp.meta_model import HyperparameterPredictor, DatasetFeatureExtractor
from cifar_core import create_model, create_optimizer, get_cifar_loaders

logger = logging.getLogger(__name__)


class CIFARMetaModelOptimizer:
    """
    Meta-model optimizer for CIFAR datasets.
    
    This class handles the meta-model training and hyperparameter prediction
    for CIFAR datasets using the LossLandscapeProbe framework.
    """
    
    def __init__(
        self, 
        dataset: str = 'cifar10',
        data_fraction: float = 1.0,
        batch_size: int = 128,
        configs_per_sample: int = 10,
        perturbations: int = 5,
        iterations: int = 3,
        min_resource: float = 0.1,
        max_resource: float = 0.5,
        run_dir: Optional[Path] = None
    ):
        """
        Initialize the meta-model optimizer.
        
        Args:
            dataset: Dataset name ('cifar10' or 'cifar100')
            data_fraction: Fraction of training data to use
            batch_size: Batch size for training
            configs_per_sample: Number of hyperparameter configurations to sample per iteration
            perturbations: Number of weight perturbations per configuration
            iterations: Number of meta-model training iterations
            min_resource: Minimum resource level for meta-model training
            max_resource: Maximum resource level for meta-model training
            run_dir: Directory to save results
        """
        self.dataset = dataset
        self.data_fraction = data_fraction
        self.batch_size = batch_size
        self.configs_per_sample = configs_per_sample
        self.perturbations = perturbations
        self.iterations = iterations
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.run_dir = run_dir
        
        # Initialize meta-model components
        self.meta_probing = None
        self.predictor = None
        self.feature_extractor = None
        
        # Load dataset
        self.train_loader, self.val_loader, self.num_classes = get_cifar_loaders(
            dataset=dataset,
            data_fraction=data_fraction,
            batch_size=batch_size
        )
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def _initialize_meta_model(self):
        """Initialize the meta-model components."""
        # Create feature extractor for dataset features
        self.feature_extractor = DatasetFeatureExtractor()
        
        # Create hyperparameter predictor with model directory
        model_dir = str(self.run_dir / "meta_model") if self.run_dir else None
        self.predictor = HyperparameterPredictor(model_dir=model_dir)
        
        # Define hyperparameter types for the predictor
        self.predictor.hyperparameter_types = {
            'num_channels': int,
            'dropout_rate': float,
            'optimizer': str,
            'learning_rate': float,
            'momentum': float,
            'weight_decay': float
        }
        
        # Get hyperparameter configurations to evaluate
        configs = self._sample_hyperparameter_configs(self.configs_per_sample)
        
        # Define model creation function
        def model_fn(config):
            return create_model(
                config=config,
                num_classes=self.num_classes
            )
        
        # Define dataset function
        def dataset_fn(data_fraction):
            return get_cifar_loaders(
                dataset=self.dataset,
                data_fraction=data_fraction,
                batch_size=self.batch_size
            )[:2]  # Only return train and val loaders
        
        # Define optimizer function
        def optimizer_fn(model, config):
            return create_optimizer(
                model=model,
                config=config
            )
        
        # Create meta-probing instance
        self.meta_probing = MetaProbing(
            configs=configs,
            model_fn=model_fn,
            dataset_fn=dataset_fn,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer_fn=optimizer_fn,
            max_epochs=10,  # Use a small number of epochs for meta-model training
            device=self.device,
            meta_model_dir=model_dir
        )
    
    def _get_hyperparameter_space(self):
        """
        Define the hyperparameter search space.
        
        Returns:
            Dictionary with hyperparameter ranges
        """
        return {
            'num_channels': [16, 32, 64, 128],
            'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'optimizer': ['sgd', 'adam'],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            'momentum': [0.0, 0.9, 0.95, 0.99],
            'weight_decay': [0.0, 0.0001, 0.0005, 0.001, 0.005]
        }
    
    def _sample_hyperparameter_configs(self, num_configs):
        """
        Sample hyperparameter configurations from the search space.
        
        Args:
            num_configs: Number of configurations to sample
            
        Returns:
            List of hyperparameter configurations
        """
        hp_space = self._get_hyperparameter_space()
        configs = []
        
        for _ in range(num_configs):
            config = {
                'num_channels': random.choice(hp_space['num_channels']),
                'dropout_rate': random.choice(hp_space['dropout_rate']),
                'optimizer': random.choice(hp_space['optimizer']),
                'learning_rate': random.choice(hp_space['learning_rate']),
                'momentum': random.choice(hp_space['momentum']) if random.choice(hp_space['optimizer']) == 'sgd' else 0.0,
                'weight_decay': random.choice(hp_space['weight_decay'])
            }
            configs.append(config)
        
        return configs
    
    def _evaluate_configuration(self, config, resource_level=0.1, epochs=5):
        """
        Evaluate a single hyperparameter configuration.
        
        Args:
            config: Hyperparameter configuration
            resource_level: Fraction of epochs to use
            epochs: Maximum number of epochs
            
        Returns:
            Dictionary with evaluation results
        """
        # Create model and optimizer
        model = create_model(config, num_classes=self.num_classes).to(self.device)
        optimizer = create_optimizer(model, config)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Calculate number of epochs for this resource level
        num_epochs = max(1, int(epochs * resource_level))
        
        # Train for the specified number of epochs
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_loss = running_loss / total
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_loss = running_loss / total
            val_acc = correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Return evaluation results
        return {
            'config': config,
            'train_loss': train_losses[-1],
            'train_acc': train_accs[-1],
            'val_loss': val_losses[-1],
            'val_acc': val_accs[-1],
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'epochs': num_epochs
        }
    
    def train_meta_model(self):
        """
        Train the meta-model and predict the best hyperparameters.
        
        Returns:
            Dictionary with the best predicted hyperparameters
        """
        logger.info("Starting meta-model training")
        
        # Initialize meta-model components
        self._initialize_meta_model()
        
        # Create output directory for meta-model results
        if self.run_dir:
            meta_dir = self.run_dir / "meta_model"
            meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Run meta-model optimization using the built-in method
        # This handles the iterations, evaluations, and meta-model training
        best_config = self.meta_probing.run_meta_optimization(
            min_resource=self.min_resource,
            max_resource=self.max_resource,
            num_iterations=self.iterations,
            num_initial_configs=self.configs_per_sample,
            measure_flatness=True
        )
        
        # Save the results
        if self.run_dir:
            # Save meta-model results
            with open(meta_dir / "meta_results.json", "w") as f:
                json.dump({
                    'best_config': best_config,
                    'best_val_acc': 0.0  # We don't have direct access to this from run_meta_optimization
                }, f, indent=2)
        
        logger.info(f"Meta-model training completed. Best configuration: {best_config}")
        return best_config
        
    def predict_best_hyperparameters(self):
        """
        Predict the best hyperparameters using the trained meta-model.
        
        Returns:
            Dictionary with the best predicted hyperparameters
        """
        if self.meta_probing is None:
            logger.warning("Meta-model not trained. Training now...")
            return self.train_meta_model()
        
        # Use the meta-model to predict the best hyperparameters
        best_config = self.meta_probing.predict_best_configuration(self._get_hyperparameter_space())
        logger.info(f"Predicted best configuration: {best_config}")
        return best_config
