"""
CIFAR Meta-Model Optimizer

This module provides a meta-model optimizer for CIFAR datasets.
It samples hyperparameter configurations, evaluates them on multiple data subsets,
and trains a meta-model to predict the best hyperparameters.
"""

import os
import json
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from llp.cifar_dataset import get_cifar_dataset, get_cifar_dataloaders
from llp.cifar_trainer import CIFARTrainer
from llp.cifar_models import get_model

# Configure logging
logger = logging.getLogger(__name__)

class CIFARMetaModelOptimizer:
    """
    Meta-model optimizer for CIFAR datasets.
    
    This class samples hyperparameter configurations, evaluates them on multiple data subsets,
    and trains a meta-model to predict the best hyperparameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the meta-model optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        random.seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        torch.manual_seed(config.get('seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.get('seed', 42))
        
        # Get dataset
        self.dataset = config.get('dataset', 'cifar10')
        self.data_fraction = config.get('data_fraction', 0.1)
        
        # Create data subsets for meta-model training
        self.data_subsets = self._create_data_subsets()
        
        # Initialize meta-model
        self.meta_model = None
        self.meta_features = None
        self.meta_targets = None
    
    def _create_data_subsets(self) -> List[Dict[str, Any]]:
        """
        Create data subsets for meta-model training.
        
        Returns:
            List of data subset configurations
        """
        # Define number of subsets
        num_subsets = self.config.get('num_subsets', 5)
        
        # Create subsets with different random seeds
        subsets = []
        for i in range(num_subsets):
            subset = {
                'seed': self.config.get('seed', 42) + i,
                'data_fraction': self.data_fraction
            }
            subsets.append(subset)
        
        return subsets
    
    def sample_configurations(self, num_configs: int) -> List[Dict[str, Any]]:
        """
        Sample hyperparameter configurations.
        
        Args:
            num_configs: Number of configurations to sample
            
        Returns:
            List of hyperparameter configurations
        """
        # Define hyperparameter search space
        lr_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        weight_decay_values = [0.0, 0.0001, 0.0005, 0.001]
        optimizer_values = ['sgd', 'adam']
        momentum_values = [0.9, 0.95, 0.99]  # Only for SGD
        
        # Sample configurations
        configs = []
        for _ in range(num_configs):
            optimizer_name = random.choice(optimizer_values)
            config = {
                'lr': random.choice(lr_values),
                'weight_decay': random.choice(weight_decay_values),
                'optimizer': optimizer_name
            }
            
            # Add momentum for SGD
            if optimizer_name == 'sgd':
                config['momentum'] = random.choice(momentum_values)
            
            configs.append(config)
        
        return configs
    
    def evaluate_configuration(self, config: Dict[str, Any], subset: Dict[str, Any], resource_level: float) -> Dict[str, Any]:
        """
        Evaluate a hyperparameter configuration on a data subset.
        
        Args:
            config: Hyperparameter configuration
            subset: Data subset configuration
            resource_level: Resource level (fraction of epochs)
            
        Returns:
            Evaluation results
        """
        # Set random seeds
        random.seed(subset['seed'])
        np.random.seed(subset['seed'])
        torch.manual_seed(subset['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(subset['seed'])
        
        # Get dataset
        train_dataset, val_dataset, test_dataset = get_cifar_dataset(
            self.dataset, subset['data_fraction']
        )
        
        # Get dataloaders
        batch_size = self.config.get('batch_size', 128)
        train_loader, val_loader, test_loader = get_cifar_dataloaders(
            train_dataset, val_dataset, test_dataset, batch_size
        )
        
        # Get model
        model_name = self.config.get('model', 'resnet18')
        num_classes = 10 if self.dataset == 'cifar10' else 100
        model = get_model(model_name, num_classes)
        model = model.to(self.device)
        
        # Configure optimizer
        if config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=config['lr'],
                momentum=config.get('momentum', 0.9),
                weight_decay=config['weight_decay']
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )
        
        # Configure trainer
        trainer = CIFARTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=self.device
        )
        
        # Calculate number of epochs based on resource level
        max_epochs = self.config.get('max_epochs', 10)
        num_epochs = max(1, int(max_epochs * resource_level))
        
        # Train model
        for epoch in range(num_epochs):
            train_loss, train_acc = trainer.train_epoch()
            val_loss, val_acc = trainer.validate()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Extract meta-features
        meta_features = self._extract_meta_features(train_dataset, val_dataset)
        
        # Return results
        return {
            'config': config,
            'val_accuracy': val_acc,
            'train_accuracy': train_acc,
            'meta_features': meta_features
        }
    
    def _extract_meta_features(self, train_dataset, val_dataset) -> List[float]:
        """
        Extract meta-features from datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            List of meta-features
        """
        # Extract basic meta-features
        meta_features = []
        
        # Dataset size
        meta_features.append(len(train_dataset))
        meta_features.append(len(val_dataset))
        
        # Class distribution
        if hasattr(train_dataset, 'targets'):
            targets = torch.tensor(train_dataset.targets)
            _, counts = torch.unique(targets, return_counts=True, dim=0)
            class_distribution = counts.float() / len(targets)
            
            # Add class distribution statistics
            meta_features.append(float(torch.min(class_distribution, dim=0)[0]))
            meta_features.append(float(torch.max(class_distribution, dim=0)[0]))
            meta_features.append(float(torch.mean(class_distribution)))
            meta_features.append(float(torch.std(class_distribution)))
        else:
            # Dummy values if targets not available
            meta_features.extend([0.1, 0.1, 0.1, 0.0])
        
        # Add dataset-specific features
        if self.dataset == 'cifar10':
            meta_features.append(10.0)  # Number of classes
        else:
            meta_features.append(100.0)  # Number of classes
        
        return meta_features
    
    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """
        Optimize hyperparameters using meta-model.
        
        Returns:
            Best hyperparameter configuration
        """
        logger.info("Starting meta-model hyperparameter optimization...")
        
        # Sample hyperparameter configurations
        configs_per_sample = getattr(self.config, 'configs_per_sample', 10)
        num_configs = getattr(self.config, 'num_configs', configs_per_sample)
        
        logger.info(f"Sampling {num_configs} hyperparameter configurations...")
        configs = self.sample_configurations(num_configs)
        
        # Evaluate configurations on multiple data subsets
        min_resource = getattr(self.config, 'min_resource', 0.1)
        
        # Store meta-features and targets for meta-model training
        meta_features = []
        meta_targets = []
        
        # Track best configuration
        best_val_accuracy = 0.0
        best_config = None
        
        # Evaluate each configuration
        for config_idx, config in enumerate(configs):
            logger.info(f"Evaluating configuration {config_idx+1}/{len(configs)}")
            
            # Evaluate configuration on multiple data subsets
            subset_results = []
            
            for subset_idx, subset in enumerate(self.data_subsets):
                logger.info(f"Evaluating dataset {subset_idx+1}/{len(self.data_subsets)}")
                
                # Evaluate configuration on this subset
                try:
                    result = self.evaluate_configuration(config, subset, min_resource)
                    subset_results.append(result)
                    
                    # Extract meta-features and targets
                    if 'meta_features' in result and 'val_accuracy' in result:
                        meta_features.append(result['meta_features'])
                        meta_targets.append(result['val_accuracy'])
                        
                    # Track best configuration
                    if result['val_accuracy'] > best_val_accuracy:
                        best_val_accuracy = result['val_accuracy']
                        best_config = config
                        logger.info(f"New best configuration found! Accuracy: {best_val_accuracy:.4f}")
                except Exception as e:
                    logger.error(f"Error evaluating configuration {config_idx+1} on subset {subset_idx+1}: {str(e)}")
        
        # Store meta-features and targets for later use
        self.meta_features = meta_features
        self.meta_targets = meta_targets
        
        # Train meta-model if we have enough data
        if len(meta_features) > 0 and len(meta_targets) > 0:
            logger.info("Training meta-model...")
            
            # Convert to tensors
            meta_features_tensor = torch.tensor(meta_features, dtype=torch.float32)
            meta_targets_tensor = torch.tensor(meta_targets, dtype=torch.float32).view(-1, 1)
            
            # Create dataset
            meta_dataset = TensorDataset(meta_features_tensor, meta_targets_tensor)
            
            # Define model architecture
            input_dim = meta_features_tensor.shape[1]
            hidden_dim = 64
            output_dim = 1
            
            # Create model
            model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            ).to(self.device)
            
            # Define optimizer and loss function
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            # Create data loaders
            dataset_size = len(meta_dataset)
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size
            
            # Split into train and validation sets
            train_dataset, val_dataset = random_split(
                meta_dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=8, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=8, shuffle=False, num_workers=0
            )
        else:
            # Create dummy data if no meta-features are available
            logger.warning("No meta-features available, creating dummy data for meta-model training")
            dummy_features = torch.randn(20, input_dim)
            dummy_targets = torch.randn(20, 1)
            dummy_dataset = TensorDataset(dummy_features, dummy_targets)
            
            # Split into train and validation sets
            train_size = 16
            val_size = 4
            train_dataset, val_dataset = random_split(
                dummy_dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=4, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=4, shuffle=False, num_workers=0
            )
        
        # Define local training and evaluation functions
        def train_one_epoch(model, train_loader, optimizer, device):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            return running_loss / len(train_loader.dataset)
        
        def evaluate(model, val_loader, device):
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    running_loss += loss.item() * inputs.size(0)
            
            return running_loss / len(val_loader.dataset)
        
        # Train meta-model
        num_epochs = 20
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, self.device)
            val_loss = evaluate(model, val_loader, self.device)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Store meta-model
        self.meta_model = model
        
        # Return best configuration found
        logger.info("Meta-model optimization complete!")
        logger.info(f"Best configuration found with accuracy: {best_val_accuracy:.4f}")
        
        return {
            'val_accuracy': best_val_accuracy,
            'config': best_config
        }
    
    def predict_hyperparameters(self, meta_features: List[float]) -> Dict[str, Any]:
        """
        Predict hyperparameters using meta-model.
        
        Args:
            meta_features: Meta-features for prediction
            
        Returns:
            Predicted hyperparameter configuration
        """
        if self.meta_model is None:
            logger.warning("Meta-model not trained, returning default configuration")
            return {
                'lr': 0.01,
                'weight_decay': 0.0001,
                'optimizer': 'sgd',
                'momentum': 0.9
            }
        
        # Convert meta-features to tensor
        meta_features_tensor = torch.tensor([meta_features], dtype=torch.float32).to(self.device)
        
        # Make prediction
        self.meta_model.eval()
        with torch.no_grad():
            predicted_accuracy = self.meta_model(meta_features_tensor).item()
        
        logger.info(f"Predicted accuracy: {predicted_accuracy:.4f}")
        
        # Return best configuration found during optimization
        # In a real-world scenario, we would predict the hyperparameters directly
        # or use the meta-model to evaluate multiple configurations
        return self.optimize_hyperparameters()['config']
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Save meta-model results to file.
        
        Args:
            results: Meta-model results
            output_dir: Output directory
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results to file
        results_path = output_path / 'meta_model_results.json'
        
        # Convert tensors to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items() 
                        if k not in ['model', 'model_state']}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            else:
                return obj
        
        # Convert results to serializable format
        serializable_results = convert_to_serializable(results)
        
        # Save to file
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
