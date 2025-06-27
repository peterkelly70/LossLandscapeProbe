#!/usr/bin/env python3
"""
Meta-Model Optimizer Module
==========================

Core optimizer implementation for hyperparameter optimization.
"""

import torch
import logging
import numpy as np
import random
import copy
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from .config import MetaModelConfig
from .training import train_one_epoch, evaluate_model, create_optimizer
from .meta_model import MetaModel, extract_meta_features, create_meta_model_datasets

logger = logging.getLogger(__name__)

class MetaModelOptimizer:
    """
    Base class for meta-model optimization.
    
    This class provides the core functionality for meta-model training and optimization.
    """
    
    def __init__(self, config: Optional[MetaModelConfig] = None, **kwargs):
        """
        Initialize the meta-model optimizer.
        
        Args:
            config: Configuration object with all parameters. If None, uses defaults.
            **kwargs: Alternative way to provide parameters that will override the config.
        """
        # Create config from defaults, update with provided config, then with kwargs
        self.config = MetaModelConfig()
        if config is not None:
            for key, value in config.__dict__.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Override with any kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Set random seeds for reproducibility
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.random_seed)
                torch.cuda.manual_seed_all(self.config.random_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        # Set device
        self.device = torch.device(self.config.device)
        logger.info(f"Using device: {self.device}")
        
        # Initialize data loaders
        self.train_loader = None
        self.val_loader = None
        
        # Store timing information
        self._timing_data = {
            'batch_times': [],
            'start_time': time.time(),
            'num_workers': 1
        }
        
        # Create run directory if not provided
        if self.config.run_dir is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.config.run_dir = Path(f"runs/meta_model_{timestamp}")
        elif isinstance(self.config.run_dir, str):
            self.config.run_dir = Path(self.config.run_dir)
            
        # Create directory if it doesn't exist
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run directory: {self.config.run_dir}")
    
    def optimize_hyperparameters(self, num_configs=10):
        """
        Optimize hyperparameters using meta-model approach.
        
        Args:
            num_configs: Number of configurations to evaluate
            
        Returns:
            Dictionary with optimization results
        """
        # This is a base implementation that should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement optimize_hyperparameters")
    
    def _create_meta_model(self, input_dim, output_dim):
        """
        Create a meta-model for hyperparameter prediction.
        
        Args:
            input_dim: Input dimension (meta-features)
            output_dim: Output dimension (hyperparameters)
            
        Returns:
            Meta-model instance
        """
        return MetaModel(input_dim=input_dim, output_dim=output_dim)
    
    def _train_meta_model(self, meta_model, features, targets, epochs=50):
        """
        Train the meta-model on collected data.
        
        Args:
            meta_model: Meta-model instance
            features: Input features tensor
            targets: Target values tensor
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training results
        """
        # Move model to device
        meta_model.to(self.device)
        
        # Create optimizer
        optimizer_config = {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        }
        optimizer = create_optimizer(meta_model, optimizer_config)
        
        # Create data loaders
        batch_size = min(32, features.size(0))
        dataset = torch.utils.data.TensorDataset(features, targets)
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        # Train for specified epochs
        best_loss = float('inf')
        best_model_state = None
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = train_one_epoch(
                model=meta_model,
                loader=self.train_loader,
                optimizer=optimizer,
                device=self.device
            )
            train_losses.append(train_loss)
            
            # Evaluate
            val_loss = evaluate_model(
                model=meta_model,
                loader=self.val_loader,
                device=self.device
            )
            val_losses.append(val_loss)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = copy.deepcopy(meta_model.state_dict())
        
        # Restore best model
        if best_model_state is not None:
            meta_model.load_state_dict(best_model_state)
        
        return {
            'meta_model': meta_model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_loss': best_loss
        }
