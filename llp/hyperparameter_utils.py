"""
Hyperparameter Utilities
=======================

This module provides functionality for sampling and perturbing hyperparameter
configurations for neural network training.
"""

from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
import torch
import random
from dataclasses import dataclass, field
from enum import Enum, auto
import math


class HyperparameterType(Enum):
    """Types of hyperparameters supported for sampling and perturbation."""
    FLOAT = auto()
    INT = auto()
    CATEGORICAL = auto()
    BOOL = auto()
    LOG_FLOAT = auto()  # For sampling in log space


@dataclass
class HyperparameterRange:
    """Defines the range and sampling behavior for a hyperparameter."""
    name: str
    param_type: HyperparameterType
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    categories: Optional[List[Any]] = None
    default: Any = None
    scale: str = 'linear'  # 'linear' or 'log'
    description: str = ''  # Optional description of the parameter

    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        """Validate the hyperparameter range definition."""
        if self.param_type in [HyperparameterType.FLOAT, HyperparameterType.INT, HyperparameterType.LOG_FLOAT]:
            if self.min_val is None or self.max_val is None:
                raise ValueError(f"min_val and max_val must be specified for {self.param_type}")
            if self.min_val >= self.max_val:
                raise ValueError(f"min_val ({self.min_val}) must be less than max_val ({self.max_val})")
        elif self.param_type == HyperparameterType.CATEGORICAL:
            if not self.categories:
                raise ValueError("categories must be specified for CATEGORICAL type")


def get_default_hyperparameter_space() -> Dict[str, HyperparameterRange]:
    """
    Get the default hyperparameter search space for CIFAR training.
    
    Returns:
        Dictionary mapping parameter names to their ranges
    """
    return {
        'learning_rate': HyperparameterRange(
            name='learning_rate',
            param_type=HyperparameterType.LOG_FLOAT,
            min_val=1e-4,
            max_val=1e-1,
            default=0.1,
            scale='log'
        ),
        'momentum': HyperparameterRange(
            name='momentum',
            param_type=HyperparameterType.FLOAT,
            min_val=0.8,
            max_val=0.999,
            default=0.9
        ),
        'weight_decay': HyperparameterRange(
            name='weight_decay',
            param_type=HyperparameterType.LOG_FLOAT,
            min_val=1e-5,
            max_val=1e-2,
            default=5e-4,
            scale='log'
        ),
        'batch_size': HyperparameterRange(
            name='batch_size',
            param_type=HyperparameterType.INT,
            min_val=32,
            max_val=512,
            default=128
        ),
        # Using AdamW as the sole optimizer since it generally performs better than SGD for meta-learning
        # - More robust to learning rate choices
        # - Handles different parameter updates adaptively
        # - Performs well with default settings
        # - Better weight decay handling than regular Adam
        'optimizer': HyperparameterRange(
            name='optimizer',
            param_type=HyperparameterType.CATEGORICAL,
            categories=['adamw'],
            default='adamw',
            description='Optimizer to use (AdamW is used exclusively for its robustness and performance)'
        ),
        'beta1': HyperparameterRange(
            name='beta1',
            param_type=HyperparameterType.FLOAT,
            min_val=0.8,
            max_val=0.999,
            default=0.9,
            description='Beta1 parameter for AdamW (exponential decay rate for first moment estimates)'
        ),
        'beta2': HyperparameterRange(
            name='beta2',
            param_type=HyperparameterType.FLOAT,
            min_val=0.9,
            max_val=0.9999,
            default=0.999,
            description='Beta2 parameter for AdamW (exponential decay rate for second moment estimates)'
        ),
        'eps': HyperparameterRange(
            name='eps',
            param_type=HyperparameterType.LOG_FLOAT,
            min_val=1e-8,
            max_val=1e-6,
            default=1e-8,
            description='Epsilon parameter for AdamW (term added to denominator to improve numerical stability)'
        ),
    }


class HyperparameterSampler:
    """Handles sampling and perturbation of hyperparameter configurations."""
    
    def __init__(self, hyperparam_space: Optional[Dict[str, HyperparameterRange]] = None):
        """
        Initialize the hyperparameter sampler.
        
        Args:
            hyperparam_space: Dictionary defining the hyperparameter space.
                            If None, uses the default CIFAR space.
        """
        self.hyperparam_space = hyperparam_space or get_default_hyperparameter_space()
    
    def sample_configuration(self, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sample a random hyperparameter configuration.
        
        Args:
            base_config: Optional base configuration to start from.
                        If provided, only samples parameters not in the base config.
                        
        Returns:
            Dictionary containing the sampled hyperparameter values
        """
        config = {}
        
        # Start with base config if provided
        if base_config:
            config.update(base_config)
        
        # Sample remaining parameters
        for name, param_range in self.hyperparam_space.items():
            if name in config:
                continue
                
            if param_range.param_type == HyperparameterType.FLOAT:
                value = random.uniform(param_range.min_val, param_range.max_val)
            elif param_range.param_type == HyperparameterType.INT:
                value = random.randint(int(param_range.min_val), int(param_range.max_val))
            elif param_range.param_type == HyperparameterType.LOG_FLOAT:
                log_min = math.log10(param_range.min_val)
                log_max = math.log10(param_range.max_val)
                value = 10 ** random.uniform(log_min, log_max)
            elif param_range.param_type == HyperparameterType.CATEGORICAL:
                value = random.choice(param_range.categories)
            elif param_range.param_type == HyperparameterType.BOOL:
                value = random.choice([True, False])
            else:
                raise ValueError(f"Unsupported parameter type: {param_range.param_type}")
            
            config[name] = value
        
        return config
    
    def perturb_configuration(
        self, 
        base_config: Dict[str, Any],
        perturbation_scale: float = 0.1,
        fixed_params: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a perturbed version of a base configuration.
        
        Args:
            base_config: Base configuration to perturb
            perturbation_scale: Scale of the perturbation (0.0 to 1.0)
            fixed_params: List of parameter names to keep fixed
            
        Returns:
            New configuration with perturbed values
        """
        if fixed_params is None:
            fixed_params = []
            
        new_config = base_config.copy()
        
        for name, value in base_config.items():
            if name in fixed_params or name not in self.hyperparam_space:
                continue
                
            param_range = self.hyperparam_space[name]
            
            if param_range.param_type in [HyperparameterType.FLOAT, HyperparameterType.LOG_FLOAT]:
                # For float parameters, add Gaussian noise scaled by the parameter range
                if param_range.param_type == HyperparameterType.LOG_FLOAT:
                    # Convert to log space for perturbation
                    log_value = math.log10(value)
                    log_min = math.log10(param_range.min_val)
                    log_max = math.log10(param_range.max_val)
                    log_range = log_max - log_min
                    
                    # Add noise in log space
                    noise = random.gauss(0, perturbation_scale * log_range)
                    new_log_value = np.clip(log_value + noise, log_min, log_max)
                    new_value = 10 ** new_log_value
                else:
                    # Linear scale for regular float
                    value_range = param_range.max_val - param_range.min_val
                    noise = random.gauss(0, perturbation_scale * value_range)
                    new_value = np.clip(value + noise, param_range.min_val, param_range.max_val)
                
                new_config[name] = float(new_value)
                
            elif param_range.param_type == HyperparameterType.INT:
                # For integers, round to nearest integer after perturbation
                value_range = param_range.max_val - param_range.min_val
                noise = random.gauss(0, perturbation_scale * value_range)
                new_value = int(round(np.clip(value + noise, param_range.min_val, param_range.max_val)))
                new_config[name] = new_value
                
            elif param_range.param_type == HyperparameterType.CATEGORICAL:
                # For categorical, randomly choose a different value with some probability
                if random.random() < perturbation_scale and len(param_range.categories) > 1:
                    other_categories = [c for c in param_range.categories if c != value]
                    if other_categories:  # Ensure there are other categories to choose from
                        new_config[name] = random.choice(other_categories)
            
            # For BOOL, we could flip with some probability, but leaving fixed for now
            
        return new_config
    
    def generate_perturbations(
        self, 
        base_config: Dict[str, Any],
        num_perturbations: int = 5,
        perturbation_scale: float = 0.1,
        fixed_params: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple perturbed versions of a base configuration.
        
        Args:
            base_config: Base configuration to perturb
            num_perturbations: Number of perturbations to generate
            perturbation_scale: Scale of the perturbation (0.0 to 1.0)
            fixed_params: List of parameter names to keep fixed
            
        Returns:
            List of perturbed configurations
        """
        return [
            self.perturb_configuration(base_config, perturbation_scale, fixed_params)
            for _ in range(num_perturbations)
        ]


def sample_hyperparameter_configs(
    num_configs: int,
    hyperparam_space: Optional[Dict[str, HyperparameterRange]] = None,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Sample multiple random hyperparameter configurations.
    
    Args:
        num_configs: Number of configurations to sample
        hyperparam_space: Hyperparameter space definition
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled configurations
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    sampler = HyperparameterSampler(hyperparam_space)
    return [sampler.sample_configuration() for _ in range(num_configs)]


def generate_perturbed_configs(
    base_config: Dict[str, Any],
    num_perturbations: int = 5,
    perturbation_scale: float = 0.1,
    hyperparam_space: Optional[Dict[str, HyperparameterRange]] = None,
    fixed_params: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate perturbed versions of a base configuration.
    
    Args:
        base_config: Base configuration to perturb
        num_perturbations: Number of perturbations to generate
        perturbation_scale: Scale of the perturbation (0.0 to 1.0)
        hyperparam_space: Hyperparameter space definition
        fixed_params: List of parameter names to keep fixed
        seed: Random seed for reproducibility
        
    Returns:
        List of perturbed configurations
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    sampler = HyperparameterSampler(hyperparam_space)
    return sampler.generate_perturbations(
        base_config=base_config,
        num_perturbations=num_perturbations,
        perturbation_scale=perturbation_scale,
        fixed_params=fixed_params
    )
