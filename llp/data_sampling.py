"""
Data Sampling Probes Module
==========================

This module implements various data sampling strategies for efficient hyperparameter optimization:
- Successive Halving (SHA)
- Hyperband
- Learning curve extrapolation
- Training-free proxies

These methods train models on subsets of data or for limited iterations to quickly
identify promising hyperparameter configurations.
"""

import numpy as np
import torch
from typing import Dict, List, Callable, Union, Tuple, Optional, Any
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfigEvaluation:
    """Class for storing evaluation results of a hyperparameter configuration."""
    config_id: str
    config: Dict[str, Any]
    train_size: Union[int, float]  # Size of training data used
    epochs: int  # Number of epochs trained
    val_loss: float
    val_metric: float  # Accuracy, F1, etc.
    train_time: float  # Time taken for training
    model_state: Optional[Dict] = None  # Optional saved model state


class SuccessiveHalving:
    """
    Implementation of Successive Halving Algorithm (SHA) for hyperparameter optimization.
    
    SHA allocates a small training budget to many hyperparameter configs, then 
    progressively focuses resources on the better performers.
    """
    
    def __init__(
        self,
        configs: List[Dict[str, Any]],
        train_fn: Callable,
        min_resource: int = 1,
        max_resource: int = 81,
        reduction_factor: int = 3,
        min_configs: int = 1
    ):
        """
        Initialize the Successive Halving algorithm.
        
        Args:
            configs: List of hyperparameter configurations to evaluate
            train_fn: Function that trains a model with given config and resources
                     Should have signature: train_fn(config, resource) -> ConfigEvaluation
            min_resource: Minimum resource (epochs/data fraction) to allocate
            max_resource: Maximum resource to allocate
            reduction_factor: Factor by which to reduce configs and increase resources
            min_configs: Minimum number of configurations to keep
        """
        self.configs = configs
        self.train_fn = train_fn
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.min_configs = min_configs
        self.results = []
        
    def run(self) -> List[ConfigEvaluation]:
        """
        Run the successive halving algorithm.
        
        Returns:
            List of configuration evaluations, sorted by performance
        """
        active_configs = self.configs.copy()
        current_resource = self.min_resource
        iteration = 0
        
        while len(active_configs) > self.min_configs and current_resource <= self.max_resource:
            logger.info(f"SHA Iteration {iteration}: {len(active_configs)} configs with resource {current_resource}")
            
            # Evaluate all active configurations with the current resource level
            evaluations = []
            for config in active_configs:
                eval_result = self.train_fn(config, current_resource)
                evaluations.append(eval_result)
                self.results.append(eval_result)
            
            # Sort by validation metric (assuming higher is better)
            evaluations.sort(key=lambda x: x.val_metric, reverse=True)
            
            # Keep the top 1/reduction_factor configurations
            n_keep = max(self.min_configs, len(active_configs) // self.reduction_factor)
            active_configs = [eval_result.config for eval_result in evaluations[:n_keep]]
            
            # Increase the resource for the next iteration
            current_resource *= self.reduction_factor
            iteration += 1
        
        # Final evaluation with max resources for the remaining configs
        final_evaluations = []
        for config in active_configs:
            eval_result = self.train_fn(config, self.max_resource)
            final_evaluations.append(eval_result)
            self.results.append(eval_result)
        
        # Return all results, sorted by performance
        self.results.sort(key=lambda x: x.val_metric, reverse=True)
        return self.results


class Hyperband:
    """
    Implementation of Hyperband algorithm for hyperparameter optimization.
    
    Hyperband runs multiple SHA brackets with different initial budgets to achieve
    a near-optimal trade-off between exploring many configs and exploiting promising ones.
    """
    
    def __init__(
        self,
        config_sampler: Callable[[int], List[Dict[str, Any]]],
        train_fn: Callable,
        min_resource: int = 1,
        max_resource: int = 81,
        reduction_factor: int = 3
    ):
        """
        Initialize the Hyperband algorithm.
        
        Args:
            config_sampler: Function that samples n configurations
            train_fn: Function that trains a model with given config and resources
            min_resource: Minimum resource (epochs/data fraction) to allocate
            max_resource: Maximum resource to allocate
            reduction_factor: Factor by which to reduce configs and increase resources
        """
        self.config_sampler = config_sampler
        self.train_fn = train_fn
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.results = []
        
        # Calculate the number of brackets
        self.s_max = int(np.log(max_resource / min_resource) / np.log(reduction_factor))
        self.B = (self.s_max + 1) * max_resource
        
    def run(self) -> List[ConfigEvaluation]:
        """
        Run the Hyperband algorithm.
        
        Returns:
            List of configuration evaluations, sorted by performance
        """
        for s in reversed(range(self.s_max + 1)):
            # Number of configurations
            n = int(self.B / self.max_resource * self.reduction_factor**(s) / (s + 1))
            
            # Initial resource per configuration
            r = self.max_resource * self.reduction_factor**(-s)
            
            logger.info(f"Hyperband bracket s={s}: n={n}, r={r}")
            
            # Sample n configurations
            configs = self.config_sampler(n)
            
            # Run successive halving on these configurations
            sha = SuccessiveHalving(
                configs=configs,
                train_fn=self.train_fn,
                min_resource=r,
                max_resource=self.max_resource,
                reduction_factor=self.reduction_factor
            )
            bracket_results = sha.run()
            self.results.extend(bracket_results)
        
        # Return all results, sorted by performance
        self.results.sort(key=lambda x: x.val_metric, reverse=True)
        return self.results


def learning_curve_extrapolation(
    config: Dict[str, Any],
    train_fn: Callable,
    resources: List[int],
    target_resource: int,
    fit_fn: Optional[Callable] = None
) -> Tuple[float, float]:
    """
    Extrapolate learning curve from partial training to predict final performance.
    
    Args:
        config: Hyperparameter configuration
        train_fn: Function to train model with given resources
        resources: List of resource levels to use for extrapolation
        target_resource: Target resource level to predict for
        fit_fn: Function to fit learning curve (defaults to power law)
        
    Returns:
        Tuple of (predicted_val_loss, predicted_val_metric)
    """
    # Default fit function (power law: y = a * x^b + c)
    if fit_fn is None:
        def power_law(x, a, b, c):
            return a * np.power(x, b) + c
        fit_fn = power_law
    
    # Train at each resource level
    evaluations = []
    for r in resources:
        eval_result = train_fn(config, r)
        evaluations.append(eval_result)
    
    # Extract data for curve fitting
    x = np.array([e.epochs for e in evaluations])
    y_loss = np.array([e.val_loss for e in evaluations])
    y_metric = np.array([e.val_metric for e in evaluations])
    
    # Fit curves and extrapolate
    from scipy.optimize import curve_fit
    
    try:
        # Fit loss curve
        params_loss, _ = curve_fit(fit_fn, x, y_loss)
        predicted_loss = fit_fn(target_resource, *params_loss)
        
        # Fit metric curve
        params_metric, _ = curve_fit(fit_fn, x, y_metric)
        predicted_metric = fit_fn(target_resource, *params_metric)
        
        return predicted_loss, predicted_metric
    
    except Exception as e:
        logger.warning(f"Curve fitting failed: {e}")
        # Return the best observed values as fallback
        best_idx = np.argmin(y_loss)
        return y_loss[best_idx], y_metric[best_idx]


class TrainingFreeProxy:
    """
    Implements training-free proxies to evaluate neural network architectures
    without full training.
    
    Examples include:
    - Gradient norms at initialization
    - Jacobian spectrum
    - Activation diversity
    - NEAR (Network Expressivity by Activation Rank)
    """
    
    @staticmethod
    def compute_gradient_norm(model: torch.nn.Module, data_batch, criterion) -> float:
        """
        Compute the norm of gradients at initialization.
        
        Args:
            model: PyTorch model
            data_batch: Batch of data (inputs, targets)
            criterion: Loss function
            
        Returns:
            Gradient norm
        """
        inputs, targets = data_batch
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        return total_norm
    
    @staticmethod
    def compute_activation_diversity(model: torch.nn.Module, data_batch) -> float:
        """
        Compute activation diversity score.
        
        Args:
            model: PyTorch model
            data_batch: Batch of data (inputs)
            
        Returns:
            Activation diversity score
        """
        # This is a simplified implementation
        # In practice, you would hook into each layer and compute diversity metrics
        inputs = data_batch[0]
        
        # Register hooks to capture activations
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach())
        
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.ReLU, torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            model(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute diversity (simplified)
        diversity_score = 0.0
        for act in activations:
            # Flatten activations
            flat_act = act.view(act.size(0), -1)
            # Compute variance across batch dimension
            var = torch.var(flat_act, dim=0).mean().item()
            diversity_score += var
        
        return diversity_score / len(activations) if activations else 0.0
