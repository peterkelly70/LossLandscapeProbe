"""
Two-Tier Probing Module
======================

This module combines data sampling and parameter-space probing strategies
to efficiently identify promising hyperparameter configurations and model
regions that are likely to generalize well.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Callable, Union, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import time

from .data_sampling import ConfigEvaluation, SuccessiveHalving
from .parameter_probing import perturb_weights, SAM, StochasticWeightAveraging, measure_sharpness

logger = logging.getLogger(__name__)


@dataclass
class TwoTierEvaluation(ConfigEvaluation):
    """
    Extends ConfigEvaluation with parameter-space probing metrics.
    """
    sharpness: float = 0.0  # Measure of loss landscape sharpness
    perturbation_robustness: float = 0.0  # Robustness to weight perturbations
    generalization_score: float = 0.0  # Composite score for generalization


class TwoTierProbing:
    """
    Combines data sampling and parameter-space probing for efficient hyperparameter optimization.
    """
    
    def __init__(
        self,
        configs: List[Dict[str, Any]],
        model_fn: Callable[[Dict[str, Any]], nn.Module],
        dataset_fn: Callable[[float], Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
        criterion: nn.Module,
        optimizer_fn: Callable[[nn.Module, Dict[str, Any]], torch.optim.Optimizer],
        max_epochs: int = 100,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        alpha: float = 0.1  # Weight for sharpness in generalization score
    ):
        """
        Initialize the Two-Tier Probing framework.
        
        Args:
            configs: List of hyperparameter configurations to evaluate
            model_fn: Function that creates a model from a config
            dataset_fn: Function that returns (train_loader, val_loader) for a given data fraction
            criterion: Loss function
            optimizer_fn: Function that creates an optimizer from a model and config
            max_epochs: Maximum number of epochs for full training
            device: Device to run the model on
            alpha: Weight for sharpness in generalization score
        """
        self.configs = configs
        self.model_fn = model_fn
        self.dataset_fn = dataset_fn
        self.criterion = criterion
        self.optimizer_fn = optimizer_fn
        self.max_epochs = max_epochs
        self.device = device
        self.alpha = alpha
        
        self.results = []
    
    def train_and_evaluate(
        self,
        config: Dict[str, Any],
        resource: Union[int, float],
        measure_flatness: bool = True,
        noise_std: float = 0.01,
        num_perturbations: int = 5
    ) -> TwoTierEvaluation:
        """
        Train a model with the given config and resource level, and evaluate it.
        
        Args:
            config: Hyperparameter configuration
            resource: Resource level (epochs or data fraction)
            measure_flatness: Whether to measure loss landscape flatness
            noise_std: Standard deviation for weight perturbations
            num_perturbations: Number of perturbation samples
            
        Returns:
            Evaluation results
        """
        # Determine if resource is epochs or data fraction
        if resource <= 1.0:
            # Resource is data fraction
            data_fraction = resource
            epochs = self.max_epochs
        else:
            # Resource is epochs
            data_fraction = 1.0
            epochs = min(int(resource), self.max_epochs)
        
        # Get data loaders
        train_loader, val_loader = self.dataset_fn(data_fraction)
        
        # Create model and optimizer
        model = self.model_fn(config)
        model = model.to(self.device)
        optimizer = self.optimizer_fn(model, config)
        
        # Training
        start_time = time.time()
        model.train()
        
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        train_time = time.time() - start_time
        
        # Evaluation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        # Parameter-space probing (if requested)
        sharpness = 0.0
        perturbation_robustness = 0.0
        
        if measure_flatness:
            logger.info(f"Measuring loss landscape sharpness (noise_std={noise_std}, samples={num_perturbations})...")
            sharp_start = time.time()
            
            # Measure sharpness
            sharpness = measure_sharpness(
                model, val_loader, self.criterion,
                noise_std=noise_std,
                num_samples=num_perturbations,
                device=self.device
            )
            
            # Calculate perturbation robustness (1 / sharpness)
            perturbation_robustness = 1.0 / (sharpness + 1e-10)
            
            logger.info(f"Sharpness measurement completed in {time.time() - sharp_start:.2f}s")
            logger.info(f"Sharpness: {sharpness:.4f}, Perturbation Robustness: {perturbation_robustness:.4f}")
        
        # Calculate generalization score
        generalization_score = val_accuracy - self.alpha * sharpness
        
        # Create evaluation result
        result = TwoTierEvaluation(
            config_id=str(hash(str(config))),
            config=config,
            train_size=data_fraction,
            epochs=epochs,
            val_loss=val_loss,
            val_metric=val_accuracy,
            train_time=train_time,
            sharpness=sharpness,
            perturbation_robustness=perturbation_robustness,
            generalization_score=generalization_score
        )
        
        return result
    
    def run_successive_halving(
        self,
        min_resource: int = 1,
        max_resource: int = 81,
        reduction_factor: int = 3,
        measure_flatness: bool = True,
        progress_callback: Callable[[int, int, float], None] = None
    ) -> List[TwoTierEvaluation]:
        """
        Run Successive Halving with two-tier evaluation.
        
        Args:
            min_resource: Minimum resource to allocate
            max_resource: Maximum resource to allocate
            reduction_factor: Factor by which to reduce configs and increase resources
            measure_flatness: Whether to measure loss landscape flatness
            progress_callback: Optional callback function for reporting progress
                              Called with (stage, remaining_configs, current_resource)
            
        Returns:
            List of evaluation results
        """
        def train_fn(config, resource):
            return self.train_and_evaluate(config, resource, measure_flatness)
        
        # Create a wrapper for the SHA run method to track progress
        class ProgressTrackingSHA(SuccessiveHalving):
            def __init__(self, *args, **kwargs):
                self.progress_callback = kwargs.pop('progress_callback', None)
                super().__init__(*args, **kwargs)
                
            def run(self):
                configs = self.configs.copy()
                stage = 0
                resource = self.min_resource
                
                while len(configs) > 1:
                    # Call progress callback at the start of each stage
                    if self.progress_callback:
                        self.progress_callback(stage, len(configs), resource)
                        
                    # Run the stage as normal
                    logger.info(f"Stage {stage}: Running {len(configs)} configs with resource {resource}")
                    results = []
                    
                    for config in configs:
                        result = self.train_fn(config, resource)
                        results.append(result)
                    
                    # Sort by validation metric (higher is better)
                    results.sort(key=lambda x: x.val_metric, reverse=True)
                    
                    # Keep top 1/reduction_factor configs
                    k = max(1, int(len(configs) / self.reduction_factor))
                    results = results[:k]
                    configs = [r.config for r in results]
                    
                    # Increase resource for next stage
                    resource = min(resource * self.reduction_factor, self.max_resource)
                    stage += 1
                
                # Final stage - evaluate the remaining config with max resource
                if self.progress_callback:
                    self.progress_callback(stage, len(configs), resource)
                    
                logger.info(f"Final stage: Running {len(configs)} configs with resource {resource}")
                final_results = []
                
                for config in configs:
                    result = self.train_fn(config, resource)
                    final_results.append(result)
                
                return final_results
        
        # Use our progress tracking version of SHA
        sha = ProgressTrackingSHA(
            configs=self.configs,
            train_fn=train_fn,
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
            progress_callback=progress_callback
        )
        
        results = sha.run()
        self.results = results
        
        return results
    
    def select_best_configs(
        self,
        n: int = 1,
        criterion: str = 'generalization_score'
    ) -> List[Dict[str, Any]]:
        """
        Select the best n configurations based on the specified criterion.
        
        Args:
            n: Number of configurations to select
            criterion: Criterion to use for selection ('val_metric', 'sharpness', 'generalization_score')
            
        Returns:
            List of best configurations
        """
        if not self.results:
            raise ValueError("No results available. Run a probing method first.")
        
        if criterion == 'val_metric':
            sorted_results = sorted(self.results, key=lambda x: x.val_metric, reverse=True)
        elif criterion == 'sharpness':
            sorted_results = sorted(self.results, key=lambda x: x.sharpness)
        elif criterion == 'generalization_score':
            sorted_results = sorted(self.results, key=lambda x: x.generalization_score, reverse=True)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return [result.config for result in sorted_results[:n]]
    
    def train_with_sam(
        self,
        config: Dict[str, Any],
        rho: float = 0.05,
        epochs: int = 100,
        progress_callback: Callable[[int, float, float, float], None] = None
    ) -> TwoTierEvaluation:
        """
        Train a model with Sharpness-Aware Minimization (SAM).
        
        Args:
            config: Hyperparameter configuration
            rho: Size of the neighborhood for perturbation
            epochs: Number of epochs to train
            progress_callback: Optional callback function for reporting progress
                              Called with (epoch, loss, accuracy, elapsed_time)
            
        Returns:
            Evaluation results
        """
        # Get data loaders
        train_loader, val_loader = self.dataset_fn(1.0)
        
        # Create model
        model = self.model_fn(config)
        model = model.to(self.device)
        
        # Create base optimizer
        base_optimizer = lambda params, **kwargs: self.optimizer_fn(params, config)
        
        # Create SAM optimizer
        optimizer = SAM(model.parameters(), base_optimizer, rho=rho)
        
        # Training
        start_time = time.time()
        model.train()
        logger.info(f"Starting SAM training with rho={rho}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            batches = 0
            
            # Training loop
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    return loss
                
                # SAM optimizer step
                loss = optimizer.step(closure)
                running_loss += loss.item()
                batches += 1
            
            # Calculate average loss for the epoch
            avg_loss = running_loss / batches if batches > 0 else 0.0
            
            # Quick evaluation for progress reporting
            if progress_callback is not None:
                # Do a quick evaluation on validation set
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    # Use a subset of validation data for quick feedback
                    for i, (inputs, targets) in enumerate(val_loader):
                        if i >= 5:  # Limit to 5 batches for speed
                            break
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                
                # Calculate current accuracy
                current_accuracy = correct / total if total > 0 else 0.0
                
                # Calculate elapsed time
                epoch_time = time.time() - epoch_start
                total_elapsed = time.time() - start_time
                
                # Call progress callback
                progress_callback(epoch, avg_loss, current_accuracy, total_elapsed)
                
                # Switch back to training mode
                model.train()
        
        train_time = time.time() - start_time
        
        # Evaluation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        # Measure sharpness
        sharpness = measure_sharpness(
            model, val_loader, self.criterion,
            device=self.device
        )
        
        # Calculate perturbation robustness (1 / sharpness)
        perturbation_robustness = 1.0 / (sharpness + 1e-10)
        
        # Calculate generalization score
        generalization_score = val_accuracy - self.alpha * sharpness
        
        # Create evaluation result
        result = TwoTierEvaluation(
            config_id=str(hash(str(config))),
            config=config,
            train_size=1.0,
            epochs=epochs,
            val_loss=val_loss,
            val_metric=val_accuracy,
            train_time=train_time,
            sharpness=sharpness,
            perturbation_robustness=perturbation_robustness,
            generalization_score=generalization_score,
            model_state=model.state_dict()
        )
        
        return result
    
    def train_with_swa(
        self,
        config: Dict[str, Any],
        epochs: int = 100,
        swa_start: int = 75,
        swa_freq: int = 1,
        progress_callback: Callable[[int, float, float, float, bool], None] = None
    ) -> Tuple[TwoTierEvaluation, TwoTierEvaluation]:
        """
        Train a model with Stochastic Weight Averaging (SWA).
        
        Args:
            config: Hyperparameter configuration
            epochs: Number of epochs to train
            swa_start: Epoch to start SWA from
            swa_freq: Frequency of model averaging
            progress_callback: Optional callback function for reporting progress
                              Called with (epoch, loss, accuracy, elapsed_time, is_swa_active)
            
        Returns:
            Tuple of (regular_eval, swa_eval)
        """
        # Get data loaders
        train_loader, val_loader = self.dataset_fn(1.0)
        
        # Create model and optimizer
        model = self.model_fn(config)
        model = model.to(self.device)
        optimizer = self.optimizer_fn(model, config)
        
        # Create SWA object
        swa = StochasticWeightAveraging(model, swa_start=swa_start, swa_freq=swa_freq)
        
        # Training
        start_time = time.time()
        model.train()
        logger.info(f"Starting SWA training with swa_start={swa_start}, swa_freq={swa_freq}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            batches = 0
            
            # Training loop
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batches += 1
            
            # Calculate average loss for the epoch
            avg_loss = running_loss / batches if batches > 0 else 0.0
            
            # Update SWA model
            is_swa_active = swa.update(epoch)
            
            # Quick evaluation for progress reporting
            if progress_callback is not None:
                # Do a quick evaluation on validation set
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    # Use a subset of validation data for quick feedback
                    for i, (inputs, targets) in enumerate(val_loader):
                        if i >= 5:  # Limit to 5 batches for speed
                            break
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                
                # Calculate current accuracy
                current_accuracy = correct / total if total > 0 else 0.0
                
                # Calculate elapsed time
                epoch_time = time.time() - epoch_start
                total_elapsed = time.time() - start_time
                
                # Call progress callback
                progress_callback(epoch, avg_loss, current_accuracy, total_elapsed, is_swa_active)
                
                # Switch back to training mode
                model.train()
        
        train_time = time.time() - start_time
        
        # Finalize SWA model
        swa.finalize()
        swa_model = swa.get_swa_model()
        swa_model = swa_model.to(self.device)
        
        # Evaluate regular model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        # Measure sharpness for regular model
        sharpness = measure_sharpness(
            model, val_loader, self.criterion,
            device=self.device
        )
        
        # Calculate perturbation robustness (1 / sharpness)
        perturbation_robustness = 1.0 / (sharpness + 1e-10)
        
        # Calculate generalization score
        generalization_score = val_accuracy - self.alpha * sharpness
        
        # Create evaluation result for regular model
        regular_eval = TwoTierEvaluation(
            config_id=str(hash(str(config))),
            config=config,
            train_size=1.0,
            epochs=epochs,
            val_loss=val_loss,
            val_metric=val_accuracy,
            train_time=train_time,
            sharpness=sharpness,
            perturbation_robustness=perturbation_robustness,
            generalization_score=generalization_score,
            model_state=model.state_dict()
        )
        
        # Evaluate SWA model
        swa_model.eval()
        swa_val_loss = 0.0
        swa_correct = 0
        swa_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = swa_model(inputs)
                loss = self.criterion(outputs, targets)
                swa_val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                swa_total += targets.size(0)
                swa_correct += predicted.eq(targets).sum().item()
        
        swa_val_loss /= len(val_loader)
        swa_val_accuracy = swa_correct / swa_total
        
        # Measure sharpness for SWA model
        swa_sharpness = measure_sharpness(
            swa_model, val_loader, self.criterion,
            device=self.device
        )
        
        # Calculate perturbation robustness (1 / sharpness)
        swa_perturbation_robustness = 1.0 / (swa_sharpness + 1e-10)
        
        # Calculate generalization score
        swa_generalization_score = swa_val_accuracy - self.alpha * swa_sharpness
        
        # Create evaluation result for SWA model
        swa_eval = TwoTierEvaluation(
            config_id=str(hash(str(config))) + "_swa",
            config=config,
            train_size=1.0,
            epochs=epochs,
            val_loss=swa_val_loss,
            val_metric=swa_val_accuracy,
            train_time=train_time,
            sharpness=swa_sharpness,
            perturbation_robustness=swa_perturbation_robustness,
            generalization_score=swa_generalization_score,
            model_state=swa_model.state_dict()
        )
        
        return regular_eval, swa_eval
