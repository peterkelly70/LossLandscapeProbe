"""
Meta-Model Probing Module
========================

This module extends the Two-Tier Probing framework with meta-learning capabilities
to predict optimal hyperparameters based on dataset characteristics and partial training results.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Callable, Union, Tuple, Optional, Any
import logging
import time
from pathlib import Path
import os

from .two_tier_probing import TwoTierProbing, TwoTierEvaluation
from .parameter_probing import perturb_weights, SAM, StochasticWeightAveraging, measure_sharpness as measure_sharpness
from .meta_model import HyperparameterPredictor, DatasetFeatureExtractor, TrainingResultFeatureExtractor, MetaModelTrainer

logger = logging.getLogger(__name__)


class MetaProbing(TwoTierProbing):
    """
    Extends TwoTierProbing with meta-learning capabilities to predict optimal hyperparameters.
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
        alpha: float = 0.1,  # Weight for sharpness in generalization score
        meta_model_dir: Optional[str] = None
    ):
        """
        Initialize the Meta-Probing framework.
        
        Args:
            configs: List of hyperparameter configurations to evaluate
            model_fn: Function that creates a model from a config
            dataset_fn: Function that returns (train_loader, val_loader) for a given data fraction
            criterion: Loss function
            optimizer_fn: Function that creates an optimizer from a model and config
            max_epochs: Maximum number of epochs for full training
            device: Device to run the model on
            alpha: Weight for sharpness in generalization score
            meta_model_dir: Directory to save/load meta-model files
        """
        super().__init__(
            configs=configs,
            model_fn=model_fn,
            dataset_fn=dataset_fn,
            criterion=criterion,
            optimizer_fn=optimizer_fn,
            max_epochs=max_epochs,
            device=device,
            alpha=alpha
        )
        
        # Initialize meta-model components
        self.meta_model_dir = meta_model_dir or os.path.join(os.path.dirname(__file__), '../models/meta')
        self.predictor = HyperparameterPredictor(model_dir=self.meta_model_dir)
        self.meta_trainer = MetaModelTrainer(predictor=self.predictor)
        
        # Track training history for meta-model
        self.training_histories = {}
        
        logger.info(f"Initialized MetaProbing with meta-model directory: {self.meta_model_dir}")
    
    def train_and_evaluate(
        self,
        config: Dict[str, Any],
        resource: Union[int, float],
        measure_flatness: bool = True,
        noise_std: float = 0.01,
        num_perturbations: int = 5
    ) -> TwoTierEvaluation:
        """
        Train a model with the given config and resource level, evaluate it,
        and collect data for the meta-model.
        
        Args:
            config: Hyperparameter configuration
            resource: Resource level (epochs or data fraction)
            measure_flatness: Whether to measure loss landscape flatness
            noise_std: Standard deviation for weight perturbations
            num_perturbations: Number of perturbation samples
            
        Returns:
            Evaluation results
        """
        # Initialize training history for this config
        config_id = str(hash(str(config)))
        if config_id not in self.training_histories:
            self.training_histories[config_id] = []
        
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
        
        # Track metrics for each epoch
        epoch_metrics = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            batches = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batches += 1
            
            # Calculate average loss for this epoch
            avg_loss = running_loss / batches if batches > 0 else 0.0
            
            # Quick validation on a subset of validation data
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                # Use at most 10 batches for quick validation
                for i, (inputs, targets) in enumerate(val_loader):
                    if i >= 10:
                        break
                    
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_loss = val_loss / min(10, len(val_loader)) if len(val_loader) > 0 else 0.0
            val_accuracy = correct / total if total > 0 else 0.0
            
            # Record metrics for this epoch
            epoch_metrics.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'resource_level': data_fraction,
                'elapsed_time': time.time() - epoch_start
            })
            
            # Switch back to training mode
            model.train()
        
        train_time = time.time() - start_time
        
        # Full validation
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
        
        # Measure sharpness if requested
        sharpness = 0.0
        perturbation_robustness = 0.0
        
        if measure_flatness:
            logger.info(f"Measuring loss landscape sharpness (noise_std={noise_std}, samples={num_perturbations})...")
            sharpness_start = time.time()
            
            # Call the measure_sharpness function from parameter_probing
            sharpness = measure_sharpness(
                model, val_loader, self.criterion,
                noise_std=noise_std,
                num_samples=num_perturbations,
                device=self.device
            )
            
            # Calculate perturbation robustness (1 / sharpness)
            perturbation_robustness = 1.0 / (sharpness + 1e-10)
            
            logger.info(f"Sharpness: {sharpness:.4f}, Perturbation Robustness: {perturbation_robustness:.4f}")
            logger.info(f"Sharpness measurement took {time.time() - sharpness_start:.2f}s")
        
        # Calculate generalization score
        generalization_score = val_accuracy - self.alpha * sharpness
        
        # Create evaluation result
        eval_result = TwoTierEvaluation(
            config_id=config_id,
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
        
        # Add training history for meta-model
        self.training_histories[config_id].extend(epoch_metrics)
        
        # Process evaluation result for meta-model
        self.meta_trainer.process_evaluation_result(
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            training_history=epoch_metrics,
            sharpness=sharpness,
            performance=generalization_score
        )
        
        return eval_result
    
    def train_meta_model(self):
        """
        Train the meta-model on all collected examples.
        """
        logger.info("Training meta-model from collected examples...")
        self.meta_trainer.train_meta_model()
        logger.info("Meta-model training complete")
    
    def predict_optimal_hyperparameters(self) -> Dict[str, Any]:
        """
        Predict optimal hyperparameters for the current dataset.
        
        Returns:
            Dictionary of predicted optimal hyperparameters
        """
        logger.info("Predicting optimal hyperparameters for the current dataset...")
        
        # Get data loaders for the full dataset
        train_loader, val_loader = self.dataset_fn(1.0)
        
        # Make prediction
        predicted_config = self.meta_trainer.predict_hyperparameters(train_loader, val_loader)
        
        logger.info(f"Predicted optimal hyperparameters: {predicted_config}")
        return predicted_config
    
    def run_meta_optimization(self, 
                             min_resource: float = 0.1, 
                             max_resource: float = 0.5,
                             reduction_factor: int = 2,
                             measure_flatness: bool = True,
                             progress_callback: Callable[[int, int, float], None] = None,
                             num_initial_configs: int = 6,
                             num_iterations: int = 3):
        """
        Run meta-model-guided hyperparameter optimization.
        
        This method:
        1. Evaluates initial configurations on small subsets of data
        2. Trains a meta-model to predict optimal hyperparameters
        3. Generates and evaluates new configurations based on predictions
        4. Repeats the process for multiple iterations
        
        Args:
            min_resource: Minimum resource level (data fraction)
            max_resource: Maximum resource level (data fraction)
            reduction_factor: Factor by which to reduce configs/increase resources
            measure_flatness: Whether to measure loss landscape flatness
            progress_callback: Callback for reporting progress
            num_initial_configs: Number of initial configurations to evaluate
            num_iterations: Number of meta-model training iterations
            
        Returns:
            Best configuration found
        """
        logger.info(f"Running meta-model-guided hyperparameter optimization...")
        logger.info(f"Starting with {len(self.configs)} configurations")
        logger.info(f"Resource range: {min_resource:.1f} to {max_resource:.1f} of data")
        logger.info(f"Reduction factor: {reduction_factor}")
        
        # Track overall progress
        total_configs = len(self.configs)
        start_time = time.time()
        
        # Initial configurations to evaluate
        current_configs = self.configs[:num_initial_configs]
        current_resource = min_resource
        
        # Track all evaluated configurations
        all_evaluations = []
        
        # Run meta-optimization iterations
        for iteration in range(num_iterations):
            logger.info(f"Meta-optimization iteration {iteration+1}/{num_iterations}")
            
            # Evaluate current configurations
            stage_start = time.time()
            stage_evaluations = []
            
            if progress_callback:
                progress_callback(
                    iteration, 
                    len(current_configs), 
                    current_resource
                )
            
            # Evaluate each configuration
            for i, config in enumerate(current_configs):
                logger.info(f"Evaluating configuration {i+1}/{len(current_configs)} at resource level {current_resource:.2f}")
                eval_result = self.train_and_evaluate(
                    config=config,
                    resource=current_resource,
                    measure_flatness=measure_flatness
                )
                stage_evaluations.append(eval_result)
                all_evaluations.append(eval_result)
            
            # Report stage completion
            stage_time = time.time() - stage_start
            logger.info(f"Completed evaluation of {len(current_configs)} configurations in {stage_time:.2f}s")
            
            # Train meta-model on all evaluations so far
            self.train_meta_model()
            
            # Increase resource level for next iteration
            current_resource = min(current_resource * reduction_factor, max_resource)
            
            # Generate new configurations using meta-model predictions
            if iteration < num_iterations - 1:  # Skip for the last iteration
                predicted_config = self.predict_optimal_hyperparameters()
                
                # Create variations of the predicted config
                new_configs = [predicted_config]
                
                # Add some variations with small perturbations
                for _ in range(num_initial_configs - 1):
                    variation = self._create_config_variation(predicted_config)
                    new_configs.append(variation)
                
                current_configs = new_configs
                logger.info(f"Generated {len(current_configs)} new configurations for next iteration")
        
        # Select best configuration based on generalization score
        all_evaluations.sort(key=lambda x: x.generalization_score, reverse=True)
        best_config = all_evaluations[0].config
        best_score = all_evaluations[0].generalization_score
        
        total_time = time.time() - start_time
        logger.info(f"Meta-optimization complete in {total_time:.2f}s")
        logger.info(f"Best configuration found with generalization score: {best_score:.4f}")
        
        return best_config
    
    def _create_config_variation(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a variation of a configuration by applying small perturbations.
        
        Args:
            base_config: Base configuration to vary
            
        Returns:
            New configuration with variations
        """
        variation = base_config.copy()
        
        for key, value in variation.items():
            if isinstance(value, (int, float)):
                # Apply random perturbation
                if isinstance(value, int):
                    # For integers, perturb by ±1 or ±2
                    variation[key] = value + np.random.choice([-2, -1, 1, 2])
                    # Ensure positive values for certain parameters
                    if key in ['batch_size', 'hidden_size', 'num_layers'] and variation[key] <= 0:
                        variation[key] = 1
                else:
                    # For floats, perturb by ±10%
                    perturbation = value * (1 + np.random.uniform(-0.1, 0.1))
                    variation[key] = perturbation
                    # Ensure positive values for certain parameters
                    if key in ['learning_rate', 'weight_decay'] and variation[key] <= 0:
                        variation[key] = value * 0.1
        
        return variation
