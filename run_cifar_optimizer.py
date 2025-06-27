#!/usr/bin/env python3
"""
Direct implementation of CIFAR meta-model optimizer with progress bars.
This script runs the optimizer without modifying the original code.
"""
import os
import sys
import copy
import logging
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the original CIFARMetaModelOptimizer
from llp.cifar_meta_model import CIFARMetaModelOptimizer

def convert_to_serializable(obj):
    """Convert PyTorch tensors to Python types for JSON serialization."""
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

class CIFARMetaModelOptimizerWithProgress(CIFARMetaModelOptimizer):
    """
    Extended version of CIFARMetaModelOptimizer with progress bars.
    """
    
    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """
        Train the meta-model using hyperparameter configurations and their evaluations.
        
        This method samples hyperparameter configurations, evaluates them on data subsets,
        extracts meta-features, and trains a meta-model to predict optimal hyperparameters.
        
        Returns:
            Dictionary containing the best hyperparameters found
        """
        # Try to import tqdm for progress bars
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            
        logger.info("Starting meta-model hyperparameter optimization...")
        print("🔍 Starting meta-model hyperparameter optimization...")
        
        # Sample hyperparameter configurations to evaluate
        # Use configs_per_sample attribute if available, otherwise default to 10
        num_configs = getattr(self.config, 'configs_per_sample', 
                         getattr(self.config, 'num_configs', 10))
        
        logger.info(f"Sampling {num_configs} hyperparameter configurations...")
        print(f"🔍 Sampling {num_configs} hyperparameter configurations...")
        configs = self._sample_hyperparameter_configs(
            num_configs=num_configs
            # Note: The method uses self.config.random_seed internally
        )
        
        # Initialize storage for evaluation results
        best_val_accuracy = 0.0
        best_config = None
        meta_features = []
        meta_targets = []
        
        # Evaluate each configuration on multiple data subsets
        total_configs = len(configs)
        total_subsets = len(self.data_subsets) if hasattr(self, 'data_subsets') else 1
        total_evaluations = total_configs * total_subsets
        evaluation_counter = 0
        
        # Store these values as instance variables for progress tracking
        self.total_evaluations = total_evaluations
        self.current_evaluation = 0
        
        print(f"⏳ Total configurations to evaluate: {total_configs}")
        print(f"⏳ Total data subsets: {total_subsets}")
        print(f"⏳ Total evaluations: {total_evaluations}")
        
        # Create progress bar for configurations
        config_pbar = None
        if use_tqdm:
            config_pbar = tqdm(total=total_configs, desc="Meta-Model Configs")
        
        for config_idx, config in enumerate(configs):
            logger.info(f"Evaluating configuration {config_idx+1}/{total_configs} [{(config_idx/total_configs)*100:.1f}% configurations]")
            print(f"\n🔄 Evaluating configuration {config_idx+1}/{total_configs} [{(config_idx/total_configs)*100:.1f}% configurations]")
            print(f"📊 Configuration: {convert_to_serializable(config)}")
            
            # Create progress bar for datasets
            dataset_pbar = None
            if use_tqdm:
                dataset_pbar = tqdm(total=total_subsets, desc="Dataset Progress")
            
            # Evaluate on multiple data subsets
            subset_results = []
            
            for subset_idx in range(total_subsets):
                # Update counter for overall progress tracking
                evaluation_counter += 1
                self.current_evaluation = evaluation_counter
                
                # Simple overall progress calculation
                overall_progress = (evaluation_counter / total_evaluations) * 100
                
                logger.info(f"Evaluating dataset {subset_idx+1}/{total_subsets} "
                          f"[Overall Progress: {evaluation_counter}/{total_evaluations} ({overall_progress:.1f}%)]")
                print(f"📈 Evaluating dataset {subset_idx+1}/{total_subsets} "
                      f"[Overall Progress: {evaluation_counter}/{total_evaluations} ({overall_progress:.1f}%)]")
                
                # Update dataset progress bar
                if use_tqdm and dataset_pbar is not None:
                    dataset_pbar.update(1)
                
                # Use reduced resource level for faster evaluation
                resource_level = getattr(self.config, 'min_resource', 0.2)
                
                # Evaluate this configuration on this subset
                try:
                    result = self._evaluate_configuration(
                        config=config,
                        subset_idx=subset_idx,
                        resource_level=resource_level
                    )
                    subset_results.append(result)
                    print(f"✅ Evaluation complete - Accuracy: {result.get('val_accuracy', 0.0):.4f}")
                except Exception as e:
                    logger.error(f"Error evaluating configuration on subset {subset_idx+1}: {str(e)}")
                    print(f"❌ Error evaluating configuration on subset {subset_idx+1}: {str(e)}")
                    # Add a placeholder result with error information
                    result = {
                        'error': str(e),
                        'val_accuracy': 0.0,
                        'config': config,
                        'subset_idx': subset_idx
                    }
                    subset_results.append(result)
            
            # Close dataset progress bar
            if use_tqdm and dataset_pbar is not None:
                dataset_pbar.close()
                
            # Log memory usage if using CUDA
            if self.device.type == 'cuda':
                logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # Update configuration progress bar
            if use_tqdm and config_pbar is not None:
                config_pbar.update(1)
            
            # Extract meta-features and targets from results
            for result in subset_results:
                if 'error' not in result and 'meta_features' in result:
                    meta_features.append(result['meta_features'])
                    meta_targets.append(result['val_accuracy'])
                    
                    # Track best configuration
                    if result['val_accuracy'] > best_val_accuracy:
                        best_val_accuracy = result['val_accuracy']
                        best_config = config
                        print(f"🏆 New best configuration found! Accuracy: {best_val_accuracy:.4f}")
        
        # Close configuration progress bar
        if use_tqdm and config_pbar is not None:
            config_pbar.close()
            
        # Store meta-features and targets for later use
        self.meta_features = meta_features
        self.meta_targets = meta_targets
        
        # Return early if no meta-features were collected
        if not meta_features:
            logger.warning("No meta-features collected, returning best configuration found")
            return {'val_accuracy': best_val_accuracy, 'config': convert_to_serializable(best_config or {})}
        
        # Train meta-model with the collected data
        # Use reduced resource level for faster evaluation
        resource_level = getattr(self.config, 'min_resource', 0.2)
        
        # Define number of epochs based on resource level
        max_epochs = getattr(self.config, 'max_epochs', 10)  # Default to 10 if not specified
        actual_epochs = int(max_epochs * resource_level)
        
        # Use a simple model for meta-model training
        input_dim = len(meta_features[0]) if meta_features else 10
        output_dim = 1  # For regression tasks like hyperparameter optimization
        
        # Create a simple MLP model
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim)
        ).to(self.device)
        
        # Prepare optimizer configuration
        optimizer_config = {
            'optimizer': 'adam',  # Default to Adam
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        }
        
        # Create dataset from meta-features and meta-targets
        meta_features_tensor = torch.tensor(meta_features, dtype=torch.float32)
        meta_targets_tensor = torch.tensor(meta_targets, dtype=torch.float32).view(-1, 1)
        meta_dataset = TensorDataset(meta_features_tensor, meta_targets_tensor)
        
        # Create data loaders
        batch_size = min(32, len(meta_dataset))
        train_loader = DataLoader(
            meta_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            meta_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config['learning_rate'], 
                                    weight_decay=optimizer_config['weight_decay'])
        
        # Train meta-model
        print(f"\n🧠 Training meta-model for {actual_epochs} epochs...")
        
        # Create progress bar for epochs
        epoch_pbar = None
        if use_tqdm:
            epoch_pbar = tqdm(total=actual_epochs, desc="Meta-Model Training")
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(actual_epochs):
            # Train one epoch
            model.train()
            train_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.mse_loss(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            # Evaluate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    val_loss += F.mse_loss(output, target).item()
            val_loss /= len(val_loader)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{actual_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Update progress bar
            if use_tqdm and epoch_pbar is not None:
                epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
                epoch_pbar.update(1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"📝 New best meta-model found! Validation loss: {best_val_loss:.4f}")
        
        # Close epoch progress bar
        if use_tqdm and epoch_pbar is not None:
            epoch_pbar.close()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        # Store meta-model
        self.meta_model = model
        
        # Return best configuration found
        best_config_found = best_config if best_config is not None else {}
        
        # Ensure we only return serializable values
        best_config_serializable = convert_to_serializable(best_config_found)
        
        # Add best accuracy to the output
        results = {
            'val_accuracy': best_val_accuracy,
            'config': best_config_serializable
        }
        
        return results

def main():
    """Run the CIFAR meta-model with progress bars."""
    print("CIFAR Meta-Model with Progress Bars")
    print("=" * 40)
    
    # Create an instance of the optimizer with progress bars
    print("Creating optimizer...")
    optimizer = CIFARMetaModelOptimizerWithProgress()
    
    # Run the hyperparameter optimization with progress bars
    print("Running hyperparameter optimization with progress bars...")
    result = optimizer.optimize_hyperparameters()
    
    # Print the result
    print("\nOptimization complete!")
    print(f"Best configuration found: {result}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
