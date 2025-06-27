#!/usr/bin/env python3
"""
CIFAR Meta-Model Trainer with Progress Bars

This script runs the actual CIFAR meta-model training with enhanced progress bars.
"""

import os
import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import importlib
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('meta_model_progress.log')
    ]
)
logger = logging.getLogger(__name__)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False
    logger.warning("tqdm not installed. Progress bars will be text-based only.")

def convert_to_serializable(obj):
    """Convert PyTorch tensors to Python types for JSON serialization."""
    import torch
    
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

def log_cuda_memory():
    """Log CUDA memory usage."""
    import torch
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"CUDA Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        print(f"🧠 CUDA Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def run_training():
    """Run the actual CIFAR meta-model training with progress bars."""
    # Add the project root to the path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Import necessary modules
    try:
        from train_with_meta import main as train_main
        import torch
        import sys
        
        # Add the fixed version to sys.path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import the fixed version of CIFARMetaModelOptimizer
        from llp.fixed_cifar_meta_model import CIFARMetaModelOptimizer
        
        # Save the original optimize_hyperparameters method
        original_optimize_hyperparameters = CIFARMetaModelOptimizer.optimize_hyperparameters
        
        # Define the enhanced optimize_hyperparameters method with progress bars
        def enhanced_optimize_hyperparameters(self) -> Dict[str, Any]:
            """
            Enhanced version of optimize_hyperparameters with progress bars.
            """
            logger.info("Starting meta-model hyperparameter optimization...")
            print("\n" + "="*80)
            print("🔍 STARTING META-MODEL HYPERPARAMETER OPTIMIZATION")
            print("="*80)
            
            # Log CUDA memory usage
            log_cuda_memory()
            
            # Sample hyperparameter configurations
            configs_per_sample = getattr(self.config, 'configs_per_sample', 10)
            num_configs = getattr(self.config, 'num_configs', configs_per_sample)
            
            logger.info(f"Sampling {num_configs} hyperparameter configurations...")
            print(f"🔍 Sampling {num_configs} hyperparameter configurations...")
            
            configs = self.sample_configurations(num_configs)
            
            # Create progress bar for configurations
            config_pbar = None
            if USE_TQDM:
                config_pbar = tqdm(total=len(configs), desc="Meta-Model Configs")
            
            # Evaluate configurations on multiple data subsets
            min_resource = getattr(self.config, 'min_resource', 0.1)
            
            # Print overall information
            print(f"\n📋 EVALUATION PLAN:")
            print(f"  ⏳ Total configurations to evaluate: {len(configs)}")
            print(f"  ⏳ Total data subsets: {len(self.data_subsets)}")
            print(f"  ⏳ Total evaluations: {len(configs) * len(self.data_subsets)}")
            print(f"  ⏳ Resource level: {min_resource}")
            print(f"="*80)
            
            # Store meta-features and targets for meta-model training
            meta_features = []
            meta_targets = []
            
            # Track best configuration
            best_val_accuracy = 0.0
            best_config = None
            
            # Evaluate each configuration
            for config_idx, config in enumerate(configs):
                logger.info(f"Evaluating configuration {config_idx+1}/{len(configs)} [{(config_idx/len(configs))*100:.1f}% configurations]")
                print(f"\n{'*'*80}")
                print(f"🔄 EVALUATING CONFIGURATION {config_idx+1}/{len(configs)} [{(config_idx/len(configs))*100:.1f}% COMPLETE]")
                print(f"{'*'*80}")
                print(f"📊 Configuration: {json.dumps(convert_to_serializable(config), indent=2)}")
                
                # Create progress bar for datasets
                dataset_pbar = None
                if USE_TQDM:
                    dataset_pbar = tqdm(total=len(self.data_subsets), desc="Dataset Progress")
                
                # Evaluate configuration on multiple data subsets
                subset_results = []
                
                for subset_idx, subset in enumerate(self.data_subsets):
                    # Calculate overall progress
                    evaluation_counter = config_idx * len(self.data_subsets) + subset_idx + 1
                    total_evaluations = len(configs) * len(self.data_subsets)
                    overall_progress = (evaluation_counter / total_evaluations) * 100
                    
                    logger.info(f"Evaluating dataset {subset_idx+1}/{len(self.data_subsets)} "
                              f"[Overall Progress: {evaluation_counter}/{total_evaluations} ({overall_progress:.1f}%)]")
                    print(f"\n📈 EVALUATING DATASET {subset_idx+1}/{len(self.data_subsets)}")
                    print(f"   [OVERALL PROGRESS: {evaluation_counter}/{total_evaluations} ({overall_progress:.1f}%)]")
                    
                    # Update dataset progress bar
                    if USE_TQDM and dataset_pbar is not None:
                        dataset_pbar.update(1)
                    
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
                            print(f"\n🏆 NEW BEST CONFIGURATION FOUND! ACCURACY: {best_val_accuracy:.4f}")
                        
                        print(f"   ✅ EVALUATION COMPLETE - ACCURACY: {result['val_accuracy']:.4f}")
                    except Exception as e:
                        logger.error(f"Error evaluating configuration {config_idx+1} on subset {subset_idx+1}: {str(e)}")
                        print(f"   ❌ ERROR: {str(e)}")
                
                # Close dataset progress bar
                if USE_TQDM and dataset_pbar is not None:
                    dataset_pbar.close()
                
                # Update configuration progress bar
                if USE_TQDM and config_pbar is not None:
                    config_pbar.update(1)
            
            # Close configuration progress bar
            if USE_TQDM and config_pbar is not None:
                config_pbar.close()
            
            # Store meta-features and targets for later use
            self.meta_features = meta_features
            self.meta_targets = meta_targets
            
            # Train meta-model if we have enough data
            if len(meta_features) > 0 and len(meta_targets) > 0:
                # Define number of epochs based on resource level
                max_epochs = getattr(self.config, 'max_epochs', 10)  # Default to 10 if not specified
                actual_epochs = int(max_epochs * min_resource)
                
                logger.info(f"Training meta-model for {actual_epochs} epochs...")
                print(f"\n{'='*80}")
                print(f"🧠 TRAINING META-MODEL FOR {actual_epochs} EPOCHS")
                print(f"{'='*80}")
                
                # Create progress bar for epochs
                epoch_pbar = None
                if USE_TQDM:
                    epoch_pbar = tqdm(total=actual_epochs, desc="Meta-Model Training")
                
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
                
                # Prepare optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
                
                # Prepare data loaders for meta-model training
                meta_features_tensor = torch.tensor(meta_features, dtype=torch.float32)
                meta_targets_tensor = torch.tensor(meta_targets, dtype=torch.float32).view(-1, 1)
                
                # Split data into train and validation sets (80/20)
                n_samples = len(meta_features_tensor)
                n_train = int(0.8 * n_samples)
                
                # Shuffle indices
                indices = torch.randperm(n_samples)
                train_indices = indices[:n_train]
                val_indices = indices[n_train:]
                
                # Create datasets
                train_dataset = torch.utils.data.TensorDataset(
                    meta_features_tensor[train_indices], 
                    meta_targets_tensor[train_indices]
                )
                val_dataset = torch.utils.data.TensorDataset(
                    meta_features_tensor[val_indices], 
                    meta_targets_tensor[val_indices]
                )
                
                # Create data loaders
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=32, shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=32, shuffle=False
                )
                
                # Train meta-model
                best_val_loss = float('inf')
                best_model_state = None
                
                for epoch in range(actual_epochs):
                    # Training
                    model.train()
                    train_loss = 0.0
                    for X, y in train_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(X)
                        loss = torch.nn.functional.mse_loss(outputs, y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item() * X.size(0)
                    
                    train_loss /= len(train_loader.dataset)
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for X, y in val_loader:
                            X, y = X.to(self.device), y.to(self.device)
                            outputs = model(X)
                            loss = torch.nn.functional.mse_loss(outputs, y)
                            val_loss += loss.item() * X.size(0)
                    
                    val_loss /= len(val_loader.dataset)
                    
                    # Log progress
                    logger.info(f"Epoch {epoch+1}/{actual_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                    print(f"\n📊 EPOCH {epoch+1}/{actual_epochs}:")
                    print(f"   TRAIN LOSS: {train_loss:.4f}")
                    print(f"   VAL LOSS: {val_loss:.4f}")
                    
                    # Update progress bar
                    if USE_TQDM and epoch_pbar is not None:
                        epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
                        epoch_pbar.update(1)
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = copy.deepcopy(model.state_dict())
                        logger.info(f"New best meta-model found! Validation loss: {best_val_loss:.4f}")
                        print(f"\n📝 NEW BEST META-MODEL FOUND! VALIDATION LOSS: {best_val_loss:.4f}")
                
                # Close epoch progress bar
                if USE_TQDM and epoch_pbar is not None:
                    epoch_pbar.close()
                
                # Load best model
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                
                # Store meta-model
                self.meta_model = model
            
            # Return best configuration found
            logger.info(f"Meta-model optimization complete!")
            print(f"\n{'='*80}")
            print(f"✅ META-MODEL OPTIMIZATION COMPLETE!")
            print(f"{'='*80}")
            logger.info(f"Best configuration found with accuracy: {best_val_accuracy:.4f}")
            print(f"🏆 BEST CONFIGURATION FOUND WITH ACCURACY: {best_val_accuracy:.4f}")
            print(f"{'='*80}")
            
            # Prepare result with basic Python types for JSON serialization
            result = {
                'val_accuracy': best_val_accuracy,
                'config': convert_to_serializable(best_config)
            }
            
            return result
        
        # Patch the optimize_hyperparameters method
        CIFARMetaModelOptimizer.optimize_hyperparameters = enhanced_optimize_hyperparameters
        
        # Run the training script
        print("\n" + "="*80)
        print("🚀 RUNNING CIFAR META-MODEL TRAINING WITH ENHANCED PROGRESS BARS")
        print("="*80)
        
        # Run the training
        train_main()
        
        # Restore the original method
        CIFARMetaModelOptimizer.optimize_hyperparameters = original_optimize_hyperparameters
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        
    except ImportError as e:
        logger.error(f"Error importing required modules: {str(e)}")
        print(f"❌ ERROR: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error running training: {str(e)}")
        print(f"❌ ERROR: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    run_training()
