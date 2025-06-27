#!/usr/bin/env python3
"""
Simple Meta-Model Trainer with Clear Progress Bars

This script provides a simplified implementation of the meta-model training process
with clear, visible progress bars and logging.
"""

import os
import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from llp.fixed_cifar_meta_model import CIFARMetaModelOptimizer

class SimpleProgressBar:
    """Simple text-based progress bar"""
    
    def __init__(self, total, desc="Progress", bar_length=50):
        self.total = total
        self.desc = desc
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()
        self.print_progress()
    
    def update(self, n=1):
        self.current += n
        self.print_progress()
    
    def print_progress(self):
        percent = min(100.0, (self.current / self.total) * 100)
        filled_length = int(self.bar_length * self.current // self.total)
        bar = '#' * filled_length + '-' * (self.bar_length - filled_length)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total / self.current - 1)
            eta_str = f"ETA: {int(eta//60):02d}:{int(eta%60):02d}"
        else:
            eta_str = "ETA: --:--"
            
        print(f"\r{self.desc}: [{bar}] {self.current}/{self.total} ({percent:.1f}%) {eta_str}", end="", flush=True)
        
        if self.current == self.total:
            print()  # Add newline when complete
    
    def close(self):
        if self.current < self.total:
            self.current = self.total
            self.print_progress()
        print()  # Add newline

def patch_optimize_hyperparameters(original_method):
    """Patch the optimize_hyperparameters method to add progress bars"""
    
    def patched_optimize_hyperparameters(self):
        print("\n" + "=" * 80)
        print("🔍 STARTING META-MODEL HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        
        # Get hyperparameter space
        hp_space = self._get_hyperparameter_space()
        
        # Sample initial configurations
        configs = self._sample_hyperparameter_configs(self.config.configs_per_sample)
        total_configs = len(configs)
        
        print(f"\n📊 Evaluating {total_configs} hyperparameter configurations")
        print(f"📈 Using {len(self.data_subsets)} data subsets for evaluation")
        
        # Create progress bar for configurations
        config_bar = SimpleProgressBar(total_configs, desc="Configurations")
        
        # Track best configuration and results
        best_val_accuracy = float('-inf')
        best_config_found = None
        all_results = []
        
        # Evaluate each configuration
        for config_idx, config in enumerate(configs):
            print(f"\n🔄 Configuration {config_idx+1}/{total_configs}:")
            print(f"   Learning rate: {config.get('learning_rate', 'N/A')}")
            print(f"   Weight decay: {config.get('weight_decay', 'N/A')}")
            print(f"   Optimizer: {config.get('optimizer', 'N/A')}")
            
            # Evaluate on multiple data subsets
            subset_results = []
            subset_bar = SimpleProgressBar(len(self.data_subsets), desc="Data Subsets")
            
            for subset_idx in range(len(self.data_subsets)):
                # Evaluate configuration on this subset
                result = self._evaluate_configuration(
                    config, 
                    subset_idx=subset_idx,
                    resource_level=self.config.min_resource
                )
                
                # Print result
                print(f"   Subset {subset_idx+1}: Accuracy = {result['val_accuracy']:.4f}")
                
                # Store result
                subset_results.append(result)
                subset_bar.update(1)
            
            # Calculate average accuracy across subsets
            avg_accuracy = sum(r['val_accuracy'] for r in subset_results) / len(subset_results)
            print(f"   Average accuracy: {avg_accuracy:.4f}")
            
            # Track best configuration
            if avg_accuracy > best_val_accuracy:
                best_val_accuracy = avg_accuracy
                best_config_found = config.copy()
                print(f"   📈 New best configuration found! Accuracy: {best_val_accuracy:.4f}")
            
            # Store results for meta-model training
            result_entry = {
                'config': config,
                'results': subset_results,
                'avg_val_accuracy': avg_accuracy
            }
            all_results.append(result_entry)
            
            # Update progress bar
            config_bar.update(1)
        
        # Close configuration progress bar
        config_bar.close()
        
        print("\n" + "=" * 80)
        print("🧠 TRAINING META-MODEL")
        print("=" * 80)
        
        # Extract meta-features and prepare datasets
        X_train, y_train = [], []
        for result in all_results:
            config = result['config']
            accuracy = result['avg_val_accuracy']
            
            # Extract features from configuration
            features = [
                config.get('learning_rate', 0.01),
                config.get('weight_decay', 0.0),
                config.get('momentum', 0.0),
                1.0 if config.get('optimizer', '') == 'sgd' else 0.0,
                1.0 if config.get('optimizer', '') == 'adam' else 0.0,
            ]
            
            X_train.append(features)
            y_train.append(accuracy)
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        
        # Create simple meta-model (MLP)
        input_dim = X_train.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Train meta-model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Create data loader
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Train for a few epochs
        num_epochs = 100
        epoch_bar = SimpleProgressBar(num_epochs, desc="Training Epochs")
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
            
            epoch_bar.update(1)
        
        # Close epoch progress bar
        epoch_bar.close()
        
        # Store meta-model
        self.meta_model = model
        
        print("\n" + "=" * 80)
        print(f"✅ META-MODEL OPTIMIZATION COMPLETE")
        print(f"🏆 Best configuration found with accuracy: {best_val_accuracy:.4f}")
        print("=" * 80)
        
        # Return best configuration
        return {
            'learning_rate': best_config_found.get('learning_rate', 0.01),
            'weight_decay': best_config_found.get('weight_decay', 0.0001),
            'momentum': best_config_found.get('momentum', 0.9),
            'optimizer': best_config_found.get('optimizer', 'sgd'),
            'best_val_accuracy': best_val_accuracy
        }
    
    return patched_optimize_hyperparameters

def run_training():
    """Run meta-model training with enhanced progress bars"""
    # Create output directory
    output_dir = Path("reports/meta_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize meta-model optimizer
    meta_optimizer = CIFARMetaModelOptimizer(
        dataset="cifar10",
        data_fraction=0.1,  # Use 10% of data for faster training
        batch_size=128,
        configs_per_sample=10,
        perturbations=5,
        iterations=3,
        min_resource=0.1,
        max_resource=0.5,
        run_dir=str(output_dir)
    )
    
    # Patch the optimize_hyperparameters method
    meta_optimizer.optimize_hyperparameters = patch_optimize_hyperparameters(
        meta_optimizer.optimize_hyperparameters
    ).__get__(meta_optimizer)
    
    try:
        # Run meta-model optimization
        print("\n🚀 Starting meta-model training for CIFAR-10 with 10% sample size")
        print(f"💾 Logs will be saved to: {output_dir}/training.log")
        
        # Train meta-model and get best hyperparameters
        best_config = meta_optimizer.optimize_hyperparameters()
        
        # Save results
        results_file = output_dir / "meta_model_results.json"
        with open(results_file, "w") as f:
            # Convert any non-serializable objects to basic Python types
            serializable_config = {k: float(v) if isinstance(v, torch.Tensor) else v 
                                  for k, v in best_config.items()}
            json.dump(serializable_config, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        print(f"🏆 Best hyperparameters: {json.dumps(serializable_config, indent=2)}")
        
    except Exception as e:
        print(f"\n❌ Error during meta-model training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Import here to avoid circular imports
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    
    run_training()
