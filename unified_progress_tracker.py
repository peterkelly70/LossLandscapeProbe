#!/usr/bin/env python3
"""
Unified Progress Tracker for CIFAR Meta-Model Training

This script provides clear, visible progress bars and detailed logging for
CIFAR meta-model hyperparameter optimization and training.

It can be used in two ways:
1. As a standalone script to run the training with progress bars
2. As a module to be imported and used by other scripts

Features:
- Simple ASCII progress bars visible in any terminal
- Detailed logging with emojis for better visual feedback
- Real-time monitoring of training progress
- Fail-fast approach with minimal dependencies
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("progress_tracker")

# Simple progress bar class
class SimpleProgressBar:
    """Simple ASCII progress bar that works in any terminal"""
    
    def __init__(self, total: int, desc: str = "Progress", bar_length: int = 50):
        self.total = max(1, total)  # Avoid division by zero
        self.desc = desc
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.print_progress()
    
    def update(self, n: int = 1) -> None:
        """Update progress bar by n steps"""
        self.current = min(self.total, self.current + n)
        self.last_update_time = time.time()
        self.print_progress()
    
    def print_progress(self) -> None:
        """Print the progress bar to the console"""
        percent = min(100.0, (self.current / self.total) * 100)
        filled_length = int(self.bar_length * self.current // self.total)
        bar = '#' * filled_length + '-' * (self.bar_length - filled_length)
        
        elapsed = time.time() - self.start_time
        if self.current > 0 and self.current < self.total:
            eta = elapsed * (self.total / self.current - 1)
            eta_min, eta_sec = divmod(int(eta), 60)
            eta_str = f"ETA: {eta_min:02d}:{eta_sec:02d}"
        elif self.current == 0:
            eta_str = "ETA: --:--"
        else:
            eta_str = f"Time: {int(elapsed//60):02d}:{int(elapsed%60):02d}"
            
        print(f"\r{self.desc}: [{bar}] {self.current}/{self.total} ({percent:.1f}%) {eta_str}", end="", flush=True)
        
        if self.current == self.total:
            print()  # Add newline when complete
    
    def close(self) -> None:
        """Close the progress bar"""
        if self.current < self.total:
            self.current = self.total
            self.print_progress()
        print()  # Add newline

# Patch functions for the CIFARMetaModelOptimizer
def patch_optimize_hyperparameters(original_method):
    """
    Patch the optimize_hyperparameters method to add progress bars
    
    Args:
        original_method: The original optimize_hyperparameters method
        
    Returns:
        A patched method with progress bars
    """
    def patched_method(self):
        print("\n" + "=" * 80)
        print("🔍 STARTING META-MODEL HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        
        # Get the number of configurations to evaluate
        num_configs = len(self.configurations)
        config_bar = SimpleProgressBar(num_configs, desc="Configurations")
        
        # Store original evaluate_configuration method
        original_evaluate = self.evaluate_configuration
        
        # Define patched evaluate_configuration method
        def patched_evaluate(config, subset_indices):
            config_str = ", ".join([f"{k}: {v}" for k, v in config.items()])
            print(f"\n🔄 Configuration {config_bar.current + 1}/{num_configs}:")
            print(f"   {config_str}")
            
            # Create progress bar for data subsets
            subset_bar = SimpleProgressBar(len(subset_indices), desc="Data Subsets")
            
            # Store original results
            results = []
            
            # Evaluate on each subset with progress tracking
            for i, subset_idx in enumerate(subset_indices):
                result = original_evaluate(config, [subset_idx])
                subset_bar.update(1)
                
                # Print accuracy for this subset
                accuracy = result[0]['accuracy'] * 100 if result else 0
                print(f"   Subset {i+1}: Accuracy = {accuracy:.2f}%")
                
                results.extend(result)
            
            subset_bar.close()
            
            # Calculate and print average accuracy
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results) * 100
            print(f"   Average accuracy: {avg_accuracy:.2f}%")
            
            # Update configuration progress bar
            config_bar.update(1)
            
            return results
        
        # Replace evaluate_configuration with patched version
        self.evaluate_configuration = patched_evaluate
        
        try:
            # Call original method
            result = original_method(self)
            
            # Print best configuration found
            if 'best_val_accuracy' in result:
                print(f"\n📈 Best configuration found with accuracy: {result['best_val_accuracy']:.4f}")
            
            # Print completion message
            print("\n" + "=" * 80)
            print("✅ META-MODEL OPTIMIZATION COMPLETE")
            print("=" * 80)
            
            return result
        finally:
            # Restore original method
            self.evaluate_configuration = original_evaluate
            
            # Close progress bar if needed
            if config_bar.current < config_bar.total:
                config_bar.close()
    
    return patched_method

def patch_train_meta_model(original_method):
    """
    Patch the train_meta_model method to add progress bars
    
    Args:
        original_method: The original train_meta_model method
        
    Returns:
        A patched method with progress bars
    """
    def patched_method(self, X, y, epochs=20, batch_size=32):
        print("\n" + "=" * 80)
        print("🧠 TRAINING META-MODEL")
        print("=" * 80)
        
        # Create progress bar for epochs
        epoch_bar = SimpleProgressBar(epochs, desc="Training Epochs")
        
        # Store original epoch end callback
        original_epoch_end = getattr(self, '_epoch_end_callback', None)
        
        # Define patched epoch end callback
        def patched_epoch_end(epoch, loss):
            epoch_bar.update(1)
            print(f"   Epoch {epoch+1}/{epochs}: Loss = {loss:.6f}")
            
            # Call original callback if it exists
            if original_epoch_end:
                original_epoch_end(epoch, loss)
        
        # Set epoch end callback
        self._epoch_end_callback = patched_epoch_end
        
        try:
            # Call original method
            result = original_method(self, X, y, epochs, batch_size)
            
            # Print completion message
            print("\n" + "=" * 80)
            print("✅ META-MODEL TRAINING COMPLETE")
            print("=" * 80)
            
            return result
        finally:
            # Restore original callback if needed
            if original_epoch_end:
                self._epoch_end_callback = original_epoch_end
            elif hasattr(self, '_epoch_end_callback'):
                delattr(self, '_epoch_end_callback')
            
            # Close progress bar if needed
            if epoch_bar.current < epoch_bar.total:
                epoch_bar.close()
    
    return patched_method

# Function to patch the CIFARMetaModelOptimizer class
def patch_cifar_meta_model_optimizer():
    """
    Patch the CIFARMetaModelOptimizer class to add progress bars
    
    Returns:
        The patched CIFARMetaModelOptimizer class
    """
    try:
        # Try to import the fixed version first
        from llp.fixed_cifar_meta_model import CIFARMetaModelOptimizer
    except ImportError:
        try:
            # Fall back to original version
            from llp.cifar_meta_model import CIFARMetaModelOptimizer
        except ImportError:
            logger.error("❌ Failed to import CIFARMetaModelOptimizer")
            raise
    
    # Patch methods
    CIFARMetaModelOptimizer.optimize_hyperparameters = patch_optimize_hyperparameters(
        CIFARMetaModelOptimizer.optimize_hyperparameters
    )
    
    CIFARMetaModelOptimizer.train_meta_model = patch_train_meta_model(
        CIFARMetaModelOptimizer.train_meta_model
    )
    
    return CIFARMetaModelOptimizer

# Function to patch the CIFARTrainer class
def patch_cifar_trainer():
    """
    Patch the CIFARTrainer class to add progress bars
    
    Returns:
        The patched CIFARTrainer class
    """
    try:
        from llp.cifar_trainer import CIFARTrainer
    except ImportError:
        logger.error("❌ Failed to import CIFARTrainer")
        raise
    
    # Store original train method
    original_train = CIFARTrainer.train
    
    # Define patched train method
    def patched_train(self):
        print("\n" + "=" * 80)
        print("🏋️ TRAINING MODEL WITH BEST HYPERPARAMETERS")
        print("=" * 80)
        
        # Get the number of epochs
        num_epochs = self.config.get('epochs', 100)
        
        # Create progress bar for epochs
        epoch_bar = SimpleProgressBar(num_epochs, desc="Final Training")
        
        # Store original epoch end callback
        original_epoch_end = getattr(self, 'on_epoch_end', None)
        
        # Define patched epoch end callback
        def patched_epoch_end(epoch, train_metrics, val_metrics):
            # Update progress bar
            epoch_bar.update(1)
            
            # Print metrics
            train_acc = train_metrics.get('accuracy', 0) * 100
            val_acc = val_metrics.get('accuracy', 0) * 100
            print(f"   Epoch {epoch+1}/{num_epochs}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")
            
            # Call original callback if it exists
            if original_epoch_end:
                original_epoch_end(epoch, train_metrics, val_metrics)
        
        # Replace the epoch end callback
        self.on_epoch_end = patched_epoch_end
        
        try:
            # Call original method
            result = original_train(self)
            
            # Print completion message
            print("\n" + "=" * 80)
            print("✅ MODEL TRAINING COMPLETE")
            if 'best_val_acc' in result:
                print(f"🏆 Best validation accuracy: {result['best_val_acc']:.2f}%")
            print("=" * 80)
            
            return result
        finally:
            # Restore original callback
            if original_epoch_end:
                self.on_epoch_end = original_epoch_end
            
            # Close progress bar if needed
            if epoch_bar.current < epoch_bar.total:
                epoch_bar.close()
    
    # Replace the train method
    CIFARTrainer.train = patched_train
    
    return CIFARTrainer

# Function to patch the unified_cifar_training.py script
def patch_unified_cifar_training():
    """
    Apply all patches to the unified_cifar_training.py script
    """
    # Patch the CIFARMetaModelOptimizer class
    patched_optimizer = patch_cifar_meta_model_optimizer()
    
    # Patch the CIFARTrainer class
    patched_trainer = patch_cifar_trainer()
    
    # Return patched classes
    return patched_optimizer, patched_trainer

# Function to run the unified_cifar_training.py script with patches
def run_training_with_progress(args=None):
    """
    Run the unified_cifar_training.py script with progress bars
    
    Args:
        args: Command line arguments
    """
    # Apply patches
    patch_unified_cifar_training()
    
    # Import and run the main function
    try:
        from unified_cifar_training import main
        main(args)
    except ImportError:
        logger.error("❌ Failed to import unified_cifar_training")
        raise

# Main function
def main():
    """Main function to parse arguments and run the script"""
    parser = argparse.ArgumentParser(description="Run CIFAR training with progress bars")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"], help="Dataset to use")
    parser.add_argument("--sample-size", type=int, default=10, help="Percentage of training data to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for final training")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model")
    parser.add_argument("--generate-report", action="store_true", help="Generate a report")
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.outdir is None:
        args.outdir = f"reports/{args.dataset}/{args.dataset}_{args.sample_size}"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    
    # Add file handler to logger
    file_handler = logging.FileHandler(os.path.join(args.outdir, "training.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    # Print welcome message
    print("\n" + "=" * 80)
    print(f"🚀 STARTING CIFAR TRAINING WITH VISIBLE PROGRESS")
    print(f"   Dataset: {args.dataset}")
    print(f"   Sample size: {args.sample_size}%")
    print(f"   Epochs: {args.epochs}")
    print(f"   Output directory: {args.outdir}")
    print("=" * 80 + "\n")
    
    # Run training with progress
    run_training_with_progress(args)

if __name__ == "__main__":
    main()
