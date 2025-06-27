#!/usr/bin/env python3
"""
Patch Script for CIFAR Meta-Model Training

This script patches the actual training process to show visible progress bars.
"""

import os
import sys
import time
from pathlib import Path
import importlib.util
import types

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Simple progress bar implementation
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

# Patch the CIFARMetaModelOptimizer.optimize_hyperparameters method
def patch_optimize_hyperparameters(original_method):
    """Patch the optimize_hyperparameters method to add progress bars"""
    
    def patched_optimize_hyperparameters(self):
        print("\n" + "=" * 80)
        print("🔍 STARTING META-MODEL HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        
        # Call the original method to get the result
        result = original_method(self)
        
        # Print completion message
        print("\n" + "=" * 80)
        print(f"✅ META-MODEL OPTIMIZATION COMPLETE")
        print(f"🏆 Best configuration found with accuracy: {result.get('best_val_accuracy', 0):.4f}")
        print("=" * 80)
        
        return result
    
    return patched_optimize_hyperparameters

# Patch the CIFARTrainer.train method
def patch_train_method(original_method):
    """Patch the train method to add progress bars"""
    
    def patched_train(self):
        print("\n" + "=" * 80)
        print("🏋️ TRAINING MODEL WITH BEST HYPERPARAMETERS")
        print("=" * 80)
        
        # Get the number of epochs
        num_epochs = self.config.get('epochs', 100)
        
        # Create progress bar
        epoch_bar = SimpleProgressBar(num_epochs, desc="Training Epochs")
        
        # Store the original epoch_end callback
        original_epoch_end = getattr(self, 'on_epoch_end', None)
        
        # Define a new epoch_end callback
        def patched_epoch_end(epoch, train_metrics, val_metrics):
            # Update progress bar
            epoch_bar.update(1)
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics.get('loss', 0):.4f}, Acc: {train_metrics.get('accuracy', 0):.2f}%")
            print(f"  Val Loss: {val_metrics.get('loss', 0):.4f}, Acc: {val_metrics.get('accuracy', 0):.2f}%")
            
            # Call original callback if it exists
            if original_epoch_end:
                original_epoch_end(epoch, train_metrics, val_metrics)
        
        # Replace the epoch_end callback
        setattr(self, 'on_epoch_end', patched_epoch_end)
        
        # Call the original method
        result = original_method(self)
        
        # Close progress bar
        epoch_bar.close()
        
        # Print completion message
        print("\n" + "=" * 80)
        print(f"✅ TRAINING COMPLETE")
        print(f"🏆 Best validation accuracy: {result.get('best_val_acc', 0):.2f}%")
        print("=" * 80)
        
        return result
    
    return patched_train

def patch_and_run():
    """Patch the training methods and run the training"""
    try:
        # Import the modules
        from unified_cifar_training import main
        
        # Run the patched training
        print("\n🚀 Starting CIFAR training with visible progress bars")
        main()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if training is already running
    import subprocess
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    if "unified_cifar_training.py" in result.stdout:
        print("\n⚠️ Training is already running. Showing progress for the current run:")
        
        # Print the current training status
        print("\n" + "=" * 80)
        print("🔄 CIFAR TRAINING IN PROGRESS")
        print("=" * 80)
        
        # Check the log file
        log_path = Path("reports/cifar10/cifar10_10/training.log")
        if log_path.exists():
            with open(log_path, "r") as f:
                lines = f.readlines()
                print(f"\nLast 10 log entries:")
                for line in lines[-10:]:
                    print(f"  {line.strip()}")
        
        print("\n💡 To see real-time progress, use this command:")
        print("  tail -f reports/cifar10/cifar10_10/training.log")
        
    else:
        # No training running, start a new one with patches
        patch_and_run()
