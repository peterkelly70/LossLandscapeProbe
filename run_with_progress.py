#!/usr/bin/env python3
"""
Script to run the CIFAR meta-model with progress bars.
"""
import os
import sys
import importlib.util
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_optimizer_with_progress():
    """Load the optimizer_with_progress module."""
    from llp.optimizer_with_progress import optimize_hyperparameters
    return optimize_hyperparameters

def patch_cifar_meta_model():
    """Patch the CIFARMetaModelOptimizer class with the progress bar implementation."""
    from llp.cifar_meta_model import CIFARMetaModelOptimizer
    
    # Load the optimize_hyperparameters function from our implementation
    optimize_fn = load_optimizer_with_progress()
    
    # Replace the method in the class
    CIFARMetaModelOptimizer.optimize_hyperparameters = optimize_fn
    
    return CIFARMetaModelOptimizer

def main():
    """Run the CIFAR meta-model with progress bars."""
    print("CIFAR Meta-Model with Progress Bars")
    print("=" * 40)
    
    # Patch the CIFARMetaModelOptimizer class
    CIFARMetaModelOptimizer = patch_cifar_meta_model()
    
    # Create an instance of the optimizer
    print("Creating optimizer...")
    optimizer = CIFARMetaModelOptimizer()
    
    # Run the hyperparameter optimization with progress bars
    print("Running hyperparameter optimization with progress bars...")
    result = optimizer.optimize_hyperparameters()
    
    # Print the result
    print("\nOptimization complete!")
    print(f"Best configuration found: {result}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
