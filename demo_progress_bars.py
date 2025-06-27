#!/usr/bin/env python3
"""
Demo script for CIFAR meta-model with progress bars.
This script demonstrates the progress bars implementation without modifying the original code.
"""
import os
import sys
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the optimizer with clean implementation
from llp.optimize_hyperparameters_clean import optimize_hyperparameters

# Import the original CIFARMetaModelOptimizer
from llp.cifar_meta_model import CIFARMetaModelOptimizer

class EnhancedOptimizer(CIFARMetaModelOptimizer):
    """Enhanced optimizer with progress bars."""
    
    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """Override with the clean implementation."""
        return optimize_hyperparameters(self)

def main():
    """Run the CIFAR meta-model with progress bars."""
    print("CIFAR Meta-Model with Progress Bars Demo")
    print("=" * 40)
    
    # Create an instance of the enhanced optimizer
    print("Creating optimizer...")
    optimizer = EnhancedOptimizer()
    
    # Run the hyperparameter optimization with progress bars
    print("Running hyperparameter optimization with progress bars...")
    result = optimizer.optimize_hyperparameters()
    
    # Print the result
    print("\nOptimization complete!")
    print(f"Best configuration found: {result}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
