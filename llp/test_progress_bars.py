"""
Test script for the CIFAR meta-model progress bars implementation.
"""
import sys
import os
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llp.cifar_meta_model import CIFARMetaModelOptimizer
from llp.meta_model_integration import integrate_with_meta_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Test the progress bar implementation for the meta-model optimization."""
    print("Testing CIFAR meta-model progress bars implementation...")
    
    # Integrate the optimized method with the meta-model optimizer
    CIFARMetaModelOptimizer = integrate_with_meta_model(CIFARMetaModelOptimizer)
    
    # Create a meta-model optimizer instance
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
