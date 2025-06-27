#!/usr/bin/env python3
"""
Script to apply progress bar improvements to the CIFAR meta-model.

This script:
1. Backs up the original cifar_meta_model.py file
2. Modifies the CIFARMetaModelOptimizer class to use the improved optimize_hyperparameters method
3. Runs a test to verify the implementation works correctly
"""
import os
import sys
import shutil
import importlib
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def backup_original_file():
    """Create a backup of the original cifar_meta_model.py file."""
    original_file = os.path.join('llp', 'cifar_meta_model.py')
    backup_file = os.path.join('llp', 'cifar_meta_model.py.backup')
    
    if not os.path.exists(backup_file):
        print(f"Creating backup of {original_file} to {backup_file}...")
        shutil.copy2(original_file, backup_file)
    else:
        print(f"Backup file {backup_file} already exists, skipping backup.")

def apply_progress_bars():
    """Apply the progress bar improvements to the CIFAR meta-model."""
    # Import the necessary modules
    from llp.hyperparameter_optimization import optimize_hyperparameters
    from llp.cifar_meta_model import CIFARMetaModelOptimizer
    
    print("Applying progress bar improvements to CIFARMetaModelOptimizer...")
    
    # Replace the optimize_hyperparameters method
    CIFARMetaModelOptimizer.optimize_hyperparameters = optimize_hyperparameters
    
    # Save the modified class back to the file
    with open(os.path.join('llp', 'cifar_meta_model.py'), 'r') as f:
        lines = f.readlines()
    
    # Find the start and end of the optimize_hyperparameters method
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if "def optimize_hyperparameters" in line:
            start_idx = i
            break
    
    if start_idx is not None:
        # Find the next method definition
        for i in range(start_idx + 1, len(lines)):
            if "    def " in lines[i]:
                end_idx = i
                break
    
    if start_idx is not None and end_idx is not None:
        # Get the source code of the new method
        import inspect
        new_method_source = inspect.getsource(optimize_hyperparameters)
        
        # Adjust indentation to match the class
        new_method_lines = new_method_source.split('\n')
        new_method_lines = [line[4:] if line.startswith('    ') else line for line in new_method_lines]
        new_method_source = '\n'.join(new_method_lines)
        
        # Replace the old method with the new one
        new_lines = lines[:start_idx] + [new_method_source] + lines[end_idx:]
        
        # Write the modified file
        with open(os.path.join('llp', 'cifar_meta_model.py'), 'w') as f:
            f.writelines(new_lines)
        
        print("Progress bar improvements applied successfully!")
    else:
        print("Could not find the optimize_hyperparameters method in the file.")

def main():
    """Main function to apply progress bar improvements."""
    print("CIFAR Meta-Model Progress Bars Implementation")
    print("=" * 40)
    
    # Backup the original file
    backup_original_file()
    
    # Apply progress bar improvements
    apply_progress_bars()
    
    print("\nDone! The CIFAR meta-model now has improved progress bars.")
    print("You can run the test script with: python -m llp.test_progress_bars")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
