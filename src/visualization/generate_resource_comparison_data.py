#!/usr/bin/env python3
"""
Generate sample resource comparison data for CIFAR-10 and CIFAR-100

This script creates sample data files that demonstrate how using different
percentages of the dataset (10%, 20%, 30%, 40%) affects model performance.
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Set up paths
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Date-time string for unique filenames
DATE_STR = datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_resource_comparison_data(dataset='cifar10'):
    """Generate sample resource comparison data for the given dataset."""
    print(f"\nGenerating {dataset.upper()} resource comparison data...")
    
    # Define resource levels (dataset percentages)
    resource_levels = [0.1, 0.2, 0.3, 0.4]  # 10%, 20%, 30%, 40%
    
    # Sample data for each resource level
    data = []
    
    for level in resource_levels:
        # Simulate meta-model time increasing with dataset size
        meta_time = level * 120 + np.random.normal(0, 10)
        
        # Simulate training time increasing with dataset size
        training_time = level * 600 + np.random.normal(0, 30)
        
        # Simulate accuracy increasing with dataset size but with diminishing returns
        accuracy_base = 65 + (level * 100) * 0.5
        final_accuracy = min(accuracy_base + np.random.normal(0, 2), 95)
        
        # Simulate best hyperparameters
        if dataset == 'cifar10':
            best_num_channels = int(16 + level * 64)
            best_dropout_rate = max(0.1, 0.5 - level)
            best_learning_rate = 0.001 if level <= 0.2 else 0.0005
            best_weight_decay = 0.0005 if level >= 0.3 else 0.001
            best_optimizer = 'adam' if level >= 0.2 else 'sgd'
            best_momentum = 0.9 if best_optimizer == 'sgd' else 0.0
        else:  # cifar100
            best_num_channels = int(32 + level * 96)
            best_dropout_rate = max(0.15, 0.6 - level)
            best_learning_rate = 0.001 if level <= 0.3 else 0.0005
            best_weight_decay = 0.001 if level >= 0.2 else 0.002
            best_optimizer = 'adam'
            best_momentum = 0.0
        
        # Number of perturbations and subsets
        num_perturbations = 10
        num_subsets = 6
        
        # Add to data
        data.append({
            'resource_level': level,
            'meta_time': meta_time,
            'training_time': training_time,
            'total_time': meta_time + training_time,
            'final_test_acc': final_accuracy,
            'best_num_channels': best_num_channels,
            'best_dropout_rate': best_dropout_rate,
            'best_learning_rate': best_learning_rate,
            'best_weight_decay': best_weight_decay,
            'best_optimizer': best_optimizer,
            'best_momentum': best_momentum,
            'best_epoch': int(30 + level * 40),
            'num_perturbations': num_perturbations,
            'num_subsets': num_subsets
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, f'{dataset}_resource_level_comparison_{DATE_STR}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved resource comparison data to {csv_path}")
    
    # Generate training history for each resource level
    generate_training_histories(dataset, resource_levels, DATE_STR)
    
    return df

def generate_training_histories(dataset, resource_levels, date_str):
    """Generate sample training history data for each resource level."""
    for level in resource_levels:
        # Number of epochs increases with resource level
        num_epochs = int(30 + level * 40)
        
        # Generate sample history data
        epochs = list(range(1, num_epochs + 1))
        
        # Training loss decreases with epochs, faster for smaller datasets
        decay_rate = 0.15 if level <= 0.2 else 0.1
        train_loss = [2.5 * np.exp(-decay_rate * epoch) + 0.3 + np.random.normal(0, 0.05) for epoch in epochs]
        
        # Test loss follows similar pattern but with higher floor
        test_loss = [2.3 * np.exp(-decay_rate * epoch) + 0.5 + np.random.normal(0, 0.1) for epoch in epochs]
        
        # Training accuracy increases with epochs
        max_train_acc = min(95, 80 + level * 40)
        train_acc = [max_train_acc * (1 - np.exp(-0.1 * epoch)) + np.random.normal(0, 1) for epoch in epochs]
        train_acc = [min(acc, 99) for acc in train_acc]  # Cap at 99%
        
        # Test accuracy increases with epochs but plateaus lower than training
        max_test_acc = min(90, 65 + level * 50)
        test_acc = [max_test_acc * (1 - np.exp(-0.08 * epoch)) + np.random.normal(0, 1.5) for epoch in epochs]
        test_acc = [min(acc, max_test_acc + 2) for acc in test_acc]  # Cap at max_test_acc + 2
        
        # Create history dictionary
        history = {
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        }
        
        # Create metadata
        metadata = {
            'dataset': dataset,
            'resource_level': level,
            'num_epochs': num_epochs,
            'date_generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'history': history
        }
        
        # Save to JSON
        json_path = os.path.join(RESULTS_DIR, f'{dataset}_resource_level_{level}_{date_str}.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved training history for resource level {level} to {json_path}")

def main():
    """Main function to generate all sample data."""
    print("Generating Resource Level Comparison Sample Data")
    print("===============================================")
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate CIFAR-10 data
    generate_resource_comparison_data('cifar10')
    
    # Generate CIFAR-100 data
    generate_resource_comparison_data('cifar100')
    
    print("\nSample data generation complete.")
    print("Now run visualize_resource_comparison.py to generate visualizations.")

if __name__ == "__main__":
    main()
