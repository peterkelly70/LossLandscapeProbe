#!/usr/bin/env python3
"""
Generate Placeholder Results
============================

This script generates placeholder result files for CIFAR-10 and CIFAR-100
to allow reports to be generated even when actual training hasn't been completed.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def generate_cifar10_results(output_path):
    """Generate placeholder CIFAR-10 results"""
    # Create synthetic training history
    epochs = list(range(1, 51))  # 50 epochs
    train_losses = [2.0 * np.exp(-0.05 * e) + 0.3 + 0.1 * np.random.randn() for e in epochs]
    train_accs = [0.4 * (1 - np.exp(-0.05 * e)) + 0.5 + 0.05 * np.random.randn() for e in epochs]
    val_losses = [1.8 * np.exp(-0.04 * e) + 0.4 + 0.15 * np.random.randn() for e in epochs]
    val_accs = [0.35 * (1 - np.exp(-0.04 * e)) + 0.55 + 0.04 * np.random.randn() for e in epochs]
    
    # Ensure values are in reasonable range
    train_accs = [min(max(acc, 0.1), 0.99) for acc in train_accs]
    val_accs = [min(max(acc, 0.1), 0.99) for acc in val_accs]
    train_losses = [max(loss, 0.1) for loss in train_losses]
    val_losses = [max(loss, 0.1) for loss in val_losses]
    
    # Create sample predictions
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Generate 20 sample predictions
    sample_predictions = []
    for i in range(20):
        true_class = np.random.randint(0, 10)
        pred_probs = np.random.dirichlet(np.ones(10) * 0.5) 
        pred_class = np.argmax(pred_probs)
        
        # Make predictions more accurate than random
        if np.random.rand() < 0.7:  # 70% accuracy
            pred_class = true_class
            # Adjust probabilities to favor the true class
            pred_probs = np.random.dirichlet(np.ones(10) * 0.5)
            pred_probs = pred_probs / pred_probs.sum() * 0.3  # Scale down
            pred_probs[true_class] = 0.7 + 0.3 * np.random.rand()  # High probability for true class
            pred_probs = pred_probs / pred_probs.sum()  # Normalize
        
        sample_predictions.append({
            'image_id': i,
            'true_class': int(true_class),
            'true_label': class_names[true_class],
            'pred_class': int(pred_class),
            'pred_label': class_names[pred_class],
            'probabilities': pred_probs.tolist(),
            'correct': pred_class == true_class
        })
    
    # Create confusion matrix (with realistic patterns)
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for i in range(10):
        # Diagonal elements (correct predictions) should be higher
        confusion_matrix[i, i] = 80 + np.random.randint(-10, 10)
        
        # Off-diagonal elements (errors) should be lower and have patterns
        # e.g., cars might be confused with trucks more often
        for j in range(10):
            if i != j:
                # Base error rate
                confusion_matrix[i, j] = np.random.randint(1, 10)
                
                # Add common confusion patterns
                if (i == 0 and j == 9) or (i == 9 and j == 0):  # airplane vs truck
                    confusion_matrix[i, j] += np.random.randint(5, 10)
                elif (i == 1 and j == 9) or (i == 9 and j == 1):  # automobile vs truck
                    confusion_matrix[i, j] += np.random.randint(10, 20)
                elif (i == 2 and j == 3) or (i == 3 and j == 2):  # bird vs cat
                    confusion_matrix[i, j] += np.random.randint(5, 15)
                elif (i == 3 and j == 5) or (i == 5 and j == 3):  # cat vs dog
                    confusion_matrix[i, j] += np.random.randint(15, 25)
                elif (i == 4 and j == 7) or (i == 7 and j == 4):  # deer vs horse
                    confusion_matrix[i, j] += np.random.randint(10, 20)
    
    # Normalize confusion matrix to ensure row sums are reasonable
    row_sums = confusion_matrix.sum(axis=1)
    for i in range(10):
        target_sum = 100  # Each class has 100 test examples
        if row_sums[i] > 0:
            confusion_matrix[i] = np.round(confusion_matrix[i] * target_sum / row_sums[i]).astype(int)
    
    # Create results dictionary
    results = {
        'dataset': 'CIFAR-10',
        'model_name': 'CNN-MetaModel',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'epochs': epochs,
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'test_acc': val_accs[-1],  # Use last validation accuracy as test accuracy
        'test_loss': val_losses[-1],  # Use last validation loss as test loss
        'class_names': class_names,
        'confusion_matrix': confusion_matrix.tolist(),
        'sample_predictions': sample_predictions,
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': 128,
            'optimizer': 'Adam',
            'weight_decay': 0.0005,
            'dropout': 0.2,
            'architecture': 'CNN with 3 convolutional layers'
        },
        'is_placeholder': True  # Flag to indicate this is placeholder data
    }
    
    # Save results
    torch.save(results, output_path)
    print(f"Generated placeholder CIFAR-10 results: {output_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Generate placeholder results for CIFAR-10 and CIFAR-100')
    parser.add_argument('--cifar10', type=str, default='cifar10_results.pth', help='Output path for CIFAR-10 results')
    args = parser.parse_args()
    
    # Generate CIFAR-10 results
    generate_cifar10_results(args.cifar10)
    
    print("Placeholder results generated successfully.")

if __name__ == '__main__':
    main()
