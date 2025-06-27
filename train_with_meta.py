#!/usr/bin/env python3
"""
CIFAR Training with Meta-Model Hyperparameter Optimization
========================================================

This script implements the full training pipeline with meta-model hyperparameter optimization:
1. Trains a meta-model to predict optimal hyperparameters
2. Uses the meta-model to predict hyperparameters for the target task
3. Trains the final model using the predicted hyperparameters
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

# Import the meta-model implementation
from llp.fixed_cifar_meta_model import CIFARMetaModelOptimizer

# Define MetaModelConfig class if it doesn't exist in fixed version
class MetaModelConfig:
    def __init__(self, dataset='cifar10', batch_size=128, run_dir='./runs/meta_model'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.run_dir = run_dir
        self.num_configs = 10
        self.configs_per_sample = 10
        self.min_resource = 0.1
        self.max_epochs = 20
from llp.cifar_core import create_model, create_optimizer, get_cifar_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR with Meta-Model')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'],
                      help='Dataset to use (default: cifar10)')
    parser.add_argument('--data-dir', default='./data', help='Dataset directory')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                      help='Fraction of training data to use (0.0 to 1.0)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--model', default='resnet18', help='Model architecture')
    parser.add_argument('--run-dir', default='./runs', help='Directory to save runs')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    return parser.parse_args()

def train_final_model(model, train_loader, val_loader, hyperparams, num_epochs, device):
    """Train the final model with the given hyperparameters."""
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_name=hyperparams.get('optimizer', 'sgd'),
        lr=hyperparams.get('lr', 0.1),
        momentum=hyperparams.get('momentum', 0.9),
        weight_decay=hyperparams.get('weight_decay', 5e-4)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model in the run directory
            model_dir = Path('runs') / args.dataset / args.model
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'hyperparams': hyperparams,
                'dataset': args.dataset,
                'model_arch': args.model
            }, model_path)
            print(f'Saved best model to {model_path}')
        
        # Update learning rate
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Val Acc: {val_acc:.2f}%')
    
    return best_val_acc

def main():
    args = parse_args()
    
    print(f"\nTraining {args.model.upper()} on {args.dataset.upper()} with {args.data_fraction*100}% of data")
    print(f"Using device: {args.device}")
    
    # Create dataset-specific run directory
    base_run_dir = Path(args.run_dir) / args.dataset
    run_dir = base_run_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Initialize and train meta-model
    print("\n=== Training Meta-Model ===")
    meta_config = MetaModelConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        run_dir=base_run_dir / 'meta_model'  # Store meta-model at dataset level
    )
    
    meta_optimizer = CIFARMetaModelOptimizer(meta_config)
    meta_optimizer.optimize_hyperparameters()
    
    # Step 2: Predict hyperparameters
    print("\n=== Predicting Hyperparameters ===")
    # Extract meta-features from the dataset
    train_loader, val_loader, num_classes = get_cifar_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        data_fraction=args.data_fraction
    )
    
    # Use meta-features from the first batch as a simple example
    # In a real implementation, we would extract proper meta-features
    sample_batch, _ = next(iter(train_loader))
    dummy_meta_features = [sample_batch.shape[0], sample_batch.shape[1], 
                          sample_batch.shape[2], sample_batch.shape[3], 
                          num_classes, args.data_fraction, args.batch_size, 0.1, 0.1, 0.1]
    
    # Predict hyperparameters using the meta-model
    best_hyperparams = meta_optimizer.predict_hyperparameters(dummy_meta_features)
    print(f"Predicted hyperparameters: {best_hyperparams}")
    
    # Step 3: Train final model with predicted hyperparameters
    print("\n=== Training Final Model ===")
    train_loader, val_loader, num_classes = get_cifar_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        data_fraction=args.data_fraction
    )
    
    model = create_model(args.model, num_classes=num_classes).to(args.device)
    
    print(f"Training with hyperparameters: {best_hyperparams}")
    best_val_acc = train_final_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        hyperparams=best_hyperparams,
        num_epochs=args.epochs,
        device=args.device
    )
    
    print(f"\n=== Training Complete ===")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save final results
    results = {
        'dataset': args.dataset,
        'model': args.model,
        'data_fraction': args.data_fraction,
        'best_val_acc': best_val_acc,
        'hyperparameters': best_hyperparams
    }
    
    with open(run_dir / 'results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
