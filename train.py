#!/usr/bin/env python3
"""
CIFAR Training with Rich Logging
===============================

This script provides a unified interface for training CIFAR models with rich
logging and progress tracking.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel

from llp.rich_training import CIFARTrainer

# Set up console
console = Console()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CIFAR models with rich logging')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                      choices=['cifar10', 'cifar100'],
                      help='CIFAR dataset to use')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                      help='Fraction of training data to use (0.0 to 1.0)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum for SGD')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                      help='Model architecture')
    parser.add_argument('--optimizer', type=str, default='adam',
                      choices=['adam', 'sgd'],
                      help='Optimizer to use')
    
    # Directory arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                      help='Directory to store datasets')
    parser.add_argument('--run-dir', type=str, default='./runs',
                      help='Directory to save logs and checkpoints')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu). Auto-detected if None.')
    
    return parser.parse_args()

def print_config(config: Dict[str, Any]):
    """Print configuration in a nice format."""
    table = ""
    for key, value in config.items():
        table += f"[bold cyan]{key:>20}:[/] {value}\n"
    
    console.print(Panel(
        table.strip(),
        title="[bold]Training Configuration",
        border_style="blue",
        expand=False
    ))

def main():
    """Main training function."""
    args = parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create run directory
    run_dir = Path(args.run_dir) / f"{args.dataset}_{int(args.data_fraction*100)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        'dataset': args.dataset,
        'data_fraction': args.data_fraction,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'model': args.model,
        'optimizer': args.optimizer,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'device': args.device,
    }
    
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Print config
    print_config(config)
    
    # Create trainer
    trainer = CIFARTrainer(
        dataset=args.dataset,
        data_fraction=args.data_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        run_dir=run_dir,
        device=args.device
    )
    
    # Start training
    console.rule("[bold blue]Starting Training")
    
    model_config = {
        'model_name': args.model,
        'optimizer': args.optimizer,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
    }
    
    history = trainer.train(model_config)
    
    # Test the model
    console.rule("[bold green]Testing Model")
    test_loss, test_acc = trainer.test()
    
    console.print(Panel(
        f"[bold]Test Results:\n"
        f"  Loss: {test_loss:.4f}\n"
        f"  Accuracy: {test_acc:.2f}%",
        title="[bold green]Final Test Results",
        border_style="green"
    ))
    
    # Save final metrics
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'training_history': history
    }
    
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    console.print(f"\n[bold green]Training complete! Check {run_dir} for logs and checkpoints.")

if __name__ == '__main__':
    main()
