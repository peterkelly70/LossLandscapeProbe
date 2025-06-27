"""
Rich-enabled training module for CIFAR datasets.

This module provides enhanced training capabilities with rich progress bars
and logging for better visualization of the training process.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

from .cifar_core import create_model, create_optimizer, get_cifar_loaders

# Set up rich console
console = Console()
logger = logging.getLogger(__name__)

class RichTrainingProgress:
    """Track and display training progress with rich."""
    
    def __init__(self, total_epochs: int, dataset_size: int, batch_size: int):
        self.progress = Progress(
            SpinnerColumn(),
            "•",
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )
        
        self.epoch_task = self.progress.add_task(
            "[cyan]Training", total=total_epochs
        )
        
        # Calculate total batches
        total_batches = (dataset_size + batch_size - 1) // batch_size
        self.batch_task = self.progress.add_task(
            "[green]Batch", total=total_batches, visible=False
        )
        
    def start(self):
        """Start the progress display."""
        self.progress.start()
        
    def update_epoch(self, epoch: int, metrics: Dict[str, float] = None):
        """Update epoch progress and display metrics."""
        self.progress.update(self.epoch_task, completed=epoch + 1)
        if metrics:
            self._log_metrics(metrics)
    
    def update_batch(self, batch_idx: int, total_batches: int):
        """Update batch progress."""
        self.progress.update(
            self.batch_task, 
            completed=batch_idx + 1,
            total=total_batches,
            visible=True
        )
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics in a rich table."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(style="green")
        
        for name, value in metrics.items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            table.add_row(f"{name}:", str(value))
            
        console.print(Panel(table, title="[bold]Metrics", border_style="blue"))
    
    def stop(self):
        """Stop the progress display."""
        self.progress.stop()

class CIFARTrainer:
    """
    Enhanced CIFAR trainer with rich progress tracking.
    
    This class handles the training of CIFAR models with rich progress bars
    and logging for better visualization.
    """
    
    def __init__(
        self,
        dataset: str = 'cifar10',
        data_fraction: float = 1.0,
        batch_size: int = 128,
        epochs: int = 50,
        run_dir: Optional[Path] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the trainer.
        
        Args:
            dataset: Name of the dataset ('cifar10' or 'cifar100')
            data_fraction: Fraction of training data to use (0.0 to 1.0)
            batch_size: Batch size for training
            epochs: Number of training epochs
            run_dir: Directory to save logs and checkpoints
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.dataset = dataset
        self.data_fraction = data_fraction
        self.batch_size = batch_size
        self.epochs = epochs
        self.run_dir = Path(run_dir) if run_dir else Path.cwd() / 'runs'
        self.device = device
        
        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize model and data loaders
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader, self.val_loader, self.test_loader = self._get_data_loaders()
    
    def _setup_logging(self):
        """Set up rich logging to file and console."""
        # Clear existing handlers
        logger.handlers = []
        
        # Set up rich console handler
        console_handler = RichHandler(
            console=console,
            show_time=False,
            rich_tracebacks=True,
            show_path=False,
        )
        console_handler.setLevel(logging.INFO)
        
        # Set up file handler
        log_file = self.run_dir / 'training.log'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        
        # Log startup information
        logger.info(f"Starting training with dataset: {self.dataset}")
        logger.info(f"Using {self.data_fraction*100:.1f}% of training data")
        logger.info(f"Batch size: {self.batch_size}, Epochs: {self.epochs}")
        logger.info(f"Device: {self.device}")
    
    def _get_data_loaders(self):
        """Get data loaders for training, validation, and testing."""
        return get_cifar_loaders(
            dataset=self.dataset,
            batch_size=self.batch_size,
            data_fraction=self.data_fraction,
            val_split=0.1,
            num_workers=4,
            download=True
        )
    
    def train_epoch(self, epoch: int, progress: RichTrainingProgress):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss += loss.item()
            
            # Update progress
            progress.update_batch(batch_idx, len(self.train_loader))
            
            # Log batch progress every 10% of the dataset
            if (batch_idx + 1) % max(1, len(self.train_loader) // 10) == 0:
                batch_loss = running_loss / (batch_idx + 1)
                batch_acc = 100. * correct / total
                logger.info(
                    f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}%"
                )
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model on the validation set."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, model_config: Dict[str, Any]):
        """
        Train the model with the given configuration.
        
        Args:
            model_config: Dictionary containing model and optimizer configuration
        """
        # Initialize model and optimizer
        self.model = create_model(
            model_name=model_config.get('model_name', 'resnet18'),
            num_classes=10 if self.dataset == 'cifar10' else 100,
            pretrained=False
        ).to(self.device)
        
        self.optimizer = create_optimizer(
            model=self.model,
            optimizer_name=model_config.get('optimizer', 'adam'),
            lr=model_config.get('lr', 0.001),
            weight_decay=model_config.get('weight_decay', 1e-4),
            momentum=model_config.get('momentum', 0.9)
        )
        
        # Set up learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.epochs
        )
        
        # Initialize progress tracking
        progress = RichTrainingProgress(
            total_epochs=self.epochs,
            dataset_size=len(self.train_loader.dataset),
            batch_size=self.batch_size
        )
        
        # Training loop
        best_val_acc = 0.0
        train_history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        
        try:
            progress.start()
            
            for epoch in range(self.epochs):
                # Train for one epoch
                train_loss, train_acc = self.train_epoch(epoch, progress)
                
                # Validate
                val_loss, val_acc = self.validate()
                
                # Update learning rate
                scheduler.step()
                
                # Update progress
                progress.update_epoch(epoch, {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': scheduler.get_last_lr()[0]
                })
                
                # Save history
                train_history['loss'].append(train_loss)
                train_history['acc'].append(train_acc)
                train_history['val_loss'].append(val_loss)
                train_history['val_acc'].append(val_acc)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(epoch, val_acc, is_best=True)
                
                # Save checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self._save_checkpoint(epoch, val_acc)
                
        except KeyboardInterrupt:
            logger.info("Training interrupted. Saving model...")
        finally:
            progress.stop()
            
        return train_history
    
    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_acc: Current validation accuracy
            is_best: Whether this is the best model so far
        """
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': {
                'dataset': self.dataset,
                'data_fraction': self.data_fraction,
                'batch_size': self.batch_size,
            }
        }
        
        # Save checkpoint
        filename = f"checkpoint_epoch{epoch+1}.pth.tar"
        if is_best:
            filename = 'model_best.pth.tar'
        
        torch.save(state, self.run_dir / filename)
        logger.info(f"Checkpoint saved to {self.run_dir / filename}")
    
    def test(self):
        """Test the model on the test set."""
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
            
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total
        
        logger.info(f"Test Results - Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
        return test_loss, test_acc
