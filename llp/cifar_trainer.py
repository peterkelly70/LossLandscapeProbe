"""
CIFAR Training Module with Rich Logging
=======================================

This module provides a clean implementation for training CIFAR models
with rich logging and progress tracking.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

class CIFARTrainer:
    """
    Trainer for CIFAR models with rich progress tracking.
    
    This class provides a high-level interface for training CIFAR models with rich logging
    and progress tracking. It supports both the new interface (with explicit loaders and optimizers)
    and a compatibility interface that matches the old CIFARTrainer from cifar_training.py.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dataset: str = "cifar10",
        data_fraction: float = 1.0,
        batch_size: int = 128,
        epochs: int = 100,
        run_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        # Original API parameters (kept for backward compatibility)
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[str] = None,
        num_classes: Optional[int] = None,
    ):
        """Initialize the trainer with either new or legacy interface.
        
        New interface (compatible with unified_cifar_training.py):
            config: Dictionary of hyperparameters
            dataset: Dataset name ('cifar10' or 'cifar100')
            data_fraction: Fraction of training data to use (0.0 to 1.0)
            batch_size: Batch size for training
            epochs: Number of epochs to train for
            run_dir: Directory to save logs and checkpoints
            checkpoint_dir: Directory to save model checkpoints
            
        Legacy interface (kept for backward compatibility):
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            device: Device to run training on
            num_classes: Number of classes in the dataset
        """
        # Initialize common attributes
        self.run_dir = Path(run_dir) if run_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.epochs = epochs
        self.config = config or {}
        self.criterion = nn.CrossEntropyLoss()
        
        # Handle legacy interface (direct model and loaders provided)
        if model is not None and train_loader is not None and val_loader is not None:
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader or val_loader  # Use val_loader as fallback
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
            self.num_classes = num_classes or 10
            self.model = self.model.to(self.device)
        else:
            # New interface - initialize from config
            self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
            self.dataset = dataset
            self.batch_size = batch_size
            self.data_fraction = data_fraction
            
            # Initialize model and data loaders
            self._initialize_model_and_loaders()
        
        # Create directories if they don't exist
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_model_and_loaders(self):
        """Initialize model and data loaders from configuration."""
        from .cifar_core import create_model, create_optimizer, get_cifar_loaders
        
        # Determine number of classes based on dataset
        self.num_classes = 100 if self.dataset == 'cifar100' else 10
        
        # Create model
        self.model = create_model(self.config, num_classes=self.num_classes)
        self.model = self.model.to(self.device)
        
        # Create data loaders
        self.train_loader, self.val_loader, _ = get_cifar_loaders(
            dataset=self.dataset,
            data_fraction=self.data_fraction,
            batch_size=self.batch_size
        )
        
        # Use validation loader as test loader if not specified
        self.test_loader = self.val_loader
        
        # Create optimizer
        self.optimizer = create_optimizer(self.model, self.config)
        
        # Set up learning rate scheduler if specified in config
        if 'lr_scheduler' in self.config:
            if self.config['lr_scheduler'] == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.epochs
                )
            elif self.config['lr_scheduler'] == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, 
                    step_size=self.config.get('lr_step_size', 30),
                    gamma=self.config.get('lr_gamma', 0.1)
                )
    
    def train_epoch(self, epoch: int, progress: Optional[Progress] = None) -> Tuple[float, float]:
        """Train the model for one epoch with enhanced logging.
        
        Args:
            epoch: Current epoch number
            progress: Optional rich Progress instance
            
        Returns:
            Tuple of (train_loss, train_accuracy)
        """
        # Log epoch header
        logger.info("\n" + "="*80)
        logger.info(f"EPOCH {epoch + 1:03d}/{self.epochs}")
        logger.info("-"*80)
        logger.info(f"{'Step':<8} {'Loss':<12} {'Acc %':<8} {'LR':<12} {'Time':<8}")
        logger.info("-"*80)
        
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        last_log_time = start_time
        
        # Create progress bar if not provided
        if progress is None:
            progress = Progress(
                SpinnerColumn(),
                "•",
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                TimeRemainingColumn(),
                "•",
                MofNCompleteColumn(),
                console=console,
            )
            progress.start()
            task = progress.add_task("", total=len(self.train_loader))
        else:
            task = progress.add_task(f"[cyan]Epoch {epoch+1} [yellow](Train)", total=len(self.train_loader))
            
        # Log initial step
        step_time = time.time() - start_time
        lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"{'0':<8} {'-':<12} {'-':<8} {lr:.2e} {step_time:.2f}s")
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress
            progress.update(
                task,
                advance=1,
                description=f"[cyan]Epoch {epoch+1} [yellow](Train)",
            )
            
            # Log step progress every 10% of the dataset
            if (i + 1) % max(1, len(self.train_loader) // 10) == 0 or (i + 1) == len(self.train_loader):
                step_time = time.time() - start_time
                avg_loss = train_loss / (i + 1)
                acc = 100. * correct / total
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"{i+1:<8} {avg_loss:<12.4f} {acc:<8.2f} {lr:.2e} {step_time:.2f}s")
        
        # Calculate epoch metrics
        avg_loss = train_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        epoch_time = time.time() - start_time
        
        # Log epoch summary
        logger.info("-"*80)
        logger.info(f"{'Epoch':<8} {'Loss':<12} {'Acc %':<8} {'Time'}")
        logger.info("-"*80)
        logger.info(f"{epoch + 1:<8} {avg_loss:<12.4f} {accuracy:<8.2f} {epoch_time:.2f}s")
        logger.info("="*80 + "\n")
        
        # Clean up progress if we created it
        if progress is not None and len(progress.tasks) == 1:
            progress.stop()
        
        return avg_loss, accuracy
    
    def validate(self, progress: Optional[Progress] = None) -> Tuple[float, float]:
        """Validate the model on the validation set.
        
        Args:
            progress: Optional rich Progress instance
            
        Returns:
            Tuple of (val_loss, val_accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Create progress bar if not provided
            if progress is None:
                progress = Progress(
                    SpinnerColumn(),
                    "•",
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=None),
                    TaskProgressColumn(),
                    "•",
                    TimeRemainingColumn(),
                    console=console,
                )
                task = progress.add_task("[green]Validating", total=len(self.val_loader))
                progress.start()
            else:
                task = progress.add_task("[green]Validating", total=len(self.val_loader))
            
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress
                progress.update(task, advance=1)
            
            # Calculate epoch metrics
            avg_loss = val_loss / len(self.val_loader)
            accuracy = 100. * correct / total
            
            # Clean up progress if we created it
            if progress is not None and len(progress.tasks) == 1:
                progress.stop()
            
            return avg_loss, accuracy
    
    def train(self) -> Dict[str, Any]:
        """Train the model for the specified number of epochs.
        
        Returns:
            Dictionary containing training metrics and history
        """
        best_val_acc = 0.0
        best_epoch = 0
        
        # Initialize metrics tracking
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        logger.info(f"Starting training for {self.epochs} epochs")
        
        # Set up progress bar
        with Progress(
            SpinnerColumn(),
            "•",
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeRemainingColumn(),
            "•",
            "Epoch: {task.fields[epoch]}",
            console=console
        ) as progress:
            epoch_task = progress.add_task(
                "[cyan]Training...", 
                total=self.epochs,
                epoch=0
            )
            
            for epoch in range(1, self.epochs + 1):
                # Update progress bar
                progress.update(epoch_task, advance=1, epoch=epoch)
                
                # Train for one epoch
                train_loss, train_acc = self.train_epoch(epoch, progress)
                
                # Validate
                val_loss, val_acc = self.validate(progress=progress)
                
                # Update learning rate scheduler if available
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Track metrics
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    if self.checkpoint_dir:
                        self.save_checkpoint(
                            self.checkpoint_dir / "best_model.pt",
                            val_acc,
                            epoch
                        )
                
                # Log metrics
                progress.print(
                    f"Epoch {epoch}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                    f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})"
                )
        
        # Save final model if checkpoint directory is specified
        if self.checkpoint_dir:
            self.save_checkpoint(
                self.checkpoint_dir / "final_model.pt",
                val_accs[-1] if val_accs else 0.0,
                self.epochs
            )
        
        # Prepare metrics
        metrics = {
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'final_epoch': self.epochs
        }
        
        # Save metrics if run directory is specified
        if self.run_dir:
            metrics_path = self.run_dir / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved training metrics to {metrics_path}")
        
        return metrics
    
    def test(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Test the model on the test set.
        
        Args:
            test_loader: Optional test data loader. If None, uses self.test_loader
            
        Returns:
            Dictionary of test metrics including loss, accuracy, and predictions
        """
        self.model.eval()
        test_loader = test_loader or self.test_loader
        
        if test_loader is None:
            raise ValueError("No test_loader provided and self.test_loader is None")
            
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store predictions and targets for further analysis
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        
        metrics = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'correct': correct,
            'total': total,
            'predictions': all_preds,
            'targets': all_targets
        }
        
        # Log results
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save metrics if run directory is specified
        if self.run_dir:
            # Save metrics as JSON
            metrics_path = self.run_dir / "test_metrics.json"
            # Convert numpy arrays to lists for JSON serialization
            metrics_to_save = metrics.copy()
            metrics_to_save['predictions'] = all_preds
            metrics_to_save['targets'] = all_targets
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
            logger.info(f"Saved test metrics to {metrics_path}")
            
            # Save predictions as numpy arrays
            np.save(self.run_dir / 'test_predictions.npy', np.array(all_preds))
            np.save(self.run_dir / 'test_targets.npy', np.array(all_targets))
        
        return metrics
        
    def predict(self, dataloader: DataLoader, return_probs: bool = False) -> Dict[str, Any]:
        """Generate predictions for a given dataloader.
        
        Args:
            dataloader: DataLoader for generating predictions
            return_probs: If True, returns class probabilities; otherwise returns class indices
            
        Returns:
            Dictionary containing predictions and optionally probabilities
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    inputs, targets = batch
                    all_targets.extend(targets.cpu().numpy())
                else:
                    inputs = batch[0]
                
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                if return_probs:
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy())
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
        
        result = {
            'predictions': np.array(all_preds),
            'targets': np.array(all_targets) if all_targets else None
        }
        
        if return_probs:
            result['probabilities'] = np.vstack(all_probs)
            
        return result
    
    def fit(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model for the specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        # Create progress bars
        with Progress(
            SpinnerColumn(),
            "•",
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            for epoch in range(num_epochs):
                # Train for one epoch
                train_loss, train_acc = self.train_epoch(epoch, progress)
                
                # Validate
                val_loss, val_acc = self.validate(progress)
                
                # Update learning rate if using scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Save model if validation accuracy improves
                if val_acc > best_val_acc and self.run_dir:
                    best_val_acc = val_acc
                    self.save_checkpoint(
                        self.run_dir / 'best_model.pth',
                        val_acc,
                        epoch
                    )
                
                # Update history
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Log metrics
                progress.console.print(Panel(
                    f"Epoch {epoch+1}/{num_epochs}\n"
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n"
                    f"Best Val Acc: {best_val_acc:.2f}%",
                    title=f"Epoch {epoch+1} Summary",
                    border_style="blue"
                ))
        
        return history
    
    def save_checkpoint(self, path: Union[str, Path], val_acc: float, epoch: int):
        """Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            val_acc: Validation accuracy
            epoch: Current epoch number
            
        Returns:
            Path where the checkpoint was saved
        """
        if isinstance(path, str):
            path = Path(path)
            
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'val_acc': val_acc,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'dataset': getattr(self, 'dataset', 'cifar10'),
            'num_classes': self.num_classes,
            'class_to_idx': getattr(self, 'class_to_idx', {i: str(i) for i in range(self.num_classes)})
        }
        
        try:
            # Save checkpoint
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint to {path} (Epoch {epoch}, Val Acc: {val_acc:.2f}%)")
            
            # Also save a copy as 'latest.pth' in the same directory
            latest_path = path.parent / 'latest.pth'
            torch.save(checkpoint, latest_path)
            
            return path
            
        except Exception as e:
            logger.error(f"Error saving checkpoint to {path}: {str(e)}")
            raise
    
    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            path: Path to the checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If there's an error loading the checkpoint
        """
        if isinstance(path, str):
            path = Path(path)
            
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
            
        logger.info(f"Loading checkpoint from {path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # Restore model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:  # For backward compatibility
                self.model.load_state_dict(checkpoint['state_dict'])
            
            # Restore optimizer state
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore scheduler state
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Update config if present in checkpoint
            if 'config' in checkpoint:
                self.config.update(checkpoint['config'])
            
            # Log success
            val_acc = checkpoint.get('val_acc', 0.0)
            epoch = checkpoint.get('epoch', 0)
            logger.info(f"Successfully loaded checkpoint from {path} (Epoch {epoch}, Val Acc: {val_acc:.2f}%)")
            
            return checkpoint
            
        except Exception as e:
            error_msg = f"Error loading checkpoint from {path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
            
    def lr_find(
        self, 
        start_lr: float = 1e-7, 
        end_lr: float = 10, 
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0
    ) -> Dict[str, List[float]]:
        """Learning rate range test.
        
        Args:
            start_lr: Starting learning rate
            end_lr: Maximum learning rate to test
            num_iter: Number of iterations to run
            smooth_f: Loss smoothing factor
            diverge_th: Threshold for stopping the test if loss diverges
            
        Returns:
            Dictionary containing learning rates and corresponding losses
        """
        # Save original model and optimizer state
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        
        # Initialize learning rates and loss tracking
        lrs = np.logspace(
            np.log10(start_lr), 
            np.log10(end_lr), 
            num=num_iter
        )
        losses = []
        best_loss = float('inf')
        
        # Set model to training mode
        self.model.train()
        
        # Get a batch of data
        data_iter = iter(self.train_loader)
        inputs, targets = next(data_iter)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Progress bar
        with Progress(
            SpinnerColumn(),
            "•",
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Finding learning rate...", total=num_iter)
            
            for i, lr in enumerate(lrs):
                # Update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Smooth the loss
                if losses:
                    loss = smooth_f * loss.item() + (1 - smooth_f) * losses[-1]
                else:
                    loss = loss.item()
                
                # Track best loss
                if loss < best_loss:
                    best_loss = loss
                
                # Stop if loss diverges
                if loss > diverge_th * best_loss:
                    logger.warning(f"Stopping early, loss has diverged (loss: {loss:.4f}, best: {best_loss:.4f})")
                    break
                
                losses.append(loss)
                progress.update(task, advance=1)
        
        # Restore original model and optimizer state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        
        return {
            'lrs': lrs[:len(losses)],
            'losses': losses
        }
    
    def get_learning_rate(self) -> float:
        """Get the current learning rate from the optimizer."""
        return self.optimizer.param_groups[0]['lr']
    
    def set_learning_rate(self, lr: float):
        """Set the learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_scheduler(self, name: str, **kwargs) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get a learning rate scheduler.
        
        Args:
            name: Name of the scheduler ('step', 'cosine', 'plateau', 'one_cycle')
            **kwargs: Additional arguments for the scheduler
            
        Returns:
            Learning rate scheduler or None if name is not recognized
        """
        if name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get('T_max', self.epochs),
                eta_min=kwargs.get('eta_min', 0)
            )
        elif name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.1),
                patience=kwargs.get('patience', 10),
                verbose=kwargs.get('verbose', True)
            )
        elif name == 'one_cycle':
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=kwargs.get('max_lr', 0.1),
                epochs=kwargs.get('epochs', self.epochs),
                steps_per_epoch=len(self.train_loader),
                pct_start=kwargs.get('pct_start', 0.3),
                anneal_strategy=kwargs.get('anneal_strategy', 'cos'),
                div_factor=kwargs.get('div_factor', 25.0),
                final_div_factor=kwargs.get('final_div_factor', 1e4)
            )
        return None

def create_cifar_loaders(
    dataset: str = 'cifar10',
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    data_fraction: float = 1.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create CIFAR data loaders.
    
    Args:
        dataset: Dataset name ('cifar10' or 'cifar100')
        data_dir: Directory to store the dataset
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        data_fraction: Fraction of training data to use (0.0 to 1.0)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=test_transform
        )
        num_classes = 10
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            data_dir, train=False, download=True, transform=test_transform
        )
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Split training set into train/val (90/10)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Apply data fraction if needed
    if data_fraction < 1.0:
        subset_size = int(len(train_dataset) * data_fraction)
        train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, num_classes
