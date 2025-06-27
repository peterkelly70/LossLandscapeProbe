#!/usr/bin/env python3
"""
CIFAR Training Module
===================

Model training with predicted hyperparameters for CIFAR datasets.
This module handles:
- Training the model with the best hyperparameters
- Evaluation and testing
- Saving model checkpoints and metrics
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import random
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Union

from .cifar_core import create_model, create_optimizer, get_cifar_loaders

logger = logging.getLogger(__name__)


class CIFARTrainer:
    """
    Trainer for CIFAR datasets.
    
    This class handles the training of CIFAR models with the best hyperparameters
    predicted by the meta-model.
    """
    
    def _setup_logging(self):
        """Set up logging to file in the appropriate directory."""
        if not self.run_dir:
            return
        
        # Set up file handler for logging
        sample_size = int(self.data_fraction * 100)
        log_file = self.run_dir / f"{self.dataset}_{sample_size}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Add file handler to the logger
        logger.addHandler(file_handler)
        
    def __init__(
        self, 
        config: Dict[str, Any],
        dataset: str = 'cifar10',
        data_fraction: float = 1.0,
        batch_size: int = 128,
        epochs: int = 100,
        run_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Hyperparameter configuration
            dataset: Dataset name ('cifar10' or 'cifar100')
            data_fraction: Fraction of training data to use
            batch_size: Batch size for training
            epochs: Number of epochs for training
            run_dir: Directory to save results and reports
            checkpoint_dir: Directory to save model checkpoints (defaults to run_dir/checkpoints)
        """
        self.config = config
        self.dataset = dataset
        self.data_fraction = data_fraction
        self.batch_size = batch_size
        self.epochs = epochs
        self.run_dir = run_dir
        
        # Determine checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        elif run_dir:
            self.checkpoint_dir = run_dir / "checkpoints"
        else:
            self.checkpoint_dir = None
            
        # Set up logging
        self._setup_logging()
        
        # Load dataset
        self.train_loader, self.val_loader, self.num_classes = get_cifar_loaders(
            dataset=dataset,
            data_fraction=data_fraction,
            batch_size=batch_size
        )
        
        # Initialize device
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        logger.info(f"Using device: {self.device}")
        
        # Create model and optimizer
        self.model = create_model(config, num_classes=self.num_classes).to(self.device)
        self.optimizer = create_optimizer(self.model, config)
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize metrics
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch):
        """
        Train for one epoch with enhanced progress reporting.
        
        Args:
            epoch: Current epoch number
            
        Returns:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar if not provided
        if progress is None:
            from tqdm import tqdm
            progress = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                          desc=f'Epoch {epoch + 1}/{self.epochs}', ncols=100)
        
        for batch_idx, (inputs, targets) in progress:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss += loss.item()
            
            # Update progress bar if it's a tqdm instance
            if hasattr(progress, 'set_postfix'):
                progress.set_postfix({
                    'loss': f"{running_loss / (batch_idx + 1):.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })
            
            # If using rich progress, update the batch progress
            if hasattr(progress, 'update_batch'):
                progress.update_batch(batch_idx, len(self.train_loader))
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc
    
    def validate(self, epoch, progress: Optional[Any] = None):
        """Validate the model with rich progress tracking."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar if not provided
        if progress is None:
            from tqdm import tqdm
            progress = tqdm(enumerate(self.val_loader), total=len(self.val_loader), 
                          desc='Validating', ncols=100)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in progress:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar if it's a tqdm instance
                if hasattr(progress, 'set_postfix'):
                    progress.set_postfix({
                        'loss': f"{val_loss / (batch_idx + 1):.4f}",
                        'acc': f"{100. * correct / total:.2f}%"
                    })
                
                # If using rich progress, update the batch progress
                if hasattr(progress, 'update_batch'):
                    progress.update_batch(batch_idx, len(self.val_loader))
                    current_loss = running_loss / total
                    current_acc = 100. * correct / total
                    
                    log_msg = (
                        f"Validating | "
                        f"Batch: {batch_idx+1:4d}/{total_batches} [{progress:6.1f}%] | "
                        f"Time: {time.time()-start_time:.1f}s | "
                        f"ETA: {eta_str} | "
                        f"Loss: {current_loss:.4f} | "
                        f"Acc: {current_acc:6.2f}%"
                    )
                    logger.info(log_msg)
                    # Also print to console for better visibility
                    print(f"\r{log_msg}", end="" if batch_idx < total_batches - 1 else "\n")
        
        # Calculate final metrics
        val_loss = running_loss / total
        val_acc = correct / total
        
        logger.info(f"Validation complete - Loss: {val_loss:.4f}, Acc: {100*val_acc:.2f}%")
        return val_loss, val_acc
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting training with config: {self.config}")
        logger.info(f"Training for {self.epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            # Update metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Check if this is the best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                
                # Save the best model
                if self.checkpoint_dir:
                    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pth")
                    logger.info(f"Saved best model at epoch {epoch} with validation accuracy {val_acc:.4f}")
            
            # Log progress
            logger.info(f"Epoch {epoch}/{self.epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                       f"Best Val Acc: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
            
            # Save checkpoint every 10 epochs
            if self.checkpoint_dir and epoch % 10 == 0:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'best_val_acc': self.best_val_acc,
                    'best_epoch': self.best_epoch
                }, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")
        
        # Training completed
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.checkpoint_dir / "final_model.pth")
        
        # Save metrics
        metrics = {
            'train_loss': self.train_losses,
            'train_acc': self.train_accs,
            'val_loss': self.val_losses,
            'val_acc': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'training_time': training_time,
            'config': self.config
        }
        
        if self.run_dir:
            with open(self.run_dir / "training_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        
        return metrics
    
    def plot_training_history(self):
        """
        Plot the training history.
        
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        ax1.plot(self.train_losses, label='Train')
        ax1.plot(self.val_losses, label='Validation')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.train_accs, label='Train')
        ax2.plot(self.val_accs, label='Validation')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the figure
        if self.run_dir:
            plt.savefig(self.run_dir / "training_history.png")
        
        return fig
    
    def _denormalize_image(self, img_tensor):
        """Denormalize image tensor for visualization."""
        # CIFAR-10/100 mean and std
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(img_tensor.device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(img_tensor.device)
        
        # Reverse normalization: (x - mean) / std -> x * std + mean
        img_tensor = img_tensor * std + mean
        
        # Clip to [0, 1] range
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Convert to numpy and scale to 0-255
        img_np = img_tensor.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
        img_np = (img_np * 255).astype(np.uint8)
        
        return img_np

    def _save_prediction_samples(self, images, true_labels, pred_labels, class_names, num_samples=100):
        """
        Save actual sample images from the dataset with prediction results.
        
        Args:
            images: Tensor of input images (B, C, H, W)
            true_labels: Tensor of true labels
            pred_labels: Tensor of predicted labels
            class_names: List of class names
            num_samples: Maximum number of samples to save
                
        Returns:
            Path to the generated HTML file
        """
        if not self.run_dir:
            return
            
        # Create dataset directory first (e.g., cifar10, cifar100)
        dataset_dir = self.run_dir / self.dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Then create sample size subdirectory (e.g., 10, 20, etc.)
        sample_size = int(self.data_fraction * 100)
        sample_dir = dataset_dir / str(sample_size)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create samples directory inside the sample size directory
        samples_dir = sample_dir / "prediction_samples"
        samples_dir.mkdir(exist_ok=True)
            
        # Save confusion matrix
        cm = confusion_matrix(true_labels.numpy(), pred_labels.numpy())
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, 
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(samples_dir / "confusion_matrix.png", bbox_inches='tight')
        plt.close()

        # Create HTML content with properly escaped CSS
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Prediction Samples</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style type="text/css">
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            background-color: #f9f9f9;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
            margin-top: 1.5em;
        }
        .confusion-matrix {
            margin: 20px 0;
        }
        .sample {
            display: inline-block;
            margin: 10px;
            text-align: center;
        }
        .sample img {
            max-width: 64px;
            max-height: 64px;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        .correct {
            border: 2px solid #2ecc71 !important;
        }
        .incorrect {
            border: 2px solid #e74c3c !important;
        }
        .stats {
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Prediction Samples</h1>
        <div class="stats">
            <p>Model: {model_name}</p>
            <p>Dataset: {dataset}</p>
            <p>Sample Size: {sample_size}% of training data</p>
            <p>Accuracy on this sample: {accuracy:.2f}%</p>
        </div>
        <div class="samples">
            {samples_html}
        </div>
    </div>
</body>
</html>
""".format(
            model_name=self.config.get('model_name', 'Unknown'),
            dataset=self.dataset,
            sample_size=int(self.data_fraction * 100),
            accuracy=sample_accuracy,
            samples_html=samples_html
        )
            text-align: center;
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .confusion-matrix img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .samples-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .sample {
            border: 3px solid #4CAF50;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            background: white;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .sample:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .sample.incorrect { 
            border-color: #f44336;
        }
        .sample img { 
            width: 100%; 
            height: auto;
            image-rendering: pixelated;
            border-radius: 4px;
        }
        .sample-info {
            margin-top: 10px;
            padding: 5px;
        }
        .true-label {
            font-weight: bold;
            font-size: 14px;
            margin: 5px 0;
        }
        .prediction {
            font-size: 13px;
            color: #666;
        }
        .correct { 
            color: #4CAF50;
            font-weight: bold;
        }
        .incorrect { 
            color: #f44336;
            font-weight: bold;
        }
        .stats {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .stats p {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Prediction Samples</h1>
        <p>Showing {num_samples} random samples. Green border = correct prediction, Red border = incorrect prediction.</p>
        
        <div class="stats">
            <h3>Dataset: {self.dataset.upper()}</h3>
            <p>Model trained on {int(self.data_fraction*100)}% of training data</p>
        </div>
        
        <div class="samples-container">
"""
        
        # Randomly sample indices to show using numpy's Generator with fixed seed for reproducibility
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(images), size=min(num_samples, len(images)), replace=False)
        correct_count = 0
            
        for i, idx in enumerate(indices):
            img_tensor = images[idx]
            true_label = int(true_labels[idx].item())
            pred_label = int(pred_labels[idx].item())
            is_correct = (true_label == pred_label)
                
            if is_correct:
                correct_count += 1
                
            # Denormalize and convert to numpy array
            img_np = self._denormalize_image(img_tensor)
            
            # Convert to PIL Image and save
            img = Image.fromarray(img_np)
            img_filename = f"sample_{idx:04d}.png"
            img_path = samples_dir / img_filename
            img.save(img_path)
                
            # Add to HTML
            html_content += f"""
            <div class="sample {"correct" if is_correct else "incorrect"}">
                <img src="{img_filename}" alt="Sample {i+1}">
                <div class="sample-info">
                    <div class="true-label">{class_names[true_label]}</div>
                    <div class="prediction">→ <span class="{"correct" if is_correct else "incorrect"}">{class_names[pred_label]}</span></div>
                </div>
            </div>
"""
            
        # Calculate accuracy for this sample
        sample_accuracy = correct_count / len(indices) * 100
            
        # Close HTML
        html_content += f"""
        </div>
            
        <div class="stats">
            <h3>Sample Statistics</h3>
            <p>Accuracy on shown samples: <strong>{sample_accuracy:.1f}%</strong> ({correct_count} correct out of {len(indices)})</p>
        </div>
            
        <div class="confusion-matrix">
            <h2>Confusion Matrix</h2>
            <img src="confusion_matrix.png" alt="Confusion Matrix">
            <p>The confusion matrix shows how often the model's predictions match the true labels.</p>
        </div>
    </div>
</body>
</html>
"""
            
        # Save HTML
        with open(samples_dir / "index.html", "w") as f:
            f.write(html_content)
            
        logger.info(f"Saved {len(indices)} prediction samples to {samples_dir}")
        return samples_dir / "index.html"
        
    def _generate_test_report(self, test_metrics, _samples_html_path=None, _class_names=None):
        """
        Generate an HTML test report with metrics and links to samples.
        
        Args:
            test_metrics: Dictionary containing test metrics
            _samples_html_path: Path to the prediction samples HTML file
            _class_names: List of class names (unused, kept for backward compatibility)
        """
        if not self.run_dir:
            return
            
        # Create dataset directory first (e.g., cifar10, cifar100)
        dataset_dir = self.run_dir / self.dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Then create sample size subdirectory (e.g., 10, 20, etc.)
        sample_size = int(self.data_fraction * 100)
        sample_dir = dataset_dir / str(sample_size)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = sample_dir / f"{self.dataset}_{sample_size}_test_report.html"
        
        # Calculate class accuracy table
        class_acc_rows = ""
        for class_name, acc in test_metrics['class_acc'].items():
            class_acc_rows += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{acc*100:.2f}%</td>
                </tr>"""
        
        # Create HTML report
        html_content = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>{self.dataset.upper()} Test Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0;
                    padding: 20px;
                    color: #333;
                    line-height: 1.6;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{ 
                    color: #2c3e50;
                    margin-top: 1.5em;
                }}
                .metrics {{ 
                    margin: 20px 0;
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                }}
                table {{ 
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{ 
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{ 
                    background-color: #3498db;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .confusion-matrix {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .confusion-matrix img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .samples-link {{
                    margin: 30px 0;
                    padding: 15px;
                    background: #e8f4fc;
                    border-left: 5px solid #3498db;
                }}
                .samples-link a {{
                    color: #3498db;
                    text-decoration: none;
                    font-weight: bold;
                }}
                .samples-link a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{self.dataset.upper()} Test Report</h1>
                <p>Model trained on {int(self.data_fraction*100)}% of training data</p>
                
                <div class="metrics">
                    <h2>Overall Metrics</h2>
                    <p>Test Loss: {test_metrics['test_loss']:.4f}</p>
                    <p>Test Accuracy: {test_metrics['test_acc']*100:.2f}%</p>
                </div>
                
                <div class="confusion-matrix">
                    <h2>Confusion Matrix</h2>
                    <img src="prediction_samples/confusion_matrix.png" alt="Confusion Matrix">
                    <p>The confusion matrix shows how often the model's predictions match the true labels.</p>
                </div>
                
                <h2>Class-wise Accuracy</h2>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Accuracy</th>
                    </tr>
                    {class_acc_rows}
                </table>
                
                <div class="samples-link">
                    <h2>Prediction Samples</h2>
                    <p>View detailed prediction samples: <a href="prediction_samples/index.html">Open Prediction Samples</a></p>
                </div>
            </div>
        </body>
        </html>"""
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Test report generated at {report_path}")
        return report_path

    def _get_class_names(self):
        """Get class names based on the dataset type."""
        if self.dataset == 'cifar10':
            return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Using the 20 superclasses for CIFAR-100
        return [
            'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
            'household electrical devices', 'household furniture', 'insects', 'large carnivores',
            'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
            'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
            'trees', 'vehicles 1', 'vehicles 2'
        ]

    def _load_best_model(self):
        """Load the best model if available."""
        best_model_filename = "best_model.pth"
        best_model_path = self.checkpoint_dir / best_model_filename if self.checkpoint_dir else None
        
        if best_model_path and best_model_path.exists():
            logger.info(f"Loading best model from {best_model_path}")
            self.model.load_state_dict(torch.load(best_model_path))
            return True
        
        logger.info("Best model not found, using current model state")
        return False

    def _evaluate_batch(self, inputs, targets, running_loss, class_correct, class_total, 
                       all_images, all_true_labels, all_pred_labels):
        """Evaluate a single batch and update metrics."""
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        running_loss[0] += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        
        # Update overall accuracy
        total_correct = predicted.eq(targets).sum().item()
        
        # Store samples for visualization
        all_images.append(inputs.cpu())
        all_true_labels.append(targets.cpu())
        all_pred_labels.append(predicted.cpu())
        
        # Update per-class accuracy
        correct_mask = (predicted == targets).squeeze()
        for i in range(targets.size(0)):
            label = targets[i].item()
            class_correct[label] += correct_mask[i].item()
            class_total[label] += 1
        
        return total_correct, inputs.size(0)

    def _calculate_class_accuracy(self, class_correct, class_total, class_names):
        """Calculate and log per-class accuracy."""
        class_acc = {}
        for i in range(self.num_classes):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                class_name = class_names[i] if i < len(class_names) else f"Class {i}"
                logger.info(f"Accuracy of {class_name}: {100 * acc:.2f}%")
                class_acc[class_name] = acc
        return class_acc

    def _save_test_results(self, test_metrics, all_images, all_true_labels, all_pred_labels, class_names):
        """Save test results and generate reports."""
        if not self.run_dir:
            return
            
        # Create dataset directory first (e.g., cifar10, cifar100)
        dataset_dir = self.run_dir / self.dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Then create sample size subdirectory (e.g., 10, 20, etc.)
        sample_size = int(self.data_fraction * 100)
        sample_dir = dataset_dir / str(sample_size)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create samples directory inside the sample size directory
        samples_dir = sample_dir / "prediction_samples"
        samples_dir.mkdir(exist_ok=True)
        
        # Save metrics in the sample size directory
        metrics_path = sample_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test metrics saved to {metrics_path}")
        
        # Save prediction samples and generate reports
        html_path = self._save_prediction_samples(
            all_images, all_true_labels, all_pred_labels, 
            class_names, num_samples=200
        )
        self._generate_test_report(test_metrics, html_path, class_names)

    def test(self):
        """
        Test the model on the test set.
        
        Returns:
            Dictionary with test metrics
        """
        logger.info("Testing the model")
        
        # Get class names and ensure we have enough
        class_names = self._get_class_names()
        if len(class_names) < self.num_classes:
            class_names.extend([f'Class {i}' for i in range(len(class_names), self.num_classes)])
        
        # Load best model if available
        self._load_best_model()
        
        # Initialize metrics
        self.model.eval()
        running_loss = [0.0]  # Using list for mutable reference
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        # Store samples for visualization
        all_images = []
        all_true_labels = []
        all_pred_labels = []
        
        # Evaluate on test set
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            
            for inputs, targets in self.val_loader:
                batch_correct, batch_size = self._evaluate_batch(
                    inputs, targets, running_loss, 
                    class_correct, class_total,
                    all_images, all_true_labels, all_pred_labels
                )
                total_correct += batch_correct
                total_samples += batch_size
        
        # Calculate metrics
        test_loss = running_loss[0] / total_samples
        test_acc = total_correct / total_samples
        
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Calculate per-class accuracy
        class_acc = self._calculate_class_accuracy(class_correct, class_total, class_names)
        
        # Prepare test metrics
        test_metrics = {
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'class_acc': class_acc
        }
        
        # Save results and generate reports
        all_images = torch.cat(all_images)
        all_true_labels = torch.cat(all_true_labels)
        all_pred_labels = torch.cat(all_pred_labels)
        
        self._save_test_results(
            test_metrics, all_images, all_true_labels, all_pred_labels, class_names
        )
        
        return test_metrics
