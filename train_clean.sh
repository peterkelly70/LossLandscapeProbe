#!/bin/bash

# Set environment variables
export PYTHONPATH="$PWD:$PYTHONPATH"

# Default parameters
DATASET="cifar10"
SAMPLE_SIZE=0.1
EPOCHS=100
BATCH_SIZE=128
LEARNING_RATE=0.1
WEIGHT_DECAY=5e-4
MOMENTUM=0.9
NUM_WORKERS=4
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --momentum)
            MOMENTUM="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

echo "Starting CIFAR training with the following parameters:"
echo "  Dataset: $DATASET"
echo "  Sample size: $SAMPLE_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Momentum: $MOMENTUM"
echo "  Num workers: $NUM_WORKERS"
echo "  Random seed: $SEED"

# Run the training script
python3 - << EOF
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet18
from pathlib import Path
from datetime import datetime

# Rich imports for better console output
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

# Initialize console
console = Console()

# Set random seeds for reproducibility
torch.manual_seed($SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(f"[bold]Using device:[/] {device}")

# Import from our package
try:
    from llp import get_model, get_data_loaders
    from llp.models import CIFARModel, SimpleCNN
    from llp.data import get_data_loaders_simple
    console.print("[green]✓ Successfully imported llp modules[/]")
except ImportError as e:
    console.print(f"[bold red]Error importing llp module: {e}[/]")
    console.print("Python path:", sys.path)
    sys.exit(1)

# Define parameters from shell variables
DATASET = "$DATASET"
SAMPLE_SIZE = float("$SAMPLE_SIZE")
EPOCHS = int("$EPOCHS")
BATCH_SIZE = int("$BATCH_SIZE")
LEARNING_RATE = float("$LEARNING_RATE")
WEIGHT_DECAY = float("$WEIGHT_DECAY")
MOMENTUM = float("$MOMENTUM")
NUM_WORKERS = int("$NUM_WORKERS")
SEED = int("$SEED")

# Create output directories
model_dir = Path(f"models/{DATASET}/{DATASET}_{int(float($SAMPLE_SIZE)*100)}")
model_dir.mkdir(parents=True, exist_ok=True)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Set up logging
log_file = log_dir / f"training_{DATASET}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
console.print(f"[bold]Log file:[/] {log_file}")

def log(message):
    """Log message to console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    console.print(log_message)
    with open(log_file, 'a') as f:
        f.write(f"{log_message}\n")

# Set up data loaders
try:
    log(f"Loading {DATASET} dataset...")
    train_loader, val_loader = get_data_loaders(
        dataset=DATASET,
        batch_size=int(BATCH_SIZE),
        num_workers=int(NUM_WORKERS),
        sample_size=float(SAMPLE_SIZE)
    )
    log(f"  Training samples: {len(train_loader.dataset)}")
    log(f"  Validation samples: {len(val_loader.dataset)}")
except Exception as e:
    log(f"[bold red]Error loading data: {e}[/]")
    sys.exit(1)

# Initialize model
model = CIFARModel(num_classes=10 if DATASET == 'cifar10' else 100).to(device)
optimizer = optim.SGD(
    model.parameters(),
    lr=float($LEARNING_RATE),
    momentum=float($MOMENTUM),
    weight_decay=float($WEIGHT_DECAY)
)
criterion = nn.CrossEntropyLoss()

# Training function
def train_epoch(model, loader, optimizer, criterion, epoch, progress):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    task = progress.add_task(f"[cyan]Epoch {epoch + 1}", total=len(loader))
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress
        progress.update(task, advance=1, description=f"[cyan]Epoch {epoch + 1} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Validation function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Set up progress tracking
with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("•"),
    TextColumn("ETA: {task.fields[eta]}", justify="right"),
    TimeRemainingColumn(),
    console=console
) as progress:
    # Main training loop
    best_val_acc = 0.0
    
    for epoch in range(int($EPOCHS)):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch, progress)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Log metrics
        log(f"Epoch {epoch+1:3d}/{int($EPOCHS)} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_dir / 'best_model.pth')
            log(f"[green]Saved best model with validation accuracy: {val_acc:.2f}%[/]")
    
    # Save final model
    torch.save({
        'epoch': int($EPOCHS),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, model_dir / 'final_model.pth')
    log(f"[green]Training completed. Final model saved to {model_dir}/final_model.pth[/]")

EOF
