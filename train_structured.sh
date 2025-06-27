#!/bin/bash

# Set environment variables
export PYTHONPATH="$PWD:$PYTHONPATH"

# Default parameters
DATASET="cifar10"
SAMPLE_SIZE=10
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
        --dataset) DATASET="$2"; shift 2 ;;
        --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --momentum) MOMENTUM="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Convert sample size to float for Python
SAMPLE_SIZE_FLOAT=$(echo "scale=2; $SAMPLE_SIZE/100" | bc)

# Set up directory names
MODEL_DIR="models/${DATASET}/${DATASET}_${SAMPLE_SIZE}/model"
META_MODEL_DIR="models/${DATASET}/${DATASET}_${SAMPLE_SIZE}/meta_models"
REPORT_DIR="reports/${DATASET}/${DATASET}_${SAMPLE_SIZE}"

# Create directories
mkdir -p "$MODEL_DIR"
mkdir -p "$META_MODEL_DIR"
mkdir -p "$REPORT_DIR"

echo "Starting training with the following parameters:"
echo "  Dataset: $DATASET"
echo "  Sample size: $SAMPLE_SIZE%"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Momentum: $MOMENTUM"
echo "  Num workers: $NUM_WORKERS"
echo "  Random seed: $SEED"
echo "  Model directory: $MODEL_DIR"
echo "  Report directory: $REPORT_DIR"

# Run the training script
python3 - << EOF
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

# Set up paths
MODEL_DIR = "$MODEL_DIR"
META_MODEL_DIR = "$META_MODEL_DIR"
REPORT_DIR = "$REPORT_DIR"

# Create directories
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(META_MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

# Set up logging
LOG_FILE = os.path.join(REPORT_DIR, "training.log")
console.print(f"[bold]Log file:[/] {LOG_FILE}")

def log(message):
    """Log message to console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    console.print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{log_message}\n")

# Import from our package
try:
    from llp import get_model, get_data_loaders
    from llp.models import CIFARModel, SimpleCNN
    from llp.data import get_data_loaders_simple
    log("Successfully imported llp modules")
except ImportError as e:
    log(f"[bold red]Error importing llp module: {e}")
    log(f"Python path: {sys.path}")
    sys.exit(1)

# Set up data loaders
try:
    log(f"Loading {os.path.basename(REPORT_DIR)} dataset...")
    train_loader, val_loader = get_data_loaders_simple(
        dataset="$DATASET",
        batch_size=int("$BATCH_SIZE"),
        num_workers=int("$NUM_WORKERS"),
        sample_size=float("$SAMPLE_SIZE_FLOAT")
    )
    log(f"Training samples: {len(train_loader.dataset)}")
    log(f"Validation samples: {len(val_loader.dataset)}")
except Exception as e:
    log(f"[bold red]Error loading data: {e}")
    sys.exit(1)

# Initialize model
model = CIFARModel(num_classes=10 if "$DATASET" == 'cifar10' else 100).to(device)
optimizer = optim.SGD(
    model.parameters(),
    lr=float("$LEARNING_RATE"),
    momentum=float("$MOMENTUM"),
    weight_decay=float("$WEIGHT_DECAY")
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
        progress.update(task, advance=1, description=f"[cyan]Epoch {epoch + 1} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {100.*correct/total:.2f}%")
    
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
    
    for epoch in range(int("$EPOCHS")):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch, progress)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Log metrics
        log(f"Epoch {epoch+1:3d}/{int('$EPOCHS')} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join("$MODEL_DIR", 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            log(f"[green]Saved best model to {checkpoint_path} with validation accuracy: {val_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join("$MODEL_DIR", 'final_model.pth')
    torch.save({
        'epoch': int("$EPOCHS"),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, final_model_path)
    log(f"[green]Training completed. Final model saved to {final_model_path}")

    # Generate test report
    test_report_path = os.path.join("$REPORT_DIR", 'test_report.html')
    with open(test_report_path, 'w') as f:
        f.write(f"""
        <html>
        <head><title>Test Report - {os.path.basename(REPORT_DIR)}</title></head>
        <body>
            <h1>Training Report</h1>
            <h2>Model: {os.path.basename(REPORT_DIR)}</h2>
            <h3>Final Metrics</h3>
            <ul>
                <li>Best Validation Accuracy: {best_val_acc:.2f}%</li>
                <li>Final Validation Accuracy: {val_acc:.2f}%</li>
                <li>Final Validation Loss: {val_loss:.4f}</li>
            </ul>
            <h3>Training Parameters</h3>
            <ul>
                <li>Epochs: {$EPOCHS}</li>
                <li>Batch Size: {$BATCH_SIZE}</li>
                <li>Learning Rate: {$LEARNING_RATE}</li>
                <li>Weight Decay: {$WEIGHT_DECAY}</li>
                <li>Momentum: {$MOMENTUM}</li>
            </ul>
        </body>
        </html>
        """)
    log(f"[green]Test report generated at {test_report_path}")

EOF
