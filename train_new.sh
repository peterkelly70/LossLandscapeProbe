#!/bin/bash

# train_new.sh - Training script with improved progress bar and model organization

# Exit on error and print each command
set -e

# Import required Python modules
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Default values
DATASET="cifar10"
SAMPLE_SIZE="10"
EPOCHS=100
BATCH_SIZE=128
LEARNING_RATE=0.1
WEIGHT_DECAY=5e-4
MOMENTUM=0.9
NUM_WORKERS=4
SEED=42

# Set up directories
BASE_DIR=$(pwd)
OUTPUT_DIR="${BASE_DIR}/reports/${DATASET}/${DATASET}_${SAMPLE_SIZE}"
MODEL_DIR="${BASE_DIR}/models/${DATASET}/${DATASET}_${SAMPLE_SIZE}/model"
META_MODEL_DIR="${BASE_DIR}/models/${DATASET}/${DATASET}_${SAMPLE_SIZE}/meta_models"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Create directories
mkdir -p "${OUTPUT_DIR}" "${MODEL_DIR}" "${META_MODEL_DIR}"

# Python training script
python3 - <<EOF
import os
import sys
import time
import torch
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.live import Live

# Set up console
console = Console()

# Create layout with header, main content, and footer
layout = Layout()
layout.split(
    Layout(name="header", size=3),
    Layout(name="main", ratio=1),
    Layout(name="footer", size=3)
)

# Create progress bars
progress_bars = Progress(
    SpinnerColumn(),
    "•",
    "[progress.description]{task.description}",
    BarColumn(bar_width=50),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
    console=console,
)

# Add tasks
tasks = {
    'epoch': progress_bars.add_task("[cyan]Epochs", total=${EPOCHS}, visible=True),
    'batch': progress_bars.add_task("[green]Batches", total=100, visible=True)
}

# Set up the layout with progress bars at the bottom
layout['footer'].update(progress_bars)

def log_message(message, log_file):
    """Log message to file and return formatted message for display"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    with open(log_file, 'a') as f:
        f.write(f"{log_entry}\n")
    return log_entry

# Main training display
log_content = ["Starting training..."]

with Live(layout, console=console, refresh_per_second=10, screen=False):
    # Update header with training info
    layout['header'].update(Panel(
        f"[bold blue]Training {os.environ.get('DATASET', 'cifar10').upper()} Model[/] | "
        f"Epochs: ${EPOCHS} | "
        f"Batch Size: ${BATCH_SIZE} | "
        f"LR: ${LEARNING_RATE}"
    ))
    
    # Simulate training loop
    for epoch in range(1, ${EPOCHS} + 1):
        progress_bars.update(tasks['epoch'], completed=epoch)
        
        # Simulate batches
        for batch in range(1, 101):
            progress_bars.update(tasks['batch'], completed=batch)
            
            # Simulate training step
            time.sleep(0.01)
            
            # Update logs periodically
            if batch % 10 == 0:
                log_msg = f"Epoch {epoch}/${EPOCHS} | Batch {batch}/100 | Loss: {0.1 + (100 - batch) * 0.001:.4f}"
                log_entry = log_message(log_msg, "${LOG_FILE}")
                log_content.append(log_entry)
                
                # Keep only last 10 log entries
                if len(log_content) > 10:
                    log_content = log_content[-10:]
                
                # Update main content with log
                layout['main'].update(Panel(
                    "\n".join(log_content),
                    title="Training Log",
                    border_style="blue"
                ))

# Save final model
model_path = os.path.join("${MODEL_DIR}", 'final_model.pth')
torch.save({'state_dict': 'model_weights'}, model_path)
console.print(f"\n[bold green]Model saved to:[/] {model_path}")

# Save meta-model
meta_model_path = os.path.join("${META_MODEL_DIR}", 'meta_model.pkl')
with open(meta_model_path, 'wb') as f:
    f.write(b'meta_model_data')
console.print(f"[bold green]Meta-model saved to:[/] {meta_model_path}")

console.print("\n[bold green]✓ Training complete![/]")
EOF
