#!/usr/bin/env python3
"""
Active Training Monitor for CIFAR Meta-Model

This script actively monitors the training process and displays clear progress indicators.
It follows the fail-fast approach by directly checking process status and log files.
"""

import os
import sys
import time
import subprocess
import re
from pathlib import Path
from datetime import datetime

# ANSI color codes for better visibility
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(message):
    """Print a formatted header message"""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE}{message}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 80}{RESET}")

def print_status(message, status="INFO"):
    """Print a status message with appropriate color"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if status == "INFO":
        color = BLUE
    elif status == "SUCCESS":
        color = GREEN
    elif status == "WARNING":
        color = YELLOW
    elif status == "ERROR":
        color = RED
    else:
        color = RESET
    
    print(f"{timestamp} - {color}{status}{RESET} - {message}")

def check_training_process():
    """Check if the training process is running"""
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    return "unified_cifar_training.py" in result.stdout

def get_log_content(log_path, last_n_lines=None):
    """Get the content of the log file"""
    if not log_path.exists():
        return []
    
    with open(log_path, "r") as f:
        lines = f.readlines()
    
    if last_n_lines:
        return lines[-last_n_lines:]
    return lines

def extract_progress_from_logs(log_lines):
    """Extract progress information from log lines"""
    progress_info = {
        "phase": "Unknown",
        "configs_evaluated": 0,
        "total_configs": 0,
        "current_epoch": 0,
        "total_epochs": 0,
        "best_accuracy": 0.0,
        "last_update": None
    }
    
    # Regular expressions to extract information
    meta_model_start = re.compile(r"Starting meta-model hyperparameter optimization")
    config_pattern = re.compile(r"Evaluating configuration (\d+)/(\d+)")
    epoch_pattern = re.compile(r"Epoch (\d+)/(\d+)")
    accuracy_pattern = re.compile(r"accuracy: (\d+\.\d+)")
    
    for line in log_lines:
        # Check for phase changes
        if "Starting meta-model" in line:
            progress_info["phase"] = "Meta-Model Optimization"
        elif "Starting training with predicted" in line:
            progress_info["phase"] = "Model Training"
        elif "Testing the model" in line:
            progress_info["phase"] = "Model Testing"
        
        # Extract configuration progress
        config_match = config_pattern.search(line)
        if config_match:
            progress_info["configs_evaluated"] = int(config_match.group(1))
            progress_info["total_configs"] = int(config_match.group(2))
        
        # Extract epoch progress
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            progress_info["current_epoch"] = int(epoch_match.group(1))
            progress_info["total_epochs"] = int(epoch_match.group(2))
        
        # Extract accuracy
        accuracy_match = accuracy_pattern.search(line)
        if accuracy_match:
            accuracy = float(accuracy_match.group(1))
            progress_info["best_accuracy"] = max(progress_info["best_accuracy"], accuracy)
        
        # Update timestamp
        if line.strip():
            timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if timestamp_match:
                progress_info["last_update"] = timestamp_match.group(1)
    
    return progress_info

def print_progress_bar(current, total, desc="Progress", length=50):
    """Print a text-based progress bar"""
    percent = min(100.0, (current / total) * 100) if total > 0 else 0
    filled_length = int(length * current // total) if total > 0 else 0
    bar = '#' * filled_length + '-' * (length - filled_length)
    
    print(f"\r{BOLD}{desc}{RESET}: [{BLUE}{bar}{RESET}] {current}/{total} ({percent:.1f}%)", end="", flush=True)
    
    if current == total:
        print()  # Add newline when complete

def monitor_training():
    """Monitor the training process and display progress"""
    print_header("CIFAR META-MODEL TRAINING MONITOR")
    
    log_path = Path("reports/cifar10/cifar10_10/training.log")
    if not log_path.exists():
        print_status(f"Log file not found: {log_path}", "ERROR")
        print_status("Make sure training has started", "WARNING")
        return
    
    print_status(f"Monitoring log file: {log_path}")
    
    try:
        last_configs_evaluated = 0
        last_current_epoch = 0
        
        while True:
            # Check if training process is running
            is_running = check_training_process()
            
            # Get latest log content
            log_lines = get_log_content(log_path)
            
            # Extract progress information
            progress = extract_progress_from_logs(log_lines)
            
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Print header
            print_header("CIFAR META-MODEL TRAINING MONITOR")
            
            # Print status
            status = "RUNNING" if is_running else "STOPPED"
            status_color = GREEN if is_running else RED
            print(f"Status: {status_color}{status}{RESET}")
            print(f"Phase: {BOLD}{progress['phase']}{RESET}")
            print(f"Last update: {progress['last_update'] or 'N/A'}")
            print(f"Best accuracy: {GREEN}{progress['best_accuracy']:.4f}{RESET}")
            print()
            
            # Print progress bars
            if progress["total_configs"] > 0:
                print_progress_bar(
                    progress["configs_evaluated"], 
                    progress["total_configs"], 
                    "Configurations"
                )
                print()
            
            if progress["total_epochs"] > 0:
                print_progress_bar(
                    progress["current_epoch"], 
                    progress["total_epochs"], 
                    "Training Epochs"
                )
                print()
            
            # Print recent log entries
            print(f"\n{BOLD}Recent Log Entries:{RESET}")
            recent_logs = get_log_content(log_path, 10)
            for line in recent_logs:
                print(f"  {line.strip()}")
            
            # Check for updates to show activity
            if progress["configs_evaluated"] > last_configs_evaluated:
                last_configs_evaluated = progress["configs_evaluated"]
                print_status(f"Configuration {progress['configs_evaluated']}/{progress['total_configs']} evaluated", "SUCCESS")
            
            if progress["current_epoch"] > last_current_epoch:
                last_current_epoch = progress["current_epoch"]
                print_status(f"Completed epoch {progress['current_epoch']}/{progress['total_epochs']}", "SUCCESS")
            
            # Wait before checking again
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        print_status("You can restart monitoring at any time by running this script again", "INFO")

if __name__ == "__main__":
    monitor_training()
