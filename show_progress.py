#!/usr/bin/env python3
"""
Simple Progress Monitor for CIFAR Training
Shows real-time progress of the training process
"""

import os
import sys
import time
import subprocess
from datetime import datetime

# Check if training is running
def is_training_running():
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    return "unified_cifar_training.py" in result.stdout

# Get the last N lines of the log file
def get_last_lines(file_path, n=20):
    try:
        result = subprocess.run(["tail", "-n", str(n), file_path], capture_output=True, text=True)
        return result.stdout.splitlines()
    except Exception:
        return []

# Main monitoring function
def monitor_progress():
    log_file = "reports/cifar10/cifar10_10/training.log"
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print("Make sure training has started.")
        return
    
    print(f"🔍 Monitoring training progress from: {log_file}")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        last_size = 0
        while True:
            # Clear screen for better visibility
            os.system('clear')
            
            # Check if training is still running
            running = is_training_running()
            status = "🟢 RUNNING" if running else "🔴 STOPPED"
            
            # Get current file size
            current_size = os.path.getsize(log_file)
            
            # Print header with status
            print(f"{'=' * 80}")
            print(f"CIFAR TRAINING PROGRESS MONITOR - Status: {status}")
            print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 80}\n")
            
            # Show activity indicator if file size has changed
            if current_size > last_size:
                print(f"📊 Training is active - Log file growing: {current_size - last_size} bytes since last check")
                last_size = current_size
            
            # Get and display the last lines of the log
            lines = get_last_lines(log_file)
            
            # Extract and display progress information
            config_info = None
            epoch_info = None
            accuracy_info = None
            
            for line in lines:
                if "Evaluating configuration" in line:
                    config_info = line
                if "Epoch" in line and "/" in line:
                    epoch_info = line
                if "accuracy:" in line or "Accuracy:" in line:
                    accuracy_info = line
            
            print("\n📋 PROGRESS SUMMARY:")
            if config_info:
                print(f"  ▶️ {config_info.strip()}")
            if epoch_info:
                print(f"  ▶️ {epoch_info.strip()}")
            if accuracy_info:
                print(f"  ▶️ {accuracy_info.strip()}")
            
            print("\n📜 RECENT LOG ENTRIES:")
            for line in lines:
                print(f"  {line.strip()}")
            
            # Wait before checking again
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped. Training continues in the background.")
        print("Run this script again anytime to resume monitoring.")

if __name__ == "__main__":
    monitor_progress()
