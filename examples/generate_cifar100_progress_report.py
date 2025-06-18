#!/usr/bin/env python3
"""
Generate CIFAR-100 Training Progress Report
==========================================

This script generates a visualization report for CIFAR-100 training progress
using the visualize_cifar100_progress.py script. It takes the CIFAR-100 training
log data and generates an HTML report with visualizations of training metrics.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    parser = argparse.ArgumentParser(description='Generate CIFAR-100 training progress report')
    parser.add_argument('--log-file', type=str, default='cifar100_training.log',
                        help='Path to CIFAR-100 training log file')
    parser.add_argument('--output', type=str, default='reports/cifar100_progress_report.html',
                        help='Output HTML file path')
    parser.add_argument('--save-pth', type=str, default='cifar100_transfer_results.pth',
                        help='Path to save training data as .pth file')
    args = parser.parse_args()
    
    # Get project directory
    project_dir = Path(__file__).parent.parent.absolute()
    
    # Check if log file exists
    log_file_path = os.path.join(project_dir, args.log_file)
    if not os.path.exists(log_file_path):
        print(f"Error: CIFAR-100 training log file not found at {log_file_path}")
        print("Please provide the log file path with --log-file or create a file named 'cifar100_training.log'")
        print("in the project root directory with the CIFAR-100 training log data.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.join(project_dir, args.output))
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the visualization script
    visualize_script = os.path.join(project_dir, 'examples', 'visualize_cifar100_progress.py')
    cmd = [
        sys.executable,
        visualize_script,
        '--log', log_file_path,
        '--output', os.path.join(project_dir, args.output),
        '--save-pth', os.path.join(project_dir, args.save_pth)
    ]
    
    print(f"Generating CIFAR-100 training progress report...")
    print(f"Log file: {log_file_path}")
    print(f"Output: {os.path.join(project_dir, args.output)}")
    print(f"PTH file: {os.path.join(project_dir, args.save_pth)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"CIFAR-100 training progress report generated successfully at {os.path.join(project_dir, args.output)}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating CIFAR-100 training progress report: {e}")
        return

if __name__ == "__main__":
    main()
