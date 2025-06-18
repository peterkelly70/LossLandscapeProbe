#!/usr/bin/env python3
"""
Unified visualization script for training progress across different datasets and resource levels.
This script can handle:
- CIFAR-10 (full dataset)
- CIFAR-10 with different resource levels (10%, 40%, etc.)
- CIFAR-100 (full dataset)
- CIFAR-100 with different resource levels (10%, 40%, etc.)

It generates HTML reports with interactive visualizations of training metrics.
"""

import os
import re
import sys
import json
import argparse
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import LLP modules
from llp.utils import setup_logging

# Configure logging
logger = setup_logging('visualize_progress')

# Define constants
RESOURCE_LEVELS = ['10', '20', '30', '40', '100']
DATASETS = ['cifar10', 'cifar100']
OUTPUT_DIR = os.path.join(project_root, 'reports')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize training progress for various models and datasets.')
    parser.add_argument('--log', type=str, help='Path to training log file')
    parser.add_argument('--output', type=str, help='Path to output HTML report')
    parser.add_argument('--dataset', type=str, choices=DATASETS, help='Dataset type (cifar10 or cifar100)')
    parser.add_argument('--resource-level', type=str, choices=RESOURCE_LEVELS, help='Resource level (10, 20, 30, 40, or 100 for full dataset)')
    parser.add_argument('--title', type=str, help='Custom title for the report')
    return parser.parse_args()


def detect_dataset_from_log(log_path):
    """Auto-detect dataset type and resource level from log filename."""
    filename = os.path.basename(log_path).lower()
    
    # Detect dataset
    dataset = None
    for ds in DATASETS:
        if ds in filename:
            dataset = ds
            break
    
    # Detect resource level
    resource_level = '100'  # Default to full dataset
    for level in RESOURCE_LEVELS:
        if f"_{level}" in filename or f"-{level}" in filename:
            resource_level = level
            break
            
    return dataset, resource_level


def parse_training_log(log_path, dataset):
    """Parse training log file to extract metrics."""
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Initialize data structures
    epochs = []
    losses = []
    accuracies = []
    
    # Different regex patterns based on dataset
    if dataset == 'cifar10':
        # Pattern for CIFAR-10 logs
        pattern = r'Epoch (\d+).*?loss: ([\d\.]+).*?accuracy: ([\d\.]+)'
    else:  # cifar100
        # Pattern for CIFAR-100 logs
        pattern = r'Epoch (\d+).*?loss: ([\d\.]+).*?accuracy: ([\d\.]+)'
    
    # Find all matches
    matches = re.findall(pattern, log_content)
    
    # Extract data
    for match in matches:
        epoch, loss, accuracy = match
        epochs.append(int(epoch))
        losses.append(float(loss))
        accuracies.append(float(accuracy))
    
    return {
        'epochs': epochs,
        'losses': losses,
        'accuracies': accuracies
    }


def generate_plots(data, dataset, resource_level, output_dir):
    """Generate plots for training metrics."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('ggplot')
    
    # Generate loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['epochs'], data['losses'], 'b-', linewidth=2)
    plt.title(f'{dataset.upper()} Training Loss (Resource Level: {resource_level}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, f'{dataset}_{resource_level}_loss.png')
    plt.savefig(loss_plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Generate accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['epochs'], data['accuracies'], 'g-', linewidth=2)
    plt.title(f'{dataset.upper()} Training Accuracy (Resource Level: {resource_level}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0, 1.0)  # Accuracy is between 0 and 1
    accuracy_plot_path = os.path.join(output_dir, f'{dataset}_{resource_level}_accuracy.png')
    plt.savefig(accuracy_plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return {
        'loss_plot': os.path.basename(loss_plot_path),
        'accuracy_plot': os.path.basename(accuracy_plot_path)
    }


def generate_html_report(data, plots, dataset, resource_level, output_path, title=None):
    """Generate HTML report with interactive visualizations."""
    if title is None:
        title = f"{dataset.upper()} Training Progress (Resource Level: {resource_level}%)"
    
    # Get relative paths for plots
    loss_plot_rel = plots['loss_plot']
    accuracy_plot_rel = plots['accuracy_plot']
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }}
        .chart-container {{
            width: 48%;
            margin-bottom: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            padding: 15px;
            border-radius: 5px;
            background-color: #fff;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metrics {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            width: 30%;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="metrics">
        <div class="metric-card">
            <h3>Total Epochs</h3>
            <div class="metric-value">{len(data['epochs'])}</div>
        </div>
        <div class="metric-card">
            <h3>Final Loss</h3>
            <div class="metric-value">{data['losses'][-1]:.4f}</div>
        </div>
        <div class="metric-card">
            <h3>Final Accuracy</h3>
            <div class="metric-value">{data['accuracies'][-1]:.4f}</div>
        </div>
    </div>
    
    <div class="container">
        <div class="chart-container">
            <h2>Training Loss</h2>
            <img src="{loss_plot_rel}" alt="Training Loss Plot">
        </div>
        <div class="chart-container">
            <h2>Training Accuracy</h2>
            <img src="{accuracy_plot_rel}" alt="Training Accuracy Plot">
        </div>
    </div>
    
    <h2>Training Metrics</h2>
    <table>
        <thead>
            <tr>
                <th>Epoch</th>
                <th>Loss</th>
                <th>Accuracy</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Add table rows for each epoch (limit to last 50 for large datasets)
    max_rows = 50
    start_idx = max(0, len(data['epochs']) - max_rows)
    for i in range(start_idx, len(data['epochs'])):
        html_content += f"""
            <tr>
                <td>{data['epochs'][i]}</td>
                <td>{data['losses'][i]:.6f}</td>
                <td>{data['accuracies'][i]:.6f}</td>
            </tr>"""
    
    # Complete the HTML
    html_content += """
        </tbody>
    </table>
    
    <script>
        // Add any interactive JavaScript here if needed
    </script>
</body>
</html>
"""
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    # Save data for future reference
    data_path = output_path.replace('.html', '.pth')
    torch.save({
        'dataset': dataset,
        'resource_level': resource_level,
        'epochs': data['epochs'],
        'losses': data['losses'],
        'accuracies': data['accuracies'],
        'timestamp': datetime.datetime.now().isoformat()
    }, data_path)
    
    return output_path, data_path


def main():
    """Main function to run the visualization."""
    args = parse_args()
    
    # Determine log file path
    if args.log:
        log_path = args.log
    else:
        # Try to find log files
        log_files = []
        for dataset in DATASETS:
            log_files.extend(list(project_root.glob(f"{dataset}*.log")))
        
        if not log_files:
            logger.error("No log files found. Please specify a log file with --log.")
            sys.exit(1)
        
        # Use the most recent log file
        log_path = str(sorted(log_files, key=os.path.getmtime, reverse=True)[0])
        logger.info(f"Using most recent log file: {log_path}")
    
    # Determine dataset and resource level
    if args.dataset and args.resource_level:
        dataset = args.dataset
        resource_level = args.resource_level
    else:
        dataset, resource_level = detect_dataset_from_log(log_path)
        if not dataset:
            logger.error("Could not detect dataset from log filename. Please specify with --dataset.")
            sys.exit(1)
    
    logger.info(f"Processing {dataset.upper()} dataset with resource level {resource_level}%")
    
    # Parse training log
    data = parse_training_log(log_path, dataset)
    if not data['epochs']:
        logger.error(f"No training data found in log file: {log_path}")
        sys.exit(1)
    
    logger.info(f"Found {len(data['epochs'])} epochs of training data")
    
    # Determine output path
    if args.output:
        output_path = args.output
        output_dir = os.path.dirname(output_path)
    else:
        output_dir = OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{dataset}_{resource_level}_progress_report.html")
    
    # Generate plots
    plots = generate_plots(data, dataset, resource_level, output_dir)
    
    # Generate HTML report
    html_path, data_path = generate_html_report(
        data, plots, dataset, resource_level, output_path, title=args.title
    )
    
    logger.info(f"Generated HTML report: {html_path}")
    logger.info(f"Saved data file: {data_path}")


if __name__ == "__main__":
    main()
