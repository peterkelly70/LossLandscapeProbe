#!/usr/bin/env python3
"""
Generate Resource Comparison Report
==================================

This script generates an HTML report comparing the results of training with
different sample percentages. It visualizes the relationship between sample
levels and model performance, showing how the meta-model approach performs
with different dataset sizes.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path so we can import the LLP package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llp.utils.logging_utils import setup_logger
import logging

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

# Constants
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_comparison_plots(data, output_dir):
    """Generate plots comparing results across sample percentages.
    
    Args:
        data: Dictionary with results for each sample percentage
        output_dir: Directory to save the plots
        
    Returns:
        Dictionary mapping plot types to their file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_files = {}
    
    # Extract sample percentages for display
    sample_sizes = sorted([float(level) for level in data.keys()])
    resource_percentages = [int(level * 100) for level in sample_sizes]
    
    # Extract metrics
    accuracies = []
    training_times = []
    
    for level in sample_sizes:
        level_str = str(level)
        result = data[level_str]['eval_result']
        accuracies.append(result['accuracy'] * 100)  # Convert to percentage
        training_times.append(result['training_time'])
    
    # Plot 1: Accuracy vs Sample Size
    plt.figure(figsize=(10, 6))
    plt.plot(resource_percentages, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Sample Size (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Test Accuracy vs Sample Size', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(resource_percentages)
    
    # Add value labels
    for i, acc in enumerate(accuracies):
        plt.annotate(f'{acc:.2f}%', 
                    (resource_percentages[i], acc),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    accuracy_plot_path = os.path.join(output_dir, 'accuracy_vs_resource.png')
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files['accuracy'] = accuracy_plot_path
    
    # Plot 2: Training Time vs Sample Size
    plt.figure(figsize=(10, 6))
    plt.plot(resource_percentages, training_times, 'o-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Sample Size (%)', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Training Time vs Sample Size', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(resource_percentages)
    
    # Add value labels
    for i, time in enumerate(training_times):
        plt.annotate(f'{time:.1f}s', 
                    (resource_percentages[i], time),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    time_plot_path = os.path.join(output_dir, 'time_vs_resource.png')
    plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files['time'] = time_plot_path
    
    # Plot 3: Efficiency (Accuracy/Time) vs Sample Size
    efficiency = [acc / time for acc, time in zip(accuracies, training_times)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(resource_percentages, efficiency, 'o-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Sample Size (%)', fontsize=12)
    plt.ylabel('Efficiency (Accuracy % / Second)', fontsize=12)
    plt.title('Training Efficiency vs Sample Size', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(resource_percentages)
    
    # Add value labels
    for i, eff in enumerate(efficiency):
        plt.annotate(f'{eff:.3f}', 
                    (resource_percentages[i], eff),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    efficiency_plot_path = os.path.join(output_dir, 'efficiency_vs_resource.png')
    plt.savefig(efficiency_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files['efficiency'] = efficiency_plot_path
    
    return plot_files


def generate_hyperparameter_comparison_table(data):
    """Generate an HTML table comparing hyperparameters across sample percentages.
    
    Args:
        data: Dictionary with results for each sample percentage
        
    Returns:
        HTML string with the comparison table
    """
    # Extract sample percentages and sort them
    sample_sizes = sorted([float(level) for level in data.keys()])
    
    # Get all unique hyperparameter keys
    all_hyperparams = set()
    for level_str in data:
        all_hyperparams.update(data[level_str]['best_config'].keys())
    
    # Sort hyperparameters for consistent display
    sorted_hyperparams = sorted(list(all_hyperparams))
    
    # Build the HTML table
    html = '<table class="table table-bordered table-hover">\n'
    html += '  <thead>\n'
    html += '    <tr>\n'
    html += '      <th>Hyperparameter</th>\n'
    
    # Add column headers for each sample percentage
    for level in sample_sizes:
        html += f'      <th>{int(level * 100)}%</th>\n'
    
    html += '    </tr>\n'
    html += '  </thead>\n'
    html += '  <tbody>\n'
    
    # Add rows for each hyperparameter
    for param in sorted_hyperparams:
        html += '    <tr>\n'
        html += f'      <td><strong>{param}</strong></td>\n'
        
        # Add values for each sample percentage
        for level in sample_sizes:
            level_str = str(level)
            value = data[level_str]['best_config'].get(param, 'N/A')
            html += f'      <td>{value}</td>\n'
        
        html += '    </tr>\n'
    
    html += '  </tbody>\n'
    html += '</table>\n'
    
    return html


def generate_html_report(dataset_name, data, plot_files, timestamp, output_path=None):
    """Generate an HTML report with the comparison results.
    
    Args:
        dataset_name: Name of the dataset (cifar10 or cifar100)
        data: Dictionary with results for each sample percentage
        plot_files: Dictionary with paths to the generated plots
        timestamp: Timestamp for the report
        
    Returns:
        Path to the generated HTML report
    """
    # Format the dataset name for display
    dataset_display = "CIFAR-10" if dataset_name.lower() == "cifar10" else "CIFAR-100"
    
    # Generate the hyperparameter comparison table
    hyperparameter_table = generate_hyperparameter_comparison_table(data)
    
    # Build the HTML report
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_display} Sample Size Comparison</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            padding: 20px;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #111;
            color: #eee;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #4cf;
        }}
        .container {{
            background-color: #222;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            border: 1px solid #444;
        }}
        .header {{
            margin-bottom: 30px;
            border-bottom: 1px solid #444;
            padding-bottom: 20px;
        }}
        .plot-container {{
            margin-bottom: 40px;
            background-color: #222;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #444;
        }}
        .plot-container img {{
            background-color: #fff;
            border-radius: 4px;
        }}
        .table-container {{
            margin-bottom: 40px;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #444;
            font-size: 0.9em;
            color: #aaa;
        }}
        .table {{
            color: #eee;
        }}
        .table-bordered {{
            border-color: #444;
        }}
        .table-hover tbody tr:hover {{
            background-color: #2a2a2a;
        }}
        .table thead th {{
            background-color: #1a2a3a;
            border-bottom: 2px solid #444;
            color: #eee;
        }}
        .lead {{
            color: #aaa;
        }}
        a {{
            color: #4cf;
        }}
        a:hover {{
            color: #6df;
        }}
        .navigation {{
            background-color: #222;
            padding: 10px 0;
            margin-bottom: 20px;
            border-radius: 8px;
        }}
        .navigation ul {{
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            justify-content: center;
        }}
        .navigation li {{
            margin: 0 10px;
        }}
        .navigation a {{
            color: #eee;
            text-decoration: none;
            padding: 5px 10px;
            border-radius: 4px;
        }}
        .navigation a:hover {{
            background-color: #333;
        }}
        .navigation a.active {{
            background-color: #264c73;
            color: #4cf;
        }}
        .refresh-button {{
            background-color: #264c73;
            color: #4cf;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            border: 1px solid #4cf;
            text-decoration: none;
            display: inline-block;
        }}
        .refresh-button:hover {{
            background-color: #1a3a5a;
        }}
        .status {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            background-color: #264c73;
            color: #4cf;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="navigation">
            <ul>
                <li><a href="index.html">Home</a></li>
                <li><a href="cifar10_progress.html">CIFAR-10 Progress</a></li>
                <li><a href="cifar100_progress.html">CIFAR-100 Progress</a></li>
                <li><a href="meta_model_progress.php">Meta-Model Progress</a></li>
                <li><a href="about.html">About</a></li>
            </ul>
        </div>
        
        <div class="header">
            <div>
                <h1>{dataset_display} Sample Size Comparison <span class="status">COMPLETE</span></h1>
                <p class="lead">Comparing meta-model performance across different sample percentages</p>
            </div>
            <div>
                <p>Generated on: {timestamp}</p>
                <a href="?refresh=true" class="refresh-button">Refresh Data</a>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <h2>Performance Metrics</h2>
                <p>The following plots show how model performance varies with different sample percentages (percentage of training data used).</p>
            </div>
        </div>
        
        <div class="plot">
            <div class="row">
                <div class="col-md-12">
                    <h3>Test Accuracy vs Sample Size</h3>
                    <img src="{os.path.basename(plot_files['accuracy'])}" alt="Accuracy vs Sample Size" class="img-fluid">
                    <p class="mt-2">This plot shows how test accuracy changes with different sample percentages.</p>
                </div>
            </div>
        </div>
        
        <div class="plot">
            <div class="row">
                <div class="col-md-12">
                    <h3>Training Time vs Sample Size</h3>
                    <img src="{os.path.basename(plot_files['time'])}" alt="Training Time vs Sample Size" class="img-fluid">
                    <p class="mt-2">This plot shows how training time changes with different sample percentages.</p>
                </div>
            </div>
        </div>
        
        <div class="plot">
            <div class="row">
                <div class="col-md-12">
                    <h3>Training Efficiency vs Sample Size</h3>
                    <img src="{os.path.basename(plot_files['efficiency'])}" alt="Efficiency vs Sample Size" class="img-fluid">
                    <p class="mt-2">This plot shows the efficiency (accuracy per second) for different sample percentages.</p>
                </div>
            </div>
        </div>
        
        <div class="table-container">
            <div class="row">
                <div class="col-12">
                    <h2>Hyperparameter Comparison</h2>
                    <p>The table below shows the best hyperparameters found by the meta-model for each sample percentage.</p>
                    {hyperparameter_table}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by LossLandscapeProbe - <a href="https://loss.computer-wizard.com.au/">https://loss.computer-wizard.com.au/</a></p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    
    # Save the HTML report
    if output_path:
        report_path = output_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        report_path = os.path.join(REPORTS_DIR, f"{dataset_name}_resource_comparison_{timestamp}.html")
    
    with open(report_path, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML report generated at {report_path}")
    return report_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Generate resource comparison report')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], required=True,
                        help='Dataset name (cifar10 or cifar100)')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the JSON file with comparison data')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Custom output path for the HTML report')
    
    args = parser.parse_args()
    
    # Load the comparison data
    with open(args.data_file, 'r') as f:
        comparison_data = json.load(f)
    
    # Extract data
    dataset_name = comparison_data['dataset']
    timestamp = comparison_data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
    results = comparison_data['results']
    
    # Create a directory for the plots
    plots_dir = os.path.join(REPORTS_DIR, f"{dataset_name}_comparison_plots_{timestamp}")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate the plots
    plot_files = generate_comparison_plots(results, plots_dir)
    
    # Generate the HTML report
    report_path = generate_html_report(dataset_name, results, plot_files, timestamp, args.output_path)
    
    logger.info(f"Resource comparison report generated at {report_path}")
    return report_path


if __name__ == '__main__':
    main()
