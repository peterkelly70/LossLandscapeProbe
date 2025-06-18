#!/usr/bin/env python3
"""
Generate Training Plots
======================

This script generates visualizations of training progress, including:
- Loss curves
- Accuracy curves
- Comparison between different datasets/configurations
- Hyperparameter visualization
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
from io import BytesIO
from datetime import datetime
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def plot_to_base64(fig):
    """Convert a matplotlib figure to base64 string for embedding in HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

def generate_loss_accuracy_plot(train_losses, test_accs, title):
    """Generate a plot showing training loss and test accuracy"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot training loss
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, color=color, marker='o', linestyle='-', markersize=3, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for test accuracy
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Test Accuracy (%)', color=color)
    ax2.plot(epochs, [acc * 100 for acc in test_accs], color=color, marker='s', linestyle='-', markersize=3, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add grid and title
    ax1.grid(True, alpha=0.3)
    plt.title(title)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    fig.tight_layout()
    return fig

def generate_dataset_comparison_plot(results_dict):
    """Generate a plot comparing performance across datasets"""
    datasets = list(results_dict.keys())
    
    # Handle different result formats - some use 'best_acc', others use 'val_acc' or 'test_acc'
    accuracies = []
    for dataset in datasets:
        result = results_dict[dataset]
        if 'best_acc' in result:
            accuracies.append(result['best_acc'] * 100)
        elif 'val_acc' in result:
            # Use the maximum validation accuracy
            accuracies.append(max(result['val_acc']) * 100)
        elif 'test_acc' in result:
            # Use the maximum test accuracy
            accuracies.append(max(result['test_acc']) * 100)
        else:
            # Fallback to train_acc if nothing else is available
            accuracies.append(max(result.get('train_acc', [0])) * 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(datasets, accuracies, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'][:len(datasets)])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Performance Comparison Across Datasets')
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    return fig

def generate_hyperparameter_table_html(results_dict):
    """Generate an HTML table showing hyperparameter configurations and results"""
    html = """
    <div class="table-container">
        <h3>Hyperparameter Configurations</h3>
        <table class="hyperparameter-table">
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th>Num Channels</th>
                    <th>Dropout Rate</th>
                    <th>Optimizer</th>
                    <th>Learning Rate</th>
                    <th>Weight Decay</th>
                    <th>Best Accuracy</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for dataset, result in results_dict.items():
        # Check if best_config exists, otherwise create an empty dict
        config = result.get('best_config', {})
        html += f"""
                <tr>
                    <td>{dataset}</td>
                    <td>{config.get('num_channels', 'N/A')}</td>
                    <td>{config.get('dropout_rate', 'N/A')}</td>
                    <td>{config.get('optimizer', 'N/A')}</td>
                    <td>{config.get('learning_rate', 'N/A')}</td>
                    <td>{config.get('weight_decay', 'N/A')}</td>
                    <td>{result.get('best_acc', 0)*100:.2f}%</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    return html

def generate_training_report(results_dict, output_path):
    """Generate an HTML report with training plots and hyperparameter information"""
    
    # Create plots
    plots = {}
    for dataset, result in results_dict.items():
        if 'train_losses' in result and 'test_accs' in result:
            fig = generate_loss_accuracy_plot(
                result['train_losses'], 
                result['test_accs'], 
                f"Training Progress - {dataset}"
            )
            plots[f"{dataset}_training"] = plot_to_base64(fig)
    
    # Create comparison plot if we have multiple datasets
    if len(results_dict) > 1:
        comparison_fig = generate_dataset_comparison_plot(results_dict)
        plots["dataset_comparison"] = plot_to_base64(comparison_fig)
    
    # Generate hyperparameter table
    hyperparameter_table = generate_hyperparameter_table_html(results_dict)
    
    # Create HTML report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LossLandscapeProbe Training Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot {{
            max-width: 100%;
            height: auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .table-container {{
            margin: 30px 0;
            overflow-x: auto;
        }}
        .hyperparameter-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .hyperparameter-table th, .hyperparameter-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .hyperparameter-table th {{
            background-color: #f2f2f2;
        }}
        .hyperparameter-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .metadata {{
            margin-top: 30px;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <h1>LossLandscapeProbe Training Report</h1>
    <p>This report shows training progress and results for different datasets using the meta-model approach.</p>
    <p><em>Note: Test visualizations show only a small sample (typically 200 images) out of the full 10,000 test images in each dataset. Training metrics are calculated on the complete datasets.</em></p>
    
    <h2>Training Progress</h2>
    """
    
    # Add individual training plots
    for dataset, result in results_dict.items():
        if f"{dataset}_training" in plots:
            html_content += f"""
    <div class="plot-container">
        <h3>{dataset} Training Progress</h3>
        <img class="plot" src="data:image/png;base64,{plots[f"{dataset}_training"]}" alt="{dataset} Training Plot">
    </div>
            """
    
    # Add comparison plot if available
    if "dataset_comparison" in plots:
        html_content += f"""
    <div class="plot-container">
        <h3>Dataset Performance Comparison</h3>
        <img class="plot" src="data:image/png;base64,{plots["dataset_comparison"]}" alt="Dataset Comparison Plot">
    </div>
        """
    
    # Add hyperparameter table
    html_content += hyperparameter_table
    
    # Add metadata
    html_content += f"""
    <div class="metadata">
        <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Save the HTML report
    os.makedirs('reports', exist_ok=True)
    output_path = 'reports/training_report.html'  # Fixed filename to avoid clutter
    fixed_output_path = 'training_report.html'  # Copy in project root for convenience
    
    with open(output_path, 'w') as f:
        f.write(html_content)
        
    with open(fixed_output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Training report generated at: {output_path}")
    print(f"Training report also saved to: {fixed_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate training plots and reports')
    parser.add_argument('--output', type=str, default='training_report.html',
                        help='Output HTML file path')
    args = parser.parse_args()
    
    # Automatically discover all *_results.pth files in the current directory
    results_dict = {}
    for fname in os.listdir('.'):
        if fname.endswith('_results.pth'):
            try:
                dataset_key = fname.replace('_results.pth', '').replace('_', ' ').upper()
                results_dict[dataset_key] = torch.load(fname)
                print(f"Loaded results: {fname}")
            except Exception as e:
                print(f"Warning: could not load {fname}: {e}")
    
    # Generate report if we have results
    if results_dict:
        generate_training_report(results_dict, args.output)
    else:
        print("No results found. Please run training scripts first.")

if __name__ == "__main__":
    main()
