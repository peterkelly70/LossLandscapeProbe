#!/usr/bin/env python3
"""
Generate CIFAR-100 Transfer Report
=================================

This script generates a detailed report on the transfer learning experiment
from CIFAR-10 to CIFAR-100, showing how well the meta-model generalizes
across datasets.
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

def generate_comparison_plot(cifar10_results, cifar100_results):
    """Generate a plot comparing CIFAR-10 and CIFAR-100 performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    datasets = ['CIFAR-10', 'CIFAR-100']
    accuracies = [
        cifar10_results.get('best_acc', 0) * 100,
        cifar100_results.get('best_acc', 0) * 100
    ]
    
    # Create bar chart
    bars = ax.bar(datasets, accuracies, color=['#3498db', '#e74c3c'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Add labels and title
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Meta-Model Transfer Performance')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add horizontal line for reference (state-of-the-art for SimpleCNN)
    ax.axhline(y=93, color='green', linestyle='--', alpha=0.7, label='SOTA for SimpleCNN on CIFAR-10')
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='SOTA for SimpleCNN on CIFAR-100')
    
    ax.legend()
    fig.tight_layout()
    return fig

def generate_training_curves_plot(cifar10_results, cifar100_results):
    """Generate a plot comparing training curves between CIFAR-10 and CIFAR-100"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot training loss
    if 'train_losses' in cifar10_results and 'train_losses' in cifar100_results:
        epochs_10 = range(1, len(cifar10_results['train_losses']) + 1)
        epochs_100 = range(1, len(cifar100_results['train_losses']) + 1)
        
        ax1.plot(epochs_10, cifar10_results['train_losses'], 'b-', label='CIFAR-10')
        ax1.plot(epochs_100, cifar100_results['train_losses'], 'r-', label='CIFAR-100')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot test accuracy
    if 'test_accs' in cifar10_results and 'test_accs' in cifar100_results:
        epochs_10 = range(1, len(cifar10_results['test_accs']) + 1)
        epochs_100 = range(1, len(cifar100_results['test_accs']) + 1)
        
        ax2.plot(epochs_10, [acc * 100 for acc in cifar10_results['test_accs']], 'b-', label='CIFAR-10')
        ax2.plot(epochs_100, [acc * 100 for acc in cifar100_results['test_accs']], 'r-', label='CIFAR-100')
        ax2.set_title('Test Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def generate_hyperparameter_comparison_table(cifar10_results, cifar100_results):
    """Generate an HTML table comparing hyperparameters between CIFAR-10 and CIFAR-100"""
    cifar10_config = cifar10_results.get('best_config', {})
    cifar100_config = cifar100_results.get('best_config', {})
    
    html = """
    <div class="table-container">
        <h3>Meta-Model Predicted Hyperparameters</h3>
        <table class="hyperparameter-table">
            <thead>
                <tr>
                    <th>Hyperparameter</th>
                    <th>CIFAR-10</th>
                    <th>CIFAR-100</th>
                    <th>Change</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # List of hyperparameters to compare
    hyperparams = ['num_channels', 'dropout_rate', 'optimizer', 'learning_rate', 'momentum', 'weight_decay']
    
    for param in hyperparams:
        cifar10_value = cifar10_config.get(param, 'N/A')
        cifar100_value = cifar100_config.get(param, 'N/A')
        
        # Determine if there's a change and what kind
        if cifar10_value == cifar100_value:
            change = '<span style="color: green;">Same</span>'
        elif param in ['num_channels', 'learning_rate', 'weight_decay'] and cifar10_value != 'N/A' and cifar100_value != 'N/A':
            if cifar100_value > cifar10_value:
                change = f'<span style="color: orange;">Increased (+{cifar100_value - cifar10_value})</span>'
            else:
                change = f'<span style="color: orange;">Decreased ({cifar100_value - cifar10_value})</span>'
        else:
            change = '<span style="color: orange;">Changed</span>'
        
        html += f"""
                <tr>
                    <td>{param}</td>
                    <td>{cifar10_value}</td>
                    <td>{cifar100_value}</td>
                    <td>{change}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    return html

def generate_dataset_comparison_table():
    """Generate an HTML table comparing CIFAR-10 and CIFAR-100 datasets"""
    html = """
    <div class="table-container">
        <h3>Dataset Comparison</h3>
        <table class="dataset-table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>CIFAR-10</th>
                    <th>CIFAR-100</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Number of Classes</td>
                    <td>10</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td>Images per Class (Train)</td>
                    <td>5,000</td>
                    <td>500</td>
                </tr>
                <tr>
                    <td>Images per Class (Test)</td>
                    <td>1,000</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td>Total Training Images</td>
                    <td>50,000</td>
                    <td>50,000</td>
                </tr>
                <tr>
                    <td>Total Test Images</td>
                    <td>10,000</td>
                    <td>10,000</td>
                </tr>
                <tr>
                    <td>Image Size</td>
                    <td>32x32 RGB</td>
                    <td>32x32 RGB</td>
                </tr>
                <tr>
                    <td>Hierarchical Classes</td>
                    <td>No</td>
                    <td>Yes (20 superclasses)</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    return html

def generate_transfer_insights():
    """Generate HTML content with insights about transfer learning"""
    html = """
    <div class="insights-container">
        <h3>Transfer Learning Insights</h3>
        <div class="insight-box">
            <h4>Why Transfer Learning is Challenging</h4>
            <ul>
                <li><strong>Increased Complexity:</strong> CIFAR-100 has 10x more classes than CIFAR-10, making the classification task inherently more difficult.</li>
                <li><strong>Less Data per Class:</strong> While the total dataset size is the same, CIFAR-100 has only 500 images per class (vs 5,000 for CIFAR-10).</li>
                <li><strong>Feature Extraction:</strong> The meta-model was trained to extract features from CIFAR-10, which may not be optimal for CIFAR-100's more diverse class structure.</li>
                <li><strong>Hyperparameter Sensitivity:</strong> Optimal hyperparameters can vary significantly between datasets with different characteristics.</li>
            </ul>
        </div>
        
        <div class="insight-box">
            <h4>Potential Improvements</h4>
            <ul>
                <li><strong>Dataset-Specific Meta-Models:</strong> Train dedicated meta-models for each dataset.</li>
                <li><strong>Meta-Feature Engineering:</strong> Design better dataset features that generalize across different datasets.</li>
                <li><strong>Transfer Learning Techniques:</strong> Incorporate knowledge distillation or model adaptation techniques.</li>
                <li><strong>Hierarchical Classification:</strong> Leverage CIFAR-100's superclass structure in the meta-model.</li>
            </ul>
        </div>
    </div>
    """
    return html

def generate_cifar100_transfer_report(cifar10_results, cifar100_results, output_path):
    """Generate an HTML report comparing CIFAR-10 and CIFAR-100 performance"""
    
    # Create plots
    plots = {}
    
    # Comparison plot
    comparison_fig = generate_comparison_plot(cifar10_results, cifar100_results)
    plots["accuracy_comparison"] = plot_to_base64(comparison_fig)
    
    # Training curves plot
    curves_fig = generate_training_curves_plot(cifar10_results, cifar100_results)
    plots["training_curves"] = plot_to_base64(curves_fig)
    
    # Generate tables
    hyperparameter_table = generate_hyperparameter_comparison_table(cifar10_results, cifar100_results)
    dataset_table = generate_dataset_comparison_table()
    insights = generate_transfer_insights()
    
    # Create HTML report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CIFAR-100 Transfer Learning Report</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #eee;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #111;
        }}
        h1, h2, h3, h4 {{
            color: #4cf;
        }}
        .plot-container {{
            margin: 20px 0;
            text-align: center;
            background-color: #222;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #444;
        }}
        .plot {{
            max-width: 100%;
            height: auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            background-color: #fff;
            border-radius: 4px;
        }}
        .table-container {{
            margin: 30px 0;
            overflow-x: auto;
            background-color: #222;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #444;
        }}
        .hyperparameter-table, .dataset-table {{
            width: 100%;
            border-collapse: collapse;
            color: #eee;
        }}
        .hyperparameter-table th, .hyperparameter-table td,
        .dataset-table th, .dataset-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #444;
        }}
        .hyperparameter-table th, .dataset-table th {{
            background-color: #1a2a3a;
            color: #eee;
        }}
        .hyperparameter-table tr:hover, .dataset-table tr:hover {{
            background-color: #2a2a2a;
        }}
        .metadata {{
            margin-top: 30px;
            font-size: 0.9em;
            color: #aaa;
        }}
        .insights-container {{
            margin: 30px 0;
            background-color: #222;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #444;
        }}
        .insight-box {{
            background-color: #1a2a3a;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .summary-box {{
            background-color: #222;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            border: 1px solid #444;
        }}
        strong {{
            color: #4cf;
        }}
    </style>
</head>
<body>
    <h1>CIFAR-100 Transfer Learning Report</h1>
    <p>This report analyzes how well the meta-model trained on CIFAR-10 transfers to the CIFAR-100 dataset.</p>
    
    <div class="summary-box">
        <h3>Summary</h3>
        <p>The meta-model trained on CIFAR-10 was applied to CIFAR-100 without retraining. The model achieved 
        <strong>{cifar100_results.get('best_acc', 0)*100:.2f}%</strong> accuracy on CIFAR-100, compared to 
        <strong>{cifar10_results.get('best_acc', 0)*100:.2f}%</strong> on CIFAR-10.</p>
        <p>This represents a transfer efficiency of 
        <strong>{(cifar100_results.get('best_acc', 0)/cifar10_results.get('best_acc', 1))*100:.2f}%</strong> 
        {"(Note: CIFAR-10 accuracy is zero or missing)" if cifar10_results.get('best_acc', 0) == 0 else ""}.</p>
        <p><strong>Note:</strong> Accuracy is based on the full test set of 10,000 samples. Only a subset of images is shown here.</p>
    </div>
    
    <h2>Performance Comparison</h2>
    <div class="plot-container">
        <img class="plot" src="data:image/png;base64,{plots["accuracy_comparison"]}" alt="Accuracy Comparison">
    </div>
    
    <h2>Training Curves</h2>
    <div class="plot-container">
        <img class="plot" src="data:image/png;base64,{plots["training_curves"]}" alt="Training Curves">
    </div>
    
    {hyperparameter_table}
    
    {dataset_table}
    
    {insights}
    
    <div class="metadata">
        <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>LossLandscapeProbe Framework</p>
    </div>
</body>
</html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    # Save a copy in the reports directory with a fixed name for the website
    reports_dir = os.path.join('reports', 'cifar100_transfer')
    os.makedirs(reports_dir, exist_ok=True)
    fixed_output_path = os.path.join(reports_dir, 'latest_test_report.html')
    with open(fixed_output_path, 'w') as f:
        f.write(html_content)
    
    print(f"CIFAR-100 transfer report generated at: {output_path}")
    print(f"CIFAR-100 transfer report also saved to: {fixed_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate CIFAR-100 transfer learning report')
    reports_dir = os.path.join('reports', 'cifar100_transfer')
    os.makedirs(reports_dir, exist_ok=True)
    parser.add_argument('--output', type=str, default=os.path.join(reports_dir, 'cifar100_transfer_report.html'),
                        help='Output HTML file path')
    args = parser.parse_args()
    
    # Load results
    cifar10_path = 'cifar10_results.pth'
    cifar100_path = 'cifar100_transfer_results.pth'
    
    # Check if result files exist
    missing_files = []
    if not os.path.exists(cifar10_path):
        missing_files.append(cifar10_path)
    if not os.path.exists(cifar100_path):
        missing_files.append(cifar100_path)
    
    # If any files are missing, generate a placeholder report
    if missing_files:
        print(f"Warning: The following result files are missing: {', '.join(missing_files)}")
        print("Generating placeholder report instead.")
        
        # Create placeholder HTML report
        placeholder_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CIFAR-100 Transfer Learning Report (Placeholder)</title>
            <style>
                body {{ 
                    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; 
                    margin: 40px; 
                    line-height: 1.6; 
                    background-color: #111; 
                    color: #eee; 
                }}
                h1, h3 {{ color: #4cf; }}
                .placeholder-box {{ 
                    background-color: #222; 
                    border-left: 4px solid #e74c3c; 
                    padding: 15px; 
                    margin: 20px 0; 
                    border: 1px solid #444;
                    border-radius: 5px;
                }}
                .metadata {{ margin-top: 30px; font-size: 0.9em; color: #aaa; }}
            </style>
        </head>
        <body>
            <h1>CIFAR-100 Transfer Learning Report</h1>
            
            <div class="placeholder-box">
                <h3>Training Not Completed</h3>
                <p>This is a placeholder report. The CIFAR-100 transfer learning experiment has not been completed yet.</p>
                <p>Missing files: {', '.join(missing_files)}</p>
                <p>Please run the CIFAR-100 transfer learning experiment first to generate a complete report.</p>
            </div>
            
            <div class="metadata">
                <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>LossLandscapeProbe Framework</p>
            </div>
        </body>
        </html>
        """
        
        # Write placeholder HTML to file
        with open(args.output, 'w') as f:
            f.write(placeholder_html)
        
        # Save a copy in the reports directory with a fixed name for the website
        reports_dir = os.path.join('reports', 'cifar100_transfer')
        os.makedirs(reports_dir, exist_ok=True)
        fixed_output_path = os.path.join(reports_dir, 'latest_test_report.html')
        with open(fixed_output_path, 'w') as f:
            f.write(placeholder_html)
        
        print(f"Placeholder CIFAR-100 transfer report generated at: {args.output}")
        print(f"Placeholder report also saved to: {fixed_output_path}")
        sys.exit(0)
    
    # Load results if files exist
    try:
        cifar10_results = torch.load(cifar10_path)
        cifar100_results = torch.load(cifar100_path)
    except Exception as e:
        print(f"Error loading result files: {e}")
        sys.exit(1)
    
    # Generate report
    try:
        generate_cifar100_transfer_report(cifar10_results, cifar100_results, args.output)
        print(f"CIFAR-100 transfer report successfully generated at: {args.output}")
        sys.exit(0)
    except Exception as e:
        print(f"Error generating CIFAR-100 transfer report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
