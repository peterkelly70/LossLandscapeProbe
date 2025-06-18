#!/usr/bin/env python3
"""
Generate Meta-Model Training Progress Report
===========================================

This script generates an HTML report visualizing the meta-model training progress,
showing the hyperparameter configurations being tested, their performance,
and the convergence of the meta-model over iterations.
"""

import os
import re
import sys
import json
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence

# Add the parent directory to the path so we can import the LLP package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup basic logging if llp module is not available
import logging
try:
    from llp.utils.logging_utils import setup_logger
    setup_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

logger = logging.getLogger(__name__)

# Constants
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


def parse_meta_model_log(dataset_name, sample_size):
    """Parse the meta-model training log to extract configurations and performance.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        sample_size: Resource level (e.g., 0.1 for 10%)
        
    Returns:
        Dictionary with parsed data
    """
    # Determine the model type for directory structure
    model_type = f"cifa{10 if dataset_name == 'cifar10' else 100}"
    if float(sample_size) < 1.0:
        model_type = f"{model_type}_{int(float(sample_size)*100)}"
    
    # Determine the log file name
    log_file = f"{dataset_name}_meta_model_{int(float(sample_size)*100)}pct.log"
    
    # Check in the logs directory first (where most logs are stored)
    project_root = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(project_root, "logs")
    log_path = os.path.join(logs_dir, log_file)
    
    logger.info(f"Looking for meta-model log at: {log_path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Check if the file exists in the logs directory
    if os.path.exists(log_path):
        logger.info(f"Found meta-model log at: {log_path}")
    else:
        logger.info(f"Log not found at: {log_path}")
        
        # Check alternative locations if not found
        alternative_paths = [
            os.path.join(project_root, "reports", model_type, log_file),  # New preferred location
            os.path.join(project_root, log_file),  # Root directory
            os.path.join(project_root, 'examples', log_file),  # Examples directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', log_file),
            os.path.join('/var/www/html/loss.computer-wizard.com.au/reports', model_type, log_file)  # Web server location
        ]
        
        for alt_path in alternative_paths:
            logger.info(f"Checking alternative path: {alt_path}")
            if os.path.exists(alt_path):
                logger.info(f"Found meta-model log at alternative path: {alt_path}")
                log_path = alt_path
                break
        else:
            logger.warning(f"Log file {log_file} not found in any location")
            return None
    
    # Parse the log file
    iterations = []
    configs = []
    performances = []
    losses = []
    accuracies = []
    epochs = []
    
    current_iteration = None
    current_configs = []
    current_performances = []
    current_losses = []
    current_accuracies = []
    current_epochs = []
    
    # Regular expressions for parsing
    iteration_regex = re.compile(r'Iteration (\d+), testing (\d+) configurations')
    config_regex = re.compile(r'Configuration (\d+): (\{.*\})')
    result_regex = re.compile(r'Training result: validation_accuracy=([0-9.]+)')
    best_config_regex = re.compile(r'Best configuration from iteration (\d+): (\{.*\})')
    best_acc_regex = re.compile(r'Best validation accuracy: ([0-9.]+)')
    completed_regex = re.compile(r'Meta-model training completed')
    
    with open(log_path, 'r') as f:
        for line in f:
            # Look for iteration markers
            iteration_match = iteration_regex.search(line)
            if iteration_match:
                # Save previous iteration data if exists
                if current_iteration is not None:
                    iterations.append(current_iteration)
                    configs.append(current_configs)
                    performances.append(current_performances)
                    losses.append(current_losses)
                    accuracies.append(current_accuracies)
                    epochs.append(current_epochs)
                
                # Extract iteration number
                parts = line.split("iteration")
                if len(parts) > 1:
                    try:
                        current_iteration = int(parts[1].strip().split()[0])
                        current_configs = []
                        current_performances = []
                        current_losses = []
                        current_accuracies = []
                        current_epochs = []
                    except ValueError:
                        logger.warning(f"Could not parse iteration number from: {line}")
            
            # Look for configuration evaluations
            elif "Evaluating configuration" in line and ":" in line:
                config_part = line.split("configuration")[1].split(":")[0].strip()
                try:
                    config_idx = int(config_part)
                except ValueError:
                    continue
                
                # Try to extract the configuration details
                if "{" in line and "}" in line:
                    config_str = line.split("{")[1].split("}")[0]
                    config_str = "{" + config_str + "}"
                    try:
                        config = eval(config_str)
                        current_configs.append(config)
                        # Initialize empty lists for this configuration's metrics
                        current_losses.append([])
                        current_accuracies.append([])
                        current_epochs.append([])
                    except:
                        logger.warning(f"Could not parse configuration from: {line}")
            
            # Look for performance results
            elif "Configuration" in line and "achieved validation accuracy" in line:
                parts = line.split("achieved validation accuracy")
                if len(parts) > 1:
                    try:
                        accuracy = float(parts[1].strip().split()[0])
                        current_performances.append(accuracy)
                    except ValueError:
                        logger.warning(f"Could not parse accuracy from: {line}")
            
            # Look for epoch-level metrics
            elif "Epoch" in line and ("loss:" in line or "accuracy:" in line):
                # Try to extract the configuration index
                config_match = re.search(r"Configuration\s+(\d+)", line)
                if config_match:
                    config_idx = int(config_match.group(1))
                    
                    # Extract epoch number
                    epoch_match = re.search(r"Epoch\s+(\d+)", line)
                    if epoch_match:
                        epoch = int(epoch_match.group(1))
                        
                        # Extract loss
                        loss_match = re.search(r"loss:\s+([0-9.]+)", line)
                        if loss_match:
                            loss = float(loss_match.group(1))
                            
                            # Extract accuracy
                            acc_match = re.search(r"accuracy:\s+([0-9.]+)", line)
                            if acc_match:
                                acc = float(acc_match.group(1))
                                
                                # Add to the current metrics if the config index is valid
                                if config_idx < len(current_losses):
                                    current_losses[config_idx].append(loss)
                                    current_accuracies[config_idx].append(acc)
                                    current_epochs[config_idx].append(epoch)
    
    # Save the last iteration
    if current_iteration is not None and current_configs:
        iterations.append(current_iteration)
        configs.append(current_configs)
        performances.append(current_performances)
        losses.append(current_losses)
        accuracies.append(current_accuracies)
        epochs.append(current_epochs)
    
    # Return the parsed data
    return {
        'iterations': iterations,
        'configs': configs,
        'performances': performances,
        'losses': losses,
        'accuracies': accuracies,
        'epochs': epochs
    }


def generate_meta_model_plots(data, output_dir):
    """Generate plots visualizing the meta-model training progress.
    
    Args:
        data: Dictionary with parsed meta-model data
        output_dir: Directory to save the plots
        
    Returns:
        Dictionary mapping plot types to their file paths
    """
    if not data or not data['iterations']:
        logger.warning("No data to plot")
        return {}
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the plot files dictionary
    plot_files = {}
    
    # Plot 1: Performance distribution per iteration
    plt.figure(figsize=(10, 6))
    
    for i, (iteration, performances) in enumerate(zip(data['iterations'], data['performances'])):
        if performances:  # Only plot if there are performances
            plt.boxplot(performances, positions=[iteration], widths=0.6)
    
    plt.xlabel('Meta-Model Iteration')
    plt.ylabel('Validation Accuracy')
    plt.title('Performance Distribution per Iteration')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    performance_dist_file = os.path.join(output_dir, 'performance_distribution.png')
    plt.savefig(performance_dist_file)
    plt.close()
    
    plot_files['performance_distribution'] = performance_dist_file
    
    # Plot 2: Best performance per iteration
    plt.figure(figsize=(10, 6))
    
    best_performances = []
    for performances in data['performances']:
        if performances:  # Only add if there are performances
            best_performances.append(max(performances))
    
    plt.plot(data['iterations'][:len(best_performances)], best_performances, 'o-', linewidth=2)
    plt.xlabel('Meta-Model Iteration')
    plt.ylabel('Best Validation Accuracy')
    plt.title('Best Performance per Iteration')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    best_performance_file = os.path.join(output_dir, 'best_performance.png')
    plt.savefig(best_performance_file)
    plt.close()
    
    plot_files['best_performance'] = best_performance_file
    
    # Plot 3: Loss curves for each configuration in the last iteration
    if 'losses' in data and 'epochs' in data and data['iterations']:
        last_iteration_idx = len(data['iterations']) - 1
        
        if last_iteration_idx >= 0 and last_iteration_idx < len(data['losses']):
            last_losses = data['losses'][last_iteration_idx]
            last_epochs = data['epochs'][last_iteration_idx]
            last_configs = data['configs'][last_iteration_idx] if 'configs' in data and last_iteration_idx < len(data['configs']) else []
            last_performances = data['performances'][last_iteration_idx] if 'performances' in data and last_iteration_idx < len(data['performances']) else []
            
            plt.figure(figsize=(12, 6))
            for i, (config_losses, config_epochs) in enumerate(zip(last_losses, last_epochs)):
                if config_losses and config_epochs:  # Only plot if there is data
                    label = f"Config {i+1}"
                    if i < len(last_performances):
                        label += f" (acc: {last_performances[i]:.4f})"
                    plt.plot(config_epochs, config_losses, '-o', label=label, alpha=0.7)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss Curves for Iteration {data["iterations"][last_iteration_idx]}')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            
            # Save the plot
            loss_curves_file = os.path.join(output_dir, 'loss_curves.png')
            plt.savefig(loss_curves_file)
            plt.close()
            
            plot_files['loss_curves'] = loss_curves_file
    
    # Plot 4: Accuracy curves for each configuration in the last iteration
    if 'accuracies' in data and 'epochs' in data and data['iterations']:
        last_iteration_idx = len(data['iterations']) - 1
        
        if last_iteration_idx >= 0 and last_iteration_idx < len(data['accuracies']):
            last_accuracies = data['accuracies'][last_iteration_idx]
            last_epochs = data['epochs'][last_iteration_idx]
            last_configs = data['configs'][last_iteration_idx] if 'configs' in data and last_iteration_idx < len(data['configs']) else []
            last_performances = data['performances'][last_iteration_idx] if 'performances' in data and last_iteration_idx < len(data['performances']) else []
            
            plt.figure(figsize=(12, 6))
            for i, (config_accuracies, config_epochs) in enumerate(zip(last_accuracies, last_epochs)):
                if config_accuracies and config_epochs:  # Only plot if there is data
                    label = f"Config {i+1}"
                    if i < len(last_performances):
                        label += f" (acc: {last_performances[i]:.4f})"
                    plt.plot(config_epochs, config_accuracies, '-o', label=label, alpha=0.7)
            
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'Training Accuracy Curves for Iteration {data["iterations"][last_iteration_idx]}')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            
            # Save the plot
            accuracy_curves_file = os.path.join(output_dir, 'accuracy_curves.png')
            plt.savefig(accuracy_curves_file)
            plt.close()
            
            plot_files['accuracy_curves'] = accuracy_curves_file
    
    # Plot 5: Hyperparameter importance
    if data['iterations'] and data['configs'] and data['performances']:
        last_iteration = data['iterations'][-1]
        last_configs = data['configs'][-1]
        last_performances = data['performances'][-1]
        
        if last_configs and last_performances and len(last_configs) == len(last_performances):
            # Extract hyperparameters and create a DataFrame
            hyperparams = {}
            
            # Get all hyperparameter names
            for config in last_configs:
                for key in config.keys():
                    hyperparams[key] = []
            
            # Extract values for each hyperparameter
            for config in last_configs:
                for key in hyperparams.keys():
                    hyperparams[key].append(config.get(key, None))
            
            # Create DataFrame
            df = pd.DataFrame(hyperparams)
            df['performance'] = last_performances
            
            try:
                # Calculate correlation between hyperparameters and performance
                numeric_df = df.select_dtypes(include=[np.number])
                if 'performance' in numeric_df.columns and len(numeric_df.columns) > 1:
                    correlations = numeric_df.corr()['performance'].drop('performance')
                    
                    plt.figure(figsize=(10, 6))
                    correlations.sort_values().plot(kind='barh')
                    plt.title('Hyperparameter Importance (Correlation with Performance)')
                    plt.xlabel('Correlation with Validation Accuracy')
                    plt.grid(True, linestyle='--', alpha=0.5)
                    
                    # Save the plot
                    importance_file = os.path.join(output_dir, 'hyperparameter_importance.png')
                    plt.savefig(importance_file)
                    plt.close()
                    
                    plot_files['hyperparameter_importance'] = importance_file
            except Exception as e:
                logger.warning(f"Could not generate hyperparameter importance plot: {e}")
    
    return plot_files


def generate_meta_model_report(log_file, output_dir, meta_model_path=None):
    """Generate a report visualizing the meta-model training progress.
    
    Args:
        log_file: Path to the meta-model training log file
        output_dir: Directory to save the report
        meta_model_path: Optional path to the meta-model directory
        
    Returns:
        Path to the generated HTML report
    """
    logger.info(f"Generating meta-model report from {log_file}")
    
    # Parse the log file
    data = parse_meta_model_log(log_file)
    
    # Extract feature importance if meta_model is available
    if meta_model_path:
        try:
            from llp.meta_model import HyperparameterPredictor
            meta_model = HyperparameterPredictor.load(meta_model_path)
            
            # Extract feature importance for each hyperparameter
            feature_importance = meta_model.get_feature_importance()
            data['feature_importance'] = feature_importance
            
            # Extract feature interactions for top features
            feature_interactions = {}
            for hyperparam in meta_model.models.keys():
                # Get top 3 features for this hyperparameter
                if hyperparam in feature_importance and feature_importance[hyperparam]:
                    top_features = sorted(feature_importance[hyperparam].items(), key=lambda x: x[1], reverse=True)[:3]
                    top_feature_names = [f[0] for f in top_features]
                    
                    # Get interactions for top features
                    hyperparam_interactions = {}
                    for feature in top_feature_names:
                        interaction = meta_model.get_hyperparameter_interaction(hyperparam, feature)
                        if interaction:
                            hyperparam_interactions[feature] = interaction
                    
                    if hyperparam_interactions:
                        feature_interactions[hyperparam] = hyperparam_interactions
            
            data['feature_interactions'] = feature_interactions
            logger.info(f"Extracted feature importance and interactions for {len(feature_importance)} hyperparameters")
        except Exception as e:
            logger.warning(f"Could not extract feature importance from meta-model: {e}")
    
    # Generate plots
    plot_files = generate_meta_model_plots(data, output_dir)
    
    # Generate HTML report
    report_path = os.path.join(output_dir, 'meta_model_report.html')
    
    with open(report_path, 'w') as f:
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Meta-Model Training Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .plot-container {{
                    margin-bottom: 30px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .highlight {{
                    background-color: #e3f2fd;
                    font-weight: bold;
                }}
                .section {{
                    margin-top: 40px;
                    border-top: 1px solid #eee;
                    padding-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Meta-Model Training Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Training Progress</h2>
        ''')
        
        # Add performance plots
        if 'best_performance' in plot_files:
            f.write(f'''
                <div class="plot-container">
                    <h3>Best Performance per Iteration</h3>
                    <img src="{os.path.basename(plot_files['best_performance'])}" alt="Best Performance per Iteration" style="max-width:100%;">
                </div>
            ''')
            
        if 'performance_distribution' in plot_files:
            f.write(f'''
                <div class="plot-container">
                    <h3>Performance Distribution per Iteration</h3>
                    <img src="{os.path.basename(plot_files['performance_distribution'])}" alt="Performance Distribution" style="max-width:100%;">
                </div>
            ''')
        
        # Add loss curves plot
        if 'loss_curves' in plot_files:
            f.write(f'''
                <div class="plot-container">
                    <h3>Training Loss Curves</h3>
                    <img src="{os.path.basename(plot_files['loss_curves'])}" alt="Training Loss Curves" style="max-width:100%;">
                </div>
            ''')
            
        # Add accuracy curves plot
        if 'accuracy_curves' in plot_files:
            f.write(f'''
                <div class="plot-container">
                    <h3>Training Accuracy Curves</h3>
                    <img src="{os.path.basename(plot_files['accuracy_curves'])}" alt="Training Accuracy Curves" style="max-width:100%;">
                </div>
            ''')
        
        # Add hyperparameter importance plot
        if 'hyperparameter_importance' in plot_files:
            f.write(f'''
                <div class="plot-container">
                    <h3>Hyperparameter Importance</h3>
                    <img src="{os.path.basename(plot_files['hyperparameter_importance'])}" alt="Hyperparameter Importance" style="max-width:100%;">
                </div>
            ''')
        
        # Add feature importance section (NEW)
        feature_importance_plots = [p for p in plot_files.keys() if p.startswith('feature_importance_')]
        if feature_importance_plots:
            f.write('''
                <div class="section">
                    <h2>Feature Importance Analysis</h2>
                    <p>This section shows which dataset and training features most influence each hyperparameter prediction.</p>
            ''')
            
            for key in feature_importance_plots:
                hyperparam = key.replace('feature_importance_', '')
                f.write(f'''
                    <div class="plot-container">
                        <h3>Feature Importance for {hyperparam}</h3>
                        <img src="{os.path.basename(plot_files[key])}" alt="Feature Importance for {hyperparam}" style="max-width:100%;">
                    </div>
                ''')
                
                # Add related feature interaction plots if available
                interaction_plots = [p for p in plot_files.keys() if p.startswith(f'interaction_{hyperparam}_')]
                if interaction_plots:
                    f.write(f'''
                        <h4>Feature Interactions for {hyperparam}</h4>
                        <p>These plots show how the top features affect the predicted value of {hyperparam}.</p>
                        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                    ''')
                    
                    for iplot in interaction_plots:
                        feature = iplot.replace(f'interaction_{hyperparam}_', '')
                        f.write(f'''
                            <div style="flex: 1; min-width: 300px;">
                                <h5>{feature}</h5>
                                <img src="{os.path.basename(plot_files[iplot])}" alt="Interaction {feature} → {hyperparam}" style="width:100%;">
                            </div>
                        ''')
                    
                    f.write('</div>')
            
            f.write('</div>')
        
        # Add hyperparameter exploration section
        f.write('<div class="section"><h2>Hyperparameter Exploration</h2>')
        
        hyperparam_plots = [p for p in plot_files.keys() if p.startswith('hyperparam_')]
        for key in hyperparam_plots:
            hyperparam = key.replace('hyperparam_', '')
            f.write(f'''
            <div class="plot-container">
                <h3>Hyperparameter: {hyperparam}</h3>
                <img src="{os.path.basename(plot_files[key])}" alt="{hyperparam} Values" style="max-width:100%;">
            </div>
            ''')
        
        f.write('</div>')
        
        # Add best configurations table
        if data['best_configs']:
            f.write('''
                <div class="section">
                    <h2>Best Configurations</h2>
                    <table>
                        <tr>
                            <th>Iteration</th>
                            <th>Performance</th>
                            <th>Configuration</th>
                        </tr>
            ''')
            
            for i, (iteration, perf, config) in enumerate(zip(data['iterations'], data['best_perfs'], data['best_configs'])):
                highlight = ' class="highlight"' if i == len(data['iterations']) - 1 else ''
                
                # Format the configuration as a readable string
                config_str = '<br>'.join([f"{k}: {v}" for k, v in config.items()])
                
                f.write(f'''
                    <tr{highlight}>
                        <td>{iteration}</td>
                        <td>{perf:.4f}</td>
                        <td>{config_str}</td>
                    </tr>
                ''')
                
            f.write('</table></div>')
        
        # Add suggested hyperparameters section for CIFAR-10 10% if applicable
        if dataset_name.lower() == 'cifar10' and abs(float(sample_size) - 0.1) < 0.01 and data['best_configs']:
            # Get the best configuration from the last iteration
            best_config = data['best_configs'][-1]
            
            f.write('''
                <div class="section">
                    <h2>Suggested Hyperparameters for CIFAR-10 10% Training</h2>
                    <p>Based on meta-model training, the following hyperparameters are suggested for optimal performance:</p>
                    <table>
                        <tr>
                            <th>Hyperparameter</th>
                            <th>Suggested Value</th>
                        </tr>
            ''')
            
            # Add each hyperparameter and its value
            for param, value in best_config.items():
                f.write(f'''
                        <tr>
                            <td>{param}</td>
                            <td>{value}</td>
                        </tr>
                ''')
            
            f.write('''
                    </table>
                    <p class="mt-3">These hyperparameters achieved the best validation accuracy during meta-model training 
                    and are recommended for training on the full dataset.</p>
                </div>
            ''')
        
        f.write('''
            </div>
        </body>
        </html>
        ''')
    
    logger.info(f"Meta-model report generated at {report_path}")
    return report_path


def generate_html_report(dataset_name, sample_size, data, plot_files, timestamp, output_path=None):
    """Generate an HTML report visualizing the meta-model training progress.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        sample_size: Resource level (e.g., 0.1 for 10%)
        data: Dictionary with parsed meta-model data
        plot_files: Dictionary with paths to the generated plots
        timestamp: Timestamp for the report
        output_path: Optional custom output path for the report
        
    Returns:
        Path to the generated HTML report
    """
    # Format the dataset name for display
    dataset_display = "CIFAR-10" if dataset_name.lower() == "cifar10" else "CIFAR-100"
    resource_percent = int(float(sample_size) * 100)
    
    # Build the HTML report
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_display} Meta-Model Training Progress ({resource_percent}%)</title>
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{dataset_display} Meta-Model Training Progress ({resource_percent}%)</h1>
            <p class="lead">Visualizing the meta-model hyperparameter optimization process</p>
            <p>Generated on: {timestamp}</p>
        </div>
"""
    
    # Add plots if available
    if 'best_per_iteration' in plot_files:
        html += f"""
        <div class="plot-container">
            <div class="row">
                <div class="col-md-12">
                    <h3>Best Performance per Iteration</h3>
                    <img src="{os.path.basename(plot_files['best_per_iteration'])}" alt="Best Performance per Iteration" class="img-fluid">
                    <p class="mt-2">This plot shows how the best validation accuracy improves across meta-model iterations.</p>
                </div>
            </div>
        </div>
"""
    
    if 'performance_distribution' in plot_files:
        html += f"""
        <div class="plot-container">
            <div class="row">
                <div class="col-md-12">
                    <h3>Performance Distribution per Iteration</h3>
                    <img src="{os.path.basename(plot_files['performance_distribution'])}" alt="Performance Distribution" class="img-fluid">
                    <p class="mt-2">This plot shows the distribution of validation accuracies for different configurations in each iteration.</p>
                </div>
            </div>
        </div>
"""
    
    if 'hyperparameter_importance' in plot_files:
        html += f"""
        <div class="plot-container">
            <div class="row">
                <div class="col-md-12">
                    <h3>Hyperparameter Importance</h3>
                    <img src="{os.path.basename(plot_files['hyperparameter_importance'])}" alt="Hyperparameter Importance" class="img-fluid">
                    <p class="mt-2">This plot shows the correlation between each hyperparameter and the validation accuracy.</p>
                </div>
            </div>
        </div>
"""
    
    # Add configuration table if data is available
    if data and data['iterations']:
        html += """
        <div class="table-container">
            <div class="row">
                <div class="col-12">
                    <h2>Best Configurations per Iteration</h2>
                    <table class="table table-bordered table-hover">
                        <thead>
                            <tr>
                                <th>Iteration</th>
                                <th>Best Configuration</th>
                                <th>Validation Accuracy</th>
                            </tr>
                        </thead>
                        <tbody>
"""
        
        for i, (iteration, configs, perfs) in enumerate(zip(data['iterations'], data['configs'], data['performances'])):
            if perfs:  # Only add if there are performances
                best_idx = np.argmax(perfs)
                best_perf = perfs[best_idx]
                best_config = configs[best_idx] if best_idx < len(configs) else "N/A"
                
                html += f"""
                            <tr>
                                <td>{iteration}</td>
                                <td><pre>{json.dumps(best_config, indent=2)}</pre></td>
                                <td>{best_perf:.4f}</td>
                            </tr>
"""
        
        html += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
"""
    
    # Add suggested hyperparameters section for CIFAR-10 10% if applicable
    if dataset_name.lower() == 'cifar10' and abs(float(sample_size) - 0.1) < 0.01 and data['best_configs']:
        # Get the best configuration from the last iteration
        best_config = data['best_configs'][-1]
        
        html += """
        <div class="table-container">
            <h3>Suggested Hyperparameters for CIFAR-10 10% Training</h3>
            <p>Based on meta-model training, the following hyperparameters are suggested for optimal performance:</p>
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>Hyperparameter</th>
                        <th>Suggested Value</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add each hyperparameter and its value
        for param, value in best_config.items():
            html += f"""
                    <tr>
                        <td>{param}</td>
                        <td>{value}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            <p class="mt-3">These hyperparameters achieved the best validation accuracy during meta-model training 
            and are recommended for training on the full dataset.</p>
        </div>
        """
    
    # Close the HTML
    html += """
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
        report_path = os.path.join(REPORTS_DIR, f"{dataset_name}_{resource_percent}pct_meta_model_report.html")
    
    with open(report_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Meta-model report generated at {report_path}")
    return report_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Generate meta-model training progress report')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], required=True,
                        help='Dataset name (cifar10 or cifar100)')
    parser.add_argument('--sample_size', type=float, required=True,
                        help='Sample size (e.g., 0.1 for 10%)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Custom output path for the HTML report')
    
    args = parser.parse_args()
    
    # Parse the meta-model log
    data = parse_meta_model_log(args.dataset, args.sample_size)
    
    if data is None or not data['iterations']:
        logger.warning(f"No meta-model data found for {args.dataset} at resource level {args.sample_size}")
        return None
    
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Determine the model type for directory structure based on project conventions
    # Base models: cifa10, cifa100
    # Resource level variants: cifa10_10, cifa10_20, etc.
    model_type = f"cifa{10 if args.dataset == 'cifar10' else 100}"
    resource_percent = int(args.sample_size * 100)
    
    if args.sample_size < 1.0:
        model_type = f"{model_type}_{resource_percent}"
    
    # Create the model-specific directory in reports
    model_reports_dir = os.path.join(REPORTS_DIR, model_type)
    os.makedirs(model_reports_dir, exist_ok=True)
    
    # Create a directory for the plots within the model-specific directory
    plots_dir = os.path.join(model_reports_dir, "meta_model_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate the plots
    plot_files = generate_meta_model_plots(data, plots_dir)
    
    # Generate the HTML report
    report_path = generate_html_report(args.dataset, args.sample_size, data, plot_files, timestamp, output_path=args.output_path)
    
    logger.info(f"Meta-model report generated at {report_path}")
    return report_path


if __name__ == '__main__':
    main()
