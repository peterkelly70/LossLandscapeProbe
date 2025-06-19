#!/usr/bin/env python3
"""
Generate Meta-Model Training Progress Report
===========================================

This script generates an HTML report visualizing the meta-model training progress,
showing the hyperparameter configurations being tested, their performance,
and the convergence of the meta-model over iterations.
"""

import os
import sys
import json
import logging
import subprocess
import re
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
import pandas as pd
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

# Default meta-model configuration values from the codebase
DEFAULT_NUM_ITERATIONS = 3  # Number of meta-model iterations
DEFAULT_NUM_CONFIGS = 10    # Number of configurations to try per iteration
DEFAULT_TOTAL_CONFIGS = DEFAULT_NUM_ITERATIONS * DEFAULT_NUM_CONFIGS


def parse_meta_model_log(log_file=None, model_type=None, resource_level=None):
    """Parse the meta-model training log to extract configurations and performance.
    
    Args:
        log_file: Path to the meta-model log file. If None, will search in default locations.
        model_type: Type of model (cifar10 or cifar100)
        resource_level: Resource level (10, 20, 30, 40, or None for base model)
        
    Returns:
        Dictionary with parsed data
    """
    # Initialize the info dictionary with training status
    info = {
        'status': 'not_found',
        'iterations': [],
        'configs': [],
        'performances': [],
        'sharpness_values': [],
        'robustness_values': [],
        'resource_level': resource_level
    }
    if log_file is None:
        # Try to find the log file in default locations
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Determine model type and resource level from output path if not provided
        if model_type is None:
            model_type = "cifar10"  # Default to cifar10
        
        # Format the log file name based on model type and resource level
        if resource_level:
            log_file_name = f"{model_type}_meta_model_{resource_level}pct.log"
            dir_name = f"{model_type}_{resource_level}"
        else:
            # For base model or transfer learning
            if model_type == "cifar100_transfer":
                log_file_name = "cifar100_transfer_meta_model.log"
                dir_name = "cifar100_transfer"
                model_type = "cifar100_transfer"
            else:
                log_file_name = f"{model_type}_meta_model.log"
                dir_name = model_type
        
        # Check both project and web server directories with correct naming
        possible_log_paths = [
            # Web server paths
            f"/var/www/html/loss.computer-wizard.com.au/reports/{dir_name}/{log_file_name}",
            # Project paths
            os.path.join(project_root, 'reports', dir_name, log_file_name),
            os.path.join(project_root, 'logs', log_file_name)
        ]
        
        # Find the most recently modified log file
        latest_mtime = 0
        log_path = None
        
        for path in possible_log_paths:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    log_path = path
        
        if log_path:
            log_file = log_path
            logger.info(f"Found meta-model log file: {log_file}")
            info['status'] = 'found'
        else:
            # Check if there's an active training process running
            if not log_file or not os.path.exists(log_file):
                try:
                    # Use ps to check for active meta-model training processes
                    cmd = ["ps", "aux"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    output = result.stdout.lower()
                    
                    # Debug: Log the output to see what processes are running
                    logger.info("Checking for active training processes...")
                    for line in output.split('\n'):
                        if "cifar" in line:
                            logger.info(f"Found potential training process: {line}")
                    
                    # Check if there's any unified_cifar_training.py process running
                    if "unified_cifar_training.py" in output and ("cifar10" in output or "cifar100" in output):
                        logger.info("Active meta-model training process detected")
                        
                        # Try to extract actual iterations and configs from the command line
                        num_iterations = DEFAULT_NUM_ITERATIONS
                        num_configs = DEFAULT_NUM_CONFIGS
                        
                        # Look for --num-iterations and --num-configs in the process command line
                        for line in output.split('\n'):
                            if ("meta_model" in line or "unified_cifar_training.py" in line) and ("cifar10" in line or "cifar100" in line):
                                # Extract iterations
                                iter_match = re.search(r'--num-iterations[= ]+(\d+)', line)
                                if iter_match:
                                    num_iterations = int(iter_match.group(1))
                                    logger.info(f"Detected {num_iterations} iterations from running process")
                                
                                # Extract configs
                                config_match = re.search(r'--num-configs[= ]+(\d+)', line)
                                if config_match:
                                    num_configs = int(config_match.group(1))
                                    logger.info(f"Detected {num_configs} configurations from running process")
                        
                        return {
                            "status": "in_progress",
                            "detected_iterations": num_iterations,
                            "detected_configs": num_configs,
                            "total_configs": num_iterations * num_configs
                        }
                    else:
                        logger.warning("No meta-model log file found and no active training process detected")
                        return {"status": "not_found"}
                except Exception as e:
                    logger.error(f"Error checking for active processes: {e}")
                    return {"status": "not_found"}
            
            logger.info(f"Found meta-model log at: {log_path}")
    else:
        log_path = log_file
        if not os.path.exists(log_path):
            logger.warning(f"Log file {log_path} not found")
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
    iteration_regex = re.compile(r'Meta-optimization iteration (\d+)/(\d+)')
    config_regex = re.compile(r'Configuration (\d+): (\{.*\})')
    evaluating_regex = re.compile(r'Evaluating configuration (\d+)/(\d+) at resource level ([0-9.]+)')
    sharpness_regex = re.compile(r'Sharpness: ([0-9.]+), Perturbation Robustness: ([0-9.]+)')
    performance_regex = re.compile(r'Added training example with performance ([0-9.]+)')
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
                
                # Extract iteration number and total iterations
                try:
                    current_iteration = int(iteration_match.group(1))
                    total_iterations = int(iteration_match.group(2))
                    current_configs = []
                    current_performances = []
                    current_losses = []
                    current_accuracies = []
                    current_epochs = []
                except ValueError:
                    logger.warning(f"Could not parse iteration number from: {line}")
            
            # Look for configuration evaluations
            evaluating_match = evaluating_regex.search(line)
            if evaluating_match:
                try:
                    config_idx = int(evaluating_match.group(1)) - 1  # 0-indexed internally
                    total_configs = int(evaluating_match.group(2))
                    resource_level = float(evaluating_match.group(3))
                    
                    # Create a placeholder config
                    config = {
                        "config_id": config_idx + 1,
                        "resource_level": resource_level
                    }
                    current_configs.append(config)
                    
                    # Initialize empty lists for this configuration's metrics
                    current_losses.append([])
                    current_accuracies.append([])
                    current_epochs.append([])
                except ValueError:
                    logger.warning(f"Could not parse configuration from: {line}")
            
            # Look for sharpness measurements
            sharpness_match = sharpness_regex.search(line)
            if sharpness_match and current_configs:
                try:
                    sharpness = float(sharpness_match.group(1))
                    robustness = float(sharpness_match.group(2))
                    
                    # Add to the most recent config
                    current_configs[-1]["sharpness"] = sharpness
                    current_configs[-1]["robustness"] = robustness
                except ValueError:
                    logger.warning(f"Could not parse sharpness from: {line}")
            
            # Look for performance results
            performance_match = performance_regex.search(line)
            if performance_match and current_configs:
                try:
                    accuracy = float(performance_match.group(1))
                    current_performances.append(accuracy)
                    
                    # Add to the most recent config
                    current_configs[-1]["performance"] = accuracy
                except ValueError:
                    logger.warning(f"Could not parse performance from: {line}")
            
            # Look for best configuration
            best_config_match = best_config_regex.search(line)
            if best_config_match:
                try:
                    best_iter = int(best_config_match.group(1))
                    best_config_str = best_config_match.group(2)
                    best_config = eval(best_config_str)
                    info["best_config"] = best_config
                    info["best_iteration"] = best_iter
                except Exception as e:
                    logger.warning(f"Could not parse best config: {e}")
            
            # Look for best accuracy
            best_acc_match = best_acc_regex.search(line)
            if best_acc_match:
                try:
                    best_acc = float(best_acc_match.group(1))
                    info["best_accuracy"] = best_acc
                except ValueError:
                    logger.warning(f"Could not parse best accuracy from: {line}")
    
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
                <p>LossLandscapeProbe Meta-Model Report</p>
                
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


def generate_html_report(info, output_path, web_output_path=None):
    """Generate an HTML report for meta-model training progress.
    
    Args:
        info: Dictionary with meta-model training information
        output_path: Path to save the HTML report in the project directory
        web_output_path: Optional path to save the HTML report in the web server directory
    """
    # Check if we have any data or if training is in progress
    if not info:
        logger.error("No meta-model information available for report generation")
        return None
        
    # Determine the status of meta-model training
    status = info.get('status', 'not_found')
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if web_output_path:
        os.makedirs(os.path.dirname(web_output_path), exist_ok=True)
    # Extract information from the parsed log data
    if status == 'in_progress':
        # Training is in progress but we don't have log data yet
        current_iteration = 0
        # Use detected values if available, otherwise use defaults
        total_iterations = info.get('detected_iterations', DEFAULT_NUM_ITERATIONS)
        configurations_evaluated = 0
        total_configurations = info.get('total_configs', total_iterations * DEFAULT_NUM_CONFIGS)
        progress_percent = 0
        iteration_progress = 0
        status_display = 'IN PROGRESS'
        
        # Get resource level from the info dictionary or output path
        if info and 'resource_level' in info:
            resource_level = f"{info['resource_level']}%" if isinstance(info['resource_level'], (int, float)) else str(info['resource_level'])
        elif output_path:
            if '_10' in output_path:
                resource_level = '10%'
            elif '_20' in output_path:
                resource_level = '20%'
            elif '_30' in output_path:
                resource_level = '30%'
            elif '_40' in output_path:
                resource_level = '40%'
            else:
                resource_level = 'Base Model'
        else:
            resource_level = 'Base Model'
    elif info and 'iterations' in info and info['iterations']:
        # We have actual log data
        current_iteration = len(info['iterations'])
        total_iterations = 3  # Default value for meta-model iterations
        
        if 'configs' in info and info['configs']:
            configurations_evaluated = sum(len(configs) for configs in info['configs'])
            total_configurations = 18  # Default value (6 configs per iteration * 3 iterations)
        else:
            configurations_evaluated = 0
            total_configurations = 18
            
        progress_percent = (configurations_evaluated / total_configurations) * 100 if total_configurations > 0 else 0
        iteration_progress = (current_iteration / total_iterations) * 100 if total_iterations > 0 else 0
        
        # Get resource level from the info dictionary
        if info and 'resource_level' in info:
            resource_level = f"{info['resource_level']}%" if isinstance(info['resource_level'], (int, float)) else str(info['resource_level'])
        else:
            resource_level = 'Base Model'
        
        # Determine status
        if current_iteration >= total_iterations:
            status_display = 'COMPLETE'
        else:
            status_display = 'IN PROGRESS'
    else:
        # No data available
        current_iteration = 0
        total_iterations = 3
        configurations_evaluated = 0
        total_configurations = 18
        progress_percent = 0
        iteration_progress = 0
        status_display = 'NOT STARTED'
        
        # Try to determine resource level from output path
        if output_path:
            if '_10' in output_path:
                resource_level = '10%'
            elif '_20' in output_path:
                resource_level = '20%'
            elif '_30' in output_path:
                resource_level = '30%'
            elif '_40' in output_path:
                resource_level = '40%'
            else:
                resource_level = 'Base Model'
        else:
            resource_level = 'Base Model'
            
        status = 'PENDING'
    
    # Build the HTML report
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meta-Model Training Progress ({resource_level})</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <meta http-equiv="refresh" content="60"><!-- Auto-refresh every 60 seconds -->
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
        .alert {{
            background-color: #1a2a3a;
            border-color: #264c73;
            color: #eee;
            border-left: 4px solid #4cf;
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
            <h1>Meta-Model Training Progress ({resource_level})</h1>
            <p class="lead">Visualizing the meta-model hyperparameter optimization process</p>
            
            <!-- Status Alert -->
            <div class="alert alert-{{'success' if status_display == 'COMPLETE' else 'warning' if status_display == 'IN PROGRESS' else 'danger'}} mb-4">
                <h4 class="alert-heading">Status: {status_display}</h4>
                <p>Meta-model training is currently {status_display.lower()}.</p>
                <div class="progress">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                         role="progressbar" style="width: {progress_percent}%" 
                         aria-valuenow="{progress_percent}" aria-valuemin="0" aria-valuemax="100">
                        {progress_percent:.1f}%
                    </div>
                </div>
                <p class="mt-2 mb-0">Iterations: {current_iteration}/{total_iterations} | Configurations evaluated: {configurations_evaluated}/{total_configurations}</p>
                <p class="text-muted"><small>This report will automatically refresh every 60 seconds to show the latest progress.</small></p>
            </div>
        </div>
"""
    
    # Add placeholder content based on status
    if status == 'in_progress' or status == 'not_found':
        html += f"""
        <div class="row mt-4">
            <div class="col-12">
                <div class="card bg-dark text-light mb-4">
                    <div class="card-header">
                        <h3>Meta-Model Training In Progress</h3>
                    </div>
                    <div class="card-body">
                        <p>The meta-model training process is currently running. This report will automatically update as training progresses.</p>
                        <p>The meta-model is exploring different hyperparameter configurations to find the optimal settings for training the model with {resource_level} of resources.</p>
                        <h4>Expected Hyperparameters:</h4>
                        <ul>
                            <li>Number of channels</li>
                            <li>Dropout rate</li>
                            <li>Optimizer choice</li>
                            <li>Learning rate</li>
                            <li>Momentum</li>
                            <li>Weight decay</li>
                        </ul>
                        <div class="alert alert-info">
                            <strong>Note:</strong> This page will automatically refresh every 60 seconds to show the latest training progress.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    # Add plots if available
    elif 'best_per_iteration' in info:
        html += f"""
        <div class="plot-container">
            <div class="row">
                <div class="col-md-12">
                    <h3>Best Performance per Iteration</h3>
                    <img src="{os.path.basename(info['best_per_iteration'])}" alt="Best Performance per Iteration" class="img-fluid">
                    <p class="mt-2">This plot shows how the best validation accuracy improves across meta-model iterations.</p>
                </div>
            </div>
        </div>
"""
    
    if 'performance_distribution' in info:
        html += f"""
        <div class="plot-container">
            <div class="row">
                <div class="col-md-12">
                    <h3>Performance Distribution per Iteration</h3>
                    <img src="{os.path.basename(info['performance_distribution'])}" alt="Performance Distribution" class="img-fluid">
                    <p class="mt-2">This plot shows the distribution of validation accuracies for different configurations in each iteration.</p>
                </div>
            </div>
        </div>
"""
    
    if 'hyperparameter_importance' in info:
        html += f"""
        <div class="plot-container">
            <div class="row">
                <div class="col-md-12">
                    <h3>Hyperparameter Importance</h3>
                    <img src="{os.path.basename(info['hyperparameter_importance'])}" alt="Hyperparameter Importance" class="img-fluid">
                    <p class="mt-2">This plot shows the correlation between each hyperparameter and the validation accuracy.</p>
                </div>
            </div>
        </div>
"""
    
    # Add content sections based on available data
    if info and info.get('iterations') and info.get('configs'):
        # Build the HTML content for configurations
        html += f"""
        <div class="card mb-4">
            <div class="card-header">
                <h2>Meta-Model Training Progress</h2>
            </div>
            <div class="card-body">
        """
        
        for iter_idx, (iteration, iter_configs, iter_performances) in enumerate(zip(info['iterations'], info['configs'], info['performances'])):
            html += f"""
                <h4>Iteration {iteration}</h4>
                <div class="table-responsive">
                    <table class="table table-dark table-striped">
                        <thead>
                            <tr>
                                <th>Config</th>
                                <th>Resource Level</th>
                                <th>Sharpness</th>
                                <th>Robustness</th>
                                <th>Performance</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            for config_idx, config in enumerate(iter_configs):
                # Get the performance if available
                performance = config.get('performance', 'N/A')
                if isinstance(performance, float):
                    perf_html = f"{performance:.4f}"
                else:
                    perf_html = str(performance)
                
                # Get the resource level
                resource_level = config.get('resource_level', 'N/A')
                if isinstance(resource_level, float):
                    resource_html = f"{resource_level:.2f}"
                else:
                    resource_html = str(resource_level)
                
                # Get the sharpness and robustness
                sharpness = config.get('sharpness', 'N/A')
                if isinstance(sharpness, float):
                    sharpness_html = f"{sharpness:.4f}"
                else:
                    sharpness_html = str(sharpness)
                    
                robustness = config.get('robustness', 'N/A')
                if isinstance(robustness, float):
                    robustness_html = f"{robustness:.4f}"
                else:
                    robustness_html = str(robustness)
                
                html += f"""
                            <tr>
                                <td>{config.get('config_id', config_idx + 1)}</td>
                                <td>{resource_html}</td>
                                <td>{sharpness_html}</td>
                                <td>{robustness_html}</td>
                                <td>{perf_html}</td>
                            </tr>
                """
            
            html += """
                        </tbody>
                    </table>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
    else:
        # No configurations available - show a helpful message
        html += f"""
        <div class="alert alert-info" role="alert">
            <h4 class="alert-heading">Meta-Model Training In Progress</h4>
            <p>The meta-model training is either in progress or has not yet generated configuration data.</p>
            <p>Once training progresses, this report will show:</p>
            <ul>
                <li>Hyperparameter configurations being evaluated</li>
                <li>Training progress across iterations</li>
                <li>Best performing configuration details</li>
            </ul>
            <hr>
            <p class="mb-0">This page will automatically update as training progresses. You can also refresh manually.</p>
        </div>"""
    
    # Add progress information
    if info:
        current_iteration = info.get('current_iteration', 0)
        total_iterations = info.get('total_iterations', 3)
        progress_percent = (current_iteration / total_iterations) * 100 if total_iterations > 0 else 0
        
        html += f"""
        <div class="progress-container">
            <h2>Training Progress</h2>
            <p>Meta-model training progress: Iteration {current_iteration} of {total_iterations}</p>
            <div class="progress mb-3">
                <div class="progress-bar bg-info" role="progressbar" style="width: {progress_percent}%" 
                     aria-valuenow="{progress_percent}" aria-valuemin="0" aria-valuemax="100">
                    {progress_percent:.1f}%
                </div>
            </div>
        </div>"""
    else:
        # No progress information available
        html += f"""
        <div class="progress-container">
            <h2>Training Progress</h2>
            <p>Waiting for meta-model training to start...</p>
            <div class="progress mb-3">
                <div class="progress-bar bg-info" role="progressbar" style="width: 0%" 
                     aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                    0%
                </div>
            </div>
        </div>"""
    
    # Add best configuration if available
    if info and 'configurations' in info and info['configurations']:
        # Find the best configuration
        best_config = max(info['configurations'], key=lambda x: x.get('performance', 0))
        
        html += f"""
        <div class="best-config-container">
            <h2>Best Configuration</h2>
            <div class="card bg-dark text-light mb-4">
                <div class="card-header bg-info text-dark">
                    <h5 class="mb-0">Best Performance: {best_config.get('performance', 'N/A'):.4f}</h5>
                </div>
                <div class="card-body">
                    <ul class="list-group list-group-flush bg-dark">
                        <li class="list-group-item bg-dark text-light">Num Channels: {best_config.get('num_channels', 'N/A')}</li>
                        <li class="list-group-item bg-dark text-light">Dropout Rate: {best_config.get('dropout_rate', 'N/A')}</li>
                        <li class="list-group-item bg-dark text-light">Optimizer: {best_config.get('optimizer', 'N/A')}</li>
                        <li class="list-group-item bg-dark text-light">Learning Rate: {best_config.get('learning_rate', 'N/A')}</li>
                        <li class="list-group-item bg-dark text-light">Momentum: {best_config.get('momentum', 'N/A')}</li>
                        <li class="list-group-item bg-dark text-light">Weight Decay: {best_config.get('weight_decay', 'N/A')}</li>
                    </ul>
                </div>
            </div>
        </div>"""
    else:
        # No best configuration available yet
        html += f"""
        <div class="best-config-container">
            <h2>Best Configuration</h2>
            <div class="alert alert-info">
                <p>No configurations have been evaluated yet. The best configuration will be displayed here once available.</p>
            </div>
        </div>"""
    
    # Add footer
    html += f"""
        <div class="footer">
            <p>Generated by LossLandscapeProbe - <a href="https://loss.computer-wizard.com.au/">https://loss.computer-wizard.com.au/</a></p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""
    
    # Write the HTML report to the project directory
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Meta-model report generated at {output_path}")
    
    # Also write to web server directory if provided
    if web_output_path:
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(web_output_path), exist_ok=True)
            
            # Write the report
            with open(web_output_path, 'w') as f:
                f.write(html)
            
            print(f"Meta-model report also generated at {web_output_path}")
        except Exception as e:
            print(f"Error writing to web server directory: {e}")
    return output_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Generate meta-model training report')
    parser.add_argument('--log-file', type=str, help='Path to meta-model log file')
    parser.add_argument('--output', type=str, required=True, help='Output HTML file path')
    parser.add_argument('--web-output', type=str, help='Output HTML file path for web server')
    parser.add_argument('--model-type', type=str, help='Model type (cifar10 or cifar100)')
    parser.add_argument('--resource-level', type=str, help='Resource level (10, 20, 30, 40, or 100)')
    parser.add_argument('--num-iterations', type=int, default=3, 
                        help='Number of iterations for placeholder reports (default: 3)')
    parser.add_argument('--num-configs', type=int, default=5, 
                        help='Number of configurations per iteration for placeholder reports (default: 5)')
    args = parser.parse_args()
    
    # Extract model type and resource level from output path
    model_type = None
    resource_level = None
    
    # Try to determine model type and resource level from output path
    if args.output:
        # Extract model type and resource level from output path using regex
        output_path = args.output
        model_match = re.search(r'reports/(cifar\d+)(?:_(\d+))?(?:_transfer)?', output_path)
        if model_match:
            model_type = model_match.group(1)
            if model_match.group(2):
                resource_level = int(model_match.group(2))
            
            # Check for transfer learning
            if 'transfer' in output_path:
                model_type = f"{model_type}_transfer"
    
    # Use default paths if not provided
    if not args.output or not args.web_output:
        # Try to find the project root directory
        try:
            # Get project root directory
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Determine the log file path if not provided
            log_file_path = args.log_file
            
            if not args.output:
                # Default output path in project directory
                if model_type and resource_level:
                    args.output = os.path.join(project_dir, 'reports', f'{model_type}_{resource_level}', 'meta_model_report.html')
                elif model_type == "cifar100_transfer":
                    args.output = os.path.join(project_dir, 'reports', 'cifar100_transfer', 'meta_model_report.html')
                elif model_type:
                    args.output = os.path.join(project_dir, 'reports', model_type, 'meta_model_report.html')
                else:
                    args.output = os.path.join(project_dir, 'reports', 'cifar10', 'meta_model_report.html')
            
            if not args.web_output:
                # Default web output path
                web_server_dir = '/var/www/html/loss.computer-wizard.com.au/reports'
                if model_type and resource_level:
                    args.web_output = os.path.join(web_server_dir, f'{model_type}_{resource_level}', 'meta_model_report.html')
                elif model_type == "cifar100_transfer":
                    args.web_output = os.path.join(web_server_dir, 'cifar100_transfer', 'meta_model_report.html')
                elif model_type:
                    args.web_output = os.path.join(web_server_dir, model_type, 'meta_model_report.html')
                else:
                    args.web_output = os.path.join(web_server_dir, 'cifar10', 'meta_model_report.html')
                
        except Exception as e:
            logger.error(f"Error setting up default paths: {e}")
            return
    
    # If model_type is still None, try to extract it from the output path again
    if model_type is None and args.output:
        dir_name = os.path.basename(os.path.dirname(args.output))
        if dir_name.startswith('cifar'):
            parts = dir_name.split('_')
            model_type = parts[0]
            if len(parts) > 1 and parts[1].isdigit():
                resource_level = int(parts[1])
            elif len(parts) > 1 and parts[1] == 'transfer':
                model_type = f"{model_type}_transfer"
    
    logger.info(f"Generating meta-model report for model type: {model_type}, resource level: {resource_level}")
    
    # Parse the meta-model log file
    info = parse_meta_model_log(args.log_file, model_type, resource_level)
    status = info.get('status', 'not_found')
    
    # If we're generating a placeholder report, use the command-line arguments
    if status == 'in_progress' or status == 'not_found':
        info['detected_iterations'] = args.num_iterations
        info['detected_configs'] = args.num_configs
        info['total_configs'] = args.num_iterations * args.num_configs
    
    # Generate the HTML report
    report_path = generate_html_report(info, args.output, args.web_output)
    
    logger.info(f"Meta-model report generated at {report_path}")
    return report_path


if __name__ == '__main__':
    main()
