#!/usr/bin/env python3
"""
Unified CIFAR Training Script
=============================

This script provides a unified approach to train models on CIFAR-10 and CIFAR-100 datasets
using the meta-model approach for hyperparameter optimization. It supports:

1. Training on different resource levels (10%, 20%, 30%, 40%)
2. Running in comparison mode to evaluate multiple resource levels
3. Training with meta-model guided hyperparameter optimization
4. Generating test reports and visualizations

Command-line parameters control all aspects of training, making this a flexible
replacement for the separate scripts previously used.
"""

import os
import sys
import json
import argparse
from datetime import datetime
import logging

# Add the project root directory to the path so we can import the LLP package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Also add the src directory to the path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define setup_logger function locally since llp.utils.logging_utils doesn't exist
def setup_logger():
    """Set up a basic logger for the script"""
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to prevent duplicate messages
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
from cifar_core import (
    run_meta_optimization, 
    train_and_evaluate,
    DEFAULT_RESOURCE_LEVELS,
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_NUM_CONFIGS,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE
)
from cifar_reporting import (
    generate_test_report,
    visualize_training_progress,
    update_website,
    generate_resource_comparison_report,
    generate_meta_model_report
)

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)


def save_result_to_file(dataset_name, resource_level, result_data):
    """Save the result data for a specific resource level to a JSON file.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        resource_level: Resource level used for training
        result_data: Dictionary with result data
        
    Returns:
        Path to the saved file
    """
    # Create the model-specific directory in reports
    model_type = f"cifa{10 if dataset_name == 'cifar10' else 100}"
    if resource_level < 1.0:
        model_type = f"{model_type}_{int(resource_level*100)}"
    
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "reports")
    model_reports_dir = os.path.join(reports_dir, model_type)
    os.makedirs(model_reports_dir, exist_ok=True)
    
    # Save the result data with a fixed filename
    result_file = os.path.join(model_reports_dir, "latest_result.json")
    with open(result_file, 'w') as f:
        # Make a copy to avoid modifying the original
        serializable_data = {}
        for key, value in result_data.items():
            if key == 'best_config':
                # Ensure all values are serializable
                serializable_data[key] = {k: str(v) for k, v in value.items()}
            elif key == 'eval_result':
                # Ensure all values are serializable
                serializable_data[key] = {k: float(v) if isinstance(v, (int, float)) else str(v) 
                                         for k, v in value.items()}
            else:
                serializable_data[key] = value
        
        json.dump(serializable_data, f, indent=4)
    
    # Also save a timestamped version for archival purposes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_file = os.path.join(model_reports_dir, f"result_{timestamp}.json")
    with open(archive_file, 'w') as f:
        json.dump(serializable_data, f, indent=4)
    
    logger.info(f"Result data saved to {result_file} and archived to {archive_file}")
    return result_file


def run_resource_level_comparison(dataset_name, resource_levels=DEFAULT_RESOURCE_LEVELS, 
                               epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                               num_iterations=DEFAULT_NUM_ITERATIONS, num_configs=DEFAULT_NUM_CONFIGS):
    """Run a comparison of different resource levels for meta-model training.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        resource_levels: List of resource levels to compare
        epochs: Number of epochs for final training
        batch_size: Batch size for training
        num_iterations: Number of meta-model iterations
        num_configs: Number of configurations to try per iteration
        
    Returns:
        Dictionary with results for each resource level
    """
    logger.info(f"Running resource level comparison for {dataset_name} with levels {resource_levels}")
    
    results = {}
    
    for resource_level in resource_levels:
        logger.info(f"\n\n==== Processing resource level {resource_level} ====\n")
        
        # Run meta-model optimization to find the best hyperparameters
        best_config = run_meta_optimization(
            dataset_name, resource_level, num_iterations=num_iterations, num_configs=num_configs
        )
        
        # Train a full model with the best hyperparameters
        eval_result = train_and_evaluate(
            best_config, dataset_name, epochs=epochs, batch_size=batch_size, resource_level=1.0
        )
        
        # Generate meta-model training progress report
        meta_report_path = generate_meta_model_report(dataset_name, resource_level)
        
        # Generate model training progress visualization
        vis_path = visualize_training_progress(dataset_name, resource_level)
        
        # Generate test report
        test_report_path = generate_test_report(best_config, dataset_name, resource_level)
        
        # Store the results
        result_data = {
            'best_config': best_config,
            'eval_result': eval_result,
            'meta_model_report_path': meta_report_path,
            'training_progress_path': vis_path,
            'test_report_path': test_report_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save the result to a file
        save_result_to_file(dataset_name, resource_level, result_data)
        
        # Store in memory for this run
        results[resource_level] = result_data
        
        # Update the comparison report incrementally
        generate_resource_comparison_report(dataset_name, resource_level=resource_level)
        
        logger.info(f"Completed processing for resource level {resource_level}")
        logger.info(f"Reports generated and stored in the reports directory.")
    
    # Generate a final comparison report with all results
    comparison_report = generate_resource_comparison_report(dataset_name, results=results)
    
    logger.info(f"Resource level comparison completed.")
    logger.info(f"All reports generated and stored in the reports directory.")
    logger.info(f"Use the deployment menu or 'mksite' to update the website with these reports.")
    return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Unified CIFAR Training Script')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10',
                        help='Dataset to use (cifar10 or cifar100)')
    
    # Training mode
    parser.add_argument('--mode', type=str, choices=['single', 'comparison', 'multisize'], default='single',
                        help='Training mode: single resource level, comparison of multiple levels, or multisize (all levels)')
    
    # Resource level for single mode
    parser.add_argument('--resource-level', type=float, default=0.1,
                        help='Resource level (fraction of dataset) for single mode')
    
    # Resource levels for comparison mode
    parser.add_argument('--resource-levels', type=str, default='0.1,0.2,0.3,0.4',
                        help='Comma-separated list of resource levels for comparison mode')
    
    # Meta-model parameters
    parser.add_argument('--num-iterations', type=int, default=DEFAULT_NUM_ITERATIONS,
                        help='Number of meta-model iterations')
    parser.add_argument('--num-configs', type=int, default=DEFAULT_NUM_CONFIGS,
                        help='Number of configurations to try per iteration')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of epochs for final training')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for training')
    
    # Reporting options
    parser.add_argument('--skip-reports', action='store_true',
                        help='Skip generating reports (useful for quick testing)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process resource levels for comparison mode
    if args.mode in ['comparison', 'multisize']:
        resource_levels = [float(level) for level in args.resource_levels.split(',')]
    else:
        resource_levels = [args.resource_level]
    
    # Run the appropriate mode
    if args.mode in ['comparison', 'multisize']:
        results = run_resource_level_comparison(
            args.dataset,
            resource_levels=resource_levels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
            num_configs=args.num_configs
        )
    else:  # single mode
        # Run meta-model optimization to find the best hyperparameters
        best_config = run_meta_optimization(
            args.dataset,
            args.resource_level,
            num_iterations=args.num_iterations,
            num_configs=args.num_configs
        )
        
        # Train a full model with the best hyperparameters
        eval_result = train_and_evaluate(
            best_config,
            args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resource_level=1.0  # Use full dataset for final training
        )
        
        # Prepare results dictionary
        result_data = {
            'best_config': best_config,
            'eval_result': eval_result,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate reports if not skipped
        if not args.skip_reports:
            # Determine model type for directory structure
            model_type = f"cifa{10 if args.dataset == 'cifar10' else 100}"
            if args.resource_level < 1.0:
                model_type = f"{model_type}_{int(args.resource_level*100)}"
            
            # Generate meta-model training progress report with fixed filename
            meta_report_path = generate_meta_model_report(args.dataset, args.resource_level, 
                                                       output_path=f"reports/{model_type}/latest_meta_report.html")
            result_data['meta_model_report_path'] = meta_report_path
            
            # Generate model training progress visualization with fixed filename
            vis_path = visualize_training_progress(args.dataset, args.resource_level,
                                                output_path=f"reports/{model_type}/latest_training_progress.html")
            result_data['training_progress_path'] = vis_path
            
            # Generate test report with fixed filename
            test_report_path = generate_test_report(best_config, args.dataset, args.resource_level,
                                                 output_path=f"reports/{model_type}/latest_test_report.html")
            result_data['test_report_path'] = test_report_path
            
            # Save the result to a file in the model-specific directory
            save_result_to_file(args.dataset, args.resource_level, result_data)
            
            # Update the comparison report incrementally
            generate_resource_comparison_report(args.dataset, 
                                              output_path=f"reports/{args.dataset}/latest_resource_comparison.html")
            
            logger.info(f"Completed processing for resource level {args.resource_level}")
            logger.info(f"Reports generated and stored in the reports/{model_type}/ directory.")
            logger.info(f"Training logs saved to reports/{model_type}/{model_type}_training_log.txt")
            logger.info(f"Use the deployment menu or 'mksite' to update the website with these reports.")
        
        results = result_data
    
    return results


if __name__ == '__main__':
    main()
