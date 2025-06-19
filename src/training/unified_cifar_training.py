#!/usr/bin/env python3
"""
Unified CIFAR Training Script
=============================

This script provides a unified approach to train models on CIFAR-10 and CIFAR-100 datasets
using the meta-model approach for hyperparameter optimization. It supports:

1. Training on different sample sizes (10%, 20%, 30%, 40%)
2. Running in comparison mode to evaluate multiple sample sizes
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
    DEFAULT_SAMPLE_SIZES,
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_NUM_CONFIGS,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE
)
from cifar_reporting import (
    generate_test_report,
    visualize_training_progress,
    update_website,
    generate_sample_size_comparison_report,
    generate_meta_model_report
)

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)


def save_result_to_file(dataset_name, sample_size, result_data):
    """Save the result data for a specific sample size to a JSON file.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        sample_size: Sample size used for training
        result_data: Dictionary with result data
        
    Returns:
        Path to the saved file
    """
    # Create the model-specific directory in reports using sample size in the name
    model_type = f"cifa{10 if dataset_name == 'cifar10' else 100}"
    if sample_size < 1.0:
        model_type = f"{model_type}_{int(sample_size*100)}"  # e.g., cifa10_10 for 10% sample size
    
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


def train_cifar(dataset='cifar10', mode='single', sample_size=0.1, epochs=DEFAULT_EPOCHS,
                batch_size=DEFAULT_BATCH_SIZE, num_iterations=DEFAULT_NUM_ITERATIONS,
                num_configs=DEFAULT_NUM_CONFIGS, max_training_hours=24.0):
    """
    Unified function to train CIFAR models with different configurations.
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        mode: 'single', 'comparison', or 'multisize'
        sample_size: Sample size as a fraction (0.0-1.0) for single mode
        epochs: Number of training epochs
        batch_size: Batch size for training
        num_iterations: Number of meta-model iterations
        num_configs: Number of configurations to try per iteration
        max_training_hours: Maximum training time in hours (default: 6)
        
    Returns:
        Training results or None if training failed
    """
    # Set up CUDA memory management
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Log initial memory usage
    def log_memory_usage():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    
    logger.info("Starting training with memory usage:")
    log_memory_usage()
    
    # Track start time for timeout
    start_time = time.time()
    max_training_seconds = max_training_hours * 3600
    
    try:
    
        if mode in ['comparison', 'multisize']:
            # For comparison/multisize mode, use the run_sample_size_comparison function
            sample_sizes = DEFAULT_SAMPLE_SIZES if mode == 'multisize' else [float(s) for s in sample_size.split(',')]
            logger.info(f"Running in {mode} mode with sample sizes: {sample_sizes}")
            
            # Calculate remaining time for the entire comparison
            elapsed = time.time() - start_time
            remaining_time = max(0, max_training_seconds - elapsed)
            
            if remaining_time <= 0:
                raise TimeoutError("Maximum training time reached before starting comparison")
            
            results = run_sample_size_comparison(
                dataset_name=dataset,
                sample_sizes=sample_sizes,
                epochs=epochs,
                batch_size=batch_size,
                num_iterations=num_iterations,
                num_configs=num_configs,
                max_training_hours=remaining_time/3600  # Convert to hours
            )
            return results
            
        else:  # single mode
            # For single mode, just train one model with the specified sample size
            logger.info(f"Running in single mode with sample size: {sample_size}")
            
            # Calculate remaining time for meta-optimization
            elapsed = time.time() - start_time
            remaining_time = max(0, max_training_seconds - elapsed)
            
            if remaining_time <= 0:
                raise TimeoutError("Maximum training time reached before starting meta-optimization")
            
            # Allocate 1/3 of remaining time to meta-optimization, 2/3 to final training
            meta_time = remaining_time / 3
            
            logger.info(f"Starting meta-optimization with max time: {meta_time/3600:.1f} hours")
            best_config = run_meta_optimization(
                dataset_name=dataset,
                sample_size=sample_size,
                num_iterations=num_iterations,
                num_configs=num_configs,
                max_training_hours=meta_time/3600
            )
            
            # Calculate remaining time for final training
            elapsed = time.time() - start_time
            remaining_time = max(0, max_training_seconds - elapsed)
            
            if remaining_time <= 0:
                raise TimeoutError("Maximum training time reached before starting final training")
            
            logger.info(f"Starting final training with max time: {remaining_time/3600:.1f} hours")
            
            # Train the final model with the best configuration
            result = train_and_evaluate(
                config=best_config,
                dataset_name=dataset,
                epochs=epochs,
                batch_size=batch_size,
                sample_size=sample_size,
                max_training_time=remaining_time
            )
            
            return result
            
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.exception("Error details:")
        
        # Log memory usage on error
        if torch.cuda.is_available():
            logger.error("Memory usage at time of error:")
            log_memory_usage()
        
        # Re-raise the exception to be handled by the caller
        raise


def run_sample_size_comparison(dataset_name, sample_sizes=DEFAULT_SAMPLE_SIZES, 
                               epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                               num_iterations=DEFAULT_NUM_ITERATIONS, num_configs=DEFAULT_NUM_CONFIGS,
                               max_training_hours=6.0):
    """Run a comparison of different sample sizes for meta-model training.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        sample_sizes: List of sample sizes to compare
        epochs: Number of epochs for final training
        batch_size: Batch size for training
        num_iterations: Number of meta-model iterations
        num_configs: Number of configurations to try per iteration
        max_training_hours: Maximum training time in hours (default: 6)
        
    Returns:
        Dictionary with results for each sample size
    """
    logger.info(f"Running sample size comparison for {dataset_name} with sizes {sample_sizes}")
    
    results = {}
    
    for sample_size in sample_sizes:
        logger.info(f"\n\n==== Processing sample size {sample_size} ====\n")
        
        # Run meta-model optimization to find the best hyperparameters
        best_config = run_meta_optimization(
            dataset_name, sample_size, num_iterations=num_iterations, num_configs=num_configs
        )
        
        # Train a full model with the best hyperparameters
        eval_result = train_and_evaluate(
            best_config, dataset_name, epochs=epochs, batch_size=batch_size, sample_size=1.0
        )
        
        # Generate meta-model training progress report
        meta_report_path = generate_meta_model_report(dataset_name, sample_size)
        
        # Generate model training progress visualization
        vis_path = visualize_training_progress(dataset_name, sample_size)
        
        # Generate test report
        test_report_path = generate_test_report(best_config, dataset_name, sample_size)
        
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
        save_result_to_file(dataset_name, sample_size, result_data)
        
        # Store in memory for this run
        results[sample_size] = result_data
        
        # Update the comparison report incrementally
        generate_sample_size_comparison_report(dataset_name, sample_size=sample_size)
        
        logger.info(f"Completed processing for sample size {sample_size}")
        logger.info(f"Reports generated and stored in the reports directory.")
    
    # Generate a final comparison report with all results
    comparison_report = generate_sample_size_comparison_report(dataset_name, results=results)
    
    logger.info(f"Sample size comparison completed.")
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
                        help='Training mode: single sample size, comparison of multiple sizes, or multisize (all sizes)')
    
    # Sample size for single mode
    parser.add_argument('--sample-size', type=float, default=0.1,
                        help='Sample size (fraction of dataset) for single mode')
    
    # Sample sizes for comparison mode
    parser.add_argument('--sample-sizes', type=str, default='0.1,0.2,0.3,0.4',
                        help='Comma-separated list of sample sizes for comparison mode')
    
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
    parser.add_argument('--max-training-hours', type=float, default=24.0,
                        help='Maximum training time in hours (default: 24, set to 0 for no limit)')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.9,
                        help='Fraction of GPU memory to use (0.0-1.0, default: 0.9)')
    
    # Reporting options
    parser.add_argument('--skip-reports', action='store_true',
                        help='Skip generating reports (useful for quick testing)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process sample sizes for comparison mode
    if args.mode in ['comparison', 'multisize']:
        sample_sizes = [float(size) for size in args.sample_sizes.split(',')]
    else:
        sample_sizes = [args.sample_size]
    
    # Run the appropriate mode
    if args.mode in ['comparison', 'multisize']:
        results = run_sample_size_comparison(
            args.dataset,
            sample_sizes=sample_sizes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
            num_configs=args.num_configs,
            max_training_hours=args.max_training_hours
        )
    else:  # single mode
        # Train a model with the current sample size
        result = train_cifar(
            dataset=args.dataset,
            mode=args.mode,
            sample_size=args.sample_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
            num_configs=args.num_configs,
            max_training_hours=args.max_training_hours
        )
        
        if result is None:
            logger.error(f"Training failed for sample size {args.sample_size}")
            return
        
        # Prepare results dictionary
        result_data = {
            'best_config': result['best_config'],
            'eval_result': result['eval_result'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate reports if not skipped
        if not args.skip_reports:
            # Determine model type for directory structure
            model_type = f"cifa{10 if args.dataset == 'cifar10' else 100}"
            if args.sample_size < 1.0:
                model_type = f"{model_type}_{int(args.sample_size*100)}"
            
            # Generate meta-model training progress report with fixed filename
            meta_report_path = generate_meta_model_report(args.dataset, args.sample_size, 
                                                       output_path=f"reports/{model_type}/latest_meta_report.html")
            result_data['meta_model_report_path'] = meta_report_path
            
            # Generate model training progress visualization with fixed filename
            vis_path = visualize_training_progress(args.dataset, args.sample_size,
                                                output_path=f"reports/{model_type}/latest_training_progress.html")
            result_data['training_progress_path'] = vis_path
            
            # Generate test report with fixed filename
            test_report_path = generate_test_report(best_config, args.dataset, args.sample_size,
                                                 output_path=f"reports/{model_type}/latest_test_report.html")
            result_data['test_report_path'] = test_report_path
            
            # Save the result to a file in the model-specific directory
            save_result_to_file(args.dataset, args.sample_size, result_data)
            
            # Update the comparison report incrementally
            generate_sample_size_comparison_report(args.dataset, 
                                               output_path=f"reports/{args.dataset}/latest_sample_size_comparison.html")
            
            logger.info(f"Completed processing for sample size {args.sample_size}")
            logger.info(f"Reports generated and stored in the reports/{model_type}/ directory.")
            logger.info(f"Training logs saved to reports/{model_type}/{model_type}_training_log.txt")
            logger.info(f"Use the deployment menu or 'mksite' to update the website with these reports.")
        
        results = result_data
    
    return results


if __name__ == '__main__':
    main()
