#!/usr/bin/env python3
"""
Unified CIFAR Training with Meta-Model Hyperparameter Optimization
==========================================================

This script implements a modular meta-model approach for CIFAR-10/100 training with
distinct phases:
1. Architecture setup
2. Hyperparameter meta-model training
3. Prediction of best hyperparameters
4. Training with the predicted hyperparameters
5. Testing to check accuracy
"""

import argparse
import logging
import time
from pathlib import Path
import random
import json
import torch
import numpy as np
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# We don't need a custom JSON encoder since we're handling all conversions
# in the optimizer class before returning the hyperparameters

# Import our rich logging utilities
from .rich_logging import (
    setup_rich_logging,
    print_header,
    print_config,
    TrainingProgress
)

# Import new report utilities
from llp.report_utils import generate_test_report

# Set up logging - only do this once at the module level
logger = setup_rich_logging(level=logging.INFO)

# Configure root logger to prevent duplicate logs
root_logger = logging.getLogger()
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules using relative imports
from .cifar_core import get_cifar_loaders, get_best_known_config, get_cifar100_transfer_config
from .cifar_meta_model import CIFARMetaModelOptimizer
from .cifar_trainer import CIFARTrainer
from .cifar_reporting import CIFARReporter


def parse_args():
    p = argparse.ArgumentParser(description="Unified CIFAR Training with Meta-Model Hyperparameter Optimization")
    
    # Dataset options
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"],
                   help="Dataset to use (cifar10 or cifar100)")
    p.add_argument("--sample-size", dest="sample_size", default="base",
                   help="Resource level (base, 10, 20, 30, 40, or transfer for cifar100)")
    p.add_argument("--data-fraction", type=float, default=1.0,
                   help="Fraction of training data to use (0.1 to 1.0)")
    
    # Meta-model options
    p.add_argument("--optimizer", choices=["meta", "successive_halving"], default="meta",
                   help="Optimization strategy (meta: meta-model, successive_halving: successive halving)")
    p.add_argument("--parallel-mode", choices=["none", "thread", "process", "gpu"], default="thread",
                   help="Parallelization mode for meta-model training")
    p.add_argument("--meta-model-only", action="store_true",
                   help="Only run the meta-model phase without final training")
    p.add_argument("--configs-per-sample", type=int, default=10,
                   help="Number of hyperparameter configurations to sample per iteration")
    p.add_argument("--perturbations", type=int, default=5,
                   help="Number of weight perturbations per configuration")
    p.add_argument("--iterations", type=int, default=3,
                   help="Number of meta-model training iterations")
    p.add_argument("--min-resource", type=float, default=0.1,
                   help="Minimum resource level for meta-model training")
    p.add_argument("--max-resource", type=float, default=0.5,
                   help="Maximum resource level for meta-model training")
    p.add_argument("--parallel-workers", type=int, default=0,
                   help="Number of parallel workers for meta-model training (0 for auto)")
    
    # Training options
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of epochs for final training")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Batch size for training")
    p.add_argument("--learning-rate", type=float, default=0.1,
                   help="Learning rate for training")
    p.add_argument("--weight-decay", type=float, default=5e-4,
                   help="Weight decay for optimization")
    p.add_argument("--momentum", type=float, default=0.9,
                   help="Momentum for optimizer")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Number of worker processes for data loading")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--skip-meta-model", action="store_true",
                   help="Skip meta-model and use default hyperparameters")
    p.add_argument("--use-saved-hyperparams", action="store_true",
                   help="Use previously saved hyperparameters instead of running meta-model")
    p.add_argument("--test", action="store_true",
                   help="Run testing and generate comprehensive report")
    
    # Output options
    p.add_argument("--outdir", default=None,
                   help="Optional explicit output directory (defaults to reports/<dataset>_<sample_size>/)")
    p.add_argument("--save-model", action="store_true", default=True,
                   help="Save the trained model")
    p.add_argument("--generate-report", action="store_true", default=True,
                   help="Generate HTML report after training")
    
    return p.parse_args()


def main():
    # Track total execution time
    start_time = time.time()
    
    args = parse_args()
    
    # Initialize best_config with default value
    best_config = get_best_known_config(args.dataset)
    
    # Convert sample_size to float for data_fraction if it's a percentage
    if args.sample_size.isdigit():
        data_fraction = float(args.sample_size) / 100.0
        dir_name = f"{args.dataset}_{args.sample_size}"
    elif args.sample_size == "base":
        data_fraction = 1.0
        dir_name = f"{args.dataset}"
    elif args.dataset == "cifar100" and args.sample_size == "transfer":
        data_fraction = 1.0
        dir_name = f"{args.dataset}_transfer"
    else:
        data_fraction = 1.0
        dir_name = f"{args.dataset}_{args.sample_size}"
    
    # Set up directory names based on dataset and sample size
    if args.dataset == "cifar100" and args.sample_size == "transfer":
        data_fraction = 1.0
        sample_dir = f"{args.dataset}_transfer"
    else:
        data_fraction = 1.0
        dir_name = f"{args.dataset}_{args.sample_size}"
        # Ensure the sample_size is properly formatted (e.g., '10' becomes '10' not '10.0')
        if dir_name.endswith('.0'):
            dir_name = dir_name[:-2]
    
    # Set up base directories
    base_dir = Path(__file__).parent.parent  # Go up to project root
    
    # Set up reports directory structure
    if args.outdir:
        run_dir = Path(args.outdir)
    else:
        # Use the format: reports/cifar10/cifar10_10/
        run_dir = base_dir / "reports" / args.dataset / dir_name
    
    # Create reports directory and ensure it exists
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model_dir to None
    model_dir = None
    
    # Set up model directory structure if saving models
    if args.save_model:
        # Use the format: models/cifar10/cifar10_10/
        model_dir = base_dir / "models" / args.dataset / dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (model_dir / "checkpoints").mkdir(exist_ok=True)
        (model_dir / "meta_model").mkdir(exist_ok=True)
        (model_dir / "logs").mkdir(exist_ok=True)
    
    # Set up rich logging with file output to the logs directory
    log_file = (model_dir / "logs" / 'training.log') if args.save_model else (run_dir / 'training.log')
    
    # Clear any existing log file to ensure we start fresh
    if log_file.exists():
        log_file.unlink()
    logger = setup_rich_logging(log_file=log_file, level=logging.INFO)
    
    # Print header with run information
    print_header("HYPERPARAMETER OPTIMIZATION")
    
    # Print configuration
    config_info = {
        "Optimizer": args.optimizer,
        "Dataset": args.dataset,
        "Sample Size": args.sample_size,
        "Batch Size": args.batch_size,
        "Epochs": args.epochs,
        "Data Fraction": f"{args.data_fraction*100:.1f}%" if hasattr(args, 'data_fraction') else "100.0%",
        "Configs/Sample": args.configs_per_sample,
        "Perturbations": args.perturbations,
        "Iterations": args.iterations,
        "Min Resource": args.min_resource,
        "Max Resource": args.max_resource,
        "Run Directory": str(run_dir.absolute())
    }
    print_config(config_info, "Training Configuration")
    
    # Write initial header
    logger.info("=" * 60)
    logger.info(f"STARTING TRAINING SESSION - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file.absolute()}")
    logger.info(f"Run directory: {run_dir.absolute()}")
    logger.info("-" * 60)
    
    logger.info(f"Starting unified CIFAR training for {args.dataset} with sample size {args.sample_size}")
    logger.info(f"Reports directory: {run_dir}")
    logger.info(f"Models directory: {model_dir if args.save_model else 'Not saving models'}")
    
    # Determine meta-model directory path
    meta_model_dir = model_dir / "meta_model" if args.save_model else run_dir / "meta_model"
    
    # Special case for CIFAR-100 transfer learning
    if args.dataset == "cifar100" and args.sample_size == "transfer":
        logger.info("Using transfer learning configuration for CIFAR-100")
        best_config = get_cifar100_transfer_config()
    # Skip meta-model if requested
    elif args.skip_meta_model:
        logger.info("Skipping meta-model, using best known configuration")
        best_config = get_best_known_config(args.dataset)
    # Use saved hyperparameters if requested
    elif args.use_saved_hyperparams:
        # Check both possible locations for saved hyperparameters
        meta_results_path = meta_model_dir / "meta_results.json"
        if not meta_results_path.exists():
            meta_results_path = run_dir / "meta_model" / "meta_results.json"
        
        if meta_results_path.exists():
            logger.info(f"Using previously saved hyperparameters from {meta_results_path}")
            try:
                with open(meta_results_path, "r") as f:
                    meta_results = json.load(f)
                    best_config = meta_results["best_config"]
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error loading saved hyperparameters: {e}")
                logger.info("Falling back to best known configuration")
                best_config = get_best_known_config(args.dataset)
        else:
            logger.warning("No saved hyperparameters found, running meta-model optimization")
            args.use_saved_hyperparams = False
    
    # Create meta-model optimizer
    meta_optimizer = CIFARMetaModelOptimizer(
        dataset=args.dataset,
        batch_size=args.batch_size,
        configs_per_sample=args.configs_per_sample,
        perturbations=args.perturbations,
        iterations=args.iterations,
        min_resource=args.min_resource,
        max_resource=args.max_resource,
        run_dir=run_dir
    )
    
    # Constants
    BEST_CONFIG_FILENAME = "best_config.json"
    META_MODEL_DIR = run_dir / "meta_model" if run_dir else None
    
    # Log optimization parameters
    logger.info("\n" + "="*50)
    logger.info("HYPERPARAMETER OPTIMIZATION")
    logger.info("="*50)
    logger.info(f"Optimizer:         {args.optimizer}")
    logger.info(f"Dataset:           {args.dataset}")
    logger.info(f"Sample Size:       {args.sample_size}")
    logger.info(f"Batch Size:        {args.batch_size}")
    logger.info(f"Epochs:            {args.epochs}")
    logger.info(f"Data Fraction:     {data_fraction*100:.1f}%")
    logger.info(f"Configs/Sample:    {args.configs_per_sample}")
    logger.info(f"Perturbations:     {args.perturbations}")
    logger.info(f"Iterations:        {args.iterations}")
    logger.info(f"Min Resource:      {args.min_resource}")
    logger.info(f"Max Resource:      {args.max_resource}")
    logger.info(f"Run Directory:     {run_dir.absolute() if run_dir else 'None'}")
    logger.info("-"*50 + "\n")
    
    # Check if we should use saved hyperparameters
    if args.use_saved_hyperparams and META_MODEL_DIR and (META_MODEL_DIR / BEST_CONFIG_FILENAME).exists():
        with open(META_MODEL_DIR / BEST_CONFIG_FILENAME, "r") as f:
            best_hyperparams = json.load(f)
        logger.info(f"Loaded saved hyperparameters: {best_hyperparams}")
        return best_hyperparams
    
    # Check if we should skip optimization
    if args.skip_meta_model:
        best_hyperparams = get_best_known_config()
        logger.info(f"Skipping optimization, using default hyperparameters: {best_hyperparams}")
        return best_hyperparams
    
    # Run selected optimization strategy
    if args.optimizer == "successive_halving":
        logger.info("Starting successive halving hyperparameter optimization...")
        # Initialize meta-probing for successive halving
        meta_probe = MetaProbing(
            configs=meta_optimizer._sample_hyperparameter_configs(args.configs_per_sample),
            model_fn=create_model,
            dataset_fn=get_cifar_loaders,
            criterion=nn.CrossEntropyLoss(),
            optimizer_fn=create_optimizer,
            max_epochs=args.iterations,
            device=meta_optimizer.device,
            meta_model_dir=str(META_MODEL_DIR) if META_MODEL_DIR else None
        )
        
        # Run successive halving
        results = meta_probe.run_successive_halving(
            min_resource=int(args.min_resource * 100),  # Convert to percentage
            max_resource=int(args.max_resource * 100),  # Convert to percentage
            reduction_factor=2,
            measure_flatness=True
        )
        
        # Get the best configuration
        if results:
            results.sort(key=lambda x: x.val_acc, reverse=True)
            best_hyperparams = results[0].config
            logger.info(f"Best configuration found with val_acc: {results[0].val_acc:.4f}")
        else:
            logger.warning("No results returned from successive halving - using default config")
            best_hyperparams = get_best_known_config()
    else:
        # Default to meta-model optimization
        logger.info("Starting meta-model hyperparameter optimization...")
        # Train meta-model and get the best hyperparameters for final training
        best_hyperparams = meta_optimizer.optimize_hyperparameters()
        
        # Check if we have valid hyperparameters
        if not best_hyperparams or len(best_hyperparams) == 0:
            logger.error("\n" + "!" * 70)
            logger.error("ERROR: No valid hyperparameters found.")
            logger.error("The meta-model optimization did not complete successfully.")
            logger.error("!" * 70 + "\n")
            # Use default hyperparameters as fallback
            best_hyperparams = {
                'learning_rate': 0.01,
                'weight_decay': 1e-4,
                'optimizer': 'sgd',
                'momentum': 0.9,
                'batch_size': 128
            }
            logger.info("Using fallback default hyperparameters instead.")
        
        logger.info("Meta-model training complete. Using best hyperparameters for final model training.")
        
        # Log the best hyperparameters we'll use for the final training run
        logger.info("Best hyperparameters for final CIFAR model training:")
        for key, value in best_hyperparams.items():
            # Skip any model state or non-serializable values in logs
            if key not in ['model_state', 'model', 'state_dict']:
                logger.info(f"  {key}: {value}")
            
        # Save the best hyperparameters to a JSON file
        best_config_path = run_dir / "meta_model" / "best_config.json"
        best_config_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Make sure we're only saving serializable data
        serializable_hyperparams = {k: v for k, v in best_hyperparams.items() 
                                  if k not in ['model_state', 'model', 'state_dict', 'optimizer_state']}
        
        with open(best_config_path, "w") as f:
            json.dump(serializable_hyperparams, f, indent=2)
        logger.info(f"Saved best configuration to {best_config_path}")
    
    # If we used the meta-model optimizer, report simplified parallel timing metrics
    if args.optimizer == "meta" and 'meta_model_optimizer' in locals():
        try:
            # Extract parallel timing information if available
            parallel_info = meta_model_optimizer.get_parallel_timing_info() if hasattr(meta_model_optimizer, 'get_parallel_timing_info') else {}
            
            if parallel_info:
                logger.info("\nParallel Execution Summary:")
                logger.info(f"  - Mode: {parallel_info.get('parallel_mode', 'Not detected')} | Workers: {parallel_info.get('num_workers', 1)}")
                
                if 'efficiency' in parallel_info and 'speedup' in parallel_info:
                    efficiency = parallel_info['efficiency']
                    logger.info(f"  - Efficiency: {efficiency:.2%} | Speedup: {parallel_info['speedup']:.2f}x")
                    
                    # Only show warnings for very poor efficiency
                    if efficiency < 0.4:
                        logger.warning("  - Low efficiency detected. Consider reducing parallel workers.")
                
                # Only log timing variance if it's problematic
                if 'timing_variance' in parallel_info and parallel_info['timing_variance'] > 0.3:
                    logger.warning(f"  - High timing variance: {parallel_info['timing_variance']:.2%}")
        except Exception as e:
            # Fail fast but don't crash the program
            logger.error(f"Error reporting parallel timing: {e}")
    
    logger.info("\nBest Configuration:")
    
    # Pretty print the best configuration
    for key, value in best_hyperparams.items():
        logger.info(f"  {key:<20}: {value}")
    
    # Stop here if only meta-model phase is requested
    if args.meta_model_only:
        logger.info("Meta-model only mode, skipping training phase")
        return best_hyperparams
    
    # Check if optimization was interrupted
    if best_hyperparams and best_hyperparams.get('interrupted', False):
        logger.warning("\n" + "!"*80)
        logger.warning("Optimization was interrupted. Exiting...")
        logger.warning("!"*80 + "\n")
        sys.exit(0)
        
    # Verify we have valid hyperparameters
    if not best_hyperparams or 'best_config' not in best_hyperparams or not best_hyperparams['best_config']:
        logger.error("\n" + "!"*80)
        logger.error("ERROR: No valid hyperparameters found.")
        logger.error("The meta-model optimization did not complete successfully.")
        logger.error("!"*80 + "\n")
        sys.exit(1)
    
    # Extract the actual hyperparameters
    hyperparams = best_hyperparams['best_config']
    
    # Ensure all required parameters are present
    required_params = ['num_channels', 'learning_rate', 'batch_size', 'weight_decay']
    missing_params = [p for p in required_params if p not in hyperparams]
    
    if missing_params:
        logger.error("\n" + "!"*80)
        logger.error(f"ERROR: Missing required hyperparameters: {', '.join(missing_params)}")
        logger.error("This indicates the optimization didn't complete successfully.")
        logger.error("="*80 + "\n")
        sys.exit(1)
    
    # Log the hyperparameters we'll be using
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    logger.info("-"*80)
    logger.info(f"Run directory: {run_dir.absolute()}")
    logger.info(f"Model directory: {model_dir.absolute() if args.save_model else 'Not saving models'}")
    logger.info(f"Log file: {log_file.absolute()}")
    logger.info("-"*80)
    for key, value in hyperparams.items():
        logger.info(f"  {key:<20}: {value}")
    logger.info("="*80 + "\n")
    
    # Determine checkpoint directory for model saving
    checkpoint_dir = Path(__file__).parent / "models" / dir_name / "checkpoints" if args.save_model else run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize trainer with appropriate directories
    trainer = CIFARTrainer(
        config=hyperparams,  # Use the actual hyperparameters, not the full results dict
        dataset=args.dataset,
        data_fraction=data_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir if args.save_model else None
    )
    
    try:
        # Phase 4: Train the model
        training_metrics = trainer.train()
        
        # Plot training history
        trainer.plot_training_history()
        
        # Phase 5: Test the model
        logger.info("Testing the model")
        test_metrics = trainer.test()
        
        # Run testing and generate comprehensive report
        if args.test:
            logger.info("\n" + "="*60)
            logger.info("RUNNING TESTING AND GENERATING REPORT")
            logger.info("="*60)
            
            # Get class names based on dataset
            class_names = [f"class_{i}" for i in range(10)]  # Default for CIFAR-10
            if args.dataset == 'cifar100':
                class_names = [f"class_{i}" for i in range(100)]
            
            # Generate comprehensive test report
            test_metrics = generate_test_report(
                model=trainer.model,
                test_loader=trainer.test_loader,
                class_names=class_names,
                output_dir=run_dir,
                dataset_name=args.dataset,
                sample_size=args.sample_size if args.sample_size.isdigit() else "100",
                device=trainer.device
            )
            
            logger.info("\n" + "="*60)
            logger.info("TESTING COMPLETE")
            logger.info("="*60)
            logger.info(f"Test accuracy: {test_metrics['accuracy']:.2%}")
            logger.info(f"Report saved to: {run_dir}/{args.dataset}_{args.sample_size}_test_report.html")
        
        # Generate reports if requested
        if args.generate_report:
            logger.info("Generating reports")
            reporter = CIFARReporter(
                dataset=args.dataset,
                sample_size=args.sample_size,
                run_dir=run_dir
            )
            reporter.generate_all_reports()
            
            # Log report locations
            logger.info(f"Training report: {run_dir}/{args.dataset}_{args.sample_size}_training_report.html")
            logger.info(f"Test report: {run_dir}/{args.dataset}_{args.sample_size}_test_report.html")
            logger.info(f"Meta-model report: {run_dir}/{args.dataset}_{args.sample_size}_meta_report.html")
        
        logger.info(f"Training completed. Best validation accuracy: {training_metrics['best_val_acc']:.4f}")
        logger.info(f"Test accuracy: {test_metrics['test_acc']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during training or testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info(f"All results saved to {run_dir}")
    if args.save_model:
        logger.info(f"Model checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
