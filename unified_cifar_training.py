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
import time
from pathlib import Path
import random
import json
import logging
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

# Initialize logger first without file handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from cifar_core import get_cifar_loaders, get_best_known_config, get_cifar100_transfer_config
from cifar_meta_model import CIFARMetaModelOptimizer
from cifar_training import CIFARTrainer
from cifar_reporting import CIFARReporter


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
    
    # Training options
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of epochs for final training")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Batch size for training")
    p.add_argument("--skip-meta-model", action="store_true",
                   help="Skip meta-model and use default hyperparameters")
    p.add_argument("--use-saved-hyperparams", action="store_true",
                   help="Use previously saved hyperparameters instead of running meta-model")
    
    # Output options
    p.add_argument("--outdir", default=None,
                   help="Optional explicit output directory (defaults to reports/<dataset>_<sample_size>/)")
    p.add_argument("--save-model", action="store_true", default=True,
                   help="Save the trained model")
    p.add_argument("--generate-report", action="store_true", default=True,
                   help="Generate HTML report after training")
    
    return p.parse_args()


def main():
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
    
    # Set up output directory based on the directory structure convention
    if args.outdir:
        run_dir = Path(args.outdir)
    else:
        run_dir = Path(__file__).parent / "reports" / dir_name
        

    
    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up file handler for logging
    log_file = run_dir / f"{args.dataset}_{args.sample_size}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Create model directory structure if saving models
    if args.save_model:
        model_dir = Path(__file__).parent / "models" / dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (model_dir / "checkpoints").mkdir(exist_ok=True)
        (model_dir / "meta_model").mkdir(exist_ok=True)
        (model_dir / "logs").mkdir(exist_ok=True)
    
    logger.info(f"Starting unified CIFAR training for {args.dataset} with sample size {args.sample_size}")
    logger.info(f"Output directory: {run_dir}")
    
    # Determine meta-model directory path based on whether we're saving models
    meta_model_dir = Path(__file__).parent / "models" / dir_name / "meta_model" if args.save_model else run_dir / "meta_model"
    
    # Log the data fraction being used
    logger.info(f"Using data fraction: {data_fraction:.1%} of training data")
    
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
    
    # Run meta-model optimization if needed
    if not args.skip_meta_model and not args.use_saved_hyperparams and args.sample_size != "transfer":
        logger.info("Starting meta-model hyperparameter optimization")
        
        # Initialize with default configuration
        best_config = get_best_known_config(args.dataset)
        
        try:
            # Phase 1: Meta-model hyperparameter optimization
            meta_optimizer = CIFARMetaModelOptimizer(
                dataset=args.dataset,
                data_fraction=args.data_fraction,
                batch_size=args.batch_size,
                configs_per_sample=args.configs_per_sample,
                perturbations=args.perturbations,
                iterations=args.iterations,
                min_resource=args.min_resource,
                max_resource=args.max_resource,
                run_dir=meta_model_dir.parent if args.save_model else run_dir
            )
            
            # Phase 2: Train meta-model and predict best hyperparameters
            best_config = meta_optimizer.train_meta_model()
            
            logger.info(f"Meta-model optimization completed. Best configuration: {best_config}")
        except Exception as e:
            logger.error(f"Error during meta-model optimization: {e}")
            logger.info("Falling back to best known configuration")
    
    # Stop here if only meta-model phase is requested
    if args.meta_model_only:
        logger.info("Meta-model only mode, skipping training phase")
        return
    
    # Phase 3: Train model with predicted hyperparameters
    logger.info(f"Starting training with predicted hyperparameters: {best_config}")
    
    # Determine checkpoint directory for model saving
    checkpoint_dir = Path(__file__).parent / "models" / dir_name / "checkpoints" if args.save_model else run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize trainer with appropriate directories
    trainer = CIFARTrainer(
        config=best_config,
        dataset=args.dataset,
        data_fraction=data_fraction,  # Use the calculated data_fraction
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
