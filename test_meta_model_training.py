#!/usr/bin/env python3
"""
Test script to verify meta-model training with example files.
This script creates sample training examples and then trains the meta-model.
"""

import os
import sys
import logging
import time
import random
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from llp.meta_model import HyperparameterPredictor, MetaModelTrainer

logger = logging.getLogger(__name__)

def create_sample_examples(model_dir, num_examples=5):
    """Create sample training examples in the specified directory."""
    os.makedirs(model_dir, exist_ok=True)
    
    predictor = HyperparameterPredictor(model_dir=model_dir)
    
    examples_added = 0
    for i in range(num_examples):
        # Create sample features
        dataset_features = {
            'num_classes': 10,
            'num_samples': 50000,
            'input_dim': 3072,
            'class_balance': 0.95 + random.random() * 0.05,
            'feature_correlation': 0.2 + random.random() * 0.3
        }
        
        training_features = {
            'epoch': 10,
            'train_loss': 0.5 - random.random() * 0.3,
            'val_loss': 0.6 - random.random() * 0.3,
            'train_acc': 0.7 + random.random() * 0.2,
            'val_acc': 0.65 + random.random() * 0.2,
            'loss_variance': 0.05 + random.random() * 0.05,
            'sharpness': 0.1 + random.random() * 0.2
        }
        
        hyperparameters = {
            'learning_rate': 10**random.uniform(-4, -2),
            'batch_size': random.choice([32, 64, 128, 256]),
            'optimizer': random.choice(['sgd', 'adam', 'adamw']),
            'weight_decay': 10**random.uniform(-5, -3),
            'dropout_rate': random.uniform(0.1, 0.5)
        }
        
        # Performance score (higher is better)
        performance = 0.7 + random.random() * 0.25
        
        # Add example
        success = predictor.add_training_example(
            dataset_features=dataset_features,
            training_features=training_features,
            hyperparameters=hyperparameters,
            performance=performance
        )
        
        if success:
            examples_added += 1
            logger.info(f"Added example {i+1} with performance {performance:.4f}")
        else:
            logger.warning(f"Failed to add example {i+1}")
    
    logger.info(f"Added {examples_added}/{num_examples} examples successfully")
    return examples_added

def test_meta_model_training():
    """Test the meta-model training process."""
    logger.info("=" * 80)
    logger.info("TESTING META-MODEL TRAINING")
    logger.info("=" * 80)
    
    # Set up model directory
    model_dir = os.path.join(project_root, 'models', 'cifar10', 'cifar10_10', 'meta_models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Create sample examples
    logger.info(f"Creating sample examples in {model_dir}")
    num_examples = 10
    examples_added = create_sample_examples(model_dir, num_examples)
    
    # Check for example files
    predictor = HyperparameterPredictor(model_dir=model_dir)
    example_files = predictor._find_example_files()
    
    if example_files:
        logger.info(f"Found {len(example_files)} example files")
        for i, file in enumerate(example_files[:5]):  # Show first 5 only
            logger.info(f"  - {os.path.basename(file)}")
        if len(example_files) > 5:
            logger.info(f"  - ... and {len(example_files) - 5} more")
    else:
        logger.warning("No example files found!")
        return False
    
    # Train meta-model
    logger.info("\nTraining meta-model...")
    trainer = MetaModelTrainer(predictor=predictor)
    
    try:
        trainer.train_meta_model()
        logger.info("✓ Meta-model training completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Meta-model training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_meta_model_training()
    if success:
        logger.info("\n✓ Test completed successfully!")
    else:
        logger.error("\n✗ Test failed!")
        sys.exit(1)
