#!/usr/bin/env python3
"""
Test script to verify meta-model training example saving functionality.
"""
import os
import sys
import logging
from llp.meta_model import HyperparameterPredictor, MetaModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_training_example_saving():
    """Test that training examples are saved correctly."""
    # Create a predictor with a test directory
    test_dir = os.path.join(os.path.dirname(__file__), 'test_meta_models')
    predictor = HyperparameterPredictor(test_dir)
    trainer = MetaModelTrainer(predictor)
    
    # Create a test example
    dataset_features = {
        'num_samples': 1000,
        'num_features': 32 * 32 * 3,  # CIFAR-10
        'num_classes': 10,
        'class_imbalance': 0.1,
        'class_entropy': 2.3,
        'feature_mean': 0.5,
        'feature_std': 0.25,
        'feature_skew': 0.1,
        'feature_kurtosis': -0.5,
        'feature_correlation_mean': 0.05,
        'feature_correlation_std': 0.1
    }
    
    training_features = {
        'initial_loss': 2.3,
        'final_loss': 0.1,
        'loss_decrease_rate': 2.2,
        'initial_accuracy': 0.1,
        'final_accuracy': 0.9,
        'accuracy_increase_rate': 0.8,
        'sharpness': 0.5,
        'perturbation_robustness': 0.8,
        'resource_level': 0.7,
        'epochs': 10
    }
    
    hyperparameters = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'optimizer': 'adam',
        'weight_decay': 0.0001,
        'dropout_rate': 0.5
    }
    
    # Add the example
    logger.info("Adding training example...")
    success = predictor.add_training_example(
        dataset_features=dataset_features,
        training_features=training_features,
        hyperparameters=hyperparameters,
        performance=0.9  # accuracy
    )
    
    if success:
        logger.info("✓ Successfully added training example")
        
        # Verify the file was created
        example_files = predictor._find_example_files()
        if example_files:
            logger.info(f"Found {len(example_files)} example files:")
            for f in example_files:
                logger.info(f"  - {f}")
        else:
            logger.error("✗ No example files found!")
            return False
            
        return True
    else:
        logger.error("✗ Failed to add training example")
        return False

if __name__ == "__main__":
    logger.info("Starting meta-model saving test...")
    if test_training_example_saving():
        logger.info("Test completed successfully!")
    else:
        logger.error("Test failed!")
        sys.exit(1)
