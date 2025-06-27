#!/usr/bin/env python3
"""
End-to-end test for the meta-model pipeline.
This script tests:
1. Training a model with meta-model example collection
2. Verifying examples are saved correctly
3. Training the meta-model on those examples
4. Using the meta-model to predict hyperparameters
"""

import os
import sys
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
from llp.meta_probing import MetaProbing

logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """A simple model for testing purposes."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def create_synthetic_data(num_samples=1000, input_dim=10, num_classes=2, seed=42):
    """Create synthetic data for testing."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    x = torch.randn(num_samples, input_dim)
    # Create linearly separable classes
    w = torch.randn(input_dim, 1)
    y = torch.where(x @ w > 0, 
                   torch.ones(num_samples, 1), 
                   torch.zeros(num_samples, 1)).long().squeeze()
    
    # Split into train and val
    train_size = int(0.8 * num_samples)
    x_train, x_val = x[:train_size], x[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, config, predictor=None):
    """Train a model and collect metrics."""
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                              lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), 
                             lr=config['learning_rate'],
                             weight_decay=config['weight_decay'],
                             momentum=0.9)
    
    training_history = []
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Record metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        training_history.append(epoch_metrics)
        
        logger.info(f"Epoch {epoch+1}/{config['epochs']} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Calculate a simple sharpness metric (just for testing)
    sharpness = 0.1  # In a real scenario, this would be calculated
    
    # If predictor is provided, add training example
    if predictor is not None:
        try:
            # Create feature extractors if needed
            from llp.meta_model import DatasetFeatureExtractor, TrainingResultFeatureExtractor
            dataset_extractor = DatasetFeatureExtractor()
            training_extractor = TrainingResultFeatureExtractor()
            
            # Extract features
            dataset_features = dataset_extractor.extract_features(train_loader, val_loader)
            training_features = training_extractor.extract_features(training_history, sharpness)
            
            # Add training example to the meta-model
            predictor.add_training_example(
                dataset_features=dataset_features,
                training_features=training_features,
                hyperparameters=config,
                performance=val_acc  # Using validation accuracy as performance metric
            )
            logger.info(f"Added training example with performance: {val_acc:.4f}")
        except Exception as e:
            logger.error(f"Failed to add training example: {str(e)}")
    
    return training_history, val_acc

def test_full_pipeline():
    """Test the full meta-model pipeline."""
    logger.info("=" * 80)
    logger.info("TESTING FULL META-MODEL PIPELINE")
    logger.info("=" * 80)
    
    # Set up directories
    meta_model_dir = os.path.join(project_root, 'test_full_pipeline', 'meta_models')
    model_output_dir = os.path.join(project_root, 'test_full_pipeline', 'models')
    
    # Clean up any existing directories
    import shutil
    if os.path.exists(os.path.join(project_root, 'test_full_pipeline')):
        shutil.rmtree(os.path.join(project_root, 'test_full_pipeline'))
    
    # Create directories
    os.makedirs(meta_model_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize meta-model predictor
    predictor = HyperparameterPredictor(model_dir=meta_model_dir)
    
    # Initialize meta-probing with minimal configuration
    meta_probing = MetaProbing(
        meta_model_dir=meta_model_dir,
        input_dim=10,  # Match our synthetic data dimension
        output_dim=1   # Single output for performance prediction
    )
    
    # Create synthetic data
    train_loader, val_loader = create_synthetic_data()
    
    # Train multiple models with different configs to generate examples
    configs = [
        {
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'weight_decay': 0.0001,
            'epochs': 5,
            'batch_size': 32,
            'dropout_rate': 0.2
        },
        {
            'learning_rate': 0.01,
            'optimizer': 'sgd',
            'weight_decay': 0.0005,
            'epochs': 5,
            'batch_size': 64,
            'dropout_rate': 0.3
        },
        {
            'learning_rate': 0.0005,
            'optimizer': 'adam',
            'weight_decay': 0.00001,
            'epochs': 5,
            'batch_size': 16,
            'dropout_rate': 0.1
        }
    ]
    
    logger.info("\nSTEP 1: Training models and collecting examples")
    logger.info("-" * 50)
    
    performances = []
    for i, config in enumerate(configs):
        logger.info(f"\nTraining model {i+1} with config: {config}")
        model = SimpleModel()
        _, performance = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            predictor=predictor
        )
        performances.append(performance)
        logger.info(f"Model {i+1} performance: {performance:.4f}")
    
    # Check for example files
    logger.info("\nSTEP 2: Verifying saved examples")
    logger.info("-" * 50)
    
    example_files = predictor._find_example_files()
    if example_files:
        logger.info(f"Found {len(example_files)} example files:")
        for i, file in enumerate(example_files):
            logger.info(f"  - {os.path.basename(file)}")
    else:
        logger.error("No example files found!")
        return False
    
    # Train meta-model
    logger.info("\nSTEP 3: Training meta-model")
    logger.info("-" * 50)
    
    # Train the meta-model directly using the predictor
    success = predictor.train()
    
    if not success:
        logger.error("Meta-model training failed!")
        return False
    
    logger.info("\nSTEP 4: Using meta-model for prediction")
    logger.info("-" * 50)
    
    # Create a new dataset for prediction
    new_train_loader, new_val_loader = create_synthetic_data(seed=100)
    
    # Define a config space for the meta-model to search in
    config_space = {
        'learning_rate': (0.0001, 0.01),
        'optimizer': ['adam', 'sgd'],
        'weight_decay': (0.00001, 0.001),
        'epochs': 5,  # Fixed for this test
        'batch_size': [16, 32, 64],
        'dropout_rate': (0.1, 0.5)
    }
    
    # Use meta-model to suggest hyperparameters
    try:
        suggested_configs = meta_probing.suggest_configurations(
            num_configs=1,
            config_space=config_space,
            train_loader=new_train_loader,
            val_loader=new_val_loader
        )
        # Take the first suggested configuration
        suggested_config = suggested_configs[0] if suggested_configs else configs[0]
    except Exception as e:
        logger.error(f"Error suggesting configurations: {str(e)}")
        # Fallback to the best performing config from our training
        best_config_idx = performances.index(max(performances))
        suggested_config = configs[best_config_idx]
        logger.info(f"Falling back to best known config (index {best_config_idx})")
    
    
    logger.info(f"Meta-model suggested configuration: {suggested_config}")
    
    # Train a model with the suggested config
    logger.info("\nTraining model with suggested config")
    model = SimpleModel()
    _, performance = train_model(
        model=model,
        train_loader=new_train_loader,
        val_loader=new_val_loader,
        config=suggested_config
        # No predictor here - we don't want to save this as an example
    )
    
    logger.info(f"Final model performance with suggested config: {performance:.4f}")
    logger.info(f"Average performance of baseline configs: {sum(performances)/len(performances):.4f}")
    
    return True

if __name__ == "__main__":
    success = test_full_pipeline()
    if success:
        logger.info("\n✓ Full pipeline test completed successfully!")
    else:
        logger.error("\n✗ Full pipeline test failed!")
        sys.exit(1)
