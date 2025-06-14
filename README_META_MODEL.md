# Meta-Model for Hyperparameter Prediction

This extension to the LossLandscapeProbe framework implements a meta-learning approach to predict optimal hyperparameters based on dataset characteristics and partial training results.

## Overview

The traditional approach to hyperparameter optimization (like Successive Halving) tries different configurations and selects the best one. Our meta-model approach goes further by:

1. **Learning from experience**: The meta-model learns patterns between dataset characteristics, training dynamics, and optimal hyperparameters
2. **Making predictions**: It can predict good hyperparameters for new datasets without extensive trial and error
3. **Iterative refinement**: It uses an iterative process to improve predictions based on evaluation results

## Key Components

### 1. Dataset Feature Extraction

Extracts statistical features from a dataset:
- Basic statistics (samples, features, classes)
- Class distribution statistics (imbalance, entropy)
- Feature statistics (mean, std, skew, kurtosis)
- Complexity measures (feature correlations)

### 2. Training Result Feature Extraction

Extracts features from partial training results:
- Training dynamics (loss decrease rate, accuracy increase rate)
- Sharpness measures (loss landscape curvature)
- Resource usage (data fraction, epochs)

### 3. Hyperparameter Predictor

A meta-model that predicts optimal hyperparameters:
- Uses Random Forest Regression models for each hyperparameter
- Trained on dataset features and training results
- Makes predictions for new datasets

### 4. Meta-Model Trainer

Manages the training of the meta-model:
- Collects examples from hyperparameter evaluations
- Processes results to extract features
- Trains the meta-model on collected examples

## Meta-Optimization Process

The `run_meta_optimization` method implements a novel approach:

1. **Initial Exploration**: Evaluates initial configurations on small subsets of data
2. **Meta-Model Training**: Trains a meta-model to predict optimal hyperparameters
3. **Prediction & Variation**: Generates new configurations based on predictions
4. **Iterative Refinement**: Repeats the process with larger data subsets
5. **Final Selection**: Returns the best configuration found

## Usage

To use the meta-model approach in your code:

```python
from llp.meta_probing import MetaProbing

# Create MetaProbing instance
meta_probing = MetaProbing(
    configs=configs,
    model_fn=create_model,
    dataset_fn=dataset_fn,
    criterion=criterion,
    optimizer_fn=create_optimizer
)

# Run meta-model optimization
best_config = meta_probing.run_meta_optimization(
    min_resource=0.1,
    max_resource=0.5,
    reduction_factor=2,
    measure_flatness=True,
    num_initial_configs=6,
    num_iterations=3
)

# Use the best config for final training
# ...
```

## Advantages Over Successive Halving

1. **Transfer Learning**: The meta-model learns patterns that can be applied to new datasets
2. **Intelligent Exploration**: Focuses on promising regions of the hyperparameter space
3. **Efficiency**: Requires fewer evaluations to find good configurations
4. **Adaptability**: Can adapt to different types of datasets and models

## Progress Reporting

The meta-model implementation includes detailed progress reporting:
- Iteration progress (current iteration, configs being evaluated)
- Resource levels (data fraction being used)
- Elapsed time
- Meta-model training progress
- Prediction details

This provides clear visibility into the meta-optimization process.
