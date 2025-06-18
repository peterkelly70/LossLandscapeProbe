# Loss Landscape Probing for Model Optimization (LLP)

A framework for efficient neural network training using a meta-model approach for hyperparameter optimization. The meta-model predicts optimal hyperparameters based on dataset characteristics and partial training results, eliminating the need for extensive hyperparameter search.

The framework achieves approximately 85% test accuracy on CIFAR-10 using a SimpleCNN architecture, which is competitive with standard CNN implementations while requiring significantly less hyperparameter tuning effort.

## Overview

Training deep neural networks on large datasets is computationally expensive, especially when tuning hyperparameters. This framework uses a meta-model approach to predict optimal hyperparameters without extensive search:

- **Meta-Model Approach**
  - Extract features from datasets and partial training results
  - Train a meta-model to predict optimal hyperparameters
  - Achieve near-optimal performance with minimal computational cost
  - Eliminate the need for traditional hyperparameter search methods

- **Visualization and Reporting**
  - Generate comprehensive test reports with visual examples
  - Analyze per-class performance metrics
  - Track training progress with loss and accuracy curves

## Features

- **Meta-Model Hyperparameter Optimization** - Predict optimal hyperparameters based on dataset characteristics and partial training results
- **Test Report Visualization** - Generate HTML reports showing test images with predictions and confidence scores
- **Per-Class Accuracy Analysis** - Detailed statistics on model performance across different categories
- **Training Progress Visualization** - Plot loss and accuracy curves during training
- **Web-Based Report Viewer** - Browse test reports and training results through an interactive web interface

## Installation

```bash
# Clone the repository
git clone https://github.com/peterkelly70/LossLandscapeProbe.git
cd LossLandscapeProbe

# Create and activate a virtual environment
python -m venv .llp_env
source .llp_env/bin/activate  # On Windows: .llp_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Here's a simple example of how to use the framework with the meta-model approach:

```python
from llp.meta_probing import MetaProbing

# Define hyperparameter configurations to search
configs = [
    {'num_channels': 32, 'dropout_rate': 0.2, 'optimizer': 'sgd', 'learning_rate': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4},
    {'num_channels': 32, 'dropout_rate': 0.2, 'optimizer': 'adam', 'learning_rate': 0.001, 'momentum': 0.0, 'weight_decay': 5e-4},
    # ...
]

# Create Meta-Probing object
probing = MetaProbing(
    configs=configs,
    model_fn=create_model,
    dataset_fn=get_data_loaders,
    criterion=loss_function,
    optimizer_fn=create_optimizer
)

# Run meta-model optimization
best_config = probing.run_meta_optimization(
    min_resource=0.1,  # Start with 10% of data
    max_resource=0.5,  # End with 50% of data
    reduction_factor=2,
    num_initial_configs=6,
    num_iterations=3
)

# Train with the meta-model predicted hyperparameters
model, history = train_with_meta_params(best_config)

# Generate test report
generate_test_report(model, test_loader, 'reports/test_report.html')
```

You can also use the traditional Successive Halving approach if preferred:

```python
from llp.two_tier_probing import TwoTierProbing

# Create Two-Tier Probing object
probing = TwoTierProbing(...)

# Run Successive Halving with two-tier evaluation
results = probing.run_successive_halving(...)
```

See the `examples` directory for more detailed examples.

## Examples

- `examples/cifar10_example.py` - Example of using the framework with meta-model approach for training a CNN on CIFAR-10
- `examples/train_with_meta_params.py` - Example of training a model using meta-model predicted hyperparameters
- `examples/generate_test_report.py` - Generate visual HTML reports showing test images with predictions

## Meta-Model Approach

The framework uses a meta-model to predict optimal hyperparameters based on dataset characteristics and partial training results. This approach eliminates the need for extensive hyperparameter search by leveraging knowledge from previous experiments.

### Meta-Model Hyperparameters

The meta-model optimization process uses the following default hyperparameters:

| Parameter | Default Value | Justification |
|-----------|---------------|---------------|
| `min_resource` | 0.1 (10%) | Provides sufficient signal for initial filtering while minimizing computation |
| `max_resource` | 0.5 (50%) | Balances accuracy of performance estimation with computational efficiency |
| `reduction_factor` | 2 | Standard value from Successive Halving literature, doubles resources between stages |
| `num_initial_configs` | 6 | Ensures diverse coverage of the hyperparameter space while remaining computationally feasible |
| `num_iterations` | 3 | Provides sufficient refinement while limiting total computation time |

These values are based on common defaults from hyperparameter optimization literature such as Successive Halving and Hyperband. They represent a reasonable starting point that balances exploration, exploitation, and computational efficiency.

### Limitations

It's important to note that the current meta-model implementation is both dataset-bound and model-bound:

- **Dataset Dependency**: The meta-model has been primarily validated on CIFAR-10 and may require retraining for other datasets with different characteristics.
- **Architecture Specificity**: The hyperparameter predictions are optimized for the SimpleCNN architecture and may not transfer well to other architectures like ResNets or Transformers.

These limitations suggest potential directions for future work, including cross-dataset validation and architecture-agnostic meta-features.

## Performance Comparison

Our SimpleCNN model with meta-model optimized hyperparameters achieves strong results on CIFAR-10:

| Model Type | Accuracy | Hyperparameter Tuning Required |
|------------|----------|--------------------------------|
| Our SimpleCNN (meta-model optimized) | 84.7% (85.5% peak) | Minimal (automated) |
| Standard CNNs (similar complexity) | 80-85% | Extensive manual tuning |
| Spiking Neural Networks (SNNs) | 90-93% | Moderate to extensive |
| State-of-the-art models (ResNet, ViT, etc.) | 95-99% | Extensive + specialized techniques |

The key advantage of our approach is achieving competitive performance for a simple architecture with minimal hyperparameter tuning effort, demonstrating the effectiveness of the meta-model approach.

## Visualization

The framework includes tools to visualize model performance and predictions:

- **Training Progress Visualization**: Plot loss and accuracy curves during training
- **Test Report Visualization**: View test images alongside their true and predicted labels with confidence scores
- **Per-Class Accuracy Statistics**: Analyze model performance across different categories
- **Live Demo**: [https://loss.computer-wizard.com.au/](https://loss.computer-wizard.com.au/) - Interactive visualization of test results

## Future Directions

Potential areas for future development include:

- **Multi-Dataset Support**: Expanding the framework with dedicated meta-models for additional datasets:
  - MNIST (handwritten digits)
  - Fashion-MNIST (clothing items)
  - SVHN (street view house numbers)
  - CIFAR-100 (100-class images)
  - ImageNet subsets (for scaling to larger images)

- **Spiking Neural Networks (SNNs)**: Extending the meta-model approach to optimize hyperparameters for SNNs, which offer energy efficiency advantages for edge devices

- **Transfer Learning**: Applying the meta-model to transfer learning scenarios to quickly adapt pre-trained models to new tasks

- **Advanced Visualization**: Enhancing the reporting system with interactive visualizations of the loss landscape

- **Meta-Feature Expansion**: Incorporating additional dataset characteristics to improve hyperparameter predictions

- **Cross-Dataset Generalization**: Researching techniques to make meta-models more transferable between related datasets

## References

- Successive Halving and Hyperband: [Li et al., 2018](https://proceedings.mlr.press/v80/li18a.html)
- Sharpness-Aware Minimization: [Foret et al., 2021](https://arxiv.org/abs/2010.01412)
- Stochastic Weight Averaging: [Izmailov et al., 2018](https://arxiv.org/abs/1803.05407)
- Meta-Model for Hyperparameter Optimization: [Kelly et al., 2025](https://loss.computer-wizard.com.au/)
- Flat Minima and Generalization: [Keskar et al., 2017](https://openreview.net/forum?id=Sy8gdB9xx)

## License

GPL-3.0 License

Copyright (c) 2025 Peter Kelly

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
