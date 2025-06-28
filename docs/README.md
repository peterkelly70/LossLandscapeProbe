# Loss Landscape Probing for Model Optimization (LLP)

A framework for efficient neural network training through meta-learning:
1. **Meta-model training** - training on different sample sizes of CIFAR datasets
2. **Loss landscape analysis** - visualizing and analyzing model performance across configurations

**GitHub Repository**: [https://github.com/peterkelly70/LossLandscapeProbe](https://github.com/peterkelly70/LossLandscapeProbe)

This framework implements a meta-learning approach to quickly identify promising hyperparameter settings for neural network training on image classification tasks.

## Overview

Training deep neural networks on large datasets is computationally expensive, especially when tuning hyperparameters. This framework provides a meta-learning approach to efficiently find promising hyperparameter configurations:

- **Meta-Model Training**
  - Train on different sample sizes of CIFAR-10/100 datasets (10%, 20%, 30%, 40%)
  - Evaluate multiple hyperparameter configurations across these samples
  - Build a meta-model that predicts optimal hyperparameters for the full dataset

- **Loss Landscape Analysis**
  - Visualize and analyze the loss landscape of trained models
  - Compare different configurations through an interactive web interface
  - Examine per-class performance metrics for detailed analysis

## Features

- **Meta-model training** - Learn from multiple dataset sample sizes to predict optimal hyperparameters
- **Interactive visualization** - Explore training results through a web-based interface
- **Per-class accuracy analysis** - Detailed breakdown of model performance by class
- **Confusion matrix visualization** - Identify patterns in model predictions
- **Sample prediction display** - View example predictions with corresponding images

## Installation

```bash
# Clone the repository
git clone https://github.com/peterkelly70/LossLandscapeProbe.git
cd LossLandscapeProbe

# Create and activate a virtual environment
python -m venv .lpm_env
source .lpm_env/bin/activate  # On Windows: .lpm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Here's a simple example of how to use the framework:

```python
from llp.meta_model import MetaModel
from llp.cifar_training import train_model

# Define hyperparameter configurations to search
configs = [
    {'learning_rate': 0.01, 'weight_decay': 5e-4, 'optimizer': 'sgd', 'momentum': 0.9},
    {'learning_rate': 0.1, 'weight_decay': 5e-4, 'optimizer': 'sgd', 'momentum': 0.9},
    # ...
]

# Create Meta-Model object
meta_model = MetaModel(
    configs=configs,
    model_fn=create_model,
    dataset_fn=get_cifar10_loaders,
    criterion=nn.CrossEntropyLoss(),
    optimizer_fn=create_optimizer
)

# Train meta-model on different sample sizes
meta_model.train(sample_sizes=[0.1, 0.2, 0.3, 0.4])

# Get predicted best configuration
best_config = meta_model.predict_optimal_hyperparameters()

# Train with the predicted optimal hyperparameters
model, history, test_acc = train_model(best_config, epochs=100)
print(f"Test accuracy with meta-model predicted params: {test_acc:.4f}")
```

See the `examples` directory for more detailed examples.

## Examples

- `examples/cifar10_example.py` - Example of using the framework to find good hyperparameters for training a CNN on CIFAR-10

## Reports

The project includes comprehensive reports for model training and testing on CIFAR-10 and CIFAR-100 datasets with various configurations:

### CIFAR-10 Reports
- cifar10_10, cifar10_20, cifar10_30, cifar10_40, cifar10_multi
- Each configuration includes:
  - Training Report: Detailed metrics and visualizations from the training process
  - Test Report: Performance evaluation on test data
  - Training Log: Raw training logs for debugging and analysis

### CIFAR-100 Reports
- cifar100_10, cifar100_20, cifar100_30, cifar100_40, cifar100_multi
- Each with the same comprehensive reporting structure

## Future Developments

### Planned Perturbation Data
Future releases will include enhanced parameter-space perturbation capabilities:
- Advanced gradient-based perturbation strategies
- Multi-scale perturbation analysis for deeper landscape understanding
- Adaptive perturbation magnitude based on training dynamics
- Integration with uncertainty quantification methods
- Visualization tools for loss landscape topology exploration

## References

- Successive Halving and Hyperband: [Li et al., 2018](https://proceedings.mlr.press/v80/li18a.html)
- Sharpness-Aware Minimization: [Foret et al., 2021](https://arxiv.org/abs/2010.01412)
- Stochastic Weight Averaging: [Izmailov et al., 2018](https://arxiv.org/abs/1803.05407)
- Entropy-SGD: [Chaudhari et al., 2017](https://openreview.net/forum?id=B1YfZsNFl)
- Flat Minima and Generalization: [Keskar et al., 2017](https://openreview.net/forum?id=Sy8gdB9xx)

## License

MIT
