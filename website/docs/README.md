# Loss Landscape Probing for Model Optimization (LLP)

A framework for efficient neural network training through two-tiered probing:
1. **Data sampling** - training on small subsets or limited iterations
2. **Parameter-space perturbations** - exploring weight-space by random or gradient-based tweaks

**GitHub Repository**: [https://github.com/peterkelly70/LossLandscapeProbe](https://github.com/peterkelly70/LossLandscapeProbe)

This framework implements various strategies described in literature to quickly identify promising hyperparameter settings or regions of the loss landscape that are likely to generalize well.

## Overview

Training deep neural networks on large datasets is computationally expensive, especially when tuning hyperparameters. This framework provides a two-tiered probing approach to efficiently find promising hyperparameter configurations and model regions:

- **Tier 1: Data Sampling Probes**
  - Train on small random subsets or limited iterations
  - Implement methods like Successive Halving, Hyperband, and learning curve extrapolation
  - Quickly filter out unpromising configurations without full training

- **Tier 2: Parameter-Space Probing**
  - Explore weight-space by random or gradient-based perturbations
  - Implement methods like Sharpness-Aware Minimization (SAM), Stochastic Weight Averaging (SWA), and Entropy-SGD
  - Find flat, generalizable regions of the loss landscape

## Features

- **Successive Halving (SHA)** - Allocate small training budget to many configs, progressively focus resources on better performers
- **Hyperband** - Run multiple SHA brackets with different initial budgets
- **Learning curve extrapolation** - Predict final performance from partial training
- **Sharpness-Aware Minimization (SAM)** - Optimize for uniformly low loss in parameter neighborhoods
- **Stochastic Weight Averaging (SWA)** - Average weights from different points along the training trajectory
- **Entropy-SGD** - Augment loss with local entropy term to favor flatter regions
- **Two-tier evaluation** - Combine data sampling and parameter perturbation metrics for better generalization prediction

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
from lpm.two_tier_probing import TwoTierProbing

# Define hyperparameter configurations to search
configs = [
    {'learning_rate': 0.01, 'weight_decay': 5e-4, ...},
    {'learning_rate': 0.1, 'weight_decay': 5e-4, ...},
    # ...
]

# Create Two-Tier Probing object
probing = TwoTierProbing(
    configs=configs,
    model_fn=create_model,
    dataset_fn=get_data_loaders,
    criterion=loss_function,
    optimizer_fn=create_optimizer
)

# Run Successive Halving with two-tier evaluation
results = probing.run_successive_halving(
    min_resource=0.1,  # Start with 10% of data
    max_resource=0.5,  # End with 50% of data
    reduction_factor=2,
    measure_flatness=True
)

# Select best configurations
best_configs = probing.select_best_configs(n=2, criterion='generalization_score')

# Train the best config with SAM
sam_result = probing.train_with_sam(config=best_configs[0])

# Train the best config with SWA
regular_result, swa_result = probing.train_with_swa(config=best_configs[0])
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
