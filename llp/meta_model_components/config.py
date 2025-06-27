#!/usr/bin/env python3
"""
Meta-Model Configuration Module
==============================

Configuration dataclasses and utilities for the meta-model.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

@dataclass
class MetaModelConfig:
    """Configuration for the Meta-Model Optimizer."""
    dataset: str = 'cifar10'
    batch_size: int = 128
    configs_per_sample: int = 10
    perturbations: int = 5
    iterations: int = 3
    min_resource: float = 0.1
    max_resource: float = 0.5
    num_data_subsets: int = 5
    subset_size: float = 0.2
    ensure_disjoint_subsets: bool = True
    perturbation_scale: float = 0.1
    run_dir: Optional[Union[str, Path]] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed: Optional[int] = 42
