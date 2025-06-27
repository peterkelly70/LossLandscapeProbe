#!/usr/bin/env python3
"""
CIFAR Meta-Model Module
======================

Hyperparameter prediction and meta-model optimization for CIFAR datasets.
This module handles:
- Meta-model training
- Hyperparameter prediction
- Configuration evaluation
"""

import torch
import logging
from pathlib import Path
import time
from typing import Dict, List, Any, Optional, Union

# Import our modular components
from .meta_model_components.config import MetaModelConfig
from .meta_model_components.cifar_optimizer import CIFARMetaModelOptimizer

# Set up logging
logger = logging.getLogger(__name__)

# Re-export the CIFARMetaModelOptimizer class for backward compatibility
__all__ = ['CIFARMetaModelOptimizer', 'MetaModelConfig']

# This file now serves as a thin wrapper around our modular components
# All functionality has been moved to the meta_model_components package
