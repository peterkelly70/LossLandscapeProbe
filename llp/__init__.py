"""
Loss Landscape Probing for Model Optimization (LPM)
===================================================

A framework for efficient neural network training through two-tiered probing:
1. Data sampling - training on small subsets or limited iterations
2. Parameter-space perturbations - exploring weight-space by random or gradient-based tweaks

This framework implements various strategies described in literature to quickly identify
promising hyperparameter settings or regions of the loss landscape.
"""

__version__ = "0.1.0"
