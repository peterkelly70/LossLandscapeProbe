"""
Parameter-Space Probing Module
=============================

This module implements various parameter-space perturbation strategies:
- Random weight perturbations
- Sharpness-Aware Minimization (SAM)
- Entropy-SGD
- Stochastic Weight Averaging (SWA)

These methods explore the loss landscape by perturbing weights to find flat,
generalizable regions that are likely to perform well.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Callable, Union, Tuple, Optional, Any
import copy
import numpy as np
import logging

logger = logging.getLogger(__name__)


def perturb_weights(model: nn.Module, noise_std: float = 0.01) -> nn.Module:
    """
    Add Gaussian noise to model weights.
    
    Args:
        model: PyTorch model
        noise_std: Standard deviation of the noise
        
    Returns:
        Perturbed model (original model is not modified)
    """
    perturbed_model = copy.deepcopy(model)
    
    with torch.no_grad():
        for param in perturbed_model.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)
    
    return perturbed_model


def measure_sharpness(
    model: nn.Module, 
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    noise_std: float = 0.01,
    num_samples: int = 5,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    max_batches: int = 10  # Limit number of batches for faster evaluation
) -> float:
    """
    Measure sharpness of the loss landscape by adding noise to weights.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        noise_std: Standard deviation of the noise
        num_samples: Number of perturbation samples
        device: Device to run the model on
        max_batches: Maximum number of batches to evaluate (for speed)
        
    Returns:
        Sharpness score (average loss increase after perturbation)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    model.eval()
    model = model.to(device)
    
    logger.info(f"Computing original loss on {min(max_batches, len(val_loader))} batches")
    
    # Compute original loss on a limited number of batches for speed
    original_loss = 0.0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if i >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            original_loss += loss.item()
    
    # Average over the number of batches actually used
    num_batches = min(max_batches, len(val_loader))
    original_loss /= num_batches
    
    # Compute loss after perturbation
    perturbed_losses = []
    for i in range(num_samples):
        logger.info(f"Evaluating perturbation {i+1}/{num_samples}")
        perturbed_model = perturb_weights(model, noise_std)
        perturbed_model.eval()
        perturbed_model = perturbed_model.to(device)
        
        perturbed_loss = 0.0
        with torch.no_grad():
            for j, (inputs, targets) in enumerate(val_loader):
                if j >= max_batches:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = perturbed_model(inputs)
                loss = criterion(outputs, targets)
                perturbed_loss += loss.item()
        
        perturbed_loss /= num_batches
        perturbed_losses.append(perturbed_loss)
        logger.info(f"Perturbation {i+1} loss: {perturbed_loss:.4f} (original: {original_loss:.4f})")
    
    # Calculate average loss increase
    avg_perturbed_loss = sum(perturbed_losses) / len(perturbed_losses)
    sharpness = avg_perturbed_loss - original_loss
    
    logger.info(f"Sharpness measurement completed: {sharpness:.4f}")
    return sharpness


class SAM(Optimizer):
    """
    Implementation of Sharpness-Aware Minimization (SAM) optimizer.
    
    SAM seeks parameters that lie in neighborhoods having uniformly low loss,
    as opposed to standard SGD which finds parameters that minimize the loss.
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        """
        Initialize SAM optimizer.
        
        Args:
            params: Model parameters
            base_optimizer: Base optimizer class (e.g., SGD, Adam)
            rho: Size of the neighborhood for perturbation
            **kwargs: Arguments for base optimizer
        """
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        # Initialize base optimizer
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Compute and apply the perturbation to the parameters.
        """
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Store original parameters
                self.state[p]["old_p"] = p.data.clone()
                
                # Apply perturbation
                e_w = p.grad * scale
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Restore the parameters to their original values and apply the base optimizer.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                
                # Restore original parameters
                p.data = self.state[p]["old_p"]
        
        # Apply base optimizer update
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        """
        Compute the gradient norm for all parameters.
        """
        norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                norm += torch.sum(p.grad ** 2)
        
        return torch.sqrt(norm)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        assert closure is not None, "SAM requires closure, but it was not provided"
        
        # First forward-backward pass
        closure()
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass
        loss = closure()
        self.second_step()
        
        return loss


class StochasticWeightAveraging:
    """
    Implementation of Stochastic Weight Averaging (SWA).
    
    SWA averages weights from different points along the training trajectory
    to find a solution in a wider, flatter part of the loss landscape.
    """
    
    def __init__(
        self,
        model: nn.Module,
        swa_start: int = 10,
        swa_freq: int = 5,
        swa_lr: Optional[float] = None
    ):
        """
        Initialize SWA.
        
        Args:
            model: PyTorch model
            swa_start: Epoch to start SWA from
            swa_freq: Frequency of model averaging
            swa_lr: Learning rate for SWA phase (if None, use the optimizer's LR)
        """
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        
        # Initialize SWA model with a copy of the current model
        self.swa_model = copy.deepcopy(model)
        self._set_swa_params_to_zero()
        
        self.n_averaged = 0
    
    def _set_swa_params_to_zero(self):
        """Set all SWA model parameters to zero."""
        for param in self.swa_model.parameters():
            param.data.zero_()
    
    def update(self, epoch: int) -> bool:
        """
        Update the SWA model if conditions are met.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Boolean indicating whether SWA is active (True if epoch >= swa_start)
        """
        is_swa_active = epoch >= self.swa_start
        
        if is_swa_active and (epoch - self.swa_start) % self.swa_freq == 0:
            # Update the SWA model parameters
            for swa_param, param in zip(self.swa_model.parameters(), self.model.parameters()):
                swa_param.data.add_(param.data)
            
            self.n_averaged += 1
            logger.info(f"SWA update at epoch {epoch}, models averaged: {self.n_averaged}")
        
        return is_swa_active
    
    def finalize(self):
        """
        Finalize the SWA model by dividing the accumulated parameters by the number of models.
        """
        if self.n_averaged > 0:
            for param in self.swa_model.parameters():
                param.data.div_(self.n_averaged)
            
            logger.info(f"SWA finalized with {self.n_averaged} models averaged")
        else:
            logger.warning("SWA finalized but no models were averaged")
    
    def get_swa_model(self) -> nn.Module:
        """
        Get the SWA model.
        
        Returns:
            SWA model
        """
        if self.n_averaged == 0:
            logger.warning("Returning SWA model but no models were averaged")
            return copy.deepcopy(self.model)
        
        return self.swa_model


class EntropyRegularizedSGD(Optimizer):
    """
    Implementation of Entropy-SGD optimizer.
    
    Entropy-SGD augments the loss with a local entropy term to favor
    flatter regions of the loss landscape.
    """
    
    def __init__(
        self, 
        params, 
        lr=0.01, 
        momentum=0.9, 
        gamma=1e-4, 
        L=5, 
        eps=1e-4
    ):
        """
        Initialize Entropy-SGD optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            momentum: Momentum factor
            gamma: Weight of the entropy term
            L: Number of Langevin dynamics steps
            eps: Step size for Langevin dynamics
        """
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            gamma=gamma, 
            L=L, 
            eps=eps
        )
        super(EntropyRegularizedSGD, self).__init__(params, defaults)
        
        # Initialize auxiliary variables
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['x_tilde'] = p.data.clone()
                self.state[p]['g_tilde'] = torch.zeros_like(p.data)
                self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        assert closure is not None, "Entropy-SGD requires closure, but it was not provided"
        
        loss = None
        
        for group in self.param_groups:
            gamma = group['gamma']
            L = group['L']
            eps = group['eps']
            
            # Initialize x_tilde with current parameters
            for p in group['params']:
                self.state[p]['x_tilde'].copy_(p.data)
            
            # Langevin dynamics (inner loop)
            for _ in range(L):
                # Evaluate loss and gradients at x_tilde
                loss = closure()
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    x_tilde = self.state[p]['x_tilde']
                    g_tilde = self.state[p]['g_tilde']
                    
                    # Compute gradient of entropy term
                    g_tilde.copy_(p.grad.data)
                    g_tilde.add_(gamma * (p.data - x_tilde))
                    
                    # Update x_tilde with Langevin dynamics
                    noise = torch.randn_like(x_tilde) * np.sqrt(2 * eps)
                    x_tilde.add_(-eps * g_tilde + noise)
            
            # Update parameters with SGD using the entropy-regularized gradient
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                # Add entropy term gradient
                d_p.add_(gamma * (p.data - self.state[p]['x_tilde']))
                
                # Apply momentum
                if group['momentum'] > 0:
                    momentum_buffer = self.state[p]['momentum_buffer']
                    momentum_buffer.mul_(group['momentum']).add_(d_p)
                    d_p = momentum_buffer
                
                # Update parameters
                p.data.add_(-group['lr'] * d_p)
        
        return loss
