"""
Progress bar utilities for the CIFAR meta-model optimization process.
"""
import logging
from typing import Optional, Any, Dict, List, Tuple
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

logger = logging.getLogger(__name__)

def convert_to_serializable(obj: Any) -> Any:
    """Convert PyTorch tensors and other non-serializable objects to basic Python types."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items() 
                if k not in ['model', 'model_state']}  # Exclude non-serializable keys
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def create_progress_bar(total: int, desc: str, use_tqdm: bool = True) -> Optional[Any]:
    """Create a progress bar if tqdm is available."""
    if use_tqdm:
        try:
            from tqdm import tqdm
            bar_format = '{desc}: {bar} {percentage:3.0f}% | {n_fmt}/{total_fmt}'
            return tqdm(total=total, desc=desc, bar_format=bar_format)
        except ImportError:
            return None
    return None

def update_progress_bar(pbar: Any, n: int = 1, refresh: bool = True) -> None:
    """Update a progress bar if it exists."""
    if pbar is not None:
        pbar.update(n)
        if refresh:
            pbar.refresh()

def close_progress_bar(pbar: Any) -> None:
    """Close a progress bar if it exists."""
    if pbar is not None:
        pbar.close()

def train_meta_model_epoch(
    model: nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> float:
    """Train the meta-model for one epoch."""
    model.train()
    total_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_meta_model(
    model: nn.Module, 
    val_loader: torch.utils.data.DataLoader, 
    device: torch.device
) -> float:
    """Evaluate the meta-model."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.mse_loss(output, target).item()
    return total_loss / len(val_loader)
