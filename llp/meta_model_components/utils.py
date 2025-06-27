#!/usr/bin/env python3
"""
Meta-Model Utilities Module
==========================

Utility functions for the meta-model.
"""

import torch
import numpy as np
import logging
from pathlib import Path
import json
import traceback
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

def convert_to_serializable(obj, _memo=None):
    """
    Convert various types to native Python types for JSON serialization.
    Handles NumPy types, PyTorch tensors, Path objects, and prevents circular references.
    
    Args:
        obj: Object to convert
        _memo: Internal use only - tracks seen objects to prevent circular references
        
    Returns:
        Object with types converted to JSON-serializable formats
    """
    if _memo is None:
        _memo = set()
    
    # Handle None
    if obj is None:
        return None
        
    # Handle basic types
    if isinstance(obj, (int, float, str, bool)):
        return obj
    
    # Handle circular references
    obj_id = id(obj)
    if obj_id in _memo:
        return "<circular reference>"
    _memo.add(obj_id)
    
    try:
        # Handle Path objects and file-like objects
        if hasattr(obj, '__fspath__'):  # Handles Path-like objects
            return str(obj)
            
        # Handle NumPy types
        if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            return obj.item()
            
        # Handle NumPy arrays
        if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
            return obj.tolist()
            
        # Handle PyTorch tensors
        if hasattr(obj, 'is_cuda') or hasattr(obj, 'is_sparse'):  # PyTorch tensor
            if obj.numel() == 1:  # Scalar tensor
                return obj.item()
            return obj.detach().cpu().numpy().tolist()
            
        # Handle dictionaries
        if isinstance(obj, dict):
            return {str(k): convert_to_serializable(v, _memo) 
                   for k, v in obj.items()}
            
        # Handle sequences (list, tuple, etc.)
        if isinstance(obj, (list, tuple, set)):
            return [convert_to_serializable(item, _memo) for item in obj]
            
        # Handle other iterables
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            try:
                return [convert_to_serializable(item, _memo) for item in obj]
            except:
                pass
                
        # Handle objects with __dict__ (custom classes)
        if hasattr(obj, '__dict__'):
            return {f"{type(obj).__name__}": convert_to_serializable(obj.__dict__, _memo)}
            
        # Last resort: convert to string
        return str(obj)
    except Exception as e:
        # If all else fails, return a string representation
        return f"<non-serializable: {type(obj).__name__}, error: {str(e)}>"

def setup_logging(run_dir):
    """
    Set up logging to use a single training.log file in the run directory.
    
    Args:
        run_dir: Directory to store log files
    """
    # Create run directory if it doesn't exist
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        
    # Configure logging
    log_file = run_dir / 'training.log'
    
    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging configured. Log file: {log_file}")
    return logger
