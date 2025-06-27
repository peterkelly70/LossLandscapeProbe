#!/usr/bin/env python3
"""
Serialization Utilities Module
============================

Utilities for serializing objects to JSON, including PyTorch tensors.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

class TensorJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that handles PyTorch tensors, NumPy arrays, and other non-serializable types.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, '__fspath__'):  # Path-like objects
            return str(obj)
        return super().default(obj)

def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file, handling PyTorch tensors and other special types.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        indent: Indentation level for the JSON file
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, cls=TensorJSONEncoder)

def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    with open(file_path, 'r') as f:
        return json.load(f)
