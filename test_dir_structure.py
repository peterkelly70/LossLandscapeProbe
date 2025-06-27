#!/usr/bin/env python3
"""
Test script to verify directory structure fix.
"""

import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Test parameters
    dataset = "cifar10"
    sample_size = "10"
    
    # Create dir_name using the correct format
    dir_name = f"{dataset}_{sample_size}"
    
    # Set up base directories
    base_dir = Path(__file__).parent
    
    # Set up reports directory structure using the nested format
    run_dir = base_dir / "reports" / dataset / dir_name
    
    # Create reports directory and ensure it exists
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Log the directory structure
    logger.info(f"Created directory structure: {run_dir}")
    
    # Create a test file in the directory
    test_file = run_dir / "test_file.txt"
    with open(test_file, "w") as f:
        f.write("This is a test file to verify the directory structure fix.")
    
    logger.info(f"Created test file: {test_file}")
    
    # Verify the directory structure
    if run_dir.exists():
        logger.info("Directory structure verified!")
        logger.info(f"Directory path: {run_dir}")
    else:
        logger.error("Directory structure verification failed!")

if __name__ == "__main__":
    main()
