#!/usr/bin/env python3
"""
LossLandscapeProbe Example Runner

This script provides a command-line interface to run different examples and experiments
in the LossLandscapeProbe framework.
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime

# Add the parent directory to the path so we can import the LLP package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llp.utils.logging_utils import setup_logger

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

# Define available examples
EXAMPLES = {
    "cifar10": {
        "description": "Train a model on CIFAR-10 using the meta-model approach",
        "script": "cifar10_example.py"
    },
    "cifar10_resource": {
        "description": "Compare different resource levels for meta-model training on CIFAR-10",
        "script": "sample_size_comparison_cifar10.py"
    },
    "cifar100_resource": {
        "description": "Compare different resource levels for meta-model training on CIFAR-100",
        "script": "sample_size_comparison_cifar100.py"
    },
    "visualize": {
        "description": "Visualize results from resource level comparison experiments",
        "script": "visualize_resource_comparison.py"
    },
    "setup_website": {
        "description": "Set up the website to display results",
        "script": "setup_website.py"
    },
    "generate_report": {
        "description": "Generate a test report for a model",
        "script": "generate_test_report.py"
    }
}

def list_examples():
    """Print a list of available examples."""
    print("\nAvailable examples:")
    print("===================")
    
    for key, example in EXAMPLES.items():
        print(f"{key:20} - {example['description']}")
    
    print("\nUse 'python run_example.py <example_name> [args]' to run an example.")
    print("Add --help after the example name to see example-specific options.")

def run_example(example_name, args=None):
    """Run the specified example with optional arguments."""
    if example_name not in EXAMPLES:
        logger.error(f"Unknown example: {example_name}")
        list_examples()
        return False
    
    example = EXAMPLES[example_name]
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), example["script"])
    
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
    
    # Build the command
    cmd = [sys.executable, script_path]
    
    # Add any additional arguments
    if args:
        cmd.extend(args)
    
    logger.info(f"Running {example_name}: {example['description']}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run the command
    try:
        start_time = datetime.now()
        subprocess.run(cmd, check=True)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Example {example_name} completed successfully in {duration:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running example {example_name}: {e}")
        return False

def main():
    """Main function to parse arguments and run examples."""
    parser = argparse.ArgumentParser(
        description="LossLandscapeProbe Example Runner",
        epilog="Use 'list' to see available examples."
    )
    parser.add_argument(
        "example", 
        nargs="?", 
        help="Name of the example to run, or 'list' to see available examples"
    )
    parser.add_argument(
        "args", 
        nargs=argparse.REMAINDER, 
        help="Arguments to pass to the example script"
    )
    
    args = parser.parse_args()
    
    if not args.example or args.example == "list":
        list_examples()
        return
    
    run_example(args.example, args.args)

if __name__ == "__main__":
    main()
