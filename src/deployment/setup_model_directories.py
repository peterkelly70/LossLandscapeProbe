#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for setting up model directories in the web directory based on the actual directory structure.
"""

import os
import sys
from pathlib import Path

def setup_model_directories(project_dir, web_dir):
    """Create model report directories in the web directory and ensure logs are saved to both locations."""
    # Base reports directory
    reports_dir = os.path.join(web_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Ensure project reports directory exists
    project_reports_dir = os.path.join(project_dir, 'reports')
    os.makedirs(project_reports_dir, exist_ok=True)
    
    # Model directories to create based on the memory structure
    model_dirs = [
        'cifar10',        # Base CIFAR-10 model
        'cifar100',       # Base CIFAR-100 model
        'cifar10_10',     # CIFAR-10 with 10% resources
        'cifar10_20',     # CIFAR-10 with 20% resources
        'cifar10_30',     # CIFAR-10 with 30% resources
        'cifar10_40',     # CIFAR-10 with 40% resources
        'cifar100_10',    # CIFAR-100 with 10% resources
        'cifar100_20',    # CIFAR-100 with 20% resources
        'cifar100_30',    # CIFAR-100 with 30% resources
        'cifar100_40',    # CIFAR-100 with 40% resources
        'cifar100_transfer'  # CIFAR-100 transfer learning
    ]
    
    # Create each model directory in both project and web reports directories
    for model_dir in model_dirs:
        # Create web model directory
        web_model_dir = os.path.join(reports_dir, model_dir)
        os.makedirs(web_model_dir, exist_ok=True)
        print(f"Created web model directory: {web_model_dir}")
        
        # Create project model directory
        project_model_dir = os.path.join(project_reports_dir, model_dir)
        os.makedirs(project_model_dir, exist_ok=True)
        print(f"Created project model directory: {project_model_dir}")
        
        # Create training_log.txt in both locations if it doesn't exist
        training_log_name = f"{model_dir}_training_log.txt"
        project_log_path = os.path.join(project_model_dir, training_log_name)
        web_log_path = os.path.join(web_model_dir, training_log_name)
        
        # If project log exists, copy to web directory
        if os.path.exists(project_log_path):
            try:
                with open(project_log_path, 'rb') as src_file:
                    content = src_file.read()
                with open(web_log_path, 'wb') as dst_file:
                    dst_file.write(content)
                print(f"Copied training log: {training_log_name} to web directory")
            except Exception as e:
                print(f"Error copying training log {training_log_name}: {e}")
        
        # Copy all other files from project to web directory
        if os.path.exists(project_model_dir) and os.path.isdir(project_model_dir):
            for file in os.listdir(project_model_dir):
                if file == training_log_name:  # Already handled above
                    continue
                    
                src_path = os.path.join(project_model_dir, file)
                if os.path.isfile(src_path):  # Only copy files, not directories
                    dst_path = os.path.join(web_model_dir, file)
                    try:
                        with open(src_path, 'rb') as src_file:
                            content = src_file.read()
                        with open(dst_path, 'wb') as dst_file:
                            dst_file.write(content)
                        print(f"Copied model file: {file} to web/{model_dir}")
                    except Exception as e:
                        print(f"Error copying {file} to web/{model_dir}: {e}")
    
    # Create symbolic links for training logs in the project root to ensure they're accessible in both places
    for model_dir in model_dirs:
        # Create symbolic links from model directories to root for common access
        project_model_dir = os.path.join(project_reports_dir, model_dir)
        training_log_name = f"{model_dir}_training_log.txt"
        project_log_path = os.path.join(project_model_dir, training_log_name)
        root_log_path = os.path.join(project_dir, training_log_name)
        
        # Create empty log file if it doesn't exist
        if not os.path.exists(project_log_path):
            try:
                with open(project_log_path, 'w') as f:
                    f.write(f"# Training log for {model_dir} - Created {Path.ctime(Path())}\n")
                print(f"Created empty training log: {project_log_path}")
            except Exception as e:
                print(f"Error creating training log {project_log_path}: {e}")
        
        # Create symbolic link in project root if it doesn't exist
        if not os.path.exists(root_log_path):
            try:
                os.symlink(project_log_path, root_log_path)
                print(f"Created symbolic link: {root_log_path} -> {project_log_path}")
            except Exception as e:
                print(f"Error creating symbolic link for {training_log_name}: {e}")
                
    print("\nModel directories and training logs setup complete.")
    print("Training logs will be saved to both project and web directories.")
    print("Symbolic links created in project root for easy access.")
    print("\nIMPORTANT: Make sure your training scripts write to these log files directly.")
    print("Example: /home/peter/Projects/LossLandscapeProbe/reports/cifar10/cifar10_training_log.txt")
