#!/usr/bin/env python3
"""
Progress Bar Logger for CIFAR Meta-Model Training

This script adds detailed progress reporting to the CIFAR meta-model training process
by intercepting and enhancing the logging output.
"""

import os
import sys
import logging
import time
from typing import Dict, Any, Optional

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False
    print("tqdm not installed. Progress bars will be text-based only.")

# Configure logging
class ProgressBarHandler(logging.Handler):
    """Custom logging handler that displays progress bars for training."""
    
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.progress_bars = {}
        self.current_config = 0
        self.total_configs = 0
        self.current_subset = 0
        self.total_subsets = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.start_time = time.time()
        
    def emit(self, record):
        """Process log records and update progress bars."""
        msg = self.format(record)
        
        # Extract progress information from log messages
        if "Starting meta-model hyperparameter optimization" in msg:
            print("\n🔍 Starting meta-model hyperparameter optimization...")
            
        elif "Sampling" in msg and "hyperparameter configurations" in msg:
            try:
                num_configs = int(msg.split("Sampling ")[1].split(" ")[0])
                self.total_configs = num_configs
                print(f"🔍 Sampling {num_configs} hyperparameter configurations...")
                
                if USE_TQDM:
                    self.progress_bars['configs'] = tqdm(
                        total=num_configs, 
                        desc="Meta-Model Configs",
                        bar_format='{desc}: {bar} {percentage:3.0f}% | {n_fmt}/{total_fmt}'
                    )
            except:
                pass
                
        elif "Evaluating configuration" in msg:
            try:
                config_idx = int(msg.split("configuration ")[1].split("/")[0]) - 1
                self.current_config = config_idx
                print(f"\n🔄 Evaluating configuration {config_idx+1}/{self.total_configs} "
                      f"[{(config_idx/self.total_configs)*100:.1f}% configurations]")
                
                if USE_TQDM and 'configs' in self.progress_bars:
                    self.progress_bars['configs'].update(1)
                    self.progress_bars['configs'].refresh()
            except:
                pass
                
        elif "Evaluating dataset" in msg:
            try:
                subset_idx = int(msg.split("dataset ")[1].split("/")[0]) - 1
                total_subsets = int(msg.split("dataset ")[1].split("/")[1].split(" ")[0])
                
                if self.total_subsets != total_subsets:
                    self.total_subsets = total_subsets
                    if USE_TQDM:
                        if 'subsets' in self.progress_bars:
                            self.progress_bars['subsets'].close()
                        self.progress_bars['subsets'] = tqdm(
                            total=total_subsets, 
                            desc="Dataset Progress",
                            bar_format='{desc}: {bar} {percentage:3.0f}% | {n_fmt}/{total_fmt}'
                        )
                
                self.current_subset = subset_idx
                
                # Extract overall progress if available
                overall_info = ""
                if "Overall Progress:" in msg:
                    overall_info = msg.split("Overall Progress:")[1].strip()
                
                print(f"📈 Evaluating dataset {subset_idx+1}/{total_subsets} "
                      f"{overall_info}")
                
                if USE_TQDM and 'subsets' in self.progress_bars:
                    self.progress_bars['subsets'].update(1)
                    self.progress_bars['subsets'].refresh()
            except:
                pass
                
        elif "Training meta-model for" in msg:
            try:
                epochs = int(msg.split("for ")[1].split(" ")[0])
                self.total_epochs = epochs
                print(f"\n🧠 Training meta-model for {epochs} epochs...")
                
                if USE_TQDM:
                    if 'epochs' in self.progress_bars:
                        self.progress_bars['epochs'].close()
                    self.progress_bars['epochs'] = tqdm(
                        total=epochs, 
                        desc="Meta-Model Training",
                        bar_format='{desc}: {bar} {percentage:3.0f}% | {n_fmt}/{total_fmt}'
                    )
            except:
                pass
                
        elif "Epoch" in msg and "train_loss=" in msg:
            try:
                epoch = int(msg.split("Epoch ")[1].split("/")[0]) - 1
                self.current_epoch = epoch
                
                # Extract loss values
                train_loss = float(msg.split("train_loss=")[1].split(",")[0])
                val_loss = float(msg.split("val_loss=")[1].split(")")[0])
                
                print(f"📊 Epoch {epoch+1}/{self.total_epochs}: "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                if USE_TQDM and 'epochs' in self.progress_bars:
                    self.progress_bars['epochs'].update(1)
                    self.progress_bars['epochs'].set_postfix(
                        train_loss=f"{train_loss:.4f}", 
                        val_loss=f"{val_loss:.4f}"
                    )
                    self.progress_bars['epochs'].refresh()
            except:
                pass
                
        elif "New best configuration found" in msg:
            try:
                accuracy = float(msg.split("Accuracy: ")[1])
                print(f"🏆 New best configuration found! Accuracy: {accuracy:.4f}")
            except:
                pass
                
        elif "New best meta-model found" in msg:
            try:
                val_loss = float(msg.split("Validation loss: ")[1])
                print(f"📝 New best meta-model found! Validation loss: {val_loss:.4f}")
            except:
                pass
                
        elif "Meta-model optimization complete" in msg:
            print(f"✅ Meta-model optimization complete!")
            
            # Close all progress bars
            for bar_name, bar in self.progress_bars.items():
                bar.close()
                
            # Print total time
            elapsed = time.time() - self.start_time
            print(f"⏱️ Total time: {elapsed:.2f} seconds")
            
        # Always print the original message to ensure all information is captured
        print(f"[LOG] {msg}")

def setup_progress_logging():
    """Set up progress bar logging for CIFAR meta-model training."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.FileHandler('meta_model_progress.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(file_handler)
    
    # Add progress bar handler
    progress_handler = ProgressBarHandler(level=logging.INFO)
    progress_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(progress_handler)
    
    # Add console handler for direct output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(console_handler)
    
    return root_logger

if __name__ == "__main__":
    # Set up progress logging
    logger = setup_progress_logging()
    
    # Print welcome message
    print("=" * 60)
    print("CIFAR Meta-Model Training with Enhanced Progress Reporting")
    print("=" * 60)
    
    # Run the original training script with enhanced logging
    try:
        # Add the project root to the path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import and run the training script
        print("Running training script with enhanced progress reporting...")
        
        # This will use the enhanced logging for all imported modules
        import train_with_meta
        train_with_meta.main()
        
    except Exception as e:
        logger.error(f"Error running training script: {str(e)}")
        print(f"Error running training script: {str(e)}")
        raise
