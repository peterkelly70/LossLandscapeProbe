#!/usr/bin/env python3
"""
CIFAR Reporting Module
====================

Report generation and visualization for CIFAR datasets.
This module handles:
- Training history visualization
- Meta-model results visualization
- Test results reporting
- HTML report generation
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import logging
from pathlib import Path
import torch
import torchvision
import random
from typing import Dict, List, Tuple, Any, Optional
import base64
from io import BytesIO
import os

logger = logging.getLogger(__name__)


class CIFARReporter:
    """
    Reporter for CIFAR datasets.
    
    This class handles the generation of reports and visualizations
    for CIFAR training and meta-model results.
    """
    
    def __init__(
        self,
        dataset: str = 'cifar10',
        sample_size: str = 'base',
        run_dir: Optional[Path] = None
    ):
        """
        Initialize the reporter.
        
        Args:
            dataset: Dataset name ('cifar10' or 'cifar100')
            sample_size: Sample size identifier ('base', '10', '20', etc.)
            run_dir: Directory with training results
        """
        self.dataset = dataset
        self.sample_size = sample_size
        self.run_dir = run_dir
        
        # Load metrics if available
        self.training_metrics = None
        self.meta_results = None
        self.test_metrics = None
        
        if run_dir:
            if (run_dir / "training_metrics.json").exists():
                with open(run_dir / "training_metrics.json", "r") as f:
                    self.training_metrics = json.load(f)
            
            if (run_dir / "meta_model" / "meta_results.json").exists():
                with open(run_dir / "meta_model" / "meta_results.json", "r") as f:
                    self.meta_results = json.load(f)
            
            if (run_dir / "test_metrics.json").exists():
                with open(run_dir / "test_metrics.json", "r") as f:
                    self.test_metrics = json.load(f)
    
    def plot_training_history(self):
        """
        Plot the training history.
        
        Returns:
            Matplotlib figure
        """
        if not self.training_metrics:
            logger.warning("No training metrics available")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        ax1.plot(self.training_metrics['train_loss'], label='Train')
        ax1.plot(self.training_metrics['val_loss'], label='Validation')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.training_metrics['train_acc'], label='Train')
        ax2.plot(self.training_metrics['val_acc'], label='Validation')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the figure
        if self.run_dir:
            plt.savefig(self.run_dir / "training_history.png")
        
        return fig
    
    def plot_meta_model_results(self):
        """
        Plot the meta-model results.
        
        Returns:
            Matplotlib figure
        """
        if not self.meta_results:
            logger.warning("No meta-model results available")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract configurations and validation accuracies
        configs = self.meta_results['all_configs']
        val_accs = self.meta_results['all_val_accs']
        best_config = self.meta_results['best_config']
        best_val_acc = self.meta_results['best_val_acc']
        
        # Group by optimizer
        sgd_indices = [i for i, c in enumerate(configs) if c['optimizer'] == 'sgd']
        adam_indices = [i for i, c in enumerate(configs) if c['optimizer'] == 'adam']
        
        # Plot configurations
        if sgd_indices:
            ax.scatter(
                [configs[i]['learning_rate'] for i in sgd_indices],
                [val_accs[i] for i in sgd_indices],
                label='SGD',
                alpha=0.7,
                s=80
            )
        
        if adam_indices:
            ax.scatter(
                [configs[i]['learning_rate'] for i in adam_indices],
                [val_accs[i] for i in adam_indices],
                label='Adam',
                alpha=0.7,
                s=80
            )
        
        # Highlight the best configuration
        if best_config:
            ax.scatter(
                best_config['learning_rate'],
                best_val_acc,
                color='red',
                s=150,
                marker='*',
                label='Best'
            )
        
        ax.set_title('Meta-Model Results')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Validation Accuracy')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        if self.run_dir:
            plt.savefig(self.run_dir / "meta_model_results.png")
        
        return fig
    
    def plot_class_accuracy(self):
        """
        Plot the per-class accuracy.
        
        Returns:
            Matplotlib figure
        """
        if not self.test_metrics or 'class_acc' not in self.test_metrics:
            logger.warning("No per-class accuracy available")
            return None
        
        # Get class names if available
        class_names = None
        if self.dataset == 'cifar10':
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
        
        class_acc = self.test_metrics['class_acc']
        classes = sorted(list(class_acc.keys()))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(
            range(len(classes)),
            [class_acc[str(c)] if isinstance(classes[0], str) else class_acc[c] for c in classes],
            align='center'
        )
        
        ax.set_title('Per-Class Accuracy')
        ax.set_ylabel('Accuracy')
        
        if class_names:
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
        else:
            ax.set_xlabel('Class')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the figure
        if self.run_dir:
            plt.savefig(self.run_dir / "class_accuracy.png")
        
        return fig
    
    def _fig_to_base64(self, fig):
        """Convert a matplotlib figure to base64 string for HTML embedding."""
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_str
    
    def generate_training_report(self):
        """
        Generate an HTML training report.
        
        Returns:
            HTML string
        """
        if not self.training_metrics:
            logger.warning("No training metrics available")
            return "No training metrics available"
        
        # Plot training history
        history_fig = self.plot_training_history()
        history_img = self._fig_to_base64(history_fig) if history_fig else ""
        
        # Get training details
        config = self.training_metrics['config']
        best_val_acc = self.training_metrics['best_val_acc']
        best_epoch = self.training_metrics['best_epoch']
        training_time = self.training_metrics.get('training_time', 0)
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.dataset}_{self.sample_size} Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metrics {{ margin: 20px 0; }}
                .metrics table {{ border-collapse: collapse; width: 100%; }}
                .metrics th, .metrics td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics th {{ background-color: #f2f2f2; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{self.dataset.upper()} Training Report</h1>
                <p>Sample size: {self.sample_size}</p>
                
                <h2>Training Configuration</h2>
                <div class="metrics">
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
        """
        
        # Add configuration parameters
        for key, value in config.items():
            html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
            """
        
        html += f"""
                    </table>
                </div>
                
                <h2>Training Results</h2>
                <div class="metrics">
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Best Validation Accuracy</td>
                            <td>{best_val_acc:.4f}</td>
                        </tr>
                        <tr>
                            <td>Best Epoch</td>
                            <td>{best_epoch}</td>
                        </tr>
                        <tr>
                            <td>Training Time</td>
                            <td>{training_time:.2f} seconds</td>
                        </tr>
                    </table>
                </div>
                
                <h2>Training History</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{history_img}" alt="Training History">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save the report
        if self.run_dir:
            report_path = self.run_dir / f"{self.dataset}_{self.sample_size}_training_report.html"
            with open(report_path, "w") as f:
                f.write(html)
            logger.info(f"Training report saved to {report_path}")
        
        return html
    
    def generate_meta_model_report(self):
        """
        Generate an HTML meta-model report.
        
        Returns:
            HTML string
        """
        if not self.meta_results:
            logger.warning("No meta-model results available")
            return "No meta-model results available"
        
        # Plot meta-model results
        meta_fig = self.plot_meta_model_results()
        meta_img = self._fig_to_base64(meta_fig) if meta_fig else ""
        
        # Get meta-model details
        best_config = self.meta_results['best_config']
        best_val_acc = self.meta_results['best_val_acc']
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.dataset}_{self.sample_size} Meta-Model Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metrics {{ margin: 20px 0; }}
                .metrics table {{ border-collapse: collapse; width: 100%; }}
                .metrics th, .metrics td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics th {{ background-color: #f2f2f2; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{self.dataset.upper()} Meta-Model Report</h1>
                <p>Sample size: {self.sample_size}</p>
                
                <h2>Best Hyperparameters</h2>
                <div class="metrics">
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
        """
        
        # Add best configuration parameters
        for key, value in best_config.items():
            html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
            """
        
        html += f"""
                        <tr>
                            <td>Best Validation Accuracy</td>
                            <td>{best_val_acc:.4f}</td>
                        </tr>
                    </table>
                </div>
                
                <h2>Meta-Model Results</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{meta_img}" alt="Meta-Model Results">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save the report
        if self.run_dir:
            report_path = self.run_dir / f"{self.dataset}_{self.sample_size}_meta_report.html"
            with open(report_path, "w") as f:
                f.write(html)
            logger.info(f"Meta-model report saved to {report_path}")
        
        return html
    
    def _get_prediction_samples_html(self, samples_info, class_names, samples_per_row=10):
        """Generate HTML for prediction samples grid."""
        if not samples_info or 'samples' not in samples_info or not samples_info['samples']:
            return "<p>No prediction samples available.</p>"
            
        html = """
        <h2>Prediction Samples</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 15px; margin: 20px 0;">
        """
        
        for sample in samples_info['samples']:
            is_correct = sample.get('correct', False)
            true_label = class_names[sample['true_label']] if sample['true_label'] < len(class_names) else str(sample['true_label'])
            pred_label = class_names[sample['pred_label']] if sample['pred_label'] < len(class_names) else str(sample['pred_label'])
            
            # Get relative path for HTML
            img_path = Path(sample['image_path'])
            if img_path.is_absolute():
                img_path = img_path.relative_to(self.run_dir)
            
            html += f"""
            <div style="
                border: 3px solid {'#4CAF50' if is_correct else '#F44336'};
                border-radius: 8px;
                padding: 5px;
                text-align: center;
                background: {'#E8F5E9' if is_correct else '#FFEBEE'};
            ">
                <img src="{img_path}" 
                     style="width: 100%; height: auto; image-rendering: pixelated;"
                     alt="Sample">
                <div style="
                    font-size: 11px;
                    margin-top: 5px;
                    word-wrap: break-word;
                    line-height: 1.2;
                ">
                    <div><strong>True:</strong> {true_label}</div>
                    <div><strong>Pred:</strong> {pred_label}</div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html

    def generate_test_report(self):
        """
        Generate an HTML test report with prediction samples.
        
        Returns:
            HTML string
        """
        if not self.test_metrics:
            logger.warning("No test metrics available")
            return "No test metrics available"
        
        # Plot class accuracy
        class_fig = self.plot_class_accuracy()
        class_img = self._fig_to_base64(class_fig) if class_fig else ""
        
        # Get test details
        test_acc = self.test_metrics['test_acc']
        test_loss = self.test_metrics['test_loss']
        
        # Load prediction samples if available
        samples_html = ""
        samples_file = self.run_dir / "prediction_samples.json"
        if samples_file.exists():
            try:
                with open(samples_file, 'r') as f:
                    samples_info = json.load(f)
                
                # Get class names
                if self.dataset == 'cifar10':
                    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                 'dog', 'frog', 'horse', 'ship', 'truck']
                else:  # cifar100
                    class_names = [
                        'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                        'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                        'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                        'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
                        'trees', 'vehicles 1', 'vehicles 2'
                    ]
                
                samples_html = self._get_prediction_samples_html(samples_info, class_names)
            except Exception as e:
                logger.error(f"Error loading prediction samples: {e}")
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.dataset.upper()} Test Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0;
                    padding: 20px;
                    color: #333;
                    line-height: 1.6;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{ 
                    color: #2c3e50;
                    margin-top: 1.5em;
                }}
                h1 {{ 
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                .metrics {{ 
                    margin: 20px 0;
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                }}
                .metrics table {{ 
                    width: 100%;
                    border-collapse: collapse;
                }}
                .metrics th, .metrics td {{ 
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .metrics th {{ 
                    background-color: #3498db;
                    color: white;
                }}
                .metrics tr:nth-child(even) {{ 
                    background-color: #f2f2f2;
                }}
                .metrics tr:hover {{ 
                    background-color: #e9f7fe;
                }}
                .plot {{ 
                    margin: 30px 0;
                    text-align: center;
                    background: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .plot img {{ 
                    max-width: 100%;
                    border-radius: 4px;
                }}
                .sample-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .sample {{
                    border: 2px solid #ddd;
                    border-radius: 8px;
                    overflow: hidden;
                    transition: transform 0.2s, box-shadow 0.2s;
                }}
                .sample:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .sample.correct {{ border-color: #4CAF50; }}
                .sample.incorrect {{ border-color: #F44336; }}
                .sample img {{
                    width: 100%;
                    height: 140px;
                    object-fit: contain;
                    background: #f8f9fa;
                    padding: 5px;
                }}
                .sample-info {{
                    padding: 10px;
                    font-size: 12px;
                    line-height: 1.4;
                }}
                .sample-info .label {{
                    font-weight: bold;
                    display: block;
                    margin-bottom: 3px;
                }}
                .sample-info .true {{ color: #4CAF50; }}
                .sample-info .pred {{ color: #F44336; }}
                .accuracy-badge {{
                    display: inline-block;
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                    margin-left: 5px;
                }}
                .correct .accuracy-badge {{ background: #E8F5E9; color: #2E7D32; }}
                .incorrect .accuracy-badge {{ background: #FFEBEE; color: #C62828; }}
                .legend {{
                    display: flex;
                    justify-content: center;
                    gap: 20px;
                    margin: 20px 0;
                }}
                .legend-item {{
                    display: flex;
                    align-items: center;
                    font-size: 14px;
                }}
                .legend-color {{
                    width: 20px;
                    height: 20px;
                    border-radius: 4px;
                    margin-right: 5px;
                }}
                .correct-color {{ background-color: #4CAF50; }}
                .incorrect-color {{ background-color: #F44336; }}
                @media (max-width: 768px) {{
                    .sample-grid {{
                        grid-template-columns: repeat(3, 1fr);
                    }}
                }}
                @media (max-width: 480px) {{
                    .sample-grid {{
                        grid-template-columns: repeat(2, 1fr);
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{self.dataset.upper()} Test Report</h1>
                
                <div class="metrics">
                    <h2>Test Results</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Test Accuracy</td>
                            <td>{test_acc:.4f}</td>
                        </tr>
                        <tr>
                            <td>Test Loss</td>
                            <td>{test_loss:.4f}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="plot">
                    <h2>Per-Class Accuracy</h2>
                    <img src="data:image/png;base64,{class_img}" alt="Class Accuracy">
                </div>
                
                {samples_html}
                
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color correct-color"></div>
                        <span>Correct Prediction</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color incorrect-color"></div>
                        <span>Incorrect Prediction</span>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save the report
        if self.run_dir:
            report_path = self.run_dir / f"{self.dataset}_{self.sample_size}_test_report.html"
            with open(report_path, "w") as f:
                f.write(html)
            logger.info(f"Test report saved to {report_path}")
        
        return html
    
    def generate_all_reports(self):
        """Generate all reports."""
        self.generate_training_report()
        self.generate_meta_model_report()
        self.generate_test_report()
        logger.info("All reports generated")
