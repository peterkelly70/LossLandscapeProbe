"""
Report Generation Utilities for CIFAR Training
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)
import torch
from torch import nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

logger = logging.getLogger(__name__)
console = Console()

def generate_test_report(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    output_dir: Path,
    dataset_name: str = "cifar10",
    sample_size: str = "10",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Generate a comprehensive test report with metrics and visualizations.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        class_names: List of class names
        output_dir: Directory to save the report
        dataset_name: Name of the dataset
        sample_size: Sample size used for training
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing all test metrics
    """
    logger.info("Generating test report...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Get predictions and true labels
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in track(test_loader, description="Generating predictions..."):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Get predicted class indices and probabilities
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(
        all_targets, 
        all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Create output dictionary
    test_metrics = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'targets': all_targets.tolist(),
        'probabilities': all_probs.tolist()
    }
    
    # Save metrics to JSON
    metrics_path = output_dir / f"{dataset_name}_{sample_size}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Generate visualizations
    _generate_confusion_matrix_plot(
        cm, 
        class_names, 
        output_dir / f"{dataset_name}_{sample_size}_confusion_matrix.png"
    )
    
    # Generate HTML report
    _generate_html_report(test_metrics, output_dir, dataset_name, sample_size)
    
    logger.info(f"Test report generated at {output_dir}")
    return test_metrics

def _generate_confusion_matrix_plot(
    cm: np.ndarray, 
    class_names: List[str], 
    output_path: Path
) -> None:
    """Generate and save a confusion matrix plot."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def _generate_html_report(
    metrics: Dict[str, Any],
    output_dir: Path,
    dataset_name: str,
    sample_size: str
) -> None:
    """Generate an HTML report with all metrics and visualizations."""
    report_path = output_dir / f"{dataset_name}_{sample_size}_test_report.html"
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title} - Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .metric-card {{ 
                background: #f5f5f5; 
                border-radius: 5px; 
                padding: 15px; 
                margin-bottom: 20px;
            }}
            .row {{ display: flex; flex-wrap: wrap; margin: 0 -10px; }}
            .col {{ flex: 1; min-width: 300px; padding: 0 10px; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title} - Test Report</h1>
            <div class="section">
                <h2>Model Performance</h2>
                <div class="row">
                    <div class="col">
                        <div class="metric-card">
                            <h3>Overall Accuracy: {accuracy:.2%}</h3>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Confusion Matrix</h2>
                <img src="{confusion_matrix_img}" alt="Confusion Matrix">
            </div>
            
            <div class="section">
                <h2>Classification Report</h2>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                    </tr>
    """.format(
        title=f"{dataset_name.upper()} ({sample_size}%)",
        accuracy=metrics['accuracy'],
        confusion_matrix_img=f"{dataset_name}_{sample_size}_confusion_matrix.png"
    )
    
    # Add classification report rows
    for class_name, scores in metrics['report'].items():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
            
        html_content += f"""
            <tr>
                <td>{class_name}</td>
                <td>{precision:.2f}</td>
                <td>{recall:.2f}</td>
                <td>{f1:.2f}</td>
                <td>{support}</td>
            </tr>
        """.format(
            class_name=class_name,
            precision=scores['precision'],
            recall=scores['recall'],
            f1=scores['f1-score'],
            support=scores['support']
        )
    
    # Add footer
    html_content += """
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated at {report_path}")
