import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent))

# Import the model architecture
from examples.cifar10_example import SimpleCNN

# Set up paths
REPORTS_DIR = Path(__file__).parent / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

def load_dataset(name='cifar10', batch_size=32):
    """Load CIFAR test data (cifar10 or cifar100)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if name == 'cifar10':
        ds = datasets.CIFAR10
    else:
        ds = datasets.CIFAR100
    testset = ds(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    return testset, testloader

def plot_to_html_img(fig):
    """Convert a matplotlib figure to an HTML image tag"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode('ascii')
    return f'<img src="data:image/png;base64,{data}" class="img-fluid"/>'

def load_trained_model(model_path):
    """Load the trained model from checkpoint"""
    # Create model with default config
    model = SimpleCNN()
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Handle case where the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def generate_test_report(model_path, dataset='cifar10', sample_count=200, eval_batch_size=128):
    """Generate an HTML test report with sample predictions using the trained model"""
    # Get current date for the report
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Load the trained model
    try:
        model = load_trained_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to random predictions")
        model_loaded = False
        model = None
    
    # Load CIFAR-10 test data
    testset, test_loader = load_dataset(dataset, batch_size=eval_batch_size)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    # Iterate over entire test set for metrics
    if model is not None:
        with torch.no_grad():
            for imgs, lbls in test_loader:
                outputs = model(imgs)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds)
                all_probs.append(torch.max(probs, 1)[0])
                all_labels.append(lbls)
    else:
        # Random predictions fallback
        for imgs, lbls in test_loader:
            batch_size = imgs.size(0)
            preds = torch.randint(0, 10, (batch_size,))
            probs = torch.tensor(np.random.uniform(0.5, 1.0, batch_size))
            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(lbls)
    
    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    # Compute overall metrics
    correct_total = (all_preds == all_labels).sum().item()
    total_samples = len(all_labels)
    accuracy = correct_total / total_samples
    avg_confidence = all_probs.mean().item()
    
    # Select random subset for visualisation
    rng = torch.Generator().manual_seed(42)
    sample_indices = torch.randperm(total_samples, generator=rng)[:sample_count]
    images = torch.stack([testset[i][0] for i in sample_indices])
    labels = all_labels[sample_indices]
    predicted = all_preds[sample_indices]
    confidences = all_probs[sample_indices].numpy()
    
    # Make predictions variable exists from earlier section now unnecessary
    if model is not None:
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1)
            confidences = torch.max(probabilities, 1)[0].numpy()
    # CIFAR-10 class names
    classes = datasets.CIFAR10.classes if dataset=='cifar10' else datasets.CIFAR100.classes
    
    # Recalculate sample-level correct for border colouring
    correct_sample = (predicted == labels).sum().item()
    # Generate sample predictions HTML
    sample_html = []
    for i in range(min(sample_count, len(images))):
        image = images[i].numpy()
        label = labels[i].item()
        pred = predicted[i].item()
        confidence = confidences[i]
        
        # Convert image to base64
        img_data = np.transpose(image, (1, 2, 0)) * 0.5 + 0.5  # Unnormalize
        img_data = (img_data * 255).astype(np.uint8)
        
        buf = BytesIO()
        plt.imsave(buf, img_data, format='png')
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        is_correct = label == pred
        card_class = 'border-success' if is_correct else 'border-danger'
        progress_class = 'bg-success' if is_correct else 'bg-danger'
        
        sample_html.append(f'''
        <div class="col-lg-2 col-md-3 col-sm-4 col-6 mb-3">
            <div class="card h-100 {card_class} prediction-card">
                <img src="data:image/png;base64,{img_str}" class="card-img-top" alt="CIFAR-10 sample">
                <div class="card-body p-2">
                    <p class="card-title mb-1 small"><strong>Actual:</strong> {classes[label]}</p>
                    <p class="card-text mb-1 small"><strong>Predicted:</strong> {classes[pred]}</p>
                    <div class="progress" style="height: 10px;">
                        <div class="progress-bar {progress_class}" role="progressbar" style="width: {confidence*100}%" 
                             aria-valuenow="{confidence*100}" aria-valuemin="0" aria-valuemax="100">
                        </div>
                    </div>
                    <p class="text-end mb-0 small">{confidence*100:.1f}%</p>
                </div>
            </div>
        </div>''')
    
    # Generate the HTML report
    html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CIFAR-10 Test Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1400px;
            }}
            .prediction-card {{
                transition: transform 0.2s ease, opacity 0.2s ease;
                transform: translateY(10px);
                opacity: 0;
            }}
            .prediction-card img {{
                max-height: 100px;
                object-fit: cover;
            }}
            .progress {{
                height: 10px;
                font-size: 0.75rem;
            }}
            .class-accuracy-container {{
                max-width: 400px;
                margin: 0 auto;
            }}
            .class-accuracy-bar {{
                height: 10px;
            }}
            .class-accuracy-label {{ 
                display: flex; 
                justify-content: space-between; 
                margin-bottom: 0.25rem;
                font-size: 0.9rem;
            }}
            /* Optimize for many images */
            .row.g-2 {{
                contain: content;
            }}
            .card-body.p-2 {{
                padding: 0.5rem !important;
            }}
        </style>
    </head>
    <body>
        <div class="container py-4">
            <!-- Header -->
            <div class="text-center mb-5">
                <h1 class="display-5 fw-bold">CIFAR-10 Test Report</h1>
                <p class="text-muted">Generated on {report_date}</p>
            </div>
            
            <!-- Model Info -->
            <div class="card mb-4">
                <div class="card-header bg-light">
                    <h5 class="mb-0"><i class="bi bi-info-circle me-2"></i>Model Information</h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <p><strong>Model:</strong> SimpleCNN</p>
                            <p><strong>Dataset:</strong> CIFAR-10 Test Set</p>
                            <p><strong>Samples:</strong> {min(sample_count, len(images))} of {len(labels)}</p>
                        </div>
                        <div class="col-md-6">
                            <p><strong>Input Size:</strong> 32x32x3</p>
                            <p><strong>Classes:</strong> {', '.join(classes)}</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Metrics -->
            <div class="row mb-4">
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body text-center">
                            <h5 class="card-title">Accuracy</h5>
                            <h2 class="text-primary">{accuracy*100:.2f}%</h2>
                            <p class="text-muted">{correct_sample} out of {len(labels)} samples</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body text-center">
                            <h5 class="card-title">Avg. Confidence</h5>
                            <h2 class="text-info">{avg_confidence*100:.2f}%</h2>
                            <p class="text-muted">Average prediction confidence</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body text-center">
                            <h5 class="card-title">Model Status</h5>
                            <h4><span class="badge bg-{'success' if model_loaded else 'danger'}">
                                {'Loaded Successfully' if model_loaded else 'Using Random Predictions'}
                            </span></h4>
                            <p class="text-muted">{model_path if model_loaded else 'Model not found'}</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Sample Predictions -->
            <div class="card mb-4">
                <div class="card-header bg-light">
                    <h5 class="mb-0"><i class="bi bi-images me-2"></i>CIFAR-10 Test Predictions</h5>
                    <p class="text-muted mb-0">Showing {min(sample_count, len(images))} random test samples with model predictions</p>
                    <p id="sample-accuracy" class="text-muted mb-0"></p>
                </div>
                <div class="card-body">
                    <div class="row g-2">
                        {''.join(sample_html)}
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <footer class="text-center text-muted mt-5">
                <p>Generated with LossLandscapeProbe &copy; {datetime.now().year}</p>
            </footer>
        </div>
        
        <script>
            // Add animation to prediction cards on scroll with lazy loading
            document.addEventListener('DOMContentLoaded', function() {{
                // Use a more efficient batch processing approach
                const cards = document.querySelectorAll('.prediction-card');
                const batchSize = 20; // Process cards in batches for better performance
                let currentBatch = 0;
                
                const observer = new IntersectionObserver(function(entries) {{
                    entries.forEach(function(entry) {{
                        if (entry.isIntersecting) {{
                            entry.target.style.opacity = '1';
                            entry.target.style.transform = 'translateY(0)';
                            observer.unobserve(entry.target); // Stop observing once visible
                        }}
                    }});
                }}, {{threshold: 0.1, rootMargin: '100px'}});
                
                // Process cards in batches to avoid layout thrashing
                function processBatch() {{
                    const start = currentBatch * batchSize;
                    const end = Math.min(start + batchSize, cards.length);
                    
                    for (let i = start; i < end; i++) {{
                        observer.observe(cards[i]);
                    }}
                    
                    currentBatch++;
                    if (start + batchSize < cards.length) {{
                        setTimeout(processBatch, 100); // Schedule next batch
                    }}
                }}
                
                processBatch(); // Start processing
                
                // Add summary at the top
                const correctCards = document.querySelectorAll('.border-success').length;
                const totalCards = cards.length;
                document.getElementById('sample-accuracy').textContent = 
                    `${{correctCards}} correct out of ${{totalCards}} samples (${{(correctCards/totalCards*100).toFixed(1)}}%)`;
            }});
        </script>
    </body>
    </html>"""
    
    # Save report
    report_path = REPORTS_DIR / 'test_report.html'
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Test report generated at: {report_path}")
    return report_path

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description='Generate CIFAR test report')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', choices=['cifar10','cifar100'], default='cifar10')
    parser.add_argument('--samples', type=int, default=200)
    args = parser.parse_args()
    if not os.path.isfile(args.model_path):
        print('Checkpoint not found:', args.model_path)
    generate_test_report(model_path=args.model_path, dataset=args.dataset, sample_count=args.samples)
