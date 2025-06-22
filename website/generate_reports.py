import os
import torch
from pathlib import Path
from datetime import datetime

def load_results():
    """Load the results from the .pth files."""
    base_dir = Path(__file__).parent.parent
    
    # Load CIFAR-100 transfer results
    cifar100_results = {
        'exists': False,
        'path': base_dir / 'reports' / 'cifar100_transfer' / 'cifar100_transfer_results.pth'
    }
    
    if cifar100_results['path'].exists():
        cifar100_results.update(torch.load(cifar100_results['path']))
        cifar100_results['exists'] = True
    
    return {
        'cifar100': cifar100_results
    }

def generate_html_report(results):
    """Generate an HTML report from the results."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LossLandscapeProbe - Training Report</title>
    <style>
        body {{ 
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; 
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto;
        }}
        header {{ 
            background: #1a365d; 
            color: white; 
            padding: 20px; 
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        h1, h2, h3 {{ 
            color: #2c5282; 
            margin-top: 0;
        }}
        .card {{ 
            background: white; 
            border-radius: 8px; 
            padding: 20px; 
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f0f5ff;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            border-left: 4px solid #4299e1;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2b6cb0;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            color: #4a5568;
        }}
        .config-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .config-table th, .config-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        .config-table th {{
            background: #f8fafc;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #718096;
            font-size: 14px;
            border-top: 1px solid #e2e8f0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>LossLandscapeProbe</h1>
            <p>Model Training Report - {timestamp}</p>
        </header>
        
        <main>
            <div class="card">
                <h2>Model Performance Summary</h2>
                <p>This report provides an overview of the model's performance on the CIFAR datasets.</p>
                
                <h3 style="margin-top: 30px;">CIFAR-100 Transfer Learning</h3>
    """
    
    # Add CIFAR-100 section if data exists
    if results['cifar100']['exists']:
        cifar100 = results['cifar100']
        html += f"""
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{cifar100['final_acc']*100:.2f}%</div>
                        <div class="metric-label">Final Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{cifar100['best_acc']*100:.2f}%</div>
                        <div class="metric-label">Best Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{cifar100['cifar10_acc']*100:.2f}%</div>
                        <div class="metric-label">CIFAR-10 Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{cifar100['relative_performance']*100:.1f}%</div>
                        <div class="metric-label">Relative Performance</div>
                    </div>
                </div>
                
                <h4>Best Configuration</h4>
                <table class="config-table">
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
        """
        
        # Add configuration parameters
        for param, value in cifar100['best_config'].items():
            html += f"""
                    <tr>
                        <td><strong>{param.replace('_', ' ').title()}</strong></td>
                        <td><code>{value}</code></td>
                    </tr>
            """
        
        # Format training progress section with proper error handling
        train_losses = cifar100.get('train_losses', [])
        test_accs = cifar100.get('test_accs', [])
        
        progress_html = """
                </table>
                
                <div style="margin-top: 30px;">
                    <h4>Training Progress</h4>
        """
        
        if train_losses:
            progress_html += f"<p>Initial training loss: {train_losses[0]:.4f} → Final training loss: {train_losses[-1]:.4f}</p>"
        if test_accs:
            progress_html += f"<p>Initial test accuracy: {test_accs[0]*100:.2f}% → Final test accuracy: {test_accs[-1]*100:.2f}%</p>"
            
        progress_html += "</div>"
        html += progress_html
    else:
        html += """
                <div class="card" style="background-color: #fff5f5; border-left: 4px solid #f56565;">
                    <h3>No CIFAR-100 Results Found</h3>
                    <p>Could not find CIFAR-100 transfer learning results. Please run the training script first.</p>
                </div>
        """
    
    # Close the HTML
    html += """
            </div>
        </main>
        
        <footer class="footer">
            <p>Generated by LossLandscapeProbe on {timestamp}</p>
            <p>For more information, please refer to the project documentation.</p>
        </footer>
    </div>
</body>
</html>
    """.format(timestamp=timestamp)
    
    return html

def save_report(html_content, filename='training_report.html'):
    """Save the HTML report to a file."""
    reports_dir = Path(__file__).parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    report_path = reports_dir / filename
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path

def main():
    print("Loading training results...")
    results = load_results()
    
    print("Generating HTML report...")
    html_content = generate_html_report(results)
    
    print("Saving report...")
    report_path = save_report(html_content)
    
    print(f"\n✅ Report generated successfully!")
    print(f"📄 Report saved to: {report_path}")
    print(f"🌐 You can open it in your web browser or access it through the web interface.")

if __name__ == "__main__":
    main()
