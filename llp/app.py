#!/usr/bin/env python3
"""
Flask web server for viewing CIFAR training reports.
"""

import os
import argparse
from pathlib import Path
from flask import Flask, send_from_directory, render_template, redirect, url_for

app = Flask(__name__)

# Default reports directory
DEFAULT_REPORTS_DIR = Path(__file__).parent / "reports"

@app.route('/')
def index():
    """Render the main index page with links to all reports."""
    reports = []
    
    # Scan the reports directory for datasets
    for dataset_dir in sorted(DEFAULT_REPORTS_DIR.glob('*')):
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        samples = []
        
        # Find all sample sizes for this dataset
        for sample_dir in sorted(dataset_dir.glob('*')):
            if not sample_dir.is_dir():
                continue
                
            sample_name = sample_dir.name
            report_path = sample_dir / f"{dataset_name}_{sample_name}_test_report.html"
            
            if report_path.exists():
                samples.append({
                    'name': f"{sample_name}% Sample",
                    'path': f"{dataset_name}/{sample_name}/report"
                })
        
        if samples:
            reports.append({
                'name': dataset_name.upper(),
                'samples': samples
            })
    
    return render_template('index.html', reports=reports)

@app.route('/<dataset>/<sample>/report')
def view_report(dataset, sample):
    """View a specific report."""
    report_dir = DEFAULT_REPORTS_DIR / dataset / sample
    report_file = f"{dataset}_{sample}_test_report.html"
    
    if not (report_dir / report_file).exists():
        return "Report not found", 404
    
    return send_from_directory(report_dir, report_file)

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from the reports directory."""n    parts = path.split('/')
    if len(parts) < 2:
        return "Invalid path", 400
        
    # Reconstruct the path to the file
    file_path = DEFAULT_REPORTS_DIR / '/'.join(parts[:-1])
    file_name = parts[-1]
    
    # Security check to prevent directory traversal
    try:
        file_path.resolve().relative_to(DEFAULT_REPORTS_DIR.resolve())
    except ValueError:
        return "Access denied", 403
    
    if not file_path.exists():
        return "Not found", 404
        
    return send_from_directory(file_path, file_name)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CIFAR Training Report Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Create a simple index template if it doesn't exist
    index_template = templates_dir / 'index.html'
    if not index_template.exists():
        with open(index_template, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>CIFAR Training Reports</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .dataset {
            margin-bottom: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .dataset h2 {
            color: #3498db;
            margin-top: 0;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        .samples {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .sample {
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            text-align: center;
            text-decoration: none;
            color: #2c3e50;
            border: 1px solid #e0e0e0;
        }
        .sample:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-color: #3498db;
        }
        .no-reports {
            text-align: center;
            color: #7f8c8d;
            padding: 40px 20px;
            font-style: italic;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
        }
        .refresh-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            font-size: 14px;
        }
        .refresh-btn:hover {
            background: #2980b9;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>CIFAR Training Reports</h1>
            <a href="/" class="refresh-btn">Refresh</a>
        </div>
        
        {% if reports %}
            {% for dataset in reports %}
                <div class="dataset">
                    <h2>{{ dataset.name }}</h2>
                    <div class="samples">
                        {% for sample in dataset.samples %}
                            <a href="{{ url_for('view_report', dataset=dataset.name.lower(), sample=sample.path.split('/')[-2]) }}" class="sample">
                                <div class="sample-name">{{ sample.name }}</div>
                            </a>
                        {% endfor %}
                    </div>
                </div>
            {% endfor %}
        {% else %}
            <div class="no-reports">
                <h2>No reports found</h2>
                <p>Run the training script to generate reports.</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
""")
    
    args = parse_args()
    
    # Ensure the reports directory exists
    DEFAULT_REPORTS_DIR.mkdir(exist_ok=True)
    
    print(f"Starting CIFAR Report Server at http://{args.host}:{args.port}")
    print(f"Serving reports from: {DEFAULT_REPORTS_DIR.absolute()}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
