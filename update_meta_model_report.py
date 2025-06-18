#!/usr/bin/env python3
import os
import re
import json
from datetime import datetime

def parse_meta_model_log(log_file):
    """Parse the meta-model training log file to extract progress information."""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return None
    
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Extract basic information
    info = {
        'start_time': None,
        'current_iteration': 0,
        'total_iterations': 3,  # Default from log
        'configurations_evaluated': 0,
        'total_configurations': 10,  # Default from log
        'current_resource_level': 0.1,
        'configurations': []
    }
    
    # Extract start time
    start_time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', log_content)
    if start_time_match:
        info['start_time'] = start_time_match.group(1)
    
    # Extract iteration information
    iteration_match = re.search(r'Meta-optimization iteration (\d+)/(\d+)', log_content)
    if iteration_match:
        info['current_iteration'] = int(iteration_match.group(1))
        info['total_iterations'] = int(iteration_match.group(2))
    
    # Extract resource level
    resource_match = re.search(r'at resource level ([\d.]+)', log_content)
    if resource_match:
        info['current_resource_level'] = float(resource_match.group(1))
    
    # Extract configurations evaluated
    config_matches = re.findall(r'Evaluating configuration (\d+)/(\d+)', log_content)
    if config_matches:
        last_match = config_matches[-1]
        info['configurations_evaluated'] = int(last_match[0])
        info['total_configurations'] = int(last_match[1])
    
    # Extract sharpness and robustness measurements
    sharpness_matches = re.findall(r'Sharpness: ([\d.]+), Perturbation Robustness: ([\d.]+)', log_content)
    for i, match in enumerate(sharpness_matches):
        if i >= len(info['configurations']):
            info['configurations'].append({})
        info['configurations'][i]['sharpness'] = float(match[0])
        info['configurations'][i]['robustness'] = float(match[1])
    
    # Extract performance metrics
    performance_matches = re.findall(r'Added training example with performance ([\d.]+)', log_content)
    for i, match in enumerate(performance_matches):
        if i >= len(info['configurations']):
            info['configurations'].append({})
        info['configurations'][i]['performance'] = float(match)
    
    # Calculate overall progress percentage
    total_steps = info['total_iterations'] * info['total_configurations']
    current_steps = (info['current_iteration'] - 1) * info['total_configurations'] + info['configurations_evaluated']
    info['progress_percent'] = min(100, round((current_steps / total_steps) * 100))
    
    return info

def generate_meta_model_report(log_file, output_path):
    """Generate an HTML report for meta-model training progress."""
    info = parse_meta_model_log(log_file)
    if not info:
        return False
    
    # Format the timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create configuration table rows
    config_rows = ""
    for i, config in enumerate(info['configurations']):
        if 'sharpness' in config and 'robustness' in config and 'performance' in config:
            config_rows += f"""
            <tr>
                <td>{i+1}</td>
                <td>{config['sharpness']:.6f}</td>
                <td>{config['robustness']:.4f}</td>
                <td>{config['performance']:.4f}</td>
            </tr>
            """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meta-Model Training Progress</title>
    <style>
        body {{
            padding: 20px;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #111;
            color: #eee;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #4cf;
        }}
        .header {{
            margin-bottom: 30px;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }}
        .status {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            background-color: #264c73;
            color: #4cf;
            margin-left: 10px;
        }}
        .card {{
            background-color: #222;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            border: 1px solid #333;
        }}
        .progress-container {{
            background-color: #333;
            border-radius: 4px;
            height: 25px;
            margin-bottom: 20px;
            position: relative;
        }}
        .progress-bar {{
            background-color: #4cf;
            height: 100%;
            border-radius: 4px;
            width: {info['progress_percent']}%;
            transition: width 0.5s ease;
        }}
        .progress-text {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            text-align: center;
            line-height: 25px;
            color: white;
            font-weight: bold;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background-color: #1a1a1a;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            border: 1px solid #333;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4cf;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 14px;
            color: #aaa;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background-color: #1a1a1a;
            color: #4cf;
        }}
        tr:nth-child(even) {{
            background-color: #1a1a1a;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #333;
            font-size: 12px;
            color: #777;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>CIFAR-10 Meta-Model Training Progress <span class="status">IN PROGRESS</span></h1>
            <p>Last updated: {timestamp}</p>
        </div>
        
        <div class="card">
            <h2>Training Progress</h2>
            <div class="progress-container">
                <div class="progress-bar"></div>
                <div class="progress-text">{info['progress_percent']}% Complete</div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{info['current_iteration']}/{info['total_iterations']}</div>
                    <div class="metric-label">Iteration</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{info['configurations_evaluated']}/{info['total_configurations']}</div>
                    <div class="metric-label">Configurations</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{info['current_resource_level']:.2f}</div>
                    <div class="metric-label">Resource Level</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(info['configurations'])}</div>
                    <div class="metric-label">Models Evaluated</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Configuration Evaluation Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Config #</th>
                        <th>Sharpness</th>
                        <th>Perturbation Robustness</th>
                        <th>Performance</th>
                    </tr>
                </thead>
                <tbody>
                    {config_rows}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>About Meta-Model Training</h2>
            <p>The meta-model is a surrogate model that predicts the performance of neural network configurations based on their loss landscape characteristics. This approach allows for efficient hyperparameter optimization without training each configuration to completion.</p>
            <p>The training process involves:</p>
            <ol>
                <li>Evaluating multiple configurations at low resource levels</li>
                <li>Measuring loss landscape sharpness and perturbation robustness</li>
                <li>Training a meta-model to predict full-resource performance</li>
                <li>Progressively eliminating poor-performing configurations</li>
            </ol>
        </div>
        
        <div class="footer">
            <p>Generated by LossLandscapeProbe Framework v1.0.0</p>
            <p><a href="https://loss.computer-wizard.com.au" style="color: #4cf;">https://loss.computer-wizard.com.au</a></p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write the HTML content to the output file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated meta-model report at {output_path}")
    return True

if __name__ == "__main__":
    # Define the log file path and output path
    log_file = "/home/peter/Projects/LossLandscapeProbe/reports/cifa10/cifar10_meta_model_10pct.log"
    output_path = "/home/peter/Projects/LossLandscapeProbe/reports/cifa10/meta_model_report.html"
    
    # Create the log file with the provided content for testing
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Use the log content provided by the user
    log_content = """2025-06-19 02:01:15,646 - llp.meta_probing - INFO - Running meta-model-guided hyperparameter optimization...
2025-06-19 02:01:15,646 - llp.meta_probing - INFO - Starting with 10 configurations
2025-06-19 02:01:15,646 - llp.meta_probing - INFO - Resource range: 0.1 to 1.0 of data
2025-06-19 02:01:15,646 - llp.meta_probing - INFO - Reduction factor: 2
2025-06-19 02:01:15,646 - llp.meta_probing - INFO - Meta-optimization iteration 1/3
2025-06-19 02:01:15,647 - llp.meta_probing - INFO - Evaluating configuration 1/6 at resource level 0.10
2025-06-19 02:09:56,277 - llp.meta_probing - INFO - Measuring loss landscape sharpness (noise_std=0.01, samples=5)...
2025-06-19 02:10:15,397 - llp.meta_probing - INFO - Sharpness: 0.0001, Perturbation Robustness: 11275.7654
2025-06-19 02:10:15,397 - llp.meta_probing - INFO - Sharpness measurement took 19.12s
2025-06-19 02:10:15,398 - llp.meta_model - INFO - Extracting dataset features...
2025-06-19 02:10:18,840 - llp.meta_model - INFO - Extracted 11 dataset features
2025-06-19 02:10:18,845 - llp.meta_model - INFO - Added training example with performance 0.1000
2025-06-19 02:10:18,850 - llp.meta_probing - INFO - Evaluating configuration 2/6 at resource level 0.10
2025-06-19 02:18:59,890 - llp.meta_probing - INFO - Measuring loss landscape sharpness (noise_std=0.01, samples=5)...
2025-06-19 02:19:19,531 - llp.meta_probing - INFO - Sharpness: 0.0002, Perturbation Robustness: 5283.3888
2025-06-19 02:19:19,532 - llp.meta_probing - INFO - Sharpness measurement took 19.64s
2025-06-19 02:19:19,532 - llp.meta_model - INFO - Extracting dataset features...
2025-06-19 02:19:23,061 - llp.meta_model - INFO - Extracted 11 dataset features
2025-06-19 02:19:23,072 - llp.meta_model - INFO - Added training example with performance 0.1000
2025-06-19 02:19:23,077 - llp.meta_probing - INFO - Evaluating configuration 3/6 at resource level 0.10
2025-06-19 02:39:00,975 - llp.meta_probing - INFO - Measuring loss landscape sharpness (noise_std=0.01, samples=5)...
2025-06-19 02:39:39,177 - llp.meta_probing - INFO - Sharpness: 0.2965, Perturbation Robustness: 3.3726
2025-06-19 02:39:39,177 - llp.meta_probing - INFO - Sharpness measurement took 38.20s
2025-06-19 02:39:39,177 - llp.meta_model - INFO - Extracting dataset features...
2025-06-19 02:39:42,740 - llp.meta_model - INFO - Extracted 11 dataset features
2025-06-19 02:39:42,752 - llp.meta_model - INFO - Added training example with performance 0.5277
2025-06-19 02:39:42,760 - llp.meta_probing - INFO - Evaluating configuration 4/6 at resource level 0.10"""
    
    with open(log_file, 'w') as f:
        f.write(log_content)
    
    # Generate the report
    generate_meta_model_report(log_file, output_path)
