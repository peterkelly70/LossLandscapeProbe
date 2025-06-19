#!/usr/bin/env python3
"""
Setup script to copy necessary files to the web server directory for the LossLandscapeProbe visualization website.
"""

import os
import sys
import json
import shutil
import re
import subprocess
from pathlib import Path
from setup_model_directories import setup_model_directories
import argparse

def copy_file(src, dest, create_dest_dir=True):
    """Copy a file and create destination directory if needed."""
    if create_dest_dir:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    try:
        # Read the source file
        with open(src, 'rb') as f_src:
            content = f_src.read()
        
        # Write to the destination file
        with open(dest, 'wb') as f_dest:
            f_dest.write(content)
        
        print(f"Copied: {src} → {dest}")
        return True
    except Exception as e:
        print(f"Error copying {src}: {e}")
        return False

def create_list_reports_php(web_dir):
    """Create the PHP script to list report files."""
    php_content = """<?php
header('Content-Type: application/json');

// Directory containing report files
$dir = __DIR__;

// Get all HTML files except index.html
$files = glob($dir . '/*.html');
$reports = [];

foreach ($files as $file) {
    $filename = basename($file);
    if ($filename !== 'index.html') {
        $reports[] = $filename;
    }
}

// Sort by filename (which should put newest first if using timestamp naming)
rsort($reports);

echo json_encode($reports);
?>
"""
    
    php_path = os.path.join(web_dir, 'list_reports.php')
    try:
        with open(php_path, 'w') as f:
            f.write(php_content)
        print(f"Created: {php_path}")
        return True
    except Exception as e:
        print(f"Error creating PHP script: {e}")
        return False

def create_list_training_reports_php(web_dir):
    """Create the PHP script to list training report files."""
    php_content = """<?php
header('Content-Type: application/json');

// Directory containing training report files
$dir = __DIR__ . '/training_reports';

// Check if directory exists
if (!is_dir($dir)) {
    echo json_encode([]);
    exit;
}

// Get all HTML files
$files = glob($dir . '/*.html');
$reports = [];

foreach ($files as $file) {
    $reports[] = basename($file);
}

// Sort by filename (which should put newest first if using timestamp naming)
rsort($reports);

echo json_encode($reports);
?>
"""
    
    php_path = os.path.join(web_dir, 'list_training_reports.php')
    try:
        with open(php_path, 'w') as f:
            f.write(php_content)
        print(f"Created: {php_path}")
        return True
    except Exception as e:
        print(f"Error creating PHP script: {e}")
        return False

def create_list_model_reports_php(web_dir):
    """Create the PHP script to list files in model report directories."""
    php_content = """<?php
// This script lists all files in a specified model directory
header('Content-Type: application/json');

// Sanitize the directory parameter to prevent directory traversal attacks
$dir = isset($_GET['dir']) ? preg_replace('/[^a-zA-Z0-9_]/', '', $_GET['dir']) : '';

if (empty($dir)) {
    echo json_encode([]);
    exit;
}

// Path to the reports directory
$reportsDir = __DIR__ . '/reports/' . $dir;

// Check if the directory exists
if (!is_dir($reportsDir)) {
    echo json_encode([]);
    exit;
}

// Get all files in the directory
$files = scandir($reportsDir);
$reportFiles = [];

foreach ($files as $file) {
    // Skip . and .. directories and hidden files
    if ($file[0] === '.') {
        continue;
    }
    
    // Add the file to the list
    $reportFiles[] = $file;
}

// Return the list of files as JSON
echo json_encode($reportFiles);
?>
"""
    
    php_path = os.path.join(web_dir, 'list_model_reports.php')
    try:
        with open(php_path, 'w') as f:
            f.write(php_content)
        print(f"Created: {php_path}")
        return True
    except Exception as e:
        print(f"Error creating PHP script: {e}")
        return False

def copy_markdown_files(project_dir, web_dir):
    """Copy markdown documentation files to the web directory."""
    # Check for doc directory first, fall back to project root
    docs_dir = os.path.join(project_dir, 'doc')
    if not os.path.exists(docs_dir):
        print("Documentation directory not found at {}, falling back to project root".format(docs_dir))
        docs_dir = project_dir
    
    # Copy README.md
    readme_path = os.path.join(docs_dir, 'README.md')
    readme_target_path = os.path.join(web_dir, 'README.md')
    
    # Copy LossLandscapeProbe_WhitePaper.md (renamed from whitepaper.md)
    whitepaper_path = os.path.join(docs_dir, 'LossLandscapeProbe_WhitePaper.md')
    whitepaper_target_path = os.path.join(web_dir, 'whitepaper.md')  # Keep the web filename as whitepaper.md for compatibility
    
    # Copy LICENSE
    license_path = os.path.join(project_dir, 'LICENSE')
    license_target_path = os.path.join(web_dir, 'LICENSE')
    
    try:
        # Check if docs directory exists
        if not os.path.exists(docs_dir):
            print(f"Documentation directory not found at {docs_dir}, falling back to project root")
            # Fall back to project root for backward compatibility
            readme_path = os.path.join(project_dir, 'README.md')
            whitepaper_path = os.path.join(project_dir, 'whitepaper.md')
        
        # Read and write README.md
        if os.path.exists(readme_path):
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            
            with open(readme_target_path, 'w') as f:
                f.write(readme_content)
            
            print(f"Copied README.md to {readme_target_path}")
        else:
            print(f"README.md not found at {readme_path}, skipping")
        
        # Read and write whitepaper
        if os.path.exists(whitepaper_path):
            with open(whitepaper_path, 'r') as f:
                whitepaper_content = f.read()
            
            with open(whitepaper_target_path, 'w') as f:
                f.write(whitepaper_content)
            
            print(f"Copied LossLandscapeProbe_WhitePaper.md to {whitepaper_target_path}")
        else:
            print(f"Whitepaper not found at {whitepaper_path}, skipping")
        
        # Read and write LICENSE
        if os.path.exists(license_path):
            with open(license_path, 'r') as f:
                license_content = f.read()
            
            with open(license_target_path, 'w') as f:
                f.write(license_content)
            
            print(f"Copied LICENSE to {license_target_path}")
        else:
            print("LICENSE not found, skipping")
    except Exception as e:
        print(f"Error copying markdown files: {e}")
        return False
    
    return True

def copy_log_files(project_dir, web_dir):
    """Copy log files from report directories to the web directory."""
    # Get all model-specific directories in the project reports directory
    project_reports_dir = os.path.join(project_dir, 'reports')
    web_reports_dir = os.path.join(web_dir, 'reports')
    
    if not os.path.exists(project_reports_dir):
        print("Project reports directory not found, skipping log file copying")
        return False
    
    # Create the web reports directory if it doesn't exist
    os.makedirs(web_reports_dir, exist_ok=True)
    
    # Get all model-specific directories
    model_dirs = [d for d in os.listdir(project_reports_dir) if os.path.isdir(os.path.join(project_reports_dir, d))]
    
    log_files_copied = 0
    
    # For each model directory, copy all log files to the corresponding web directory
    for model_dir in model_dirs:
        project_model_dir = os.path.join(project_reports_dir, model_dir)
        web_model_dir = os.path.join(web_reports_dir, model_dir)
        
        # Create the web model directory if it doesn't exist
        os.makedirs(web_model_dir, exist_ok=True)
        
        # Find all log files in the project model directory
        log_files = [f for f in os.listdir(project_model_dir) if f.endswith('.log')]
        
        # Copy each log file to the web model directory
        for log_file in log_files:
            src_path = os.path.join(project_model_dir, log_file)
            dest_path = os.path.join(web_model_dir, log_file)
            
            try:
                copy_file(src_path, dest_path)
                log_files_copied += 1
            except Exception as e:
                print(f"Error copying log file {log_file}: {e}")
    
    if log_files_copied > 0:
        print(f"Copied {log_files_copied} log files to web directory")
    else:
        print("No log files found to copy")
    
    return True

def copy_test_reports(project_dir, web_dir):
    """Copy test reports from the project directory to the web directory."""
    # Get all model-specific directories in the project reports directory
    project_reports_dir = os.path.join(project_dir, 'reports')
    web_reports_dir = os.path.join(web_dir, 'reports')
    
    if not os.path.exists(project_reports_dir):
        print("Project reports directory not found, skipping test report copying")
        return False
    
    # Create the web reports directory if it doesn't exist
    os.makedirs(web_reports_dir, exist_ok=True)
    
    # Get all model-specific directories
    model_dirs = [d for d in os.listdir(project_reports_dir) if os.path.isdir(os.path.join(project_reports_dir, d))]
    
    test_reports_copied = 0
    
    # For each model directory, copy all test report files to the corresponding web directory
    for model_dir in model_dirs:
        project_model_dir = os.path.join(project_reports_dir, model_dir)
        web_model_dir = os.path.join(web_reports_dir, model_dir)
        
        # Create the web model directory if it doesn't exist
        os.makedirs(web_model_dir, exist_ok=True)
        
        # Find all test report files in the project model directory
        test_report_files = [f for f in os.listdir(project_model_dir) 
                           if f.endswith('_test_report.html') or f == 'latest_test_report.html']
        
        # Copy each test report file to the web model directory
        for report_file in test_report_files:
            src_path = os.path.join(project_model_dir, report_file)
            dest_path = os.path.join(web_model_dir, report_file)
            
            try:
                copy_file(src_path, dest_path)
                test_reports_copied += 1
            except Exception as e:
                print(f"Error copying test report file {report_file}: {e}")
    
    if test_reports_copied > 0:
        print(f"Copied {test_reports_copied} test reports to web directory")
    else:
        print("No test reports found to copy")
    
    return True

def cleanup_old_reports(web_dir):
    """Clean up old timestamped reports and keep only the latest ones in model-specific directories."""
    try:
        # Find all timestamped test report files in the web directory
        timestamped_reports = [f for f in os.listdir(web_dir) 
                              if re.match(r'cifar\d+_test_report_\d+\.html', f)]
        
        if timestamped_reports:
            print(f"Found {len(timestamped_reports)} old timestamped reports to clean up")
            
            # Create model directories if they don't exist
            model_dirs = ['cifar10', 'cifar10_10', 'cifar10_20', 'cifar10_30', 'cifar10_40',
                         'cifar100', 'cifar100_10', 'cifar100_20', 'cifar100_30', 'cifar100_40',
                         'cifar100_transfer']
            
            for model_dir in model_dirs:
                model_report_dir = os.path.join(web_dir, 'reports', model_dir)
                os.makedirs(model_report_dir, exist_ok=True)
            
            # Delete old timestamped reports after ensuring we have the model-specific reports
            for report in timestamped_reports:
                try:
                    os.remove(os.path.join(web_dir, report))
                    print(f"Removed old report: {report}")
                except Exception as e:
                    print(f"Warning: Could not remove {report}: {e}")
        else:
            print("No old timestamped reports found")
    except Exception as e:
        print(f"Warning: Error during cleanup of old reports: {e}")

def update_test_report_headings(web_dir):
    """Update the headings in all test report HTML files to show sample size."""
    try:
        # Find all model directories - use correct naming (cifar not cifa)
        model_dirs = ['cifar10', 'cifar10_10', 'cifar10_20', 'cifar10_30', 'cifar10_40',
                     'cifar100', 'cifar100_10', 'cifar100_20', 'cifar100_30', 'cifar100_40',
                     'cifar100_transfer']
        
        updated_count = 0
        for model_dir in model_dirs:
            model_report_dir = os.path.join(web_dir, 'reports', model_dir)
            if not os.path.exists(model_report_dir):
                continue
                
            # Look for latest_test_report.html in each model directory
            report_path = os.path.join(model_report_dir, 'latest_test_report.html')
            if not os.path.exists(report_path):
                continue
                
            try:
                # Read the file content
                with open(report_path, 'r') as f:
                    content = f.read()
                
                # Check if it needs updating (to avoid unnecessary writes)
                if '<h2>Test Images and Predictions</h2>' in content:
                    # Replace the heading
                    updated_content = content.replace(
                        '<h2>Test Images and Predictions</h2>',
                        '<h2>Test Images and Predictions (200/10000)</h2>'
                    )
                    
                    # Write the updated content back to the file
                    with open(report_path, 'w') as f:
                        f.write(updated_content)
                    
                    updated_count += 1
                    print(f"Updated heading in {model_dir}/latest_test_report.html")
            except Exception as e:
                print(f"Warning: Could not update {model_dir}/latest_test_report.html: {e}")
        
        # Remove any old timestamped test report files in the web directory
        old_reports = [f for f in os.listdir(web_dir) 
                      if re.match(r'cifar\d+_test_report_\d+\.html', f)]
        
        for report_file in old_reports:
            report_path = os.path.join(web_dir, report_file)
            try:
                # Remove the old timestamped file
                os.remove(report_path)
                print(f"Removed old timestamped report: {report_file}")
            except Exception as e:
                print(f"Warning: Could not remove old report {report_file}: {e}")
        
        if updated_count > 0:
            print(f"Updated {updated_count} test report files.")
        else:
            print("No test report files were updated.")
    except Exception as e:
        print(f"Warning: Could not update test report headings: {e}")


def generate_reports(project_dir):
    """Generate all reports before deployment."""
    print("\n=== Generating Reports ===")
    
    # Generate CIFAR-10 test report
    try:
        print("\nGenerating CIFAR-10 test report...")
        test_report_script = os.path.join(project_dir, 'src', 'visualization', 'generate_test_report.py')
        result = subprocess.run([sys.executable, test_report_script], cwd=project_dir, check=False)
        # The script now handles its own success/failure messages
        # No need to print additional success message here
    except Exception as e:
        print(f"Error: Could not run CIFAR-10 test report script: {e}")
    
    # Generate training plots
    try:
        print("\nGenerating training plots...")
        training_plots_script = os.path.join(project_dir, 'src', 'visualization', 'generate_training_plots.py')
        result = subprocess.run([sys.executable, training_plots_script], cwd=project_dir, check=False)
        if result.returncode == 0:
            print("Training plots generated successfully.")
        else:
            print("Warning: Training plots generation may have encountered issues.")
    except Exception as e:
        print(f"Error: Could not run training plots script: {e}")
    
    # Generate CIFAR-100 transfer report if the experiment has completed
    try:
        print("\nChecking for CIFAR-100 results and generating report if available...")
        cifar100_report_script = os.path.join(project_dir, 'examples', 'generate_cifar100_report.py')
        result = subprocess.run([sys.executable, cifar100_report_script], cwd=project_dir, check=False)
        if result.returncode == 0:
            print("CIFAR-100 transfer report generated successfully.")
        else:
            print("Note: CIFAR-100 transfer report generation did not complete successfully.")
            print("This is expected if the CIFAR-100 transfer experiment is still running.")
    except Exception as e:
        print(f"Note: Could not run CIFAR-100 transfer report script: {e}")
        print("This is expected if the CIFAR-100 transfer experiment is still running.")
    
    # Generate CIFAR-10 progress report if log exists
    cifar10_log_path = os.path.join(project_dir, 'cifar10_training.log')
    if os.path.exists(cifar10_log_path):
        try:
            print("\nGenerating CIFAR-10 training progress report...")
            cifar10_progress_script = os.path.join(project_dir, 'src', 'visualization', 'visualize_cifar10_progress.py')
            result = subprocess.run([sys.executable, cifar10_progress_script, '--log', cifar10_log_path], cwd=project_dir, check=False)
            if result.returncode == 0:
                print("CIFAR-10 training progress report generated successfully.")
            else:
                print("Warning: CIFAR-10 training progress report generation may have encountered issues.")
        except Exception as e:
            print(f"Error: Could not run CIFAR-10 training progress script: {e}")
    else:
        print("\nNote: CIFAR-10 training log not found. Skipping progress report generation.")
        
    # Generate CIFAR-100 progress report if log exists
    cifar100_log_path = os.path.join(project_dir, 'cifar100_transfer.log')
    if os.path.exists(cifar100_log_path):
        try:
            print("\nGenerating CIFAR-100 training progress report...")
            cifar100_progress_script = os.path.join(project_dir, 'src', 'visualization', 'visualize_cifar100_progress.py')
            result = subprocess.run([sys.executable, cifar100_progress_script, '--log', cifar100_log_path], cwd=project_dir, check=False)
            if result.returncode == 0:
                print("CIFAR-100 training progress report generated successfully.")
            else:
                print("Warning: CIFAR-100 training progress report generation may have encountered issues.")
        except Exception as e:
            print(f"Error: Could not run CIFAR-100 training progress script: {e}")
    else:
        print("\nNote: CIFAR-100 training log not found. Skipping progress report generation.")
    
    # Check for meta-model logs and generate reports
    meta_model_logs = []
    logs_dir = os.path.join(project_dir, 'logs')
    if os.path.exists(logs_dir):
        for file in os.listdir(logs_dir):
            if re.match(r'.+_meta_model_\d+pct\.log$', file):
                meta_model_logs.append(os.path.join(logs_dir, file))
    
    if meta_model_logs:
        try:
            print("\nGenerating Meta-Model training progress reports...")
            for log_path in meta_model_logs:
                # Extract dataset and sample size from filename
                match = re.match(r'(.+)_meta_model_(\d+)pct\.log$', os.path.basename(log_path))
                if match:
                    dataset = match.group(1)
                    sample_size = int(match.group(2))
                    
                    print(f"Processing meta-model log for {dataset} at {sample_size}% resource level...")
                    meta_model_script = os.path.join(project_dir, 'src', 'visualization', 'generate_meta_model_report.py')
                    # Construct paths for log file and output files
                    log_file = os.path.join(web_dir, 'reports', f'{dataset}_{sample_size}' if sample_size < 100 else dataset, f'{dataset}_meta_model_{sample_size}pct.log')
                    output_path = os.path.join(project_dir, 'reports', f'{dataset}_{sample_size}' if sample_size < 100 else dataset, 'meta_model_report.html')
                    web_output_path = os.path.join(web_dir, 'reports', f'{dataset}_{sample_size}' if sample_size < 100 else dataset, 'meta_model_report.html')
                    
                    # Ensure directories exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    os.makedirs(os.path.dirname(web_output_path), exist_ok=True)
                    
                    result = subprocess.run(
                        [sys.executable, meta_model_script, 
                         '--log-file', log_file,
                         '--output', output_path,
                         '--web-output', web_output_path], 
                        cwd=project_dir, check=False
                    )
                    
                    if result.returncode == 0:
                        print(f"Meta-model report for {dataset} {sample_size}% generated successfully.")
                    else:
                        print(f"Warning: Meta-model report generation for {dataset} {sample_size}% may have encountered issues.")
        except Exception as e:
            print(f"Error: Could not run meta-model report generation: {e}")
    else:
        print("\nNote: No meta-model logs found. Skipping meta-model report generation.")
    
    print("\nReport generation complete.")

def main():
    # Get project directory
    project_dir = Path(__file__).parent.parent.parent.absolute()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Setup the LossLandscapeProbe website.')
    parser.add_argument('--web-dir', help='Web server directory (overrides config file)')
    args = parser.parse_args()
    
    # Load configuration from file
    config_path = os.path.join(project_dir, 'config', 'website_config.json')
    web_dir = "/var/www/html/loss.computer-wizard.com.au"  # Default fallback
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                web_dir = config.get('web_dir', web_dir)
                print(f"Loaded web directory from config file: {web_dir}")
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    else:
        print(f"Warning: Config file not found at {config_path}. Using default web directory.")
    
    # Command-line argument overrides config file
    if args.web_dir:
        web_dir = args.web_dir
        print(f"Using web directory from command line: {web_dir}")
        
        # Save to config for future use
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        try:
            with open(config_path, 'w') as config_file:
                json.dump({"web_dir": web_dir, "comment": "This configuration file stores the web server directory for the LossLandscapeProbe website."}, config_file, indent=4)
                print(f"Updated config file with new web directory.")
        except Exception as e:
            print(f"Warning: Could not update config file: {e}")
    
    
    print(f"Project directory: {project_dir}")
    print(f"Web directory: {web_dir}")
    
    # Generate all reports first
    generate_reports(project_dir)
    
    # Ensure web directory exists
    os.makedirs(web_dir, exist_ok=True)
    
    # Copy the improved website template as index.html
    template_path = os.path.join(project_dir, 'src', 'deployment', 'assets', 'improved_website_template.html')
    index_path = os.path.join(web_dir, 'index.html')
    copy_file(template_path, index_path)
    
    # Copy training history plot if it exists
    training_plot_path = os.path.join(project_dir, 'training_history.png')
    web_plot_path = os.path.join(web_dir, 'training_history.png')
    
    if os.path.exists(training_plot_path):
        copy_file(training_plot_path, web_plot_path)
        print(f"Copied training plot to {web_plot_path}")
    else:
        print(f"Note: Training plot not found at {training_plot_path}. The website will show a placeholder message.")
        
    # Copy PHP scripts for progress visualization
    php_scripts = [
        'generate_cifar10_progress_php.php',
        'generate_cifar100_progress_php.php',
        'generate_meta_model_progress_php.php'
    ]
    
    for script in php_scripts:
        php_script_path = os.path.join(project_dir, 'src', 'deployment', script)
        web_php_script_path = os.path.join(web_dir, script)
        
        if os.path.exists(php_script_path):
            copy_file(php_script_path, web_php_script_path)
            print(f"Copied {script} to {web_php_script_path}")
        else:
            print(f"Note: {script} not found at {php_script_path}.")
    
    # Copy CIFAR-100 transfer report if it exists
    cifar100_report_dir = os.path.join(project_dir, 'reports', 'cifar100_transfer')
    cifar100_report_path = os.path.join(cifar100_report_dir, 'latest_test_report.html')
    web_cifar100_report_dir = os.path.join(web_dir, 'reports', 'cifar100_transfer')
    os.makedirs(web_cifar100_report_dir, exist_ok=True)
    web_cifar100_report_path = os.path.join(web_cifar100_report_dir, 'latest_test_report.html')
    
    if os.path.exists(cifar100_report_path):
        copy_file(cifar100_report_path, web_cifar100_report_path)
        print(f"Copied CIFAR-100 transfer report to {web_cifar100_report_path}")
    else:
        print(f"Note: CIFAR-100 transfer report not found at {cifar100_report_path}. The website will show a placeholder message.")
        print(f"This is expected if the CIFAR-100 transfer experiment is still running.")
        
    # Copy specific progress reports to web directory for direct access
    # Check multiple possible locations for CIFAR-10 progress report
    cifar10_progress_report_paths = [
        os.path.join(project_dir, 'reports', 'cifar10', 'cifar10_progress_report.html'),
        os.path.join(project_dir, 'cifar10_progress_report.html'),
        os.path.join(project_dir, 'reports', 'cifar10_progress_report.html')
    ]
    
    cifar10_report_found = False
    for path in cifar10_progress_report_paths:
        if os.path.exists(path):
            web_progress_path = os.path.join(web_dir, 'cifar10_progress_report.html')
            copy_file(path, web_progress_path)
            print(f"Copied CIFAR-10 progress report from {path} to web directory")
            cifar10_report_found = True
            break
    
    if not cifar10_report_found:
        # Generate a placeholder progress report for CIFAR-10
        placeholder_path = os.path.join(web_dir, 'cifar10_progress_report.html')
        with open(placeholder_path, 'w') as f:
            cifar10_progress_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>CIFAR-10 Training Progress</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="navigation">
        <ul>
            <li><a href="index.html">Home</a></li>
            <li><a href="cifar10_progress.html" class="active">CIFAR-10 Progress</a></li>
            <li><a href="cifar100_progress.html">CIFAR-100 Progress</a></li>
            <li><a href="generate_meta_model_progress_php.php">Meta-Model Progress</a></li>
            <li><a href="about.html">About</a></li>
        </ul>
    </div>
    <div class="message">
        <p>Training is currently in progress. This report will be updated when training data becomes available.</p>
        <p>Check back later for visualizations of the training progress.</p>
    </div>
</body>
</html>'''
            f.write(cifar10_progress_html)
        print("Created placeholder CIFAR-10 progress report")
    
    # Check multiple possible locations for CIFAR-100 progress report
    cifar100_progress_report_paths = [
        os.path.join(project_dir, 'reports', 'cifar100', 'cifar100_progress_report.html'),
        os.path.join(project_dir, 'cifar100_progress_report.html'),
        os.path.join(project_dir, 'reports', 'cifar100_progress_report.html')
    ]
    
    cifar100_report_found = False
    for path in cifar100_progress_report_paths:
        if os.path.exists(path):
            web_progress_path = os.path.join(web_dir, 'cifar100_progress_report.html')
            copy_file(path, web_progress_path)
            print(f"Copied CIFAR-100 progress report from {path} to web directory")
            cifar100_report_found = True
            break
    
    if not cifar100_report_found:
        # Generate a placeholder progress report for CIFAR-100
        placeholder_path = os.path.join(web_dir, 'cifar100_progress_report.html')
        with open(placeholder_path, 'w') as f:
            cifar100_progress_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>CIFAR-100 Training Progress</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="navigation">
        <ul>
            <li><a href="index.html">Home</a></li>
            <li><a href="cifar10_progress.html">CIFAR-10 Progress</a></li>
            <li><a href="cifar100_progress.html" class="active">CIFAR-100 Progress</a></li>
            <li><a href="generate_meta_model_progress_php.php">Meta-Model Progress</a></li>
            <li><a href="about.html">About</a></li>
        </ul>
    </div>
    <div class="message">
        <p>Training is currently in progress. This report will be updated when training data becomes available.</p>
        <p>Check back later for visualizations of the training progress.</p>
    </div>
</body>
</html>'''
            f.write(cifar100_progress_html)
        print("Created placeholder CIFAR-100 progress report")
    else:
        print(f"Note: No reports found in {project_reports_dir}. The website will show placeholder messages.")
        
    # Also copy the list_training_reports.php script
    training_reports_php_src = os.path.join(project_dir, 'src', 'deployment', 'list_training_reports.php')
    training_reports_php_dest = os.path.join(web_dir, 'list_training_reports.php')
    if os.path.exists(training_reports_php_src):
        copy_file(training_reports_php_src, training_reports_php_dest)
        print(f"Copied list_training_reports.php to {web_dir}")
    else:
        print(f"Warning: list_training_reports.php not found at {training_reports_php_src}")


    
    # Create PHP scripts for listing reports
    create_list_reports_php(web_dir)
    create_list_training_reports_php(web_dir)
    create_list_model_reports_php(web_dir)
    
    # Copy the read_training_log.php file
    read_log_php_src = os.path.join(project_dir, 'src', 'deployment', 'assets', 'read_training_log.php')
    read_log_php_dst = os.path.join(web_dir, 'read_training_log.php')
    if os.path.exists(read_log_php_src):
        copy_file(read_log_php_src, read_log_php_dst)
        print(f"Copied read_training_log.php to {web_dir}")
    else:
        print(f"Warning: read_training_log.php not found at {read_log_php_src}")
    
    # Clean up old timestamped reports
    print("\nCleaning up old timestamped reports...")
    cleanup_old_reports(web_dir)
    
    # Update test report headings to show sample size
    print("\nUpdating test report headings...")
    update_test_report_headings(web_dir)
    
    # Copy markdown files (README, whitepaper, and LICENSE)
    copy_markdown_files(project_dir, web_dir)
    
    # Copy log files from report directories to web directory
    copy_log_files(project_dir, web_dir)
    
    # Copy test reports from project directory to web directory
    copy_test_reports(project_dir, web_dir)
    
    # Set permissions
    try:
        subprocess.run(['sudo', 'chmod', '-R', '777', web_dir])
        subprocess.run(['sudo', 'chown', '-R', 'www-data:www-data', web_dir])
        print(f"Set permissions on {web_dir}")
    except Exception as e:
        print(f"Warning: Could not set permissions: {e}")
    
    print("\nSetup complete! Visit your website to see the results.")

if __name__ == "__main__":
    main()
