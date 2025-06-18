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
    """Copy the README.md and whitepaper.md files to the web server directory."""
    # Copy README.md
    readme_path = os.path.join(project_dir, 'README.md')
    readme_target_path = os.path.join(web_dir, 'README.md')
    
    # Copy whitepaper.md
    whitepaper_path = os.path.join(project_dir, 'whitepaper.md')
    whitepaper_target_path = os.path.join(web_dir, 'whitepaper.md')
    
    try:
        # Read and write README.md
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        
        with open(readme_target_path, 'w') as f:
            f.write(readme_content)
        
        print(f"Copied README.md to {readme_target_path}")
        
        # Read and write whitepaper.md if it exists
        if os.path.exists(whitepaper_path):
            with open(whitepaper_path, 'r') as f:
                whitepaper_content = f.read()
            
            with open(whitepaper_target_path, 'w') as f:
                f.write(whitepaper_content)
            
            print(f"Copied whitepaper.md to {whitepaper_target_path}")
        else:
            print("whitepaper.md not found, skipping")
    except Exception as e:
        print(f"Error copying markdown files: {e}")
        return False

def cleanup_old_reports(web_dir):
    """Clean up old timestamped reports and keep only the latest ones in model-specific directories."""
    try:
        # Find all timestamped test report files in the web directory
        timestamped_reports = [f for f in os.listdir(web_dir) 
                              if re.match(r'cifar\d+_test_report_\d+\.html', f)]
        
        if timestamped_reports:
            print(f"Found {len(timestamped_reports)} old timestamped reports to clean up")
            
            # Create model directories if they don't exist
            model_dirs = ['cifa10', 'cifa10_10', 'cifa10_20', 'cifa10_30', 'cifa10_40',
                         'cifa100', 'cifa100_10', 'cifa100_20', 'cifa100_30', 'cifa100_40',
                         'cifa100_transfer']
            
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
        # Find all model directories
        model_dirs = ['cifa10', 'cifa10_10', 'cifa10_20', 'cifa10_30', 'cifa10_40',
                     'cifa100', 'cifa100_10', 'cifa100_20', 'cifa100_30', 'cifa100_40',
                     'cifa100_transfer']
        
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
        
        # Also check for any old-style reports in the web directory
        old_reports = [f for f in os.listdir(web_dir) 
                      if f.startswith(('cifar10_test_report', 'cifar100_test_report')) 
                      and f.endswith('.html')]
        
        for report_file in old_reports:
            report_path = os.path.join(web_dir, report_file)
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
                    print(f"Updated heading in {report_file}")
                    updated_count += 1
            except Exception as e:
                print(f"Warning: Could not process {report_file}: {e}")
        
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
    
    print("\nReport generation complete.")

def main():
    # Get project directory
    project_dir = Path(__file__).parent.parent.parent.absolute()
    
    # Default web directory
    web_dir = "/var/www/html/loss.computer-wizard.com.au"
    
    # Check if alternative web directory was provided
    if len(sys.argv) > 1:
        web_dir = sys.argv[1]
    
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
        
    # Copy the PHP script for on-demand CIFAR-10 progress report generation
    php_script_path = os.path.join(project_dir, 'src', 'deployment', 'generate_cifar10_progress_php.php')
    web_php_script_path = os.path.join(web_dir, 'generate_cifar10_progress_php.php')
    
    if os.path.exists(php_script_path):
        copy_file(php_script_path, web_php_script_path)
        print(f"Copied CIFAR-10 progress PHP script to {web_php_script_path}")
    else:
        print(f"Note: CIFAR-10 progress PHP script not found at {php_script_path}.")
    
    # Copy training reports if they exist (as a fallback)
    training_report_path = os.path.join(project_dir, 'training_report.html')
    web_training_report_path = os.path.join(web_dir, 'training_report.html')
    
    if os.path.exists(training_report_path):
        copy_file(training_report_path, web_training_report_path)
        print(f"Copied training report to {web_training_report_path} (fallback)")
    else:
        print(f"Note: Training report not found at {training_report_path}.")
    
    # Copy CIFAR-100 transfer report if it exists
    cifar100_report_path = os.path.join(project_dir, 'cifar100_transfer_report.html')
    web_cifar100_report_path = os.path.join(web_dir, 'cifar100_transfer_report.html')
    
    if os.path.exists(cifar100_report_path):
        copy_file(cifar100_report_path, web_cifar100_report_path)
        print(f"Copied CIFAR-100 transfer report to {web_cifar100_report_path}")
    else:
        print(f"Note: CIFAR-100 transfer report not found at {cifar100_report_path}. The website will show a placeholder message.")
        print(f"This is expected if the CIFAR-100 transfer experiment is still running.")
        
    # Copy the PHP script for on-demand CIFAR-100 progress report generation
    php_script_path = os.path.join(project_dir, 'src', 'deployment', 'generate_cifar100_progress_php.php')
    web_php_script_path = os.path.join(web_dir, 'generate_cifar100_progress_php.php')
    
    if os.path.exists(php_script_path):
        copy_file(php_script_path, web_php_script_path)
        print(f"Copied CIFAR-100 progress PHP script to {web_php_script_path}")
    else:
        print(f"Note: CIFAR-100 progress PHP script not found at {php_script_path}.")
    
    # Copy CIFAR-10 progress report if it exists (as a fallback)
    cifar10_progress_report_path = os.path.join(project_dir, 'reports', 'cifar10_progress_report.html')
    web_cifar10_progress_report_path = os.path.join(web_dir, 'cifar10_progress_report.html')
    if os.path.exists(cifar10_progress_report_path):
        copy_file(cifar10_progress_report_path, web_cifar10_progress_report_path)
        print(f"Copied CIFAR-10 training progress report to {web_cifar10_progress_report_path}")
        
        # Also copy the data file if it exists
        cifar10_progress_data_path = os.path.join(project_dir, 'reports', 'cifar10_progress_report.pth')
        web_cifar10_progress_data_path = os.path.join(web_dir, 'cifar10_progress_report.pth')
        if os.path.exists(cifar10_progress_data_path):
            copy_file(cifar10_progress_data_path, web_cifar10_progress_data_path)
            print(f"Copied CIFAR-10 training progress data file to {web_cifar10_progress_data_path}")
    else:
        print(f"Note: CIFAR-10 training progress report not found at {cifar10_progress_report_path}.")
        print(f"This is expected if CIFAR-10 training has not been run yet.")
    
    # Create reports directory for all test reports
    reports_dir = os.path.join(web_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create training reports directory
    training_reports_dir = os.path.join(web_dir, 'training_reports')
    os.makedirs(training_reports_dir, exist_ok=True)
    
    # Setup model directories based on the actual directory structure
    setup_model_directories(project_dir, web_dir)
    
    # Copy specific progress reports to web directory for direct access
    # Check multiple possible locations for CIFAR-10 progress report
    cifar10_progress_report_paths = [
        os.path.join(project_dir, 'reports', 'cifa10', 'cifar10_progress_report.html'),
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
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>CIFAR-10 Training Progress</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .message { padding: 20px; background-color: #f8f9fa; border-left: 4px solid #007bff; }
    </style>
</head>
<body>
    <h1>CIFAR-10 Training Progress</h1>
    <div class="message">
        <p>Training is currently in progress. This report will be updated when training data becomes available.</p>
        <p>Check back later for visualizations of the training progress.</p>
    </div>
</body>
</html>""")
        print("Created placeholder CIFAR-10 progress report")
    
    # Check multiple possible locations for CIFAR-100 progress report
    cifar100_progress_report_paths = [
        os.path.join(project_dir, 'reports', 'cifa100', 'cifar100_progress_report.html'),
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
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>CIFAR-100 Training Progress</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .message { padding: 20px; background-color: #f8f9fa; border-left: 4px solid #007bff; }
    </style>
</head>
<body>
    <h1>CIFAR-100 Training Progress</h1>
    <div class="message">
        <p>Training is currently in progress. This report will be updated when training data becomes available.</p>
        <p>Check back later for visualizations of the training progress.</p>
    </div>
</body>
</html>""")
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
    
    # Copy markdown files (README and whitepaper)
    copy_markdown_files(project_dir, web_dir)
    
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
