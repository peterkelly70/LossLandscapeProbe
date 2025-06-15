#!/usr/bin/env python3
"""
Setup script to copy necessary files to the web server directory for the LossLandscapeProbe visualization website.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

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

// Get all HTML files
$files = glob($dir . '/*.html');
$reports = [];

foreach ($files as $file) {
    $filename = basename($file);
    $reports[] = $filename;
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

def convert_readme_to_md(project_dir, web_dir):
    """Convert README.md to a web-friendly version."""
    readme_path = os.path.join(project_dir, 'README.md')
    web_readme_path = os.path.join(web_dir, 'README.md')
    
    try:
        # Read the README content
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        
        # Write it to the web directory
        with open(web_readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"Copied README: {readme_path} → {web_readme_path}")
        return True
    except Exception as e:
        print(f"Error copying README: {e}")
        return False

def update_test_report_headings(web_dir):
    """Update the headings in all test report HTML files to show sample size."""
    try:
        # Find all test report HTML files
        report_files = [f for f in os.listdir(web_dir) if f.startswith('cifar10_test_report_') and f.endswith('.html')]
        
        if not report_files:
            print("No test report files found in", web_dir)
            return
        
        updated_count = 0
        for report_file in report_files:
            report_path = os.path.join(web_dir, report_file)
            try:
                # Read the file content
                with open(report_path, 'r') as f:
                    content = f.read()
                
                # Replace the heading
                updated_content = content.replace(
                    '<h2>Test Images and Predictions</h2>',
                    '<h2>Test Images and Predictions (200/10000)</h2>'
                )
                
                # Add sample notice if it doesn't exist
                sample_notice = '''
                <div style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 10px; margin-bottom: 20px;">
                    <p><strong>Note:</strong> The accuracy statistics shown here are calculated based only on this sample, not the full test set.</p>
                </div>
                '''
                
                if '<div style="background-color: #f8f9fa; border-left: 4px solid #007bff;' not in updated_content:
                    updated_content = updated_content.replace(
                        '<h2>Test Images and Predictions (200/10000)</h2>',
                        f'<h2>Test Images and Predictions (200/10000)</h2>\n{sample_notice}'
                    )
                
                # Write the updated content back to the file
                try:
                    with open(report_path, 'w') as f:
                        f.write(updated_content)
                    print(f"Updated heading in {report_file}")
                    updated_count += 1
                except Exception as e:
                    print(f"Warning: Could not update {report_file}: {e}")
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
        test_report_script = os.path.join(project_dir, 'examples', 'generate_test_report.py')
        subprocess.run([sys.executable, test_report_script], cwd=project_dir, check=False)
        print("CIFAR-10 test report generated successfully.")
    except Exception as e:
        print(f"Warning: Could not generate CIFAR-10 test report: {e}")
    
    # Generate training plots report
    try:
        print("\nGenerating training plots report...")
        training_plots_script = os.path.join(project_dir, 'examples', 'generate_training_plots.py')
        subprocess.run([sys.executable, training_plots_script], cwd=project_dir, check=False)
        print("Training plots report generated successfully.")
    except Exception as e:
        print(f"Warning: Could not generate training plots report: {e}")
    
    # Generate CIFAR-100 transfer report if the experiment has completed
    try:
        print("\nChecking for CIFAR-100 results and generating report if available...")
        cifar100_report_script = os.path.join(project_dir, 'examples', 'generate_cifar100_report.py')
        subprocess.run([sys.executable, cifar100_report_script], cwd=project_dir, check=False)
        print("CIFAR-100 transfer report generated successfully.")
    except Exception as e:
        print(f"Note: Could not generate CIFAR-100 transfer report: {e}")
        print("This is expected if the CIFAR-100 transfer experiment is still running.")
    
    print("\nReport generation complete.")

def main():
    # Get project directory
    project_dir = Path(__file__).parent.parent.absolute()
    
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
    template_path = os.path.join(project_dir, 'examples', 'improved_website_template.html')
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
        
    # Copy training reports if they exist
    training_report_path = os.path.join(project_dir, 'training_report.html')
    web_training_report_path = os.path.join(web_dir, 'training_report.html')
    
    if os.path.exists(training_report_path):
        copy_file(training_report_path, web_training_report_path)
        print(f"Copied training report to {web_training_report_path}")
    else:
        print(f"Note: Training report not found at {training_report_path}. The website will show a placeholder message.")
    
    # Copy CIFAR-100 transfer report if it exists
    cifar100_report_path = os.path.join(project_dir, 'cifar100_transfer_report.html')
    web_cifar100_report_path = os.path.join(web_dir, 'cifar100_transfer_report.html')
    
    if os.path.exists(cifar100_report_path):
        copy_file(cifar100_report_path, web_cifar100_report_path)
        print(f"Copied CIFAR-100 transfer report to {web_cifar100_report_path}")
    else:
        print(f"Note: CIFAR-100 transfer report not found at {cifar100_report_path}. The website will show a placeholder message.")
        print(f"This is expected if the CIFAR-100 transfer experiment is still running.")
    
    # Create reports directory for all test reports
    reports_dir = os.path.join(web_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create training reports directory
    training_reports_dir = os.path.join(web_dir, 'training_reports')
    os.makedirs(training_reports_dir, exist_ok=True)
    
    # Copy existing CIFAR-10 test reports
    project_reports_dir = os.path.join(project_dir, 'reports')
    if os.path.exists(project_reports_dir):
        for report_file in os.listdir(project_reports_dir):
            if report_file.startswith('cifar10_test_report') and report_file.endswith('.html'):
                src_path = os.path.join(project_reports_dir, report_file)
                dest_path = os.path.join(reports_dir, report_file)
                copy_file(src_path, dest_path)
                print(f"Copied CIFAR-10 test report: {report_file} to reports directory")
            elif report_file.startswith('training_report_') and report_file.endswith('.html'):
                src_path = os.path.join(project_reports_dir, report_file)
                dest_path = os.path.join(training_reports_dir, report_file)
                copy_file(src_path, dest_path)
                print(f"Copied training report: {report_file} to training_reports directory")
    else:
        print(f"Note: No reports found in {project_reports_dir}. The website will show placeholder messages.")
        
    # Also copy the list_training_reports.php script
    training_reports_php_src = os.path.join(project_dir, 'examples', 'list_training_reports.php')
    training_reports_php_dest = os.path.join(web_dir, 'list_training_reports.php')
    if os.path.exists(training_reports_php_src):
        copy_file(training_reports_php_src, training_reports_php_dest)
        print(f"Copied list_training_reports.php to {web_dir}")
    else:
        print(f"Warning: list_training_reports.php not found at {training_reports_php_src}")


    
    # Create list_reports.php
    create_list_reports_php(web_dir)
    
    # Create list_training_reports.php
    create_list_training_reports_php(web_dir)
    
    # Update test report headings to show sample size
    print("\nUpdating test report headings...")
    update_test_report_headings(web_dir)
    
    # Convert and copy README
    convert_readme_to_md(project_dir, web_dir)
    
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
