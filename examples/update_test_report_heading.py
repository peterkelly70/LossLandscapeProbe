#!/usr/bin/env python3
"""
Simple script to update the heading in the test report HTML file.
This script doesn't require PyTorch or other dependencies.

Run with sudo to ensure you have permission to modify web server files.
"""

import os
import sys
import glob
from pathlib import Path

def update_test_report_heading(web_dir):
    """Update the heading in all test report HTML files."""
    # Find all test report HTML files (with or without timestamp)
    report_files = glob.glob(os.path.join(web_dir, 'cifar10_test_report*.html'))
    
    if not report_files:
        print("No test report files found in", web_dir)
        return
    
    for report_file in report_files:
        print(f"Updating heading in {report_file}...")
        
        # Read the file content
        with open(report_file, 'r') as f:
            content = f.read()
        
        # Replace the heading
        updated_content = content.replace(
            '<h2>Test Images and Predictions</h2>',
            '<h2>Test Images and Predictions (200/10000)</h2>'
        )
        
        # Write the updated content back to the file
        with open(report_file, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated heading in {report_file}")

if __name__ == "__main__":
    # Default web directory
    web_dir = "/var/www/html/loss.computer-wizard.com.au"
    
    # Check if alternative web directory was provided
    if len(sys.argv) > 1:
        web_dir = sys.argv[1]
    
    print(f"Web directory: {web_dir}")
    update_test_report_heading(web_dir)
