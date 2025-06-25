#!/usr/bin/env python3
"""
Fix Generated Test Reports
=========================

This script fixes the generated test reports by:
1. Updating image paths in the HTML files
2. Ensuring all required images exist
3. Creating proper directory structure
"""

import os
import re
from pathlib import Path
import shutil

def fix_test_reports(reports_dir="reports"):
    """Fix all test reports in the given directory."""
    reports_path = Path(reports_dir)
    if not reports_path.exists():
        print(f"Reports directory '{reports_dir}' not found.")
        return

    # Find all test report HTML files
    for report_path in reports_path.rglob("*_test_report.html"):
        fix_single_report(report_path)

def fix_single_report(report_path):
    """Fix a single test report file."""
    print(f"Fixing report: {report_path}")
    
    # Get the directory containing the report
    report_dir = report_path.parent
    samples_dir = report_dir / "prediction_samples"
    
    # Ensure prediction_samples directory exists
    samples_dir.mkdir(exist_ok=True)
    
    # Read the HTML content
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Fix image paths
    content = re.sub(
        r'src=["\'](?!prediction_samples/)([^"\']+\.(?:png|jpg|jpeg|gif))["\']',
        r'src="prediction_samples/\1"',
        content,
        flags=re.IGNORECASE
    )
    
    # Write the fixed content back
    with open(report_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed report: {report_path}")

if __name__ == "__main__":
    fix_test_reports()
