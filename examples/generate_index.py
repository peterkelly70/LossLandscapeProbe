#!/usr/bin/env python3
"""
Generate an index.html file that lists all test reports in the specified directory.
"""

import os
import glob
import datetime
import re

def extract_timestamp(filename):
    """Extract timestamp from filename for sorting."""
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return match.group(1)
    return filename

def generate_index(reports_dir, output_path):
    """Generate an index.html file listing all test reports."""
    # Get all HTML files in the directory
    html_files = glob.glob(os.path.join(reports_dir, "*.html"))
    
    # Filter out index.html itself
    html_files = [f for f in html_files if os.path.basename(f) != "index.html"]
    
    # Sort files by timestamp (newest first)
    html_files.sort(key=extract_timestamp, reverse=True)
    
    # Start HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LossLandscapeProbe Test Reports</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1, h2 {
                color: #333;
            }
            .report-list {
                list-style-type: none;
                padding: 0;
            }
            .report-item {
                background-color: #f5f5f5;
                margin-bottom: 10px;
                padding: 15px;
                border-radius: 5px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .report-item:hover {
                background-color: #e0e0e0;
            }
            .report-link {
                text-decoration: none;
                color: #0066cc;
                font-weight: bold;
                font-size: 1.1em;
            }
            .report-date {
                color: #666;
                font-size: 0.9em;
            }
            .report-preview {
                display: none;
                margin-top: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                height: 600px;
                width: 100%;
            }
            .preview-container {
                margin-top: 30px;
            }
            .button {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 5px;
            }
            .button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>LossLandscapeProbe Test Reports</h1>
        <p>Click on a report to preview it below or open it in a new tab.</p>
        
        <ul class="report-list">
    """
    
    # Add each report to the list
    for html_file in html_files:
        filename = os.path.basename(html_file)
        
        # Try to extract a timestamp and format it nicely
        timestamp_match = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', filename)
        if timestamp_match:
            year, month, day, hour, minute, second = timestamp_match.groups()
            formatted_date = f"{year}-{month}-{day} {hour}:{minute}:{second}"
        else:
            # Use file modification time if no timestamp in filename
            mod_time = os.path.getmtime(html_file)
            formatted_date = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        
        html_content += f"""
        <li class="report-item">
            <a href="{filename}" class="report-link" onclick="previewReport('{filename}'); return false;">{filename}</a>
            <span class="report-date">{formatted_date}</span>
            <a href="{filename}" class="button" target="_blank">Open in New Tab</a>
        </li>
        """
    
    # Complete the HTML
    html_content += """
        </ul>
        
        <div class="preview-container">
            <h2>Preview</h2>
            <iframe id="report-preview" class="report-preview" src=""></iframe>
        </div>
        
        <script>
            function previewReport(filename) {
                document.getElementById('report-preview').src = filename;
                document.getElementById('report-preview').style.display = 'block';
            }
            
            // Load the first report by default if available
            window.onload = function() {
                const reports = document.querySelectorAll('.report-link');
                if (reports.length > 0) {
                    previewReport(reports[0].getAttribute('href'));
                }
            };
        </script>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Index generated at: {output_path}")
    print(f"Listed {len(html_files)} report(s)")

if __name__ == "__main__":
    # Default paths
    reports_dir = "/var/www/html/loss"
    output_path = os.path.join(reports_dir, "index.html")
    
    # Generate the index
    generate_index(reports_dir, output_path)
