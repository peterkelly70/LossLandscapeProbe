from flask import Flask, render_template, send_from_directory, jsonify, send_file
import os
from pathlib import Path

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / 'reports'
STATIC_DIR = BASE_DIR / 'static'
TEMPLATES_DIR = BASE_DIR / 'templates'

# Menu structure
MENU_SECTIONS = [
    {
        'id': 'cifar10',
        'name': 'CIFAR-10',
        'menu_items': [
            {
                'type': 'submenu',
                'id': 'cifar10_mixed',
                'name': 'Mixed',
                'menu_items': [
                    {'name': 'Test Report', 'url': '#/cifar10/mixed/test'},
                    {'name': 'Training Report', 'url': '#/cifar10/mixed/training'},
                    {'name': 'Meta-Model Report', 'url': '#/cifar10/mixed/meta'},
                    {'name': 'Train Model', 'url': '#/cifar10/mixed/train'}
                ]
            },
            {
                'type': 'submenu',
                'id': 'cifar10_10',
                'name': '10% Sample',
                'menu_items': [
                    {'name': 'Test Report', 'url': '#/cifar10/10/test'},
                    {'name': 'Training Report', 'url': '#/cifar10/10/training'},
                    {'name': 'Meta-Model Report', 'url': '#/cifar10/10/meta'},
                    {'name': 'Train Model', 'url': '#/cifar10/10/train'}
                ]
            },
            # Add more CIFAR-10 variants as needed
        ]
    },
    {
        'id': 'cifar100',
        'name': 'CIFAR-100',
        'menu_items': [
            {
                'type': 'submenu',
                'id': 'cifar100_mixed',
                'name': 'Mixed',
                'menu_items': [
                    {'name': 'Test Report', 'url': '#/cifar100/mixed/test'},
                    {'name': 'Training Report', 'url': '#/cifar100/mixed/training'},
                    {'name': 'Meta-Model Report', 'url': '#/cifar100/mixed/meta'},
                    {'name': 'Train Model', 'url': '#/cifar100/mixed/train'}
                ]
            },
            # Add more CIFAR-100 variants as needed
        ]
    },
    {
        'id': 'documentation',
        'name': 'Documentation',
        'menu_items': [
            {'name': 'Whitepaper', 'url': '#/documentation/whitepaper'},
            {'name': 'README', 'url': '#/documentation/readme'}
        ]
    },
    {
        'id': 'github',
        'name': 'GitHub',
        'menu_items': [
            {'name': 'Repository', 'url': 'https://github.com/peterkelly70/LossLandscapeProbe', 'external': True}
        ]
    }
]

@app.route('/')
def index():
    """Serve the main application page."""
    return send_file('index.html')

@app.route('/docs/<path:filename>')
def serve_docs(filename):
    """Serve documentation files from the docs directory."""
    return send_from_directory('docs', filename)

@app.route('/reports/<path:path>')
def serve_report(path):
    """Serve report files from the reports directory."""
    return send_from_directory(REPORTS_DIR, path)

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files from the static directory."""
    return send_from_directory(STATIC_DIR, path)

@app.route('/api/menu')
def get_menu():
    """API endpoint to get the menu structure."""
    return jsonify(MENU_SECTIONS)

@app.route('/api/status')
def status():
    """API endpoint to check server status."""
    return jsonify({
        'status': 'running',
        'version': '1.0.0',
        'reports_dir': str(REPORTS_DIR.relative_to(BASE_DIR))
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('reports', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    app.run(host='0.0.0.0', port=8090, debug=True)
