from flask import Flask, render_template, send_from_directory, send_file, abort, redirect, url_for, request, flash
from markupsafe import Markup
import os
import shutil
import markdown
from pathlib import Path
from datetime import datetime

def render_markdown_file(file_path):
    """Render a markdown file to HTML."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Convert markdown to HTML
        html = markdown.markdown(
            content,
            extensions=[
                'extra',
                'smarty',
                'codehilite',
                'tables',
                'toc'
            ],
            output_format='html5'
        )
        return Markup(html)
    except Exception as e:
        return f"<div class='alert alert-danger'>Error rendering markdown: {str(e)}</div>"

import subprocess
from datetime import datetime

app = Flask(__name__, 
             template_folder=str(Path(__file__).parent.absolute() / 'templates'),
             static_folder=str(Path(__file__).parent.absolute() / 'static'))

# Paths
BASE_DIR = Path(__file__).parent.absolute()
REPORTS_DIR = BASE_DIR.parent / 'reports'
# Logs are stored within the report folders themselves
LOGS_ROOT = BASE_DIR.parent / 'logs'  # legacy; still created
LOGS_ROOT.mkdir(exist_ok=True)  # keep

# Ensure the directories exist
REPORTS_DIR.mkdir(exist_ok=True)


# Enable debug output
app.config['DEBUG'] = True
app.secret_key = 'replace-this-with-env-secret'

@app.route('/')
def index():
    """Render the main page with links to documentation."""
    return render_template('index.html')

def _find_file(*possible_paths):
    """Helper function to find the first existing file from a list of possible paths."""
    print("\n=== Looking for file in these locations ===")
    for i, path in enumerate(possible_paths, 1):
        try:
            # Convert to Path if not already
            full_path = Path(path) if not isinstance(path, Path) else path
            
            # Resolve any symbolic links and normalize the path
            try:
                full_path = full_path.resolve(strict=True)
            except (FileNotFoundError, RuntimeError):
                print(f"{i}. {path} (does not resolve)")
                continue
                
            # Check if path exists and is a file
            if not full_path.exists():
                print(f"{i}. {path} (does not exist)")
                continue
                
            if not full_path.is_file():
                print(f"{i}. {path} (exists but is not a file)")
                continue
                
            # Check if file is readable
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    pass
            except IOError as e:
                print(f"{i}. {path} (exists but not readable: {e})")
                continue
                
            # If we got here, the file is good
            print(f"{i}. {path} (FOUND!)")
            print(f"   Size: {full_path.stat().st_size} bytes")
            print(f"   Permissions: {oct(full_path.stat().st_mode)[-3:]}")
            return full_path
            
        except Exception as e:
            print(f"{i}. {path} (error: {str(e)})")
    
    print("=== No valid file found ===\n")
    return None

@app.route('/whitepaper')
def whitepaper():
    """Render the whitepaper markdown file as HTML."""
    print("\n=== Loading Whitepaper ===")
    doc_dir = BASE_DIR.parent / 'doc'
    whitepaper_path = doc_dir / 'LossLandscapeProbe_WhitePaper.md'
    
    print(f"Looking for whitepaper at: {whitepaper_path}")
    print(f"Absolute path: {whitepaper_path.absolute()}")
    
    if not whitepaper_path.exists():
        print("ERROR: Whitepaper not found at the expected location")
        # Try alternative locations
        alt_paths = [
            BASE_DIR.parent / 'LossLandscapeProbe_WhitePaper.md',
            BASE_DIR / 'doc' / 'LossLandscapeProbe_WhitePaper.md',
            BASE_DIR / 'LossLandscapeProbe_WhitePaper.md'
        ]
        
        for alt_path in alt_paths:
            print(f"Trying alternative path: {alt_path}")
            if alt_path.exists():
                whitepaper_path = alt_path
                print(f"Found whitepaper at alternative location: {whitepaper_path}")
                break
        else:
            abort(404, "Whitepaper not found in any expected location")
    
    try:
        print(f"Reading whitepaper from: {whitepaper_path}")
        with open(whitepaper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Successfully read {len(content)} characters from whitepaper")
        
        html = markdown.markdown(
            content,
            extensions=['extra', 'codehilite', 'tables', 'toc']
        )
        return render_template('markdown.html', title='Whitepaper', content=html)
    except Exception as e:
        print(f"ERROR reading/rendering whitepaper: {str(e)}")
        abort(500, f"Error loading whitepaper: {str(e)}")

@app.route('/readme')
def readme():
    """Render the README markdown file as HTML."""
    print("\n=== Loading README ===")
    doc_dir = BASE_DIR.parent / 'doc'
    readme_path = doc_dir / 'README.md'
    
    print(f"Looking for README at: {readme_path}")
    
    if not readme_path.exists():
        print("ERROR: README not found at the expected location")
        abort(404, "README not found")
    
    try:
        print(f"Reading README from: {readme_path}")
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Successfully read {len(content)} characters from README")
        
        html = markdown.markdown(
            content,
            extensions=['extra', 'codehilite', 'tables', 'toc']
        )
        return render_template('markdown.html', title='Documentation', content=html)
    except Exception as e:
        print(f"ERROR reading/rendering README: {str(e)}")
        abort(500, f"Error loading README: {str(e)}")

@app.route('/license')
def license():
    """Render the LICENSE file as HTML."""
    print("\n=== Loading License ===")
    doc_dir = BASE_DIR.parent / 'doc'
    license_path = doc_dir / 'LICENSE'
    
    print(f"Looking for license at: {license_path}")
    
    if not license_path.exists():
        print("ERROR: License not found at the expected location")
        abort(404, "LICENSE not found")
    
    try:
        print(f"Reading license from: {license_path}")
        with open(license_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Successfully read {len(content)} characters from license file")
        
        # Try to render as markdown, fall back to preformatted text if it fails
        try:
            html = markdown.markdown(
                content,
                extensions=['extra', 'codehilite', 'tables']
            )
        except Exception as e:
            print(f"Falling back to preformatted text for license: {str(e)}")
            html = f"<pre>{content}</pre>"
        
        return render_template('markdown.html', title='License', content=html)
    except Exception as e:
        print(f"ERROR reading/rendering license: {str(e)}")
        abort(500, f"Error loading license: {str(e)}")

@app.route('/reports/<path:filename>')
def report_file(filename):
    """Serve report files from the reports directory."""
    try:
        print(f"\n=== Serving report file: {filename} ===")
        print(f"Looking in: {REPORTS_DIR}")
        
        # Check if the file exists
        file_path = REPORTS_DIR / filename
        if file_path.exists():
            print(f"File found at: {file_path}")
            # For log files, render with the log display template
            if filename.endswith('.log'):
                with open(file_path, 'r') as f:
                    content = f.read()
                return render_template('log_display.html', log_content=content, log_file=filename)
            return send_from_directory(REPORTS_DIR, filename)
        else:
            print(f"File not found at: {file_path}")
            # Try to find the file in the old structure
            if not '/' in filename and filename.endswith('.html'):
                # Check if it's in a subdirectory
                for subdir in ['cifar10/base', 'cifar100/base']:
                    alt_path = REPORTS_DIR / subdir / filename
                    if alt_path.exists():
                        rel_path = str(alt_path.relative_to(REPORTS_DIR))
                        print(f"Found at alternative location: {rel_path}")
                        return send_from_directory(REPORTS_DIR, rel_path)
            
            # Return a friendly page instead of raw 404 when the report is missing
            print("Returning friendly report unavailable page")
            return render_template('report_unavailable.html', filename=filename), 200
    except Exception as e:
        print(f"Error serving report file: {e}")
        abort(404, f"Report not found: {str(e)}")

@app.route('/reports')
def list_reports():
    """List available reports."""
    return render_template('reports.html', active_page='reports')

@app.route('/api/reports')
def api_list_reports():
    """API endpoint to list available reports."""
    try:
        reports = []
        if not REPORTS_DIR.exists():
            return {'reports': []}
            
        # Include all files in the reports directory
        for entry in os.scandir(REPORTS_DIR):
            if entry.is_file():
                try:
                    stats = entry.stat()
                    reports.append({
                        'name': entry.name,
                        'size': stats.st_size,
                        'modified': stats.st_mtime,
                        'path': f'/reports/{entry.name}'
                    })
                except Exception as e:
                    print(f"Error processing {entry.name}: {e}")
                    continue
                    
        # Also include files in subdirectories
        for root, dirs, files in os.walk(REPORTS_DIR):
            for file in files:
                if file == '.DS_Store':
                    continue
                try:
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(REPORTS_DIR)
                    stats = full_path.stat()
                    reports.append({
                        'name': str(rel_path),
                        'size': stats.st_size,
                        'modified': stats.st_mtime,
                        'path': f'/reports/{rel_path}'
                    })
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        # Sort by modification time (newest first)
        reports.sort(key=lambda x: x['modified'], reverse=True)
        return {'reports': [r['name'] for r in reports]}
    except Exception as e:
        print(f"Error listing reports: {e}")
        return {'error': str(e)}, 500

@app.route('/logs')
def logs_page():
    # If a specific log file is requested, redirect to the direct log file path
    log_file = request.args.get('file')
    if log_file:
        dataset = request.args.get('dataset')
        sample_size = request.args.get('sample_size')
        
        if dataset and sample_size:
            # Redirect to the direct log file path
            return redirect(f"/reports/{dataset}_{sample_size}/{dataset}_{sample_size}.log")
    
    # Otherwise, show the log selection page
    return render_template('logs.html', active_page='logs')


@app.route('/api/logs')
def api_logs():
    """Return list of logs or tail of a log via query params"""
    log_file = request.args.get('file')
    lines = int(request.args.get('lines', '100'))

    dataset = request.args.get('dataset')
    sample_size = request.args.get('sample_size') or request.args.get('resource')

    def list_logs(dir_path):
        return sorted([p.name for p in dir_path.glob('*.log')], reverse=True)

    if not log_file:
        if dataset and sample_size:
            dir_path = REPORTS_DIR / f"{dataset}_{sample_size}"
            if not dir_path.exists():
                return {'files': []}
            return {'files': list_logs(dir_path)}
        else:
            # Legacy logs root
            return {'files': list_logs(LOGS_ROOT)}

    # Determine where to look
    search_dirs = []
    if dataset and sample_size:
        search_dirs.append(REPORTS_DIR / f"{dataset}_{sample_size}")
    search_dirs.append(LOGS_ROOT)

    log_path = None
    for d in search_dirs:
        cand = d / log_file
        if cand.exists():
            log_path = cand
            break
    if log_path is None:
        return '', 404
    if not log_path.exists():
        return '', 404
    try:
        with open(log_path, 'r') as f:
            data = f.readlines()[-lines:]
        return ''.join(data)
    except Exception as e:
        return f'Error reading log: {e}', 500


@app.route('/train', methods=['GET'])
def train():
    """Render form to start a new training run."""
    return render_template('train.html', active_page='train')


@app.route('/train', methods=['POST'])
def start_training():
    """Kick off training in a background subprocess."""
    dataset = request.form.get('dataset', 'cifar10')
    sample_size = request.form.get('sample_size', 'multi')
    epochs = request.form.get('epochs', '20')
    configs_per_sample = request.form.get('configs_per_sample', '10')
    perturbations = request.form.get('perturbations', '10')
    iterations = request.form.get('iterations', '3')

    # Always use the main unified script in the project root
    script_path = BASE_DIR.parent / 'unified_cifar_training.py'

    cmd = [
        'python', str(script_path),
        '--dataset', dataset,
        '--sample-size', sample_size,
        '--epochs', epochs,
        '--configs-per-sample', configs_per_sample,
        '--perturbations', perturbations,
        '--iterations', iterations
    ]
    now_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Place log next to reports: /reports/dataset_sample/
    run_report_dir = REPORTS_DIR / f"{dataset}_{sample_size}"
    run_report_dir.mkdir(parents=True, exist_ok=True)

    # Constant filename for latest log
    base_log_name = f"{dataset}_{sample_size}.log"
    log_path = run_report_dir / base_log_name

    # Archive previous log if exists
    if log_path.exists():
        LOGS_ROOT.mkdir(exist_ok=True)
        archive_ts = datetime.fromtimestamp(log_path.stat().st_mtime).strftime('%Y%m%d_%H%M%S')
        archive_name = f"{dataset}_{sample_size}_{archive_ts}.log"
        shutil.move(str(log_path), str(LOGS_ROOT / archive_name))
    try:
        log_fh = open(log_path, 'w')
        subprocess.Popen(cmd, cwd=Path(__file__).parent.parent, stdout=log_fh, stderr=log_fh)
        flash(f'Training started for {dataset} ({sample_size}). Live log: {log_path.name}', 'success')
    except Exception as e:
        flash(f'Error starting training: {e}', 'danger')
    return redirect(url_for('train'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090, debug=True)
