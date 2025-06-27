from flask import Flask, render_template_string, Markup
import markdown
from pathlib import Path

app = Flask(__name__)

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }} - LossLandscapeProbe</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .content { margin-top: 20px; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <div class="content">
        {{ content|safe }}
    </div>
</body>
</html>
"""

def load_markdown(filepath):
    """Load and render markdown file to HTML."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            md_content = f.read()
        return markdown.markdown(md_content, extensions=['fenced_code', 'tables'])
    except Exception as e:
        return f"<p>Error loading content: {str(e)}</p>"

@app.route('/')
def index():
    return """
    <h1>Documentation</h1>
    <ul>
        <li><a href="/readme">README</a></li>
        <li><a href="/whitepaper">Whitepaper</a></li>
    </ul>
    """

@app.route('/readme')
def readme():
    content = load_markdown('README.md')  # Looks in the current directory
    return render_template_string(HTML_TEMPLATE, title="README", content=Markup(content))

@app.route('/whitepaper')
def whitepaper():
    content = load_markdown('WHITEPAPER.md')  # Looks in the current directory
    return render_template_string(HTML_TEMPLATE, title="Whitepaper", content=Markup(content))

if __name__ == '__main__':
    # Create symbolic links to ensure files are found
    try:
        Path('README.md').symlink_to('website/static/docs/README.md')
        Path('WHITEPAPER.md').symlink_to('website/static/docs/WHITEPAPER.md')
    except FileExistsError:
        pass  # Links already exist
        
    print("Starting simple documentation server...")
    print("Visit http://localhost:5000")
    app.run(debug=True, port=5000)
