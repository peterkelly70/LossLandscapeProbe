#!/bin/bash

# Set up the website directory structure
WEBSITE_DIR="$(dirname "$0")/website"
REPORTS_DIR="$(dirname "$0")/reports"

# Create necessary directories
mkdir -p "$WEBSITE_DIR/static"
mkdir -p "$WEBSITE_DIR/templates"

# Create symbolic link to reports directory if it doesn't exist
if [ ! -L "$WEBSITE_DIR/reports" ]; then
    ln -sf "$REPORTS_DIR" "$WEBSITE_DIR/reports"
fi

# Install required Python packages if not already installed
if [ ! -d "$WEBSITE_DIR/venv" ]; then
    echo "Setting up Python virtual environment..."
    python3 -m venv "$WEBSITE_DIR/venv"
    source "$WEBSITE_DIR/venv/bin/activate"
    pip install --upgrade pip
    pip install -r "$WEBSITE_DIR/requirements.txt"
    deactivate
fi

echo "Website setup complete."
echo "To start the web server, run:"
echo "  ./website/server.sh"
