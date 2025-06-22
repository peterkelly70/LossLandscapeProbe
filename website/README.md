# LossLandscapeProbe Web Interface

This is the web interface for the LossLandscapeProbe framework. It provides a user-friendly way to view and interact with your training reports and documentation.

## Requirements

- Python 3.7+
- pip

## Setup

1. Run the setup script to create the necessary directories and install dependencies:

   ```bash
   ./mksite.sh
   ```

2. Start the web server:

   ```bash
   cd website
   ./start_server.sh
   ```

3. Open your web browser and navigate to:

   ```
   http://localhost:8090
   ```

## Features

- View training reports and visualizations
- Access project documentation
- Browse whitepaper and license information
- Responsive design that works on desktop and mobile devices

## Directory Structure

- `static/` - Static files (CSS, JS, images)
- `templates/` - HTML templates
- `reports/` - Symbolic link to the main reports directory
- `venv/` - Python virtual environment (created during setup)
- `app.py` - Flask application
- `requirements.txt` - Python dependencies

## Customization

You can customize the web interface by modifying the template files in the `templates/` directory. The interface uses Bootstrap 5 for styling and includes support for Markdown rendering.

## License

This project is licensed under the terms of the MIT license. See the main project's LICENSE file for more information.
