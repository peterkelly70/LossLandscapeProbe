"""
Astops Ever - Graceful Server Shutdown Handler

This module provides functionality for gracefully shutting down a Flask application.
It handles cleanup operations and ensures all resources are properly released.
"""

import os
import signal
import logging
from typing import Optional, Callable, Any
from flask import Flask, jsonify, request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('astops_ever')

class GracefulShutdown:
    """
    A class to handle graceful shutdown of a Flask application.
    
    This class provides methods to register cleanup functions and handle
    shutdown signals (SIGINT, SIGTERM) to ensure the application shuts down
    gracefully.
    """
    
    def __init__(self, app: Optional[Flask] = None):
        """
        Initialize the GracefulShutdown handler.
        
        Args:
            app: Optional Flask application instance. If provided, shutdown
                 endpoints will be registered automatically.
        """
        self.app = app
        self.cleanup_functions = []
        self.shutting_down = False
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app: Flask) -> None:
        """
        Initialize the Flask application with shutdown handlers.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Register shutdown endpoint if in debug mode
        if app.debug:
            @app.route('/shutdown', methods=['POST'])
            def shutdown():
                """Endpoint to trigger a graceful shutdown."""
                return self._shutdown_server()
    
    def register_cleanup(self, func: Callable, *args, **kwargs) -> None:
        """
        Register a cleanup function to be called during shutdown.
        
        Args:
            func: The cleanup function to call
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        self.cleanup_functions.append((func, args, kwargs))
    
    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """
        Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        if self.shutting_down:
            return
            
        self.shutting_down = True
        logger.info("Shutdown signal received. Performing cleanup...")
        
        # Run all registered cleanup functions
        for func, args, kwargs in self.cleanup_functions:
            try:
                logger.debug(f"Running cleanup function: {func.__name__}")
                func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error during cleanup in {func.__name__}: {e}")
        
        # Shutdown the server
        self._shutdown_server()
    
    def _shutdown_server(self):
        """Shutdown the Flask server."""
        if self.app is not None:
            logger.info("Shutting down server...")
            if request and hasattr(request, '_get_current_object'):
                # If called from within a request context
                func = request.environ.get('werkzeug.server.shutdown')
                if func is not None:
                    func()
                    return jsonify({"status": "Shutting down..."}), 200
            
            # Fallback for non-Werkzeug servers
            os._exit(0)
        
        return jsonify({"error": "Not running with the Werkzeug Server"}), 500


def create_app() -> Flask:
    """
    Create and configure the Flask application with graceful shutdown.
    
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Configure the application
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY') or 'dev-key-123',
        DEBUG=os.environ.get('FLASK_ENV') == 'development'
    )
    
    # Initialize graceful shutdown
    shutdown_handler = GracefulShutdown(app)
    
    # Example route
    @app.route('/')
    def index():
        return "Astops Ever - Graceful Shutdown Demo"
    
    # Example cleanup function
    def cleanup_resources():
        logger.info("Cleaning up resources...")
        # Add your cleanup code here
    
    # Register cleanup function
    shutdown_handler.register_cleanup(cleanup_resources)
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
