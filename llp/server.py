#!/usr/bin/env python3
"""
Server Management Script for LossLandscapeProbe

This script provides commands to start, stop, and check the status of the web server.

Usage:
    python server.py [command]

Commands:
    start    Start the web server
    stop     Stop the web server
    status   Check if the server is running
    restart  Restart the web server
    help     Show this help message
"""

import os
import sys
import time
import signal
import socket
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Configuration
DEFAULT_PORT = 8090
PID_FILE = "./server.pid"
LOG_FILE = "./server.log"
FLASK_APP = "website.app:create_app()"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

class ServerManager:
    """Manages the lifecycle of the Flask web server."""
    
    def __init__(self, port: int = DEFAULT_PORT):
        """Initialize the server manager.
        
        Args:
            port: Port to run the server on
        """
        self.port = port
        self.pid_file = Path(PID_FILE)
        self.log_file = Path(LOG_FILE)
    
    def start(self) -> bool:
        """Start the web server in a separate process.
        
        Returns:
            bool: True if server started successfully, False otherwise
        """
        if self.is_running():
            logger.info(f"Server is already running (PID: {self.get_pid()})")
            return True
        
        try:
            # Start the Flask app using gunicorn
            cmd = [
                "gunicorn",
                "--bind", f"0.0.0.0:{self.port}",
                "--workers", "4",
                "--worker-class", "gevent",
                "--timeout", "120",
                "--pid", str(self.pid_file.absolute()),
                "--access-logfile", "-",
                "--error-logfile", str(self.log_file.absolute()),
                FLASK_APP
            ]
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                start_new_session=True
            )
            
            # Write PID to file
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))
            
            # Wait for server to start
            if self.wait_for_server():
                logger.info(f"Server started successfully on port {self.port} (PID: {process.pid})")
                return True
            else:
                logger.error("Server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the web server.
        
        Returns:
            bool: True if server stopped successfully, False otherwise
        """
        if not self.is_running():
            logger.info("Server is not running")
            return True
            
        try:
            pid = self.get_pid()
            if not pid:
                logger.error("Could not find server PID")
                return False
                
            # Send SIGTERM to the process group
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            
            # Wait for process to terminate
            timeout = 10  # seconds
            start_time = time.time()
            while self.is_running():
                if time.time() - start_time > timeout:
                    logger.warning("Force killing server process")
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                    break
                time.sleep(0.5)
            
            # Clean up PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
                
            logger.info("Server stopped successfully")
            return True
            
        except ProcessLookupError:
            logger.warning("Server process not found")
            if self.pid_file.exists():
                self.pid_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            return False
    
    def restart(self) -> bool:
        """Restart the web server.
        
        Returns:
            bool: True if server restarted successfully, False otherwise
        """
        if self.is_running():
            if not self.stop():
                return False
            time.sleep(1)  # Give it a moment to fully stop
        return self.start()
    
    def status(self) -> Dict[str, Any]:
        """Get the status of the web server.
        
        Returns:
            Dict containing status information
        """
        pid = self.get_pid()
        is_running = self.is_running()
        
        return {
            "running": is_running,
            "pid": pid if is_running else None,
            "port": self.port,
            "pid_file": str(self.pid_file.absolute()),
            "log_file": str(self.log_file.absolute())
        }
    
    def is_running(self) -> bool:
        """Check if the server is running.
        
        Returns:
            bool: True if server is running, False otherwise
        """
        pid = self.get_pid()
        if not pid:
            return False
            
        try:
            # Check if process exists and is a gunicorn process
            with open(f"/proc/{pid}/cmdline", 'r') as f:
                cmdline = f.read()
                if 'gunicorn' in cmdline:
                    # Send signal 0 to check if process exists
                    os.kill(pid, 0)
                    return True
            return False
        except (ProcessLookupError, FileNotFoundError):
            return False
        except Exception as e:
            logger.error(f"Error checking process status: {e}")
            return False
    
    def get_pid(self) -> Optional[int]:
        """Get the server process ID.
        
        Returns:
            Optional[int]: Process ID if found, None otherwise
        """
        if not self.pid_file.exists():
            return None
            
        try:
            with open(self.pid_file, 'r') as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return None
    
    def wait_for_server(self, timeout: int = 10) -> bool:
        """Wait for the server to start.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if server is responsive, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection(('localhost', self.port), timeout=1):
                    return True
            except (socket.timeout, ConnectionRefusedError):
                time.sleep(0.1)
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Manage the LossLandscapeProbe web server')
    parser.add_argument(
        'command', 
        choices=['start', 'stop', 'restart', 'status'],
        help='Command to execute'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=DEFAULT_PORT,
        help=f'Port to run the server on (default: {DEFAULT_PORT})'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (for start command only)'
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    manager = ServerManager(port=args.port)
    
    if args.command == 'start':
        if args.debug:
            # Run in foreground with debug mode
            os.environ['FLASK_DEBUG'] = '1'
            os.environ['FLASK_ENV'] = 'development'
            os.execvp('flask', ['flask', 'run', '--port', str(args.port)])
        else:
            if not manager.start():
                sys.exit(1)
    
    elif args.command == 'stop':
        if not manager.stop():
            sys.exit(1)
    
    elif args.command == 'restart':
        if not manager.restart():
            sys.exit(1)
    
    elif args.command == 'status':
        status = manager.status()
        if status['running']:
            print(f"Server is running (PID: {status['pid']}, Port: {status['port']})")
            print(f"Log file: {status['log_file']}")
        else:
            print("Server is not running")
            if status['pid']:
                print(f"Warning: Stale PID file found at {status['pid_file']}")

if __name__ == '__main__':
    main()
