#!/usr/bin/env python3
"""
Server Management Script for LossLandscapeProbe

This script provides commands to start, stop, and check the status of the web server.

Usage:
    python -m website.server [command]

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
from typing import Optional, Tuple, Dict, Any, List

# Configuration
DEFAULT_PORT = 8090
INSTANCE_DIR = Path("./instance")
# Virtual environment in project root .llp directory
VENV_DIR = Path(__file__).parent.parent.absolute() / ".llp"
PID_FILE = INSTANCE_DIR / "server.pid"
LOG_FILE = INSTANCE_DIR / "server.log"

# Ensure instance directory exists
INSTANCE_DIR.mkdir(exist_ok=True, parents=True)

# Log virtual environment path for debugging
logging.info(f"Using virtual environment at: {VENV_DIR}")
if not VENV_DIR.exists() or not (VENV_DIR / "bin" / "gunicorn").exists():
    logging.warning(f"Gunicorn not found in virtual environment at {VENV_DIR}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE))  # Convert Path to string for logging
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
        
        # Ensure instance directory exists
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
    
    def start(self) -> bool:
        """Start the web server in a separate process.
        
        Returns:
            bool: True if server started successfully, False otherwise
        """
        # Check if server is already running
        if self.is_running():
            logger.info(f"Server is already running (PID: {self.get_pid()})")
            return True
        
        try:
            # Get the path to the gunicorn executable in the virtual environment
            gunicorn_path = VENV_DIR.absolute() / 'bin' / 'gunicorn'
            
            # Start the Flask app using gunicorn from the virtual environment
            cmd = [
                str(gunicorn_path),
                "--chdir", str(Path(__file__).parent.absolute()),
                "--bind", f"0.0.0.0:{self.port}",
                "--workers", "4",
                "--worker-class", "gevent",
                "--timeout", "120",
                "--pid", str(self.pid_file.absolute()),
                "--access-logfile", "-",
                "--error-logfile", str(self.log_file.absolute()),
                "app:app"  # Using the Flask app instance directly
            ]
            
            # Set up environment with VIRTUAL_ENV
            env = os.environ.copy()
            env['VIRTUAL_ENV'] = str(VENV_DIR.absolute())
            env['PATH'] = f"{VENV_DIR.absolute()}/bin:{env.get('PATH', '')}"
            
            # Try up to 3 ports if the specified one is in use
            original_port = self.port
            max_port_attempts = 3
            
            for attempt in range(max_port_attempts):
                try:
                    # Update the port in the command if this isn't the first attempt
                    if attempt > 0:
                        self.port = original_port + attempt
                        cmd[5] = f"0.0.0.0:{self.port}"
                        logger.info(f"Trying alternate port {self.port} (attempt {attempt+1}/{max_port_attempts})...")
                    
                    # Start the process with the updated environment
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        start_new_session=True,
                        env=env
                    )
                    
                    # Write PID to file
                    with open(self.pid_file, 'w') as f:
                        f.write(str(process.pid))
                    
                    # Wait for server to start
                    if self.wait_for_server():
                        logger.info(f"Server started successfully on port {self.port} (PID: {process.pid})")
                        return True
                    else:
                        logger.error(f"Server failed to start on port {self.port}")
                        
                        # Clean up failed attempt
                        try:
                            process.terminate()
                            if self.pid_file.exists():
                                self.pid_file.unlink()
                        except Exception as e:
                            logger.error(f"Error cleaning up failed server start: {e}")
                            
                except (OSError, subprocess.SubprocessError) as e:
                    logger.error(f"Error starting server on port {self.port}: {e}")
                    
            # If we get here, all attempts failed
            logger.error(f"Failed to start server after {max_port_attempts} attempts")
            return False
                
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the web server.
        
        Returns:
            bool: True if server stopped successfully, False otherwise
        """
        pid = self.get_pid()
        if not pid:
            logger.warning("No PID file found, server may not be running")
            return True
            
        try:
            # Send SIGTERM to the process
            os.kill(pid, signal.SIGTERM)
            
            # Wait for process to terminate
            for _ in range(10):
                try:
                    os.kill(pid, 0)  # Check if process exists
                    time.sleep(0.5)
                except ProcessLookupError:
                    break  # Process has terminated
            else:
                # Process didn't terminate, try SIGKILL
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.warning(f"Had to use SIGKILL to terminate server process {pid}")
                except ProcessLookupError:
                    pass  # Process has terminated
                    
            # Remove PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
                
            # Also check for any lingering gunicorn processes
            try:
                result = subprocess.run(['pgrep', '-f', 'gunicorn.*app:app'], 
                                       stdout=subprocess.PIPE, 
                                       text=True)
                for lingering_pid in result.stdout.strip().split('\n'):
                    if lingering_pid and lingering_pid.isdigit():
                        try:
                            lingering_pid = int(lingering_pid)
                            if lingering_pid != pid:  # Don't double-kill the main process
                                logger.warning(f"Found lingering gunicorn process {lingering_pid}, terminating")
                                os.kill(lingering_pid, signal.SIGTERM)
                        except (ValueError, ProcessLookupError):
                            pass
            except Exception as e:
                logger.warning(f"Error checking for lingering processes: {e}")
                
            logger.info(f"Server stopped (PID: {pid})")
            return True
        except ProcessLookupError:
            logger.warning(f"Process {pid} not found, removing stale PID file")
            if self.pid_file.exists():
                self.pid_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
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
            
            # Process exists but is not gunicorn, clean up stale PID file
            logger.warning(f"Found stale PID file for non-gunicorn process {pid}, cleaning up")
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False
        except (ProcessLookupError, FileNotFoundError):
            # Process doesn't exist, clean up stale PID file
            logger.warning(f"Found stale PID file for non-existent process {pid}, cleaning up")
            if self.pid_file.exists():
                self.pid_file.unlink()
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
        nargs='?',
        default='status',
        help='Command to execute (default: status)'
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
            os.chdir(Path(__file__).parent.absolute())
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
