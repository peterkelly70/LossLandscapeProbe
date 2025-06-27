"""Rich logging utilities for LossLandscapeProbe."""
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.text import Text
from typing import Optional, Dict, Any
import logging
import time
from pathlib import Path

# Global console instance
console = Console()

class TrainingProgress:
    """Track and display training progress with Rich."""
    
    def __init__(self, total_epochs: int, total_batches: int):
        self.progress = Progress(
            SpinnerColumn(),
            "•",
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=10,
        )
        
        self.epoch_task = self.progress.add_task(
            "[cyan]Epochs", total=total_epochs
        )
        self.batch_task = self.progress.add_task(
            "[green]Batches", total=total_batches, visible=False
        )
        
    def start(self):
        """Start the progress display."""
        self.progress.start()
        
    def update_epoch(self, epoch: int, metrics: Optional[Dict[str, Any]] = None):
        """Update epoch progress."""
        self.progress.update(self.epoch_task, completed=epoch + 1)
        if metrics:
            self._log_metrics(metrics)
            
    def update_batch(self, batch: int, total_batches: int):
        """Update batch progress."""
        self.progress.update(self.batch_task, completed=batch + 1, total=total_batches)
        
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics in a table."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(style="green")
        
        for name, value in metrics.items():
            table.add_row(f"{name}:", f"{value:.4f}")
            
        console.print(Panel(table, title="[bold]Metrics", border_style="blue"))

def setup_rich_logging(log_file: Optional[Path] = None, level=logging.INFO):
    """Set up Rich logging with both console and file handlers.
    
    Args:
        log_file: Optional path to log file. If None, only console logging is used.
        level: Logging level (default: logging.INFO)
        
    Returns:
        Configured logger
    """
    # Get the root logger and remove any existing handlers
    logger = logging.getLogger()
    logger.handlers = []
    
    # Prevent propagation to avoid duplicate logs from child loggers
    logger.propagate = False
    
    # Create console handler with Rich
    console_handler = RichHandler(
        console=console,
        show_time=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_path=False,
        markup=True
    )
    console_handler.setLevel(level)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    
    # Add the console handler to the root logger
    logger.addHandler(console_handler)
    
    # Add file handler if log file is provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def print_header(title: str, style: str = "bold blue"):
    """Print a styled header."""
    console.rule(f"[bold {style}]{title}")

def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """Print configuration in a table."""
    table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="green")
    
    for key, value in config.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = f"{value:.4f}"
        table.add_row(f"{key}:", str(value))
    
    console.print(Panel(table, title=f"[bold]{title}", border_style="blue"))

def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Print metrics in a table."""
    table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="green")
    
    for key, value in metrics.items():
        table.add_row(f"{key}:", f"{value:.4f}")
    
    console.print(Panel(table, title=f"[bold]{title}", border_style="green"))
