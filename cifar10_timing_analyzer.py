#!/usr/bin/env python3
"""
CIFAR10 Training Time Analyzer

This script analyzes CIFAR10 training logs to extract validation batch and epoch timings,
calculate average durations, and estimate remaining training time for ongoing runs.
"""

import re
import os
import argparse
from datetime import datetime
import statistics
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class CIFAR10TimingAnalyzer:
    """Analyzes CIFAR10 training logs for timing information."""

    def __init__(self, log_file: str):
        """
        Initialize the analyzer with a log file path.
        
        Args:
            log_file: Path to the CIFAR10 training log file
        """
        self.log_file = log_file
        self.val_batch_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - llp\.cifar_meta_model - INFO -\s+Val Batch (\d+)/(\d+)')
        self.epoch_summary_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - llp\.cifar_meta_model - INFO - Epoch (\d+)/(\d+) - Train Loss: ([\d\.]+), Val Loss: ([\d\.]+), Val Acc: ([\d\.]+)%')
        
        # Data structures to store extracted information
        self.epoch_timestamps: Dict[int, datetime] = {}
        self.val_batch_timestamps: Dict[int, Dict[int, datetime]] = {}
        self.total_batches_per_epoch = 0
        self.total_epochs = 0
        self.batch_durations: Dict[int, List[float]] = {}
        self.epoch_durations: List[float] = []

    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse timestamp string into datetime object.
        
        Args:
            timestamp_str: Timestamp string in format 'YYYY-MM-DD HH:MM:SS,mmm'
            
        Returns:
            datetime object
        """
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')

    def parse_log_file(self) -> None:
        """Parse the log file to extract validation batch and epoch timestamps."""
        print(f"Parsing log file: {self.log_file}")
        
        with open(self.log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Extract validation batch timestamps
                val_batch_match = self.val_batch_pattern.search(line)
                if val_batch_match:
                    timestamp_str, batch_num, total_batches = val_batch_match.groups()
                    batch_num, total_batches = int(batch_num), int(total_batches)
                    
                    # Determine which epoch this batch belongs to
                    # We'll assign batches to the next epoch summary we find
                    if self.total_epochs == 0 or batch_num == 1:
                        current_epoch = len(self.val_batch_timestamps) + 1
                    else:
                        current_epoch = len(self.val_batch_timestamps)
                    
                    # Initialize dict for this epoch if it doesn't exist
                    if current_epoch not in self.val_batch_timestamps:
                        self.val_batch_timestamps[current_epoch] = {}
                    
                    # Store the timestamp for this batch
                    self.val_batch_timestamps[current_epoch][batch_num] = self.parse_timestamp(timestamp_str)
                    
                    # Update total batches per epoch if needed
                    if self.total_batches_per_epoch < total_batches:
                        self.total_batches_per_epoch = total_batches
                
                # Extract epoch summary timestamps
                epoch_match = self.epoch_summary_pattern.search(line)
                if epoch_match:
                    timestamp_str, epoch_num, total_epochs, train_loss, val_loss, val_acc = epoch_match.groups()
                    epoch_num, total_epochs = int(epoch_num), int(total_epochs)
                    
                    # Store the timestamp for this epoch summary
                    self.epoch_timestamps[epoch_num] = self.parse_timestamp(timestamp_str)
                    
                    # Update total epochs if needed
                    if self.total_epochs < total_epochs:
                        self.total_epochs = total_epochs
        
        print(f"Finished parsing log file. Found data for {len(self.val_batch_timestamps)} epochs with {self.total_batches_per_epoch} batches per epoch.")

    def calculate_batch_durations(self) -> None:
        """Calculate the duration of each validation batch within epochs."""
        for epoch, batches in self.val_batch_timestamps.items():
            # Sort batch numbers to ensure we calculate durations correctly
            batch_nums = sorted(batches.keys())
            
            # Initialize list to store batch durations for this epoch
            self.batch_durations[epoch] = []
            
            # Calculate duration between consecutive batches
            for i in range(len(batch_nums) - 1):
                current_batch = batch_nums[i]
                next_batch = batch_nums[i + 1]
                
                # Calculate duration in milliseconds
                duration_ms = (batches[next_batch] - batches[current_batch]).total_seconds() * 1000
                
                # Only include reasonable durations (filter out outliers)
                if 0 < duration_ms < 1000:  # Assuming no batch should take more than 1 second
                    self.batch_durations[epoch].append(duration_ms)
        
        print(f"Calculated batch durations for {len(self.batch_durations)} epochs.")

    def calculate_epoch_durations(self) -> None:
        """Calculate the duration of each epoch based on epoch summary timestamps."""
        epoch_nums = sorted(self.epoch_timestamps.keys())
        
        for i in range(len(epoch_nums) - 1):
            current_epoch = epoch_nums[i]
            next_epoch = epoch_nums[i + 1]
            
            # Calculate duration in seconds
            duration_sec = (self.epoch_timestamps[next_epoch] - self.epoch_timestamps[current_epoch]).total_seconds()
            
            # Only include reasonable durations (filter out outliers)
            if 0 < duration_sec < 3600:  # Assuming no epoch should take more than an hour
                self.epoch_durations.append(duration_sec)
        
        print(f"Calculated durations for {len(self.epoch_durations)} epochs.")

    def get_average_batch_duration(self) -> float:
        """
        Calculate the average validation batch duration across all epochs.
        
        Returns:
            Average batch duration in milliseconds
        """
        all_durations = []
        for epoch, durations in self.batch_durations.items():
            all_durations.extend(durations)
        
        if all_durations:
            avg_duration = statistics.mean(all_durations)
            print(f"Average validation batch duration: {avg_duration:.2f} ms")
            return avg_duration
        else:
            print("No valid batch durations found.")
            return 0.0

    def get_average_epoch_duration(self) -> float:
        """
        Calculate the average epoch duration.
        
        Returns:
            Average epoch duration in seconds
        """
        if self.epoch_durations:
            avg_duration = statistics.mean(self.epoch_durations)
            print(f"Average epoch duration: {avg_duration:.2f} seconds ({avg_duration/60:.2f} minutes)")
            return avg_duration
        else:
            print("No valid epoch durations found.")
            return 0.0

    def estimate_remaining_time(self, current_epoch: int) -> Tuple[float, str]:
        """
        Estimate the remaining training time based on the current epoch.
        
        Args:
            current_epoch: The current epoch number (1-based)
            
        Returns:
            Tuple of (remaining_time_seconds, formatted_time_string)
        """
        if not self.epoch_durations:
            return 0.0, "Unable to estimate: No epoch duration data available"
        
        avg_epoch_duration = statistics.mean(self.epoch_durations)
        remaining_epochs = self.total_epochs - current_epoch
        
        if remaining_epochs <= 0:
            return 0.0, "Training complete"
        
        remaining_seconds = remaining_epochs * avg_epoch_duration
        
        # Format the remaining time
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        time_str = ""
        if hours > 0:
            time_str += f"{int(hours)} hours "
        if minutes > 0 or hours > 0:
            time_str += f"{int(minutes)} minutes "
        time_str += f"{int(seconds)} seconds"
        
        print(f"Estimated remaining time: {time_str} ({remaining_epochs} epochs remaining)")
        return remaining_seconds, time_str

    def plot_batch_durations(self, output_file: Optional[str] = None) -> None:
        """
        Plot the distribution of validation batch durations.
        
        Args:
            output_file: Optional file path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        all_durations = []
        for epoch, durations in self.batch_durations.items():
            all_durations.extend(durations)
        
        if not all_durations:
            print("No batch durations to plot.")
            return
        
        plt.hist(all_durations, bins=30, alpha=0.7, color='blue')
        plt.axvline(statistics.mean(all_durations), color='red', linestyle='dashed', linewidth=1)
        
        plt.title('Distribution of Validation Batch Durations')
        plt.xlabel('Duration (ms)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()

    def plot_epoch_durations(self, output_file: Optional[str] = None) -> None:
        """
        Plot the epoch durations over time.
        
        Args:
            output_file: Optional file path to save the plot
        """
        if not self.epoch_durations:
            print("No epoch durations to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Convert to minutes for better readability
        durations_minutes = [d / 60 for d in self.epoch_durations]
        epochs = list(range(1, len(durations_minutes) + 1))
        
        plt.plot(epochs, durations_minutes, marker='o', linestyle='-', color='blue')
        plt.axhline(statistics.mean(durations_minutes), color='red', linestyle='dashed', linewidth=1)
        
        plt.title('Epoch Durations Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Duration (minutes)')
        plt.grid(True, alpha=0.3)
        plt.xticks(epochs)
        
        if output_file:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()

    def generate_timing_report(self, output_file: Optional[str] = None) -> None:
        """
        Generate a comprehensive timing report.
        
        Args:
            output_file: Optional file path to save the report
        """
        # Calculate all timing metrics
        self.parse_log_file()
        self.calculate_batch_durations()
        self.calculate_epoch_durations()
        
        avg_batch_ms = self.get_average_batch_duration()
        avg_epoch_sec = self.get_average_epoch_duration()
        
        # Prepare the report content
        report = [
            "CIFAR10 Training Timing Report",
            "============================",
            f"Log file: {self.log_file}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Summary",
            "-------",
            f"Total epochs found: {len(self.epoch_timestamps)}",
            f"Total validation batches per epoch: {self.total_batches_per_epoch}",
            "",
            "Timing Metrics",
            "-------------",
            f"Average validation batch duration: {avg_batch_ms:.2f} ms",
            f"Average epoch duration: {avg_epoch_sec:.2f} seconds ({avg_epoch_sec/60:.2f} minutes)",
            f"Estimated full training time (10 epochs): {avg_epoch_sec*10/60:.2f} minutes",
            "",
            "Epoch-by-Epoch Breakdown",
            "----------------------"
        ]
        
        # Add epoch-by-epoch breakdown
        for epoch in sorted(self.batch_durations.keys()):
            durations = self.batch_durations[epoch]
            if durations:
                avg_duration = statistics.mean(durations)
                report.append(f"Epoch {epoch}: {len(durations)} batches, avg {avg_duration:.2f} ms per batch")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        else:
            print(report_text)
        
        return report_text

    def run_analysis(self, output_dir: Optional[str] = None) -> None:
        """
        Run the complete analysis pipeline.
        
        Args:
            output_dir: Optional directory to save output files
        """
        self.parse_log_file()
        self.calculate_batch_durations()
        self.calculate_epoch_durations()
        
        # Generate report
        report_file = os.path.join(output_dir, "cifar10_timing_report.txt") if output_dir else None
        self.generate_timing_report(report_file)
        
        # Generate plots
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            batch_plot_file = os.path.join(output_dir, "batch_durations.png")
            epoch_plot_file = os.path.join(output_dir, "epoch_durations.png")
            self.plot_batch_durations(batch_plot_file)
            self.plot_epoch_durations(epoch_plot_file)
        else:
            self.plot_batch_durations()
            self.plot_epoch_durations()


def main():
    parser = argparse.ArgumentParser(description='Analyze CIFAR10 training logs for timing information')
    parser.add_argument('log_file', help='Path to the CIFAR10 training log file')
    parser.add_argument('--output-dir', '-o', help='Directory to save output files')
    parser.add_argument('--current-epoch', '-e', type=int, help='Current epoch number for remaining time estimation')
    args = parser.parse_args()
    
    analyzer = CIFAR10TimingAnalyzer(args.log_file)
    
    if args.current_epoch:
        # Just estimate remaining time
        analyzer.parse_log_file()
        analyzer.calculate_epoch_durations()
        analyzer.estimate_remaining_time(args.current_epoch)
    else:
        # Run full analysis
        analyzer.run_analysis(args.output_dir)


if __name__ == "__main__":
    main()
