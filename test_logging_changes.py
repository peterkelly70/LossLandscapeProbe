#!/usr/bin/env python3
"""
Test script to verify the simplified parallel timing logging changes.
This script mocks the parallel timing data and logs it using the modified format.
"""

import os
import sys
import logging
import time
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_logging")

# Mock the parallel timing info that would be returned from meta_model_optimizer
def mock_get_parallel_timing_info(include_details=False) -> Dict[str, Any]:
    """
    Mock function to simulate the get_parallel_timing_info method with simplified output.
    
    Args:
        include_details: Whether to include detailed batch timing data
        
    Returns:
        Dictionary containing mock parallel timing metrics
    """
    # Basic timing info that's always included
    timing_info = {
        'parallel_mode': 'gpu_data_parallel',
        'num_workers': 4,
        'efficiency': 0.85,
        'speedup': 3.4,
        'timing_variance': 0.12,
        'total_runtime': 120.5
    }
    
    # Add detailed batch timing data if requested (old format)
    if include_details:
        timing_info.update({
            'batch_time_mean': 0.023,
            'batch_time_min': 0.018,
            'batch_time_max': 0.035
        })
    
    return timing_info

def test_logging_format():
    """Test both the old and new logging formats for comparison"""
    logger.info("\n=== Testing Original Verbose Logging Format ===")
    parallel_info = mock_get_parallel_timing_info(include_details=True)
    
    # Original verbose logging format
    logger.info("\nParallel Execution Metrics:")
    logger.info(f"  - Parallel Mode: {parallel_info.get('parallel_mode', 'Not detected')}")
    
    if 'efficiency' in parallel_info:
        efficiency = parallel_info['efficiency']
        logger.info(f"  - Parallel Efficiency: {efficiency:.2%}")
        
        if efficiency < 0.5:
            logger.warning("  - Low parallel efficiency detected. Consider reducing parallel workers.")
        elif efficiency > 0.85:
            logger.info("  - Good parallel scaling achieved.")
    
    if 'speedup' in parallel_info:
        logger.info(f"  - Estimated Speedup: {parallel_info['speedup']:.2f}x")
        
    if 'timing_variance' in parallel_info:
        logger.info(f"  - Timing Variance: {parallel_info['timing_variance']:.2%}")
        if parallel_info['timing_variance'] > 0.25:
            logger.warning("  - High variance in execution times. Check for resource contention.")
            
    # Log detailed batch timing data in original format
    if 'batch_time_mean' in parallel_info:
        logger.info(f"  - Mean Batch Time: {parallel_info['batch_time_mean']:.4f}s")
    if 'batch_time_min' in parallel_info:
        logger.info(f"  - Min Batch Time: {parallel_info['batch_time_min']:.4f}s")
    if 'batch_time_max' in parallel_info:
        logger.info(f"  - Max Batch Time: {parallel_info['batch_time_max']:.4f}s")
    
    # Wait a moment to separate the outputs
    time.sleep(1)
    
    # Test the new simplified logging format
    logger.info("\n=== Testing New Simplified Logging Format ===")
    parallel_info = mock_get_parallel_timing_info(include_details=False)
    
    # New simplified logging format
    logger.info("\nParallel Execution Summary:")
    logger.info(f"  - Mode: {parallel_info.get('parallel_mode', 'Not detected')} | Workers: {parallel_info.get('num_workers', 1)}")
    
    if 'efficiency' in parallel_info and 'speedup' in parallel_info:
        efficiency = parallel_info['efficiency']
        logger.info(f"  - Efficiency: {efficiency:.2%} | Speedup: {parallel_info['speedup']:.2f}x")
        
        # Only show warnings for very poor efficiency
        if efficiency < 0.4:
            logger.warning("  - Low efficiency detected. Consider reducing parallel workers.")
    
    # Only log timing variance if it's problematic
    if 'timing_variance' in parallel_info and parallel_info['timing_variance'] > 0.3:
        logger.warning(f"  - High timing variance: {parallel_info['timing_variance']:.2%}")
    
    # Note: detailed batch timing data is intentionally omitted in the new format

if __name__ == "__main__":
    logger.info("Testing parallel timing logging formats")
    test_logging_format()
    logger.info("Testing complete")
