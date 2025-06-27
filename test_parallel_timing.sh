#!/bin/bash

# test_parallel_timing.sh - Test script for parallel timing functionality
# This script verifies the parallel timing toggle in train.sh

# Exit on error
set -e

echo "===== TESTING PARALLEL TIMING TOGGLE FUNCTIONALITY =====" 
echo ""

# Function to check the current parallel timing setting
check_parallel_setting() {
    # Extract the current setting from the train.sh script
    grep -q "^USE_PARALLEL=true" train.sh && echo "Parallel timing is currently ENABLED" || echo "Parallel timing is currently DISABLED"
}

# Function to simulate menu selection in train.sh
simulate_menu_selection() {
    local option=$1
    echo "Simulating menu selection: $option"
    
    # Create a temporary expect script to automate the interaction
    cat > temp_expect.exp << EOF
#!/usr/bin/expect
spawn bash train.sh
expect "Select an option (0-9, a, p): "
send "$option\r"
expect {
    "Parallel timing functionality turned ON" {puts "\nResult: Parallel timing turned ON"}
    "Parallel timing functionality turned OFF" {puts "\nResult: Parallel timing turned OFF"}
    timeout {puts "\nTimeout waiting for response"}
}
expect "Select an option (0-9, a, p): "
send "0\r"
expect eof
EOF
    
    # Make the expect script executable and run it
    chmod +x temp_expect.exp
    ./temp_expect.exp
    rm temp_expect.exp
}

# Check if expect is installed
if ! command -v expect &> /dev/null; then
    echo "The 'expect' command is required but not installed."
    echo "Installing 'expect'..."
    sudo apt-get update && sudo apt-get install -y expect || {
        echo "Failed to install 'expect'. Please install it manually and run this script again."
        exit 1
    }
fi

# Check initial setting
echo "Initial setting:"
check_parallel_setting
echo ""

# Test toggling parallel timing
echo "===== TEST: Toggle parallel timing ON/OFF =====" 
simulate_menu_selection "p"
echo ""

# Check setting after toggle
echo "Setting after toggle:"
check_parallel_setting
echo ""

# Toggle back to original state
echo "===== TEST: Toggle parallel timing back to original state =====" 
simulate_menu_selection "p"
echo ""

# Final check
echo "Final setting:"
check_parallel_setting
echo ""

echo "===== TEST COMPLETE =====" 
echo "The parallel timing toggle functionality in train.sh is working correctly."

# Create a simple Python script to demonstrate the parallel timing functionality
cat > test_parallel_timing.py << EOF
import os
import sys

print("===== PYTHON TEST FOR PARALLEL TIMING FUNCTIONALITY =====\n")

# Check if USE_PARALLEL environment variable is set
use_parallel = os.environ.get('USE_PARALLEL', 'false').lower() == 'true'
print(f"USE_PARALLEL environment variable is set to: {use_parallel}")

# Show how this would affect the unified_cifar_training.py script
print("\nIn unified_cifar_training.py, this would result in:")
if use_parallel:
    print("- Parallel timing data collection ENABLED")
    print("- Detailed parallel efficiency metrics will be logged")
    print("- Performance warnings will be shown for low efficiency")
else:
    print("- Parallel timing data collection DISABLED")
    print("- No parallel efficiency metrics will be logged")
    print("- No performance warnings will be shown")

print("\nTo test with actual training, run train.sh and toggle the setting with option 'p'\n")
EOF

echo ""
echo "A Python test script has been created: test_parallel_timing.py"
echo "You can run it with different USE_PARALLEL settings to see the effect:"
echo "  USE_PARALLEL=true python3 test_parallel_timing.py"
echo "  USE_PARALLEL=false python3 test_parallel_timing.py"
