#!/bin/bash

# Script to update CIFAR-100 reports with proper class names
# This script copies the updated visible_progress.py to all CIFAR-100 report directories
# and regenerates the reports with the correct class names

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Updating CIFAR-100 Reports with Class Names ===${NC}"

# Base directory for reports
REPORTS_DIR="reports/cifar100"

# Check if the reports directory exists
if [ ! -d "$REPORTS_DIR" ]; then
    echo -e "${YELLOW}CIFAR-100 reports directory not found: $REPORTS_DIR${NC}"
    exit 1
fi

# Find all CIFAR-100 report directories
echo "Finding CIFAR-100 report directories..."
REPORT_DIRS=$(find "$REPORTS_DIR" -type d -name "cifar100_*")

if [ -z "$REPORT_DIRS" ]; then
    echo -e "${YELLOW}No CIFAR-100 report directories found${NC}"
    exit 1
fi

# Copy the updated visible_progress.py to each report directory and regenerate reports
for dir in $REPORT_DIRS; do
    echo -e "${GREEN}Updating reports in: $dir${NC}"
    
    # Copy the updated visible_progress.py
    cp visible_progress.py "$dir/"
    
    # Change to the report directory
    cd "$dir"
    
    # Get the sample size from directory name
    SAMPLE_SIZE=$(basename "$dir" | cut -d'_' -f2)
    
    # Run the visible_progress.py script to regenerate the report with class names
    echo "Regenerating report with class names for sample size $SAMPLE_SIZE%..."
    python visible_progress.py --dataset cifar100 --sample-size "$SAMPLE_SIZE"
    
    # Return to the original directory
    cd - > /dev/null
done

echo -e "${GREEN}=== CIFAR-100 Reports Updated Successfully ===${NC}"
echo "The reports now display actual class names instead of class numbers."
echo ""
echo "To view the updated reports, start the server using one of these methods:"
echo "  1. Run 'python server.py start' in the project root"
echo "  2. Or run 'cd website && ./server.sh start' for the website server"
echo "Then open http://localhost:8090 in your browser."
