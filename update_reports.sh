#!/bin/bash

# Script to update existing reports with fixed accuracy calculations and CIFAR-100 class names
# This script will copy the updated visible_progress.py to each report directory and regenerate the reports

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Updating Reports with Fixed Accuracy Calculations and CIFAR-100 Class Names ===${NC}"

# Copy the updated visible_progress.py to each report directory
echo -e "\n${GREEN}Copying updated visible_progress.py to report directories...${NC}"

# CIFAR-10 reports
for size in 10 20 30 40 multi; do
    report_dir="reports/cifar10/cifar10_${size}"
    if [ -d "$report_dir" ]; then
        echo -e "Updating ${YELLOW}$report_dir${NC}..."
        cp visible_progress.py "$report_dir/"
        
        # Regenerate the report if the directory exists
        if [ -d "$report_dir" ]; then
            echo -e "Regenerating report for ${YELLOW}cifar10_${size}${NC}..."
            cd "$report_dir"
            python visible_progress.py --dataset cifar10 --sample-size ${size} --generate-report
            cd - > /dev/null
        fi
    fi
done

# CIFAR-100 reports
for size in 10 20 30 40 multi; do
    report_dir="reports/cifar100/cifar100_${size}"
    if [ -d "$report_dir" ]; then
        echo -e "Updating ${YELLOW}$report_dir${NC}..."
        cp visible_progress.py "$report_dir/"
        
        # Regenerate the report if the directory exists
        if [ -d "$report_dir" ]; then
            echo -e "Regenerating report for ${YELLOW}cifar100_${size}${NC}..."
            cd "$report_dir"
            python visible_progress.py --dataset cifar100 --sample-size ${size} --generate-report
            cd - > /dev/null
        fi
    fi
done

echo -e "\n${GREEN}✓ All reports have been updated with fixed accuracy calculations and CIFAR-100 class names.${NC}"
echo -e "${YELLOW}Note: The reports now show:${NC}"
echo -e "${YELLOW}  - Per-class Average: The mean of individual class accuracies (red line)${NC}"
echo -e "${YELLOW}  - Overall Test Accuracy: Total correct predictions / total predictions (green line)${NC}"
echo -e "${YELLOW}  - CIFAR-100 reports now display proper class names instead of numbers${NC}"

echo -e "\n${GREEN}To view the updated reports:${NC}"
echo -e "  1. Run ${YELLOW}python server.py start${NC} in the project root"
echo -e "  2. Or run ${YELLOW}cd website && ./server.sh start${NC} for the website server"
echo -e "  3. Then open ${YELLOW}http://localhost:8090${NC} in your browser"
