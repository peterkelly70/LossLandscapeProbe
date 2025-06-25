#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

REPORTS_DIR="reports"

# Check if reports directory exists
if [ ! -d "$REPORTS_DIR" ]; then
    echo -e "${YELLOW}Reports directory does not exist: $REPORTS_DIR${NC}"
    exit 0
fi

echo -e "${YELLOW}=== Clearing Reports ===${NC}"
echo -e "This will remove all files in $REPORTS_DIR/"
echo -e "${RED}WARNING: This action cannot be undone!${NC}"

# List all subdirectories that will be affected
echo -e "\nThe following directories will be cleared:"
find "$REPORTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort

# Ask for confirmation
read -p "Are you sure you want to continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Operation cancelled.${NC}"
    exit 0
fi

# Clear the reports
echo -e "\n${YELLOW}Clearing reports...${NC}"

# Find and remove all files in reports subdirectories, but keep the directory structure
find "$REPORTS_DIR" -mindepth 2 -type f -exec rm -f {} \;

# Count remaining files
REMAINING_FILES=$(find "$REPORTS_DIR" -type f | wc -l)

if [ "$REMAINING_FILES" -eq 0 ]; then
    echo -e "${GREEN}✓ All report files have been removed.${NC}"
else
    echo -e "${YELLOW}Warning: $REMAINING_FILES files could not be removed.${NC}"
fi

echo -e "${GREEN}Done.${NC}"
