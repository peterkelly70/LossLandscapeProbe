#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Setting up LLP Development Environment ===${NC}\n"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python 3 is required but not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}pip3 is required but not installed. Please install pip3.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".llp" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3 -m venv .llp
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Failed to create virtual environment.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Virtual environment created at $(pwd)/.llp${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists at $(pwd)/.llp${NC}"
fi

# Activate the virtual environment
echo -e "\n${GREEN}Activating virtual environment and installing dependencies...${NC}"
source .llp/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "\n${YELLOW}Installing requirements from requirements.txt...${NC}"
    pip install -r requirements.txt
else
    # Install basic requirements
    echo -e "\n${YELLOW}Installing Flask...${NC}"
    pip install flask
    
    # Create a basic requirements.txt
    echo -e "\n${YELLOW}Creating requirements.txt...${NC}"
    pip freeze > requirements.txt
fi

echo -e "\n${GREEN}=== Setup Complete! ===${NC}"
echo -e "\nTo activate the environment in the future, run:"
echo -e "  ${YELLOW}source $(pwd)/.llp/bin/activate${NC}"
echo -e "\nTo deactivate the environment, simply run:"
echo -e "  ${YELLOW}deactivate${NC}"
echo -e "\nTo start the development server:"
echo -e "  ${YELLOW}cd website && ./server.sh${NC}\n"
