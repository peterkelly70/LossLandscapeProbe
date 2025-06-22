#!/bin/bash

# Colors for better UI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Function to display the menu
show_menu() {
    clear
    echo -e "${YELLOW}==================================${NC}"
    echo -e "${YELLOW}  LossLandscapeProbe Server Manager  ${NC}"
    echo -e "${YELLOW}==================================${NC}"
    
    # Show server status
    echo -e "\n${YELLOW}Server Status:${NC}"
    
    # Change to project root directory to check status
    cd "$SCRIPT_DIR/.."
    source "$SCRIPT_DIR/../.llp/bin/activate" 2>/dev/null
    
    # Check server status using the Python script
    if python -m website.server status 2>/dev/null | grep -q "Server is running"; then
        STATUS_INFO=$(python -m website.server status 2>/dev/null)
        PID=$(echo "$STATUS_INFO" | grep -o "PID: [0-9]*" | cut -d' ' -f2)
        PORT=$(echo "$STATUS_INFO" | grep -o "Port: [0-9]*" | cut -d' ' -f2)
        echo -e "  ${GREEN}●${NC} Running (PID: $PID, Port: $PORT)"
    else
        echo -e "  ${RED}●${NC} Stopped"
    fi
    
    # Return to script directory
    cd "$SCRIPT_DIR"
    
    # Show menu options
    echo -e "\n${YELLOW}Options:${NC}"
    echo "  1) Start Server"
    echo "  2) Stop Server"
    echo "  3) Restart Server"
    echo "  4) View Logs"
    echo "  5) Open in Browser"
    echo "  6) Check Status"
    echo -e "  ${YELLOW}q) Quit${NC}"
}

# Function to start the server
start_server() {
    echo -e "\n${YELLOW}Starting server...${NC}"
    # Activate virtual environment and start server
    source "$SCRIPT_DIR/../.llp/bin/activate"
    
    # Change to project root directory to ensure proper module imports
    cd "$SCRIPT_DIR/.."
    python -m website.server start
    
    # Check if server started successfully
    if python -m website.server status | grep -q "Server is running"; then
        PORT=$(python -m website.server status | grep -o "Port: [0-9]*" | cut -d' ' -f2)
        echo -e "${GREEN}Server started on port ${PORT}!${NC} Check logs at $SCRIPT_DIR/../instance/server.log"
    else
        echo -e "${RED}Failed to start server. Check logs at $SCRIPT_DIR/../instance/server.log${NC}"
    fi
    
    # Return to the original directory
    cd "$SCRIPT_DIR"
    sleep 2
}

# Function to stop the server
stop_server() {
    echo -e "\n${YELLOW}Stopping server...${NC}"
    source "$SCRIPT_DIR/../.llp/bin/activate"
    
    # Change to project root directory to ensure proper module imports
    cd "$SCRIPT_DIR/.."
    python -m website.server stop
    
    # Verify server was stopped
    if ! python -m website.server status | grep -q "Server is running"; then
        echo -e "${GREEN}Server stopped successfully.${NC}"
    else
        echo -e "${RED}Failed to stop server. Check logs at $SCRIPT_DIR/../instance/server.log${NC}"
    fi
    
    # Return to the original directory
    cd "$SCRIPT_DIR"
    sleep 1
}

# Function to view logs
view_logs() {
    echo -e "\n${YELLOW}Viewing logs (Press 'q' to return to menu)...${NC}"
    LOG_FILE="$SCRIPT_DIR/../instance/server.log"
    if [ -f "$LOG_FILE" ]; then
        # Use less with line numbers and quit-if-one-screen
        less -N +F "$LOG_FILE"
    else
        echo -e "${RED}Log file not found at $LOG_FILE${NC}"
        read -n 1 -s -r -p "Press any key to continue..."
    fi
}

# Function to open in browser
open_browser() {
    source "$SCRIPT_DIR/../.llp/bin/activate"
    
    # Change to project root directory to ensure proper module imports
    cd "$SCRIPT_DIR/.."
    if python -m website.server status | grep -q "Server is running"; then
        PORT=$(python -m website.server status | grep -o "Port: [0-9]*" | cut -d' ' -f2)
        URL="http://localhost:${PORT:-8090}"
        echo -e "\n${YELLOW}Opening $URL in your default browser...${NC}"
        xdg-open "$URL" 2>/dev/null || open "$URL" 2>/dev/null || echo -e "${RED}Could not open browser. Please open manually: $URL${NC}"
    else
        echo -e "${RED}Server is not running!${NC}"
        sleep 1
    fi
    
    # Return to the original directory
    cd "$SCRIPT_DIR"
}

# Main menu loop
while true; do
    show_menu
    read -p "Enter your choice: " choice
    
    case $choice in
        1) start_server ;;
        2) stop_server ;;
        3) 
            stop_server
            start_server
            ;;
        4) view_logs ;;
        5) open_browser ;;
        6) 
            echo -e "\n${YELLOW}Checking server status...${NC}"
            source "$SCRIPT_DIR/../.llp/bin/activate"
            
            # Change to project root directory to ensure proper module imports
            cd "$SCRIPT_DIR/.."
            python -m website.server status
            
            # Return to the original directory
            cd "$SCRIPT_DIR"
            read -n 1 -s -r -p "Press any key to continue..."
            ;;
        q|Q) 
            echo -e "\n${YELLOW}Exiting...${NC}"
            exit 0
            ;;
        *) 
            echo -e "\n${RED}Invalid option!${NC}"
            sleep 1
            ;;
    esac
done
