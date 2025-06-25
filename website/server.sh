#!/bin/bash

# Configuration
PORT=8090
APP="app.py"
PID_FILE="./flask_server.pid"
LOG_FILE="./flask_server.log"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to get the process ID of the running server
get_pid() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE" 2>/dev/null)
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "$pid"
            return 0
        fi
        # Remove stale PID file
        rm -f "$PID_FILE"
    fi
    echo ""
    return 1
}

# Function to check and install Flask
check_flask() {
    if ! python3 -c "import flask" >/dev/null 2>&1; then
        echo -e "${YELLOW}Flask is not installed. Installing...${NC}"
        if ! python3 -m pip install --user flask; then
            echo -e "${RED}Failed to install Flask. Please install it manually with:${NC}"
            echo -e "  python3 -m pip install --user flask"
            return 1
        fi
        echo -e "${GREEN}Flask installed successfully!${NC}"
    fi
    return 0
}

# Function to start the server
start_server() {
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Server is already running (PID: $pid)${NC}"
        return 1
    fi
    
    # Check if Flask is installed
    if ! check_flask; then
        return 1
    fi
    
    echo -e "${GREEN}Starting Flask server on port $PORT...${NC}"
    export FLASK_APP="$APP"
    nohup python3 -m flask run --port=$PORT --host=0.0.0.0 > "$LOG_FILE" 2>&1 & echo $! > "$PID_FILE"
    
    sleep 2  # Give it a moment to start
    
    if [ -n "$(get_pid)" ]; then
        echo -e "${GREEN}Server started successfully (PID: $(cat $PID_FILE))${NC}"
        echo -e "Logs are being written to: $LOG_FILE"
    else
        echo -e "${RED}Failed to start server. Check $LOG_FILE for details.${NC}"
        return 1
    fi
}

# Function to stop the server
stop_server() {
    local pid=$(get_pid)
    if [ -z "$pid" ]; then
        echo -e "${YELLOW}Server is not running${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Stopping Flask server (PID: $pid)...${NC}"
    kill -TERM "$pid"
    
    # Wait for the process to terminate
    local count=0
    while [ -n "$(get_pid)" ] && [ "$count" -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done
    
    if [ -n "$(get_pid)" ]; then
        echo -e "${RED}Force stopping server...${NC}"
        kill -9 "$pid"
        sleep 1
    fi
    
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
    fi
    
    echo -e "${GREEN}Server stopped${NC}"
}

# Function to show server status
status_server() {
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        echo -e "${GREEN}Server is running (PID: $pid)${NC}"
        echo -e "Process info:"
        ps -p "$pid" -o pid,ppid,user,%cpu,%mem,cmd
    else
        echo -e "${RED}Server is not running${NC}"
    fi
}

# Function to show interactive menu
show_menu() {
    clear
    echo -e "\n${YELLOW}=== Flask Server Manager ===${NC}\n"
    
    # Show current status
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        echo -e "${GREEN}● Server is running (PID: $pid)${NC}"
    else
        echo -e "${RED}● Server is not running${NC}"
    fi
    
    echo -e "\n${YELLOW}Please choose an option:${NC}"
    if [ -n "$pid" ]; then
        echo -e "  ${GREEN}1${NC}) Stop server"
        echo -e "  ${GREEN}2${NC}) Restart server"
    else
        echo -e "  ${GREEN}1${NC}) Start server"
    fi
    echo -e "  ${YELLOW}s${NC}) Show server status"
    echo -e "  ${YELLOW}l${NC}) View server logs"
    echo -e "  ${RED}q${NC}) Quit\n"
    
    read -p "Enter your choice: " choice
    
    case $choice in
        1)
            if [ -n "$pid" ]; then
                stop_server
            else
                start_server
            fi
            ;;
        2)
            if [ -n "$pid" ]; then
                stop_server
                start_server
            else
                echo -e "${YELLOW}Server is not running. Starting it now...${NC}"
                start_server
            fi
            ;;
        s|S)
            status_server
            ;;
        l|L)
            if [ -f "$LOG_FILE" ]; then
                echo -e "\n${YELLOW}=== Server Logs (last 20 lines) ===${NC}\n"
                tail -n 20 "$LOG_FILE"
            else
                echo -e "${YELLOW}No log file found.${NC}"
            fi
            ;;
        q|Q)
            echo -e "\n${YELLOW}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "\n${RED}Invalid choice. Please try again.${NC}"
            sleep 1
            return 1
            ;;
    esac
    
    echo -e "\n${YELLOW}Press any key to continue...${NC}"
    read -n 1 -s
    return 0
}

# Main script
if [ $# -eq 0 ]; then
    # Interactive mode
    while true; do
        show_menu
    done
else
    # Command-line mode
    case "$1" in
        start)
            start_server
            ;;
        stop)
            stop_server
            ;;
        restart)
            stop_server
            start_server
            ;;
        status)
            status_server
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status}"
            echo "       $0 (with no arguments for interactive mode)"
            exit 1
            ;;
    esac
fi

exit 0
