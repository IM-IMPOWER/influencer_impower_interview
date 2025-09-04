#!/bin/bash
# AIDEV-NOTE: Stop script for KOL scraper service
# Gracefully stops the service with proper cleanup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="KOL Scraper"
LOG_DIR="logs"
PID_FILE="$LOG_DIR/kol-scraper.pid"

echo -e "${YELLOW}Stopping $SERVICE_NAME...${NC}"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}PID file not found. Service may not be running.${NC}"
    
    # Try to find the process anyway
    PID=$(pgrep -f "kol-scraper" 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo -e "${YELLOW}Found running process with PID $PID${NC}"
    else
        echo -e "${GREEN}Service is not running.${NC}"
        exit 0
    fi
else
    PID=$(cat "$PID_FILE")
fi

# Check if process is actually running
if ! kill -0 "$PID" 2>/dev/null; then
    echo -e "${YELLOW}Process with PID $PID is not running. Cleaning up PID file.${NC}"
    rm -f "$PID_FILE"
    echo -e "${GREEN}Cleanup completed.${NC}"
    exit 0
fi

echo -e "${YELLOW}Found running service with PID $PID${NC}"

# Function to check if process is still running
is_running() {
    kill -0 "$1" 2>/dev/null
}

# Attempt graceful shutdown
echo -e "${YELLOW}Sending TERM signal for graceful shutdown...${NC}"
kill -TERM "$PID"

# Wait for graceful shutdown (up to 30 seconds)
echo -e "${YELLOW}Waiting for graceful shutdown...${NC}"
count=0
while is_running "$PID" && [ $count -lt 30 ]; do
    echo -n "."
    sleep 1
    ((count++))
done

if is_running "$PID"; then
    echo -e "\n${YELLOW}Graceful shutdown timed out. Sending KILL signal...${NC}"
    kill -KILL "$PID"
    
    # Wait a bit more for the process to die
    count=0
    while is_running "$PID" && [ $count -lt 5 ]; do
        sleep 1
        ((count++))
    done
    
    if is_running "$PID"; then
        echo -e "${RED}Failed to kill process $PID${NC}"
        exit 1
    else
        echo -e "${YELLOW}Process killed forcefully.${NC}"
    fi
else
    echo -e "\n${GREEN}Service stopped gracefully.${NC}"
fi

# Clean up PID file
if [ -f "$PID_FILE" ]; then
    rm -f "$PID_FILE"
    echo -e "${YELLOW}Cleaned up PID file.${NC}"
fi

# Optional: Clean up log files if requested
if [ "$1" = "--clean-logs" ]; then
    echo -e "${YELLOW}Cleaning up log files...${NC}"
    rm -f "$LOG_DIR"/*.log
    echo -e "${GREEN}Log files cleaned.${NC}"
fi

# Show final status
echo -e "${GREEN}$SERVICE_NAME stopped successfully!${NC}"

# Check if any related processes are still running
REMAINING=$(pgrep -f "kol-scraper" 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo -e "${YELLOW}Warning: Found remaining processes:${NC}"
    pgrep -f "kol-scraper" -l
    echo -e "${YELLOW}You may need to stop them manually.${NC}"
fi