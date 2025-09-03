#!/bin/bash
# AIDEV-NOTE: Startup script for KOL scraper service
# Handles environment setup, health checks, and graceful startup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="KOL Scraper"
BINARY_NAME="kol-scraper"
CONFIG_FILE="config/config.yaml"
LOG_DIR="logs"
PID_FILE="$LOG_DIR/kol-scraper.pid"

# Create log directory
mkdir -p $LOG_DIR

echo -e "${BLUE}Starting $SERVICE_NAME...${NC}"

# Check if service is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo -e "${YELLOW}Service is already running with PID $OLD_PID${NC}"
        echo "Use 'scripts/stop.sh' to stop the service first."
        exit 1
    else
        echo -e "${YELLOW}Removing stale PID file...${NC}"
        rm -f "$PID_FILE"
    fi
fi

# Check if binary exists
if [ ! -f "./$BINARY_NAME" ] && [ ! -f "./build/$BINARY_NAME" ]; then
    echo -e "${RED}Binary not found. Please build the service first using 'scripts/build.sh'${NC}"
    exit 1
fi

# Determine binary path
BINARY_PATH="./$BINARY_NAME"
if [ ! -f "$BINARY_PATH" ]; then
    BINARY_PATH="./build/$BINARY_NAME"
fi

echo -e "${YELLOW}Using binary: $BINARY_PATH${NC}"

# Check configuration file
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}Using config file: $CONFIG_FILE${NC}"
else
    echo -e "${YELLOW}Config file not found, using environment variables${NC}"
fi

# Set default environment variables if not set
export ENVIRONMENT=${ENVIRONMENT:-"development"}
export PORT=${PORT:-"8080"}
export LOG_LEVEL=${LOG_LEVEL:-"info"}
export MAX_CONCURRENT_SCRAPE=${MAX_CONCURRENT_SCRAPE:-"10"}
export QUEUE_WORKERS=${QUEUE_WORKERS:-"5"}

# Validate required environment variables
REQUIRED_VARS=("DATABASE_URL" "REDIS_URL")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "${RED}Missing required environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo -e "  - $var"
    done
    echo -e "${YELLOW}Please set these variables or use docker-compose for local development.${NC}"
    exit 1
fi

# Function to check service health
check_health() {
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}Waiting for service to become healthy...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$PORT/api/v1/health" > /dev/null 2>&1; then
            echo -e "${GREEN}Service is healthy!${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo -e "\n${RED}Service failed to become healthy after $max_attempts attempts${NC}"
    return 1
}

# Function to handle cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}Stopping service (PID: $PID)...${NC}"
            kill -TERM "$PID"
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 "$PID" 2>/dev/null && [ $count -lt 30 ]; do
                sleep 1
                ((count++))
            done
            
            # Force kill if necessary
            if kill -0 "$PID" 2>/dev/null; then
                echo -e "${RED}Force killing service...${NC}"
                kill -KILL "$PID"
            fi
        fi
        rm -f "$PID_FILE"
    fi
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGINT SIGTERM EXIT

# Check if running in background mode
BACKGROUND_MODE=false
if [ "$1" = "--background" ] || [ "$1" = "-d" ]; then
    BACKGROUND_MODE=true
fi

# Start the service
echo -e "${YELLOW}Starting service on port $PORT...${NC}"
echo -e "${YELLOW}Environment: $ENVIRONMENT${NC}"
echo -e "${YELLOW}Log level: $LOG_LEVEL${NC}"

if [ "$BACKGROUND_MODE" = true ]; then
    # Start in background
    nohup "$BINARY_PATH" > "$LOG_DIR/service.log" 2>&1 &
    SERVICE_PID=$!
    echo $SERVICE_PID > "$PID_FILE"
    
    echo -e "${GREEN}Service started in background with PID $SERVICE_PID${NC}"
    echo -e "${YELLOW}Log file: $LOG_DIR/service.log${NC}"
    echo -e "${YELLOW}PID file: $PID_FILE${NC}"
    
    # Check health
    if check_health; then
        echo -e "${GREEN}$SERVICE_NAME started successfully!${NC}"
        echo -e "${BLUE}API available at: http://localhost:$PORT${NC}"
        echo -e "${BLUE}Health check: http://localhost:$PORT/api/v1/health${NC}"
        echo -e "${BLUE}Metrics: http://localhost:$PORT/api/v1/metrics${NC}"
        
        # Don't run cleanup on successful background start
        trap - SIGINT SIGTERM EXIT
    else
        echo -e "${RED}Failed to start service${NC}"
        cleanup
        exit 1
    fi
else
    # Start in foreground
    echo -e "${YELLOW}Starting in foreground mode (Ctrl+C to stop)...${NC}"
    "$BINARY_PATH" &
    SERVICE_PID=$!
    echo $SERVICE_PID > "$PID_FILE"
    
    # Check health
    if check_health; then
        echo -e "${GREEN}$SERVICE_NAME started successfully!${NC}"
        echo -e "${BLUE}API available at: http://localhost:$PORT${NC}"
        echo -e "${BLUE}Health check: http://localhost:$PORT/api/v1/health${NC}"
        echo -e "${BLUE}Press Ctrl+C to stop the service${NC}"
        
        # Wait for the service process
        wait $SERVICE_PID
    else
        echo -e "${RED}Failed to start service${NC}"
        exit 1
    fi
fi