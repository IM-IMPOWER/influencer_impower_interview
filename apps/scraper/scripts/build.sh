#!/bin/bash
# AIDEV-NOTE: Build script for KOL scraper service
# Builds the Go binary with optimizations and runs tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building KOL Scraper Service...${NC}"

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo -e "${RED}Go is not installed. Please install Go 1.21 or later.${NC}"
    exit 1
fi

# Check Go version
GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
REQUIRED_VERSION="1.21"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$GO_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Go version $GO_VERSION is not supported. Please install Go $REQUIRED_VERSION or later.${NC}"
    exit 1
fi

# Set build variables
BUILD_DIR="./build"
BINARY_NAME="kol-scraper"
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Build flags
LDFLAGS="-w -s -X main.Version=$VERSION -X main.BuildTime=$BUILD_TIME -X main.CommitHash=$COMMIT_HASH"

echo -e "${YELLOW}Version: $VERSION${NC}"
echo -e "${YELLOW}Build Time: $BUILD_TIME${NC}"
echo -e "${YELLOW}Commit: $COMMIT_HASH${NC}"

# Create build directory
mkdir -p $BUILD_DIR

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -f $BUILD_DIR/$BINARY_NAME*

# Run tests first
echo -e "${YELLOW}Running tests...${NC}"
go test -v ./... || {
    echo -e "${RED}Tests failed. Build aborted.${NC}"
    exit 1
}

# Run linting if available
if command -v golangci-lint &> /dev/null; then
    echo -e "${YELLOW}Running linting...${NC}"
    golangci-lint run ./... || {
        echo -e "${RED}Linting failed. Build aborted.${NC}"
        exit 1
    }
fi

# Build for different platforms
echo -e "${YELLOW}Building binaries...${NC}"

# Linux AMD64
echo -e "${YELLOW}Building for Linux AMD64...${NC}"
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="$LDFLAGS" \
    -o "$BUILD_DIR/${BINARY_NAME}-linux-amd64" \
    ./main.go

# Linux ARM64
echo -e "${YELLOW}Building for Linux ARM64...${NC}"
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build \
    -ldflags="$LDFLAGS" \
    -o "$BUILD_DIR/${BINARY_NAME}-linux-arm64" \
    ./main.go

# Darwin AMD64 (macOS Intel)
echo -e "${YELLOW}Building for macOS AMD64...${NC}"
CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build \
    -ldflags="$LDFLAGS" \
    -o "$BUILD_DIR/${BINARY_NAME}-darwin-amd64" \
    ./main.go

# Darwin ARM64 (macOS Apple Silicon)
echo -e "${YELLOW}Building for macOS ARM64...${NC}"
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build \
    -ldflags="$LDFLAGS" \
    -o "$BUILD_DIR/${BINARY_NAME}-darwin-arm64" \
    ./main.go

# Windows AMD64
echo -e "${YELLOW}Building for Windows AMD64...${NC}"
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build \
    -ldflags="$LDFLAGS" \
    -o "$BUILD_DIR/${BINARY_NAME}-windows-amd64.exe" \
    ./main.go

# Create a default binary for current platform
echo -e "${YELLOW}Creating default binary...${NC}"
go build -ldflags="$LDFLAGS" -o "$BUILD_DIR/$BINARY_NAME" ./main.go

# Show build results
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${YELLOW}Built binaries:${NC}"
ls -la $BUILD_DIR/

# Calculate binary sizes
echo -e "${YELLOW}Binary sizes:${NC}"
for file in $BUILD_DIR/*; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo -e "  $(basename "$file"): $size"
    fi
done

echo -e "${GREEN}Build script completed successfully!${NC}"